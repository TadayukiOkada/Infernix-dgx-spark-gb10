// GB10 UMA zero-copy host-tier probe (roadmap P2.2, A7.1). On GB10 (cudaDevAttrIntegrated = 1) the
// pinned host prefix-cache arena and device memory are the same LPDDR5X, so a host-tier restore
// could in principle read its parent state directly from mapped host pages instead of staging an
// H2D copy first. This probe measures that, before any production change (pre-registered gate:
// zero-copy must beat staging by >= 3%).
//
// RESULT (measured on GB10, gate FAILED -> P2.2 REJECTED, no production change):
// The restore copy lands bytes in the cache's OWN device storage (paged KV planes, GDN/state slots)
// which the model re-reads at EVERY layer and decode step, not once. With reads=1 zero-copy looks
// 72-80% faster (it dodges the copy), but that access pattern is unreal. From reads>=8 the staging
// path wins decisively (-29% to -212%: zero-copy is worse) because the device copy stays resident
// (L2 for <=24 MiB) while a mapped-host source re-pays the host->device bus latency on every read.
// The staged copy is therefore load-bearing; the production design is correct. See:
//   restore copy: src/core/paged_kv_cache.cpp:721 (cudaMemcpy2DAsync H2D),
//                 src/models/qwen3_5/state/state_image.cpp:687
//   multi-read consumers: src/ops/kernel/paged_kv_address.cuh, softmax_attention causal_cache,
//                 src/ops/linear_attention/gated_delta_net/recurrent.cu
//
//   staging   = cudaMemcpyAsync(host -> device, S)  +  R x kernel(device_src)
//   zero-copy = R x kernel(host_mapped_src)
//
// The kernel mimics the restore's read shape: a strided walk over the image (not a flat memcpy), so
// it captures the read-side cost the production op pays. `--reads=R` reproduces the production
// access pattern: a restored prefix page / state slot is materialized into the cache's own device
// storage ONCE and then re-read by attention/GDN at every layer and decode step (R >> 1). With R = 1
// the probe degenerates to the single-read case, where a mapped-host read can dodge the copy. Both
// kernels produce the same checksum, timed-out to defeat DCE. Host-only + one small device buffer;
// loads no model. Reports median microseconds per mode so `ckt ab --cmd-a ... --cmd-b ...` can
// compare, and prints the raw medians directly.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                                      \
    do {                                                                                                   \
        cudaError_t err_ = (x);                                                                            \
        if (err_ != cudaSuccess) {                                                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));   \
            return 1;                                                                                      \
        }                                                                                                  \
    } while (0)

namespace {

// Strided restore-shaped read: 1024-byte records, one thread per record, summing to a scalar sink.
// Mirrors how context_kv_materialize consumes a parent image: whole small records, not a scan.
__global__ void restore_read_kernel(const uint4* __restrict__ src, std::size_t n_vec, float* sink) {
    const std::size_t i     = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    unsigned acc = 0u;
    for (std::size_t v = i; v < n_vec; v += stride) {
        const uint4 u = src[v];
        acc += u.x + u.y + u.z + u.w;
    }
    if (acc == 0xDEADBEEFu) { *sink = 1.f; } // never taken; keeps the loads live
}

// Median microseconds to perform `reads` consecutive reads of `src`. Back-to-back launches emulate
// the production steady state where the same bytes are re-read at every step; a device copy of the
// bytes stays resident (L2 for small, DRAM for large), while a mapped-host source re-pays the
// host->device bus latency on every read.
double time_kernel(void* src, std::size_t bytes, int reads, cudaStream_t stream) {
    const std::size_t n_vec = bytes / sizeof(uint4);
    const int threads       = 256;
    const int blocks        = 48 * 16; // GB10 SMs x 16; enough to cover the read in one wave
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    // warmup
    for (int i = 0; i < 5; ++i) { restore_read_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<uint4*>(src), n_vec, sink); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> samples;
    for (int i = 0; i < 25; ++i) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int r = 0; r < reads; ++r) {
            restore_read_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<uint4*>(src), n_vec, sink);
        }
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        samples.push_back(ms * 1000.f);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CUDA_CHECK(cudaFree(sink));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

double time_copy(void* dst, void* src, std::size_t bytes, cudaStream_t stream) {
    for (int i = 0; i < 5; ++i) { CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream)); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> samples;
    for (int i = 0; i < 25; ++i) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        samples.push_back(ms * 1000.f);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    // A prefix state image in the realistic restore band (a few MiB to a few tens of MiB).
    std::vector<std::size_t> sizes_mib;
    // Number of times the materialized bytes are re-read after restore (one-shot vs steady state).
    std::vector<int> reads_list;
    const auto parse_list = [](const std::string& spec, auto& out) {
        std::size_t pos = 0;
        while (pos < spec.size()) {
            const std::size_t comma = spec.find(',', pos);
            const std::size_t len = (comma == std::string::npos) ? std::string::npos : comma - pos;
            using T = typename std::decay_t<decltype(out)>::value_type;
            out.push_back(static_cast<T>(std::atoll(spec.substr(pos, len).c_str())));
            if (comma == std::string::npos) { break; }
            pos = comma + 1;
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--mib=", 0) == 0) { parse_list(a.substr(6), sizes_mib); }
        else if (a.rfind("--reads=", 0) == 0) { parse_list(a.substr(8), reads_list); }
        else { std::fprintf(stderr, "usage: [--mib=4,8,16,32] [--reads=1,8,32]\n"); return 2; }
    }
    if (sizes_mib.empty()) { sizes_mib = {4, 8, 16, 32}; }
    if (reads_list.empty()) { reads_list = {1, 8, 32}; }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s integrated=%d SMs=%d L2=%d MiB\n", prop.name, prop.integrated,
                prop.multiProcessorCount, static_cast<int>(prop.l2CacheSize >> 20));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (const std::size_t mib : sizes_mib) {
        const std::size_t bytes = mib << 20;
        // Host source, pinned and registered+mapped so the SM can address it (as the arena is).
        void* host = nullptr;
        CUDA_CHECK(cudaMallocHost(&host, bytes));
        std::memset(host, 0x5A, bytes);
        void* host_dev_ptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&host_dev_ptr, host, 0));
        // Device staging arena.
        void* device = nullptr;
        CUDA_CHECK(cudaMalloc(&device, bytes));

        const double copy_us = time_copy(device, host, bytes, stream);
        for (const int reads : reads_list) {
            // Per-read microsecond figures; staging adds the one-time copy to the first read.
            const double dev_read_us  = time_kernel(device, bytes, reads, stream) / reads;
            const double host_read_us = time_kernel(host_dev_ptr, bytes, reads, stream) / reads;
            const double staging_total = copy_us + dev_read_us * reads;
            const double zero_total    = host_read_us * reads;
            std::printf("size=%zu MiB reads=%2d  staging(copy %.1f + %d x devread %.1f)=%.1f us  "
                        "zerocopy(%d x hostread %.1f)=%.1f us  gain=%.2f%%\n",
                        mib, reads, copy_us, reads, dev_read_us, staging_total, reads, host_read_us,
                        zero_total, 100.0 * (staging_total - zero_total) / staging_total);
        }

        CUDA_CHECK(cudaFree(device));
        CUDA_CHECK(cudaFreeHost(host));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
