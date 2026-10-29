// Peak-bandwidth probe for NVIDIA GB10 Grace Blackwell (DGX Spark).
// Measures achieved read-only streaming bandwidth with maximum parallelism.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#define CUDA_CHECK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

// Pure read-streaming kernel: each thread reads uint4 chunks, sums them into a register,
// writes one byte back at the very end (to prevent dead-code elimination).
__global__ void read_stream_kernel(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        // #pragma unroll 4
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) { *sink = 1.f; } // never taken, prevents DCE
}

int main(int argc, char** argv) {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s  SM=%d  shared_per_sm=%zu KiB\n",
                prop.name, prop.multiProcessorCount, prop.sharedMemPerMultiprocessor/1024);

    // Sizes to sweep: large enough to saturate, small enough to stay in DRAM (not L2-resident)
    const size_t sizes_mb[] = {512, 1024, 2048};
    for (size_t sz_mb : sizes_mb) {
        size_t bytes = sz_mb << 20;
        size_t n_vec = bytes / sizeof(uint4);
        uint4* d_src = nullptr;
        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, bytes));
        CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
        std::vector<uint4> h(bytes / sizeof(uint4));
        for (size_t i = 0; i < h.size(); ++i) { h[i] = {static_cast<unsigned>(i),0,0,0}; }
        CUDA_CHECK(cudaMemcpy(d_src, h.data(), bytes, cudaMemcpyHostToDevice));

        // Sweep block counts to find the sweet spot
        struct Result { int blocks; int threads; double gb_s; };
        std::vector<Result> results;
        for (int threads : {256, 512, 1024}) {
            for (int mult : {1, 2, 4, 8, 16, 32}) {
                int blocks = prop.multiProcessorCount * mult;
                if (blocks * threads > 65535*8) continue; // sanity cap
                // Warmup
                auto kern = [&](int b, int t){ read_stream_kernel<<<b,t>>>(d_src, n_vec, d_sink); };
                for (int w = 0; w < 3; ++w) kern(blocks, threads);
                CUDA_CHECK(cudaDeviceSynchronize());
                // Measured runs
                constexpr int REPS = 5;
                double total_ms = 0;
                for (int r = 0; r < REPS; ++r) {
                    cudaEvent_t s_, e_;
                    CUDA_CHECK(cudaEventCreate(&s_)); CUDA_CHECK(cudaEventCreate(&e_));
                    CUDA_CHECK(cudaEventRecord(s_));
                    kern(blocks, threads);
                    CUDA_CHECK(cudaEventRecord(e_));
                    CUDA_CHECK(cudaEventSynchronize(e_));
                    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s_, e_));
                    total_ms += ms;
                    CUDA_CHECK(cudaEventDestroy(s_)); CUDA_CHECK(cudaEventDestroy(e_));
                }
                double avg_ms = total_ms / REPS;
                double gb_s = (double)bytes / (avg_ms * 1e-3) / 1e9;
                results.push_back({blocks, threads, gb_s});
            }
        }
        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b){ return a.gb_s > b.gb_s; });
        std::printf("\nSize %zu MiB — top configs:\n", sz_mb);
        for (auto& r : results) {

            std::printf("  blocks=%5d threads=%4d  →  %.1f GB/s   (%.0f%% of 273 spec)\n",
                        r.blocks, r.threads, r.gb_s, r.gb_s/273.0*100);
        }
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_sink));
    }
    return 0;
}
