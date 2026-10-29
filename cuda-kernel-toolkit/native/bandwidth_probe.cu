// bandwidth_probe.cu — sustained memory bandwidth and clock measurement.
//
// Derives the *measured* streaming bandwidth ceiling instead of trusting
// (bus_width * mem_clock) arithmetic.  On unified-memory parts such as GB10
// DGX Spark, `memory_bus_width`/`memory_clock_rate_khz` are reported by the
// runtime but do NOT describe the LPDDR5X system bus: the product spec (273
// GB/s) is roughly half of the naive calculation.  The toolkit therefore
// reports both numbers, clearly labelled, and uses the measured one for
// roofline efficiency.
//
// Three kernels, each with a known byte footprint:
//   read    : streaming read of uint4, XOR-accumulated, one byte written back
//   copy    : streaming read + write (2x footprint)
//   write   : streaming write of uint4
//
// Output: key<TAB>value records on stdout.
//
// Build: nvcc -arch=<detected> -O2 -std=c++17 -o bandwidth_probe bandwidth_probe.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(e_));                \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {

constexpr int kBlock = 256;

__global__ void k_read(const uint4* __restrict__ src, unsigned long long n_vec,
                       float* __restrict__ sink) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;  // never taken; prevents DCE
}

__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst,
                       unsigned long long n_vec) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n_vec; i += stride) dst[i] = src[i];
}

__global__ void k_write(uint4* __restrict__ dst, unsigned long long n_vec, unsigned int seed) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const uint4 v = {seed, seed, seed, seed};
    for (; i < n_vec; i += stride) dst[i] = v;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Timing {
    double median_us;
    double p10_us;
    double p90_us;
};

template <class Body>
Timing time_launch(Body&& body, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) body();
    CK(cudaDeviceSynchronize());
    std::vector<double> us;
    us.reserve(reps);
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(s));
        body();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, s, e));
        us.push_back((double)ms * 1000.0);
    }
    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    std::sort(us.begin(), us.end());
    auto pct = [&](double p) {
        if (us.size() == 1) return us[0];
        double rank = p / 100.0 * (us.size() - 1);
        size_t lo = (size_t)rank, hi = std::min(lo + 1, us.size() - 1);
        double f = rank - lo;
        return us[lo] * (1.0 - f) + us[hi] * f;
    };
    return {pct(50), pct(10), pct(90)};
}

// Sweeps block counts and threads, keeping the best sustained rate.
struct Best {
    double gbps = 0.0;
    int blocks = 0;
    int threads = 0;
    double us = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    int reps = 30, warmup = 5;
    size_t footprint_mb = 1024;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mb") == 0 && i + 1 < argc) footprint_mb = (size_t)std::atoll(argv[++i]);
    }

    cudaDeviceProp p{};
    CK(cudaSetDevice(0));
    CK(cudaGetDeviceProperties(&p, 0));
    const int sms = p.multiProcessorCount;

    std::printf("probe.schema\t1\n");
    std::printf("bandwidth.footprint_mb\t%zu\n", footprint_mb);
    std::printf("bandwidth.reps\t%d\n", reps);
    std::printf("bandwidth.warmup\t%d\n", warmup);

    // Naive bus arithmetic, clearly labelled as such and reported alongside the
    // measured value so a disagreement is visible instead of hidden.
    {
        int mem_khz = 0;
        if (cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0) == cudaSuccess && mem_khz > 0) {
            double naive = 2.0 * (double)mem_khz * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
            std::printf("bandwidth.naive_bus_gbps\t%.1f\n", naive);
        }
    }

    const size_t bytes = footprint_mb << 20;
    const unsigned long long n_vec = bytes / sizeof(uint4);
    uint4 *a = nullptr, *b = nullptr;
    float* sink = nullptr;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMalloc(&sink, sizeof(float)));
    CK(cudaMemset(a, 0x3f, bytes));
    CK(cudaMemset(b, 0x3f, bytes));
    CK(cudaDeviceSynchronize());

    const int thread_choices[] = {256, 512, 1024};
    const int mult_choices[] = {1, 2, 4, 8, 16, 24};

    // ---- read ----
    Best br;
    for (int t : thread_choices) {
        for (int m : mult_choices) {
            int blocks = sms * m;
            auto body = [&] { k_read<<<blocks, t>>>(a, n_vec, sink); };
            Timing tm = time_launch(body, warmup, reps);
            double gbps = (double)bytes / (tm.median_us * 1e-6) / 1e9;
            if (gbps > br.gbps) br = {gbps, blocks, t, tm.median_us};
        }
    }

    // ---- copy ----
    Best bc;
    for (int t : thread_choices) {
        for (int m : mult_choices) {
            int blocks = sms * m;
            auto body = [&] { k_copy<<<blocks, t>>>(a, b, n_vec); };
            Timing tm = time_launch(body, warmup, reps);
            double gbps = 2.0 * (double)bytes / (tm.median_us * 1e-6) / 1e9;
            if (gbps > bc.gbps) bc = {gbps, blocks, t, tm.median_us};
        }
    }

    // ---- write ----
    Best bw;
    for (int t : thread_choices) {
        for (int m : mult_choices) {
            int blocks = sms * m;
            auto body = [&] { k_write<<<blocks, t>>>(b, n_vec, 0x3f3f3f3fu); };
            Timing tm = time_launch(body, warmup, reps);
            double gbps = (double)bytes / (tm.median_us * 1e-6) / 1e9;
            if (gbps > bw.gbps) bw = {gbps, blocks, t, tm.median_us};
        }
    }

    std::printf("bandwidth.read.gbps\t%.1f\n", br.gbps);
    std::printf("bandwidth.read.blocks\t%d\n", br.blocks);
    std::printf("bandwidth.read.threads\t%d\n", br.threads);
    std::printf("bandwidth.read.us\t%.4f\n", br.us);
    std::printf("bandwidth.copy.gbps\t%.1f\n", bc.gbps);
    std::printf("bandwidth.copy.blocks\t%d\n", bc.blocks);
    std::printf("bandwidth.copy.threads\t%d\n", bc.threads);
    std::printf("bandwidth.write.gbps\t%.1f\n", bw.gbps);
    std::printf("bandwidth.write.blocks\t%d\n", bw.blocks);
    std::printf("bandwidth.write.threads\t%d\n", bw.threads);
    // The read path is the meaningful ceiling for memory-bound kernels that
    // stream weights/activations; report it as the single recommended constant.
    std::printf("bandwidth.sustained_read_gbps\t%.1f\n", br.gbps);

    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(sink));
    return 0;
}
