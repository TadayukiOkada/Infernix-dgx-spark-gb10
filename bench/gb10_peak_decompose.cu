// Decomposition: what exactly produces 90% of peak DRAM bandwidth?
// Tests each factor independently: occupancy, MLP (unroll depth), writes, access pattern.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

// ── Variant A: grid-stride, compiler-controlled unroll (the original probe) ──
__global__ void kernel_a(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ── Variant B: fixed unroll depth via #pragma, explicit index arithmetic ──
template <int UNROLL>
__global__ void kernel_b(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x * UNROLL + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * UNROLL;
    unsigned int acc[UNROLL] = {};
    size_t limit = n_vec - (UNROLL - 1);
    for (size_t i = base; i < limit; i += stride) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            uint4 v = src[i + u];
            acc[u] += v.x + v.y + v.z + v.w;
        }
    }
    unsigned int total = 0;
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) total += acc[u];
    if (total == 0xDEADBEEFu) *sink = 1.f;
}

// ── Variant C: same as A but WITH writes (read+write streaming) ──
__global__ void kernel_c(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
        dst[i] = v;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ── Variant D: cp.async into shared memory, then read from smem (like MoE kernels) ──
__global__ void kernel_d(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    extern __shared__ __align__(16) uint4 smem[];
    constexpr int kChunk = 256; // uint4 per chunk = 4 KiB
    const size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (size_t i = base; i < n_vec; i += stride) {
        if (threadIdx.x < kChunk && i + threadIdx.x < n_vec) {
            unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(smem + threadIdx.x));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(src + i + threadIdx.x) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        if (threadIdx.x < kChunk && i + threadIdx.x < n_vec) {
            uint4 v = smem[threadIdx.x];
            acc += v.x + v.y + v.z + v.w;
        }
        __syncthreads();
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ── Variant E: ld.global.nc.L1::no_allocate (exact PTX from inference kernels) ──
__device__ __forceinline__ uint4 ld_nc_na_uint4(const uint4* p) {
    uint4 value;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) : "l"(p));
    return value;
}
__global__ void kernel_e(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = ld_nc_na_uint4(src + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ── Variant F: one load per thread, zero MLP within thread ──
__global__ void kernel_f(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int acc = 0u;
    if (i < n_vec) {
        uint4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ═══════════════ Benchmark harness ═══════════════

template <typename LaunchFn>
double bench(LaunchFn&& launch_fn, size_t bytes) {
    for (int w = 0; w < 5; ++w) launch_fn();
    CK(cudaDeviceSynchronize());
    constexpr int REPS = 7; double total_ms = 0;
    for (int r = 0; r < REPS; ++r) {
        cudaEvent_t s_, e_; CK(cudaEventCreate(&s_)); CK(cudaEventCreate(&e_));
        CK(cudaEventRecord(s_)); launch_fn(); CK(cudaEventRecord(e_));
        CK(cudaEventSynchronize(e_));
        float ms = 0; CK(cudaEventElapsedTime(&ms, s_, e_));
        total_ms += ms;
        CK(cudaEventDestroy(s_)); CK(cudaEventDestroy(e_));
    }
    return (double)bytes / (total_ms / REPS * 1e-3) / 1e9;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s  SM=%d\n", prop.name, prop.multiProcessorCount);

    const size_t MB = 1024 << 20;
    size_t bytes = 256 * MB;  // 256 MiB buffer — well above L2 (~128 MiB) to force DRAM traffic
    size_t n_vec = bytes / sizeof(uint4);
    uint4 *d_src, *d_dst; float* d_sink;
    CK(cudaMalloc(&d_src, bytes)); CK(cudaMalloc(&d_dst, bytes)); CK(cudaMalloc(&d_sink, 4));
    CK(cudaMemset(d_src, 0x42, bytes));  // fill without host-side vector

    // Sweep over thread counts and CTA multiples for each variant
    struct Config { int threads; int mult; };
    const Config configs[] = {{128, 1}, {128, 4}, {128, 8}, {256, 1}, {256, 2}, {256, 4}, {256, 8},
                              {512, 1}, {512, 2}, {512, 4}, {1024, 1}, {1024, 2}};

    std::printf("\n%-10s %-8s %-8s | %8s %8s %8s %8s %8s %8s\n",
                "variant", "threads", "mult", "A_grid", "B_u2", "B_u4", "B_u8", "E_ldncna", "C_r+w");
    std::printf("%s\n", std::string(78, '-').c_str());

    for (auto& cfg : configs) {
        int blocks = prop.multiProcessorCount * cfg.mult;
        int t = cfg.threads;

        double a = bench([&]{ kernel_a<<<blocks, t>>>(d_src, n_vec, d_sink); }, bytes);
        double b2 = bench([&]{ kernel_b<2><<<blocks, t>>>(d_src, n_vec, d_sink); }, bytes);
        double b4 = bench([&]{ kernel_b<4><<<blocks, t>>>(d_src, n_vec, d_sink); }, bytes);
        double b8 = bench([&]{ kernel_b<8><<<blocks, t>>>(d_src, n_vec, d_sink); }, bytes);
        double e = bench([&]{ kernel_e<<<blocks, t>>>(d_src, n_vec, d_sink); }, bytes);
        double c = bench([&]{ kernel_c<<<blocks, t>>>(d_src, d_dst, n_vec, d_sink); }, bytes);

        std::printf("%-10s %-8d %-8d | %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
                    "read-only", t, cfg.mult, a, b2, b4, b8, e, c);
    }

    // Specialized tests at the sweet spot (blocks=4×SM, threads=256)
    int sb = prop.multiProcessorCount * 4, st = 256;
    std::printf("\n=== Detailed comparison at blocks=%d threads=%d ===\n", sb, st);

    double a = bench([&]{ kernel_a<<<sb, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  A grid-stride (compiler unroll):   %7.1f GB/s\n", a);

    double b2 = bench([&]{ kernel_b<2><<<sb, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  B explicit #pragma unroll 2:       %7.1f GB/s\n", b2);

    double b4 = bench([&]{ kernel_b<4><<<sb, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  B explicit #pragma unroll 4:       %7.1f GB/s\n", b4);

    double b8 = bench([&]{ kernel_b<8><<<sb, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  B explicit #pragma unroll 8:       %7.1f GB/s\n", b8);

    double e = bench([&]{ kernel_e<<<sb, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  E ld.global.nc.L1::no_allocate:    %7.1f GB/s\n", e);

    double c = bench([&]{ kernel_c<<<sb, st>>>(d_src, d_dst, n_vec, d_sink); }, bytes);
    std::printf("  C read+write streaming:            %7.1f GB/s total traffic\n", c);

    size_t smem_bytes = 256 * sizeof(uint4);
    double d_bw = bench([&]{ kernel_d<<<sb, st, smem_bytes>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  D cp.async->smem->read:            %7.1f GB/s\n", d_bw);

    double f = bench([&]{ kernel_f<<<sb * 32, st>>>(d_src, n_vec, d_sink); }, bytes);
    std::printf("  F one load/thread (no loop):       %7.1f GB/s\n", f);

    // Very high occupancy with few threads per block
    for (int ft : {32, 64}) {
        int fb = prop.multiProcessorCount * 32;
        double ff = bench([&]{ kernel_f<<<fb, ft>>>(d_src, n_vec, d_sink); }, bytes);
        std::printf("  F one load/thread threads=%d:     %7.1f GB/s\n", ft, ff);
    }

    CK(cudaFree(d_src)); CK(cudaFree(d_dst)); CK(cudaFree(d_sink));
    return 0;
}
