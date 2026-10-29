// memory.cu — Layer 2A: streaming memory bandwidth ceilings.
//
// Establishes what the GB10 memory system can actually sustain, which is the
// number that bounds every elementwise/reduction/quantization kernel later.
// Sweeps working-set size from L2-resident up through DRAM so cache behaviour
// and sustained bandwidth are separated, and covers read / write / copy /
// triad / gather patterns plus vector-width variants.
#include "gb10_lab/common.h"

namespace {

// ---- Kernels --------------------------------------------------------------

__global__ void k_read(const uint4* __restrict__ src, size_t n_vec, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;   // never taken; prevents dead-code elimination
}

__global__ void k_write(uint4* __restrict__ dst, size_t n_vec) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const uint4 val = make_uint4(0x1234ull, 0x5678ull, 0x9ABCull, 0xDEF0ull);
    for (; i < n_vec; i += stride) dst[i] = val;
}

__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n_vec) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n_vec; i += stride) dst[i] = src[i];
}

__global__ void k_triad(const float4* __restrict__ a, const float4* __restrict__ b,
                        float4* __restrict__ c, float scale, size_t n_vec) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n_vec; i += stride) {
        float4 av = a[i], bv = b[i], cv;
        cv.x = scale * av.x + bv.x; cv.y = scale * av.y + bv.y;
        cv.z = scale * av.z + bv.z; cv.w = scale * av.w + bv.w;
        c[i] = cv;
    }
}

// Gather with a fixed stride in elements — measures how much strided access
// degrades effective bandwidth versus contiguous.
__global__ void k_gather_strided(const uint4* __restrict__ src, size_t n_elem_u4,
                                 int stride_u4, float* sink) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned acc = 0u;
    for (; i < n_elem_u4; i += total) {
        size_t idx = i * stride_u4;
        if (idx >= n_elem_u4) break;
        uint4 v = src[idx];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = 1.f;
}

// ---- Harness helpers ------------------------------------------------------

struct MemConfig {
    const char* pattern;     // "read","write","copy","triad","gather"
    size_t bytes;            // logical working set
    double logical_bytes;    // actual traffic counted for GB/s
    int blocks;              // grid size
    int threads;             // block size
    int stride_u4;           // gather stride (ignored otherwise)
};

// Run one configuration, returning median microseconds and validating nothing
// (bandwidth kernels have no reference output to compare against).
double run_one(const MemConfig& cfg, int warmup, int iters) {
    const DeviceInfo& d = DeviceInfo::get();
    void* buf_a = nullptr; void* buf_b = nullptr; float* sink = nullptr;
    LAB_CHECK(cudaMalloc(&buf_a, cfg.bytes));
    if (std::strcmp(cfg.pattern, "copy") == 0 || std::strcmp(cfg.pattern, "triad") == 0)
        LAB_CHECK(cudaMalloc(&buf_b, cfg.bytes));
    LAB_CHECK(cudaMalloc(&sink, sizeof(float)));
    // Touch the buffers once so first-touch page effects are out of timing.
    LAB_CHECK(cudaMemset(buf_a, 0x11, cfg.bytes));
    if (buf_b) LAB_CHECK(cudaMemset(buf_b, 0x22, cfg.bytes));
    LAB_CHECK(cudaDeviceSynchronize());

    const uint4* ua = static_cast<const uint4*>(buf_a);
    uint4* va       = static_cast<uint4*>(buf_a);
    const uint4* ub = static_cast<const uint4*>(buf_b);
    uint4* vb       = static_cast<uint4*>(buf_b);
    const float4* fa = reinterpret_cast<const float4*>(buf_a);
    const float4* fb = reinterpret_cast<const float4*>(buf_b);
    float4* fc       = reinterpret_cast<float4*>(buf_b ? buf_b : buf_a);
    const size_t n_vec   = cfg.bytes / sizeof(uint4);
    const size_t n_fvec  = cfg.bytes / sizeof(float4);

    auto launch = [&](cudaStream_t stream) {
        switch (cfg.pattern[0]) {
            case 'r': k_read<<<cfg.blocks, cfg.threads, 0, stream>>>(ua, n_vec, sink); break;
            case 'w': k_write<<<cfg.blocks, cfg.threads, 0, stream>>>(va, n_vec); break;
            case 'c': k_copy<<<cfg.blocks, cfg.threads, 0, stream>>>(ua, vb, n_vec); break;
            case 't': k_triad<<<cfg.blocks, cfg.threads, 0, stream>>>(fa, fb, fc, 0.5f, n_fvec); break;
            case 'g': k_gather_strided<<<cfg.blocks, cfg.threads, 0, stream>>>(ua, n_vec, cfg.stride_u4, sink); break;
            default: break;
        }
    };

    std::vector<double> us = timed_iterations(launch, warmup, iters);
    const double med = percentile(us, 50.0);

    LAB_CHECK(cudaFree(buf_a));
    if (buf_b) LAB_CHECK(cudaFree(buf_b));
    LAB_CHECK(cudaFree(sink));
    return med;
}

} // namespace

void run_memory(const LabArgs& args) {
    const DeviceInfo& d = DeviceInfo::get();
    const size_t l2 = d.l2_bytes;
    const int sm = d.sm_count;

    // Working-set sizes spanning L2-resident through several multiples of DRAM.
    std::vector<std::size_t> sizes;
    sizes.push_back(std::max<size_t>(l2 / 2, 8ull << 20));   // ~L2/2
    sizes.push_back(l2);                                     // ~L2
    sizes.push_back(4ull << 20);                             // small streaming
    sizes.push_back(64ull << 20);                            // exceeds L2
    sizes.push_back(256ull << 20);                           // deep DRAM
    sizes.push_back(1024ull << 20);                          // large DRAM
    if (args.quick) { sizes = {64ull << 20, 256ull << 20}; }

    // Grid sizing: enough blocks to keep every SM busy with several warps each.
    const int threads = 256;
    const int blocks_per_sm_candidates[] = {4, 8, 16, 32};

    for (const char* pattern : {"read", "write", "copy", "triad"}) {
        double best_gb = 0.0;
        for (size_t bytes : sizes) {
            // Align down to a multiple of 16 bytes.
            bytes &= ~(size_t(15));
            const double logical = [&]() {
                if (!std::strcmp(pattern, "read"))  return (double)bytes;
                if (!std::strcmp(pattern, "write")) return (double)bytes;
                if (!std::strcmp(pattern, "copy"))  return (double)(2 * bytes);
                return (double)(3 * bytes);   // triad: 2 reads + 1 write
            }();

            double best_cfg_us = 1e30;
            int best_blocks = 0;
            for (int mult : blocks_per_sm_candidates) {
                int blocks = sm * mult;
                MemConfig cfg{pattern, bytes, logical, blocks, threads, 1};
                const double us = run_one(cfg, args.warmup, args.iters);
                if (us < best_cfg_us) { best_cfg_us = us; best_blocks = blocks; }
            }
            const double gb_s = logical / (best_cfg_us * 1e-6) / 1e9;
            best_gb = std::max(best_gb, gb_s);

            JsonLine l;
            l.kv("op", "memory")
             .kv("pattern", pattern)
             .append_fragment(",\"working_set_bytes\":" + fmt_num((double)bytes))
             .append_fragment(",\"logical_bytes\":" + fmt_num(logical))
             .append_fragment(",\"blocks\":" + fmt_num((double)best_blocks))
             .append_fragment(",\"threads\":" + fmt_num((double)threads))
             .append_fragment(",\"median_us\":" + fmt_num(best_cfg_us))
             .append_fragment(",\"gb_per_s\":" + fmt_num(gb_s))
             .append_fragment(",\"pct_of_273\":" + fmt_num(gb_s / 273.0 * 100.0));
            emit_record(args, l);
        }
        std::printf("  %-6s peak %.1f GB/s (%.0f%% of 273 spec)\n", pattern, best_gb,
                    best_gb / 273.0 * 100.0);
    }

    // Gather/stride degradation at one representative DRAM size.
    size_t gbytes = args.quick ? (128ull << 20) : (256ull << 20);
    gbytes &= ~(size_t(15));
    for (int stride : {1, 2, 4, 8}) {
        int blocks = sm * 16;
        MemConfig cfg{"gather", gbytes, (double)gbytes, blocks, threads, stride};
        const double us = run_one(cfg, args.warmup, args.iters);
        const double gb_s = (double)gbytes / (us * 1e-6) / 1e9;
        JsonLine l;
        l.kv("op", "memory").kv("pattern", "gather")
         .append_fragment(",\"stride_u4\":" + fmt_num((double)stride))
         .append_fragment(",\"working_set_bytes\":" + fmt_num((double)gbytes))
         .append_fragment(",\"median_us\":" + fmt_num(us))
         .append_fragment(",\"gb_per_s\":" + fmt_num(gb_s));
        emit_record(args, l);
        std::printf("  gather x%-2d %.1f GB/s\n", stride, gb_s);
    }

    std::printf("\nMemory ceilings complete.\n");
}
