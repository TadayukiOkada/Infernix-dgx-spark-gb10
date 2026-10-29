// Persistent-L2 efficacy probe for GB10 decode patterns. No model weights.
//
// Question: does a persisting access-policy window protect a target region from being evicted by
// the ~4 GB/token dense-weight flood? Persistence only helps if a pinned byte is READ MORE THAN
// ONCE while resident. We test two patterns:
//   ROTATING  = each round reads a DIFFERENT slice of the pool (models MoE: routed experts rotate).
//   FIXED     = every round reads the SAME region (models reusable weights: hc/shared/router).
// Each round then streams a large FLOOD to simulate dense weights evicting normal L2.
// Compare target-region read time with vs without a persisting access-policy window.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)

// Read a [base, base+span) uint4 range. Each CTA writes ONE partial sum to its own slot (no global
// atomic contention), so the kernel is bandwidth-bound, not serialized by atomics.
__global__ void read_reduce(const uint4* __restrict__ p, std::size_t n_vec, unsigned long long* out) {
    unsigned long long acc = 0;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
        uint4 v = p[i];
        acc += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w;
    }
    out[blockIdx.x] = acc; // per-CTA slot; read back never happens, keeps loads live
}

// Time ONLY the target read kernels across `rounds`, optionally interleaving a flood between the
// two reads of each round. Returns total ms spent on target reads only.
static double time_target_only(const uint4* target, std::size_t target_vec, const uint4* flood, std::size_t flood_vec,
                               int rounds, std::size_t pool_vec, bool rotating, bool flood_between, cudaStream_t s) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    unsigned long long* sink; CK(cudaMalloc(&sink, sizeof(unsigned long long) * 8192));
    const int CTAS = 4096, THR = 256;
    read_reduce<<<CTAS, THR, 0, s>>>(target, target_vec, sink);
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int r = 0; r < rounds; ++r) {
        const uint4* t = rotating ? (pool_vec ? target + (static_cast<std::size_t>(r) * target_vec) % pool_vec : target)
                                  : target;
        read_reduce<<<CTAS, THR, 0, s>>>(t, target_vec, sink);           // first read
        if (flood_between) read_reduce<<<CTAS, THR, 0, s>>>(flood, flood_vec, sink);
        read_reduce<<<CTAS, THR, 0, s>>>(t, target_vec, sink);           // second read (the reuse we measure)
    }
    CK(cudaEventRecord(b, s));
    CK(cudaStreamSynchronize(s));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(sink);
    return static_cast<double>(ms); // includes floods if flood_between
}

// Time a single read_reduce over the target region only (no flood), averaged over rounds.
static double time_single_read(const uint4* target, std::size_t target_vec, int rounds, cudaStream_t s) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    unsigned long long* sink; CK(cudaMalloc(&sink, sizeof(unsigned long long) * 8192));
    const int CTAS = 4096, THR = 256;
    read_reduce<<<CTAS, THR, 0, s>>>(target, target_vec, sink);
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int r = 0; r < rounds; ++r) read_reduce<<<CTAS, THR, 0, s>>>(target, target_vec, sink);
    CK(cudaEventRecord(b, s));
    CK(cudaStreamSynchronize(s));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(sink);
    return static_cast<double>(ms) / rounds;
}

static int set_window(const void* base, std::size_t bytes, bool persist, cudaStream_t s) {
    if (!persist) {
        cudaStreamAttrValue v{}; v.accessPolicyWindow.base_ptr = nullptr; v.accessPolicyWindow.num_bytes = 0;
        CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v));
        CK(cudaCtxResetPersistingL2Cache());
        return 0;
    }
    cudaStreamAttrValue v{};
    v.accessPolicyWindow.base_ptr  = const_cast<void*>(base);
    v.accessPolicyWindow.num_bytes = bytes;
    v.accessPolicyWindow.hitRatio  = 1.0f;
    v.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v));
    return 0;
}

int main() {
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0));
    const std::size_t carve = static_cast<std::size_t>(p.persistingL2CacheMaxSize);
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve));
    std::printf("GB10 L2 %.1f MiB, persisting carveout %.1f MiB\n", p.l2CacheSize/1048576.0, carve/1048576.0);

    const std::size_t MiB = 1u<<20;
    const std::size_t target_bytes = carve;              // pin exactly the carveout (18 MiB)
    const std::size_t pool_bytes   = 3u * carve;         // rotating pool larger than carveout
    const std::size_t flood_bytes  = 256u * MiB;         // dense-weight flood >> L2
    const std::size_t target_vec = target_bytes/16, pool_vec = pool_bytes/16, flood_vec = flood_bytes/16;

    uint4 *target, *flood;
    CK(cudaMalloc(&target, pool_bytes));                 // rotating reads inside one big pool
    CK(cudaMalloc(&flood, flood_bytes));
    CK(cudaMemset(target, 1, pool_bytes)); CK(cudaMemset(flood, 1, flood_bytes));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int R = 40;

    // Baseline: single cold-ish read of the target (what a once-per-token expert read costs).
    set_window(target, 0, false, s);
    double cold = time_single_read(target, target_vec, R, s);
    std::printf("single target read (DRAM)        : %.4f ms  (%.0f GB/s)\n",
                cold, static_cast<double>(target_bytes)/(cold*1e-3)/1e9);

    auto report = [&](const char* name, bool rotating, bool persist, bool flood_between){
        if (persist) set_window(target, target_bytes, true, s); else set_window(target, 0, false, s);
        double ms = time_target_only(target, target_vec, flood, flood_vec, R,
                                     rotating?pool_vec:0, rotating, flood_between, s);
        // Per-round cost of the SECOND (reuse) read = total minus one full read per round baseline.
        double per_round = ms / R;
        std::printf("%-28s : %.4f ms/round  (target=%zu MiB %s persist=%d flood=%d)\n",
                    name, per_round, target_bytes/MiB, rotating?"rotate":"fixed ", persist, flood_between);
        set_window(target, 0, false, s);
    };

    // Best case: re-read same fixed region with NO eviction -> should hit L2 fully.
    report("FIXED reread no-flood",  false, false, false);
    report("FIXED reread +FLOOD",    false, false, true);
    report("FIXED reread PERSIST+FLOOD", false, true,  true);
    report("ROTATE reread +FLOOD",   true,  false, true);
    report("ROTATE reread PERSIST+FLOOD", true,  true,  true);

    cudaFree(target); cudaFree(flood); cudaStreamDestroy(s);
    return 0;
}
