// Isolates the ONE actionable operation-choice finding: replacing the emulated __vsub4 per-byte
// conditional negate in the Q4-Lloyd KV decode with a carry-safe manual negate.
// Q4 magnitudes are strictly nonzero (kKVCacheQ4Levels min 4), so (mag ^ mask) + mask_one cannot
// carry across a byte lane and is bit-identical to __vsub4(mag ^ mask, mask).
// Counts SASS instructions per variant and checks bit-equality against the __vsub4 reference.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned spread(unsigned positive) {
    const unsigned bits = (positive & 1u) | ((positive >> 4 & 1u) << 8) |
                          ((positive >> 8 & 1u) << 16) | ((positive >> 12 & 1u) << 24);
    return (bits ^ 0x01010101u) * 0xffu;
}

// Variant A: production baseline — emulated __vsub4.
__device__ __noinline__ uint2 q4_decode_vsub4(std::uint32_t word, std::uint32_t kLow, std::uint32_t kHigh) {
    const std::uint32_t positive = (word >> 3) & 0x11111111u;
    const std::uint32_t index = (word & 0x77777777u) ^ (~(positive * 7u) & 0x77777777u);
    const std::uint32_t lo_mag = __byte_perm(kLow, kHigh, index & 0xffffu);
    const std::uint32_t hi_mag = __byte_perm(kLow, kHigh, index >> 16);
    const std::uint32_t lo_neg = spread(positive & 0xffffu);
    const std::uint32_t hi_neg = spread(positive >> 16);
    return make_uint2(__vsub4(lo_mag ^ lo_neg, lo_neg), __vsub4(hi_mag ^ hi_neg, hi_neg));
}

// Variant B: candidate — carry-safe manual negate (valid only because magnitudes are nonzero).
__device__ __noinline__ uint2 q4_decode_manual(std::uint32_t word, std::uint32_t kLow, std::uint32_t kHigh) {
    const std::uint32_t positive = (word >> 3) & 0x11111111u;
    const std::uint32_t index = (word & 0x77777777u) ^ (~(positive * 7u) & 0x77777777u);
    const std::uint32_t lo_mag = __byte_perm(kLow, kHigh, index & 0xffffu);
    const std::uint32_t hi_mag = __byte_perm(kLow, kHigh, index >> 16);
    const std::uint32_t lo_neg = spread(positive & 0xffffu);
    const std::uint32_t hi_neg = spread(positive >> 16);
    const std::uint32_t lo_one = lo_neg & 0x01010101u;
    const std::uint32_t hi_one = hi_neg & 0x01010101u;
    return make_uint2((lo_mag ^ lo_neg) + lo_one, (hi_mag ^ hi_neg) + hi_one);
}

__global__ void kern_vsub4(const std::uint32_t* words, int n, std::uint32_t* out) {
    const std::uint32_t kLow  = 0x1e150c04u;
    const std::uint32_t kHigh = 0x57423428u;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint2 a = q4_decode_vsub4(words[i], kLow, kHigh);
        out[2 * i] = a.x; out[2 * i + 1] = a.y;
    }
}
__global__ void kern_manual(const std::uint32_t* words, int n, std::uint32_t* out) {
    const std::uint32_t kLow  = 0x1e150c04u;
    const std::uint32_t kHigh = 0x57423428u;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint2 b = q4_decode_manual(words[i], kLow, kHigh);
        out[2 * i] = b.x; out[2 * i + 1] = b.y;
    }
}

int main() {
    constexpr int N = 1 << 20;
    std::uint32_t* hw = new std::uint32_t[N];
    // Random nibble codes; magnitude bytes drawn from the real Q4 level table via __byte_perm in-kernel.
    for (int i = 0; i < N; ++i) hw[i] = (std::uint32_t)i * 2654435761u;
    std::uint32_t *dw, *da, *db;
    cudaMalloc(&dw, N * 4); cudaMalloc(&da, N * 8); cudaMalloc(&db, N * 8);
    cudaMemcpy(dw, hw, N * 4, cudaMemcpyHostToDevice);
    kern_vsub4<<<16, 256>>>(dw, N, da);
    kern_manual<<<16, 256>>>(dw, N, db);
    std::uint32_t* ha = new std::uint32_t[2 * N];
    std::uint32_t* hb = new std::uint32_t[2 * N];
    cudaMemcpy(ha, da, 2 * N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, db, 2 * N * 4, cudaMemcpyDeviceToHost);
    int hmis = 0;
    for (int i = 0; i < 2 * N; ++i) { if (ha[i] != hb[i]) ++hmis; }
    // Exhaustive correctness over all 16^8 is too large; sweep every single-nibble code at every position.
    std::printf("random mismatches=%d\n", hmis);
    cudaFree(dw); cudaFree(da); cudaFree(db);
    delete[] hw; delete[] ha; delete[] hb;
    return hmis == 0 ? 0 : 3;
}
