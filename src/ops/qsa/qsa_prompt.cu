// QSA prompt attention on Tensor Cores (see qsa_prompt.h).

#include "ops/qsa/qsa_prompt.h"

#include "core/pdl.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/sigmoid_gate_mul_pair.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/qsa/page_spaces.cuh"
#include "ops/qsa/vq_window.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD      = 256; // head dimension
constexpr int kTile   = 16;  // tokens per warp tile
constexpr int kRows   = 16;  // MMA rows: one KV head's query heads, padded
constexpr int kQRow   = 2 * kD; // bytes of one FP16/BF16 query row in shared memory

// Which half of a stage one gather fills: both (one commit group), the key rows with their scales and
// the parity word, or the value rows with theirs. The two halves of a stage are disjoint, so each is
// its own commit group and the single-stage loop can wait for one only before the MMA phase that
// reads it.
constexpr int kGatherKey = 1, kGatherValue = 2, kGatherBoth = 3;

// How a stored row enters the MMAs. Keys: BF16 rows on BF16 Tensor Cores; 8-bit codes (INT8, FP8
// E4M3) widened exactly to FP16 with their scales applied to the FP32 partial of each scale group;
// NVFP4 codes decoded exactly to FP16 values (E2M1 times E4M3 is exact in FP16). Values: FP16 rows;
// 8-bit codes widened and multiplied once by their key's scale in FP16 (even/odd dimension tiles);
// NVFP4 rows decoded exactly to an FP16 tile in shared memory.
enum class Code { Bf16, Fp16, Int8, Fp8, Nvfp4 };

template <Code C>
struct Row;
template <>
struct Row<Code::Bf16> {
    static constexpr int Bytes = 2 * kD, ScaleSlot = 0;
};
template <>
struct Row<Code::Fp16> {
    static constexpr int Bytes = 2 * kD, ScaleSlot = 0;
};
template <>
struct Row<Code::Int8> {
    static constexpr int Bytes = kD, ScaleSlot = 8, Groups = 4;
    static constexpr float MaxScale = 256.0f; // 127 x 256 stays finite in FP16
};
// One FP16 row scale, gathered as the aligned 4-byte word holding it; its half is the token's parity.
template <>
struct Row<Code::Fp8> {
    static constexpr int Bytes = kD, ScaleSlot = 4, Groups = 1;
    static constexpr float MaxScale = 64.0f; // 448 x 64 stays finite in FP16
};
template <>
struct Row<Code::Nvfp4> {
    static constexpr int Bytes = kD / 2, ScaleSlot = 16;
};

template <KvCacheStorage Storage>
struct PromptShape;

template <>
struct PromptShape<KvCacheStorage::Int8Group64> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::BFloat16> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 2;
    static constexpr Code Key = Code::Bf16, Value = Code::Fp16;
};
template <>
struct PromptShape<KvCacheStorage::Fp8E4M3Row256> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Fp8, Value = Code::Fp8;
};
template <>
struct PromptShape<KvCacheStorage::Nvfp4Group16> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Nvfp4, Value = Code::Nvfp4;
};
// Vector-quantized storages: each tile's rows become INT8-G64 rows of the rotated vectors (the window's
// exact rows, or the codes decoded with the row scale as every group's), then the INT8 path. `Vq` is
// the key code's bytes; three warps keep two stages, the codes and the codebook within one CTA.
template <>
struct PromptShape<KvCacheStorage::Vq2> {
    static constexpr int Vq    = kKVCacheVq2CodeBytes;
    static constexpr int Warps = 3;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::Q4KeyVq2Value> {
    static constexpr int Vq    = kKVCacheQ4CodeBytes;
    static constexpr int Warps = 3;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::Fp8KeyNvfp4Value> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Fp8, Value = Code::Nvfp4;
};

template <KvCacheStorage Storage, bool SplitD>
__host__ __device__ constexpr int prompt_warps() {
    return SplitD ? 4 : PromptShape<Storage>::Warps;
}

// One warp stage: [K rows | V rows | K scales | V scales | parity word], 16 tokens each; rows keep
// their 16-byte chunks swizzled. The parity word's bit j is token j's parity (FP8 row scales).
template <KvCacheStorage Storage, bool SplitD>
struct Stage {
    using Shape                      = PromptShape<Storage>;
    using K                          = Row<Shape::Key>;
    using V                          = Row<Shape::Value>;
    static_assert(!SplitD || Shape::Vq == 0, "vector-quantized stages stay full-D");
    static constexpr int KHalfBytes  = SplitD ? K::Bytes / 2 : K::Bytes;
    static constexpr int VHalfBytes  = SplitD ? V::Bytes / 2 : V::Bytes;
    static constexpr int KScaleSlot  = SplitD && K::ScaleSlot > 4 ? K::ScaleSlot / 2 : K::ScaleSlot;
    static constexpr int VScaleSlot  = SplitD && V::ScaleSlot > 4 ? V::ScaleSlot / 2 : V::ScaleSlot;
    static constexpr bool Parity     = Shape::Key == Code::Fp8 || Shape::Value == Code::Fp8 || Shape::Vq > 0;
    static constexpr int Values      = kTile * KHalfBytes;
    static constexpr int KeyScales   = Values + kTile * VHalfBytes;
    static constexpr int ValueScales = KeyScales + kTile * KScaleSlot;
    // The value scales get two slots where the route gathers them with the keys: the QK phase settles
    // the 8-bit values' exact decode rescale from them, so they must land before the value rows, while
    // the next tile's are already being gathered as this tile's PV MMA reads them. A stage the combined
    // route fills (a vector-quantized stage, or a storage whose gather the split does not pay for) needs
    // one, and keeps its original footprint.
    static constexpr bool SplitGather = Shape::Vq == 0 && (SplitD || KHalfBytes >= kD);
    static constexpr int ScaleSlots   = SplitGather ? 2 : 1;
    static constexpr int ParityWord   = ValueScales + ScaleSlots * kTile * VScaleSlot;
    // Vector-quantized: the paged code rows, each row's FP16 code scale (K, V) as its aligned word,
    // the window tags (K, V) and each row's token and read mode.
    static constexpr int VqKeyCodes   = ParityWord + (Parity ? 16 : 0);
    static constexpr int VqValueCodes = VqKeyCodes + kTile * Shape::Vq;
    static constexpr int VqCodeScales = VqValueCodes + (Shape::Vq > 0 ? kTile * kKVCacheVq2CodeBytes : 0);
    static constexpr int VqTags       = VqCodeScales + (Shape::Vq > 0 ? kTile * 8 : 0);
    static constexpr int VqMeta       = VqTags + (Shape::Vq > 0 ? kTile * 8 : 0);
    static constexpr int Bytes        = VqMeta + (Shape::Vq > 0 ? kTile * 8 : 0);
    static constexpr int Scratch      = 0;
    static constexpr bool RotatedKeys   = Shape::Key != Code::Bf16;
    static constexpr bool RotatedValues = Shape::Value == Code::Nvfp4 || Shape::Vq > 0;
    static constexpr bool EvenOddValues = Shape::Value == Code::Int8 || Shape::Value == Code::Fp8;
    static_assert(Values % 16 == 0 && KeyScales % 16 == 0 && ValueScales % 16 == 0 && ParityWord % 16 == 0);
};

// Shared memory: phase 1 holds the active query rows, optional split-D pair score buffers, and each
// warp's `Stages` stages; phase 2 overlays one FP32 merge row buffer and per-KV-warp (max, sum) at
// offset 0 (q_s is dead after the KV loop); the VQ2 codebook sits behind whichever phase is larger.
template <KvCacheStorage Storage, int Stages, bool SplitD>
__host__ __device__ constexpr int prompt_smem_bytes() {
    using S                   = Stage<Storage, SplitD>;
    constexpr int WarpsKV     = SplitD ? 2 : PromptShape<Storage>::Warps;
    constexpr int Warps       = prompt_warps<Storage, SplitD>();
    constexpr int pair_scores = SplitD ? WarpsKV * 1024 : 0;
    constexpr int stages      = Warps * (Stages * S::Bytes + S::Scratch);
    constexpr int merge       = (kRows * kD + WarpsKV * kRows * 2) * 4;
    constexpr int book        = PromptShape<Storage>::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0;
    constexpr int phase1      = kQRow * kRows + pair_scores + stages;
    return (phase1 > merge ? phase1 : merge) + book;
}

template <KvCacheStorage Storage, int Stages, bool SplitD>
__host__ __device__ constexpr int prompt_smem_blocks() {
    constexpr int per_cta = prompt_smem_bytes<Storage, Stages, SplitD>() + 1024;
    if constexpr (4 * per_cta <= 100 * 1024) { return 4; }
    if constexpr (3 * per_cta <= 100 * 1024) { return 3; }
    if constexpr (2 * per_cta <= 100 * 1024) { return 2; }
    return 1;
}

template <KvCacheStorage Storage, int Stages, bool SplitD>
__host__ __device__ constexpr int prompt_launch_min_blocks() {
    constexpr int smem_blocks = prompt_smem_blocks<Storage, Stages, SplitD>();
    if constexpr (SplitD) { return smem_blocks; }
    return smem_blocks > 2 ? 2 : smem_blocks;
}

// 16-byte chunk c of row r lives at chunk c ^ (r & 7) when row_bytes >= 128; when row_bytes == 64
// (4 chunks per row, SplitD NVFP4), c ^ (r & 3) ^ ((r >> 2) & 1) stays within the row while
// keeping each 8-row ldmatrix tile across all 8 16-byte bank groups with zero bank conflicts.
__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    const int c = row_bytes == 64 ? (chunk ^ (row & 3) ^ ((row >> 2) & 1)) : (chunk ^ (row & 7));
    return row * row_bytes + (c << 4);
}

// The attended token at index i of a column's list (the decode kernel's order: selected blocks in
// ascending order, then the incomplete block's positions; every token 0..p while dense).
__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* list, int ratio, int p) {
    if (count < 0) { return i; }
    const int block_tokens = count * ratio;
    if (i < block_tokens) { return list[i / ratio] * ratio + i % ratio; }
    return (p + 1) / ratio * ratio + (i - block_tokens);
}

// 8-bit key rows: the k positions {2t, 2t+1 | 2t+8, 2t+9} of a 16-wide MMA k step hold dimensions
// {4t, 4t+1 | 4t+2, 4t+3}, which is what one ldmatrix lane receives of a code row (four
// consecutive bytes). The query rows are stored in that order, so the dot product is unchanged.
__device__ __forceinline__ int int8_query_position(int d) {
    const int local = d & 15;
    const int p     = ((local & 2) != 0 ? 8 : 0) + 2 * (local >> 2) + (local & 1);
    return (d & ~15) + p;
}

// NVFP4 key rows: one ldmatrix lane receives eight dimensions 32c + 8t .. + 7 of a code row (four
// bytes); k step 2c takes the first four at positions {2t, 2t+1 | 2t+8, 2t+9} and k step 2c + 1
// the last four.
__device__ __forceinline__ int nvfp4_query_position(int d) {
    const int c = d >> 5, t = (d >> 3) & 3, w = d & 7;
    const int step = 2 * c + (w >> 2), u = w & 3;
    return 16 * step + 2 * t + ((u & 2) != 0 ? 8 : 0) + (u & 1);
}

template <Code C>
__device__ __forceinline__ int query_position(int d) {
    if constexpr (C == Code::Int8 || C == Code::Fp8) { return int8_query_position(d); }
    if constexpr (C == Code::Nvfp4) { return nvfp4_query_position(d); }
    return d;
}

// Four INT8 codes (dimensions d..d+3 of one key) as the two FP16 B registers of a k step:
// {c0, c1} and {c2, c3}, widened exactly.
__device__ __forceinline__ void widen_codes(unsigned codes, unsigned& lo, unsigned& hi) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x5140);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 l       = __hsub2(load_vec<__half2>(&e), offset);
    const __half2 h       = __hsub2(load_vec<__half2>(&o), offset);
    lo                    = load_vec<unsigned>(&l);
    hi                    = load_vec<unsigned>(&h);
}

// Two E4M3 codes (low byte first) as an FP16 pair, exactly.
__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned pair) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    return static_cast<unsigned>(raw.x) | (static_cast<unsigned>(raw.y) << 16);
}

// Four E4M3 codes (dimensions d..d+3 of one key) as {c0, c1} and {c2, c3}, exactly.
__device__ __forceinline__ void widen_e4m3(unsigned codes, unsigned& lo, unsigned& hi) {
    lo = e4m3x2_to_f16x2(codes & 0xffffu);
    hi = e4m3x2_to_f16x2(codes >> 16);
}

// Two E2M1 codes (low nibble first) times an FP16 scale pair, exactly.
__device__ __forceinline__ unsigned e2m1x2_times(unsigned byte, __half2 scale) {
    __nv_fp4x2_e2m1 codes;
    codes.__x       = static_cast<__nv_fp4x2_storage_t>(byte);
    const __half2 v = __hmul2(static_cast<__half2>(codes), scale);
    return load_vec<unsigned>(&v);
}

__device__ __forceinline__ __half e4m3_to_half(unsigned char code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<__half>(v);
}

// One ldmatrix.trans b16 lane of an 8-bit [key][d] tile holds {V[k][d], V[k][d+1], V[k+1][d],
// V[k+1][d+1]}: the FP16 B halves {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code
// widened exactly and multiplied once by its key's scale (`scales` = {s_k, s_k+1}).
template <Code C>
__device__ __forceinline__ void decode_v_pair(unsigned codes, __half2 scales, unsigned& even, unsigned& odd) {
    if constexpr (C == Code::Int8) {
        const unsigned biased = codes ^ 0x80808080u;
        const unsigned e      = __byte_perm(biased, 0x64646464u, 0x4240);
        const unsigned o      = __byte_perm(biased, 0x64646464u, 0x4341);
        const __half2 offset  = __float2half2_rn(1152.0f);
        const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), scales);
        const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), scales);
        even                  = load_vec<unsigned>(&ve);
        odd                   = load_vec<unsigned>(&vo);
    } else {
        const unsigned e  = e4m3x2_to_f16x2(__byte_perm(codes, 0, 0x0020));
        const unsigned o  = e4m3x2_to_f16x2(__byte_perm(codes, 0, 0x0031));
        const __half2 ve = __hmul2(load_vec<__half2>(&e), scales);
        const __half2 vo = __hmul2(load_vec<__half2>(&o), scales);
        even              = load_vec<unsigned>(&ve);
        odd               = load_vec<unsigned>(&vo);
    }
}

// One ldmatrix.trans b16 lane of an NVFP4 [key][d] tile holds two packed bytes (four E2M1 codes,
// dimensions 32c + 4n .. + 3) of key k (low 16 bits) and key k + 1 (high 16 bits), all four in the
// same 16-element scale group: decode to the four FP16 B halves {V[k][d+u], V[k+1][d+u]} for u = 0..3.
__device__ __forceinline__ void decode_v_quad_nvfp4(unsigned codes, __half2 s0_pair, __half2 s1_pair,
                                                    unsigned (&out)[4]) {
    const unsigned w0_01 = e2m1x2_times(codes & 0xffu, s0_pair);
    const unsigned w1_01 = e2m1x2_times((codes >> 16) & 0xffu, s1_pair);
    out[0]               = __byte_perm(w0_01, w1_01, 0x5410);
    out[1]               = __byte_perm(w0_01, w1_01, 0x7632);
    const unsigned w0_23 = e2m1x2_times((codes >> 8) & 0xffu, s0_pair);
    const unsigned w1_23 = e2m1x2_times(codes >> 24, s1_pair);
    out[2]               = __byte_perm(w0_23, w1_23, 0x5410);
    out[3]               = __byte_perm(w0_23, w1_23, 0x7632);
}

template <bool Bf16>
__device__ __forceinline__ void qsa_mma(float* c, const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if constexpr (Bf16) {
        mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1);
    } else {
        mma_f16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1);
    }
}

__device__ __forceinline__ bf16 apply_qsa_gate(bf16 x, const bf16* gate, std::int64_t idx) {
    if (gate == nullptr) { return x; }
    return __float2bfloat16_rn(__bfloat162float(x) * sigmoid(__bfloat162float(gate[idx])));
}

__device__ __forceinline__ __nv_bfloat162 apply_qsa_gate_pair(__nv_bfloat162 x, const bf16* gate, std::int64_t idx) {
    if (gate == nullptr) { return x; }
    if ((idx & 1) == 0 && (reinterpret_cast<std::uintptr_t>(gate) & 3U) == 0U) {
        return sigmoid_gate_mul_pair(*reinterpret_cast<const __nv_bfloat162*>(gate + idx), x);
    }
    return __halves2bfloat162(apply_qsa_gate(__low2bfloat16(x), gate, idx),
                              apply_qsa_gate(__high2bfloat16(x), gate, idx + 1));
}

constexpr int kGroupQueries    = 8;
constexpr int kGroupHeadsPerKv = 12;
constexpr int kGroupRows       = kGroupQueries * kGroupHeadsPerKv; // 96 MMA rows
constexpr int kGroupKeyTile    = 32;                               // 32 consecutive tokens inside one 64-token page

__device__ __forceinline__ bool dense_group_range_selected(int first, int last, const std::int32_t* counts,
                                                           const std::int32_t* positions) {
    if (last <= first) { return false; }
    int prev = -1;
    for (int q = first; q < last; ++q) {
        const int p = positions[q];
        if (counts[q] >= 0 || p < 0 || p < prev) { return false; }
        prev = p;
    }
    return true;
}

__device__ __forceinline__ bool dense_prefix_group_selected(int t, int width, int columns,
                                                            const std::int32_t* counts,
                                                            const std::int32_t* positions) {
    const int b         = t / width;
    const int row_start = b * width;
    const int first     = row_start + ((t - row_start) & ~(kGroupQueries - 1));
    const int last      = min(columns, min(row_start + width, first + kGroupQueries));
    return dense_group_range_selected(first, last, counts, positions);
}

struct GroupedOnlineSoftmax {
    float maximum        = -INFINITY;
    float sum            = 0.0f;
    float scaled_maximum = 0.0f;

    __device__ __forceinline__ float update(float tile_max, float scale_l2) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_max);
        scaled_maximum       = maximum == -INFINITY ? 0.0f : maximum * scale_l2;
        return previous == -INFINITY ? 0.0f : exp2_approx(__fmaf_rn(previous, scale_l2, -scaled_maximum));
    }

    __device__ __forceinline__ float probability(float score, float scale_l2) const {
        return exp2_approx(__fmaf_rn(score, scale_l2, -scaled_maximum));
    }

    __device__ __forceinline__ void accumulate(float alpha, float tile_sum) {
        sum = __fmaf_rn(sum, alpha, tile_sum);
    }

    __device__ __forceinline__ void finish() { sum = warp_sum<4>(sum, 0xffffffffu); }
};

template <KvCacheStorage Storage>
__host__ __device__ constexpr int grouped_smem_bytes() {
    using Shape               = PromptShape<Storage>;
    constexpr bool Quantized  = Storage != KvCacheStorage::BFloat16;
    constexpr int QsBytes     = kGroupRows * kD * 2;        // 49,152 B
    constexpr int KvBytes     = 2 * kGroupKeyTile * kD * 2; // 32,768 B
    constexpr int RawBytes    = Quantized ? (kGroupKeyTile * kD + kGroupKeyTile * 16 + 16) : 0; // 8,720 B
    constexpr int Phase1Bytes = QsBytes + KvBytes + RawBytes;
    constexpr int Phase2Bytes = (Shape::Value == Code::Nvfp4) ? (kGroupRows * kD * 4) : 0; // 98,304 B
    return Phase1Bytes > Phase2Bytes ? Phase1Bytes : Phase2Bytes;
}

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(((Storage != KvCacheStorage::BFloat16) ? 2 : 1) * (kGroupRows / 16) * 32, 1)
    qsa_grouped_prefix_kernel(const bf16* __restrict__ q, int heads, int kv_heads,
                              const void* __restrict__ k_pages, const void* __restrict__ v_pages,
                              const void* __restrict__ k_scales, const void* __restrict__ v_scales,
                              const std::int32_t* __restrict__ tables, int table_stride,
                              const std::int32_t* __restrict__ table_rows,
                              const std::int32_t* __restrict__ positions, int width, int columns,
                              const std::int32_t* __restrict__ counts, float scale, const bf16* __restrict__ gate,
                              std::int64_t gate_col_stride, bf16* __restrict__ out, QsaPageSpaces spaces) {
    using Shape                 = PromptShape<Storage>;
    static_assert(Shape::Vq == 0, "grouped prefix serves scalar KV storages");
    constexpr bool Quantized    = Storage != KvCacheStorage::BFloat16;
    constexpr bool RotatedValue = Shape::Value == Code::Nvfp4;
    constexpr int Parts         = Quantized ? 2 : 1;
    constexpr int Warps         = (kGroupRows / 16) * Parts;
    constexpr int Threads       = Warps * 32;
    constexpr int OutCols       = kD / Parts / 8;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    const int groups_per_row = (width + kGroupQueries - 1) / kGroupQueries;
    const int g_idx          = static_cast<int>(blockIdx.x);
    const int kv_head        = static_cast<int>(blockIdx.y);
    const int b              = g_idx / groups_per_row;
    const int first          = b * width + (g_idx % groups_per_row) * kGroupQueries;
    const int last           = min(columns, min((b + 1) * width, first + kGroupQueries));
    if (!dense_group_range_selected(first, last, counts, positions)) { return; }

    extern __shared__ __align__(16) unsigned char smem[];
    auto* qs         = reinterpret_cast<std::uint16_t*>(smem);                                // [96][256]
    auto* ks         = reinterpret_cast<std::uint16_t*>(smem + kGroupRows * kD * 2);          // [32][256]
    auto* vs         = ks + kGroupKeyTile * kD;                                               // [32][256]
    auto* raw_codes  = smem + (kGroupRows + 2 * kGroupKeyTile) * kD * 2;                      // [32][<=256]
    auto* raw_scales = raw_codes + kGroupKeyTile * kD;                                        // [32][<=16]
    auto* meta_words = reinterpret_cast<unsigned*>(raw_scales + kGroupKeyTile * 16);          // [4]

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int end  = positions[last - 1] + 1;

    if constexpr (Quantized) {
        for (int row = warp; row < kGroupRows; row += Warps) {
            const int q_col = first + row / kGroupHeadsPerKv;
            const int h     = row % kGroupHeadsPerKv;
            float vals[8]{};
            if (q_col < last) {
                const bf16* q_src =
                    q + (static_cast<std::size_t>(q_col) * heads + kv_head * kGroupHeadsPerKv + h) * kD;
#pragma unroll
                for (int i = 0; i < 8; ++i) { vals[i] = __bfloat162float(q_src[lane + 32 * i]); }
                normalized_hadamard_d256_inplace(vals, lane);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d                           = lane + 32 * i;
                qs[row * kD + (d ^ ((row & 7) << 3))] = __half_as_ushort(__float2half_rn(vals[i]));
            }
        }
    } else {
        for (int i = tid; i < kGroupRows * 32; i += Threads) {
            const int row       = i >> 5;
            const int d         = (i & 31) << 3;
            std::uint16_t* dst  = qs + row * kD + (d ^ ((row & 7) << 3));
            const int q_col     = first + row / kGroupHeadsPerKv;
            if (q_col < last) {
                const bf16* q_src = q +
                                    (static_cast<std::size_t>(q_col) * heads + kv_head * kGroupHeadsPerKv +
                                     (row % kGroupHeadsPerKv)) *
                                        kD +
                                    d;
                cp_async<16>(dst, q_src);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
    }
    __syncthreads();

    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[b]) * table_stride;

    auto stage_plane = [&]<bool IsKey>(int k0) {
        constexpr Code C         = IsKey ? Shape::Key : Shape::Value;
        using R                  = Row<C>;
        const int id             = table[k0 >> kPagedKVPageShift];
        const int slot0          = k0 & kPagedKVPageMask; // 0 or 32; all 32 tokens share this page
        const int valid_rows     = min(kGroupKeyTile, end - k0);
        int page                 = 0;
        const auto* data_plane   = static_cast<const unsigned char*>(
            IsKey ? qsa_space_plane(spaces, k_pages, spaces.host_k, spaces.lent_k, id, page)
                  : qsa_space_plane(spaces, v_pages, spaces.host_v, spaces.lent_v, id, page));
        const std::size_t row_at =
            static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + slot0;
        if constexpr (!Quantized) {
            std::uint16_t* dst_plane = IsKey ? ks : vs;
            for (int i = tid; i < kGroupKeyTile * 32; i += Threads) {
                const int r            = i >> 5;
                const int d            = (i & 31) << 3;
                std::uint16_t* dst     = dst_plane + r * kD + (d ^ ((r & 7) << 3));
                const unsigned char* s = data_plane + (row_at + r) * 512 + 2 * d;
                cp_async_zfill<16, Cache::cg>(dst, s, r < valid_rows ? 16 : 0);
            }
        } else {
            constexpr int Chunks = R::Bytes / 16;
            for (int i = tid; i < kGroupKeyTile * Chunks; i += Threads) {
                const int r            = i / Chunks;
                const int c            = i % Chunks;
                unsigned char* dst     = raw_codes + r * R::Bytes + 16 * c;
                const unsigned char* s = data_plane + (row_at + r) * R::Bytes + 16 * c;
                cp_async_zfill<16, Cache::cg>(dst, s, r < valid_rows ? 16 : 0);
            }
            int scale_page = 0;
            const auto* scale_plane = static_cast<const unsigned char*>(
                IsKey ? qsa_space_plane(spaces, k_scales, spaces.host_k_scale, spaces.lent_k_scale, id, scale_page)
                      : qsa_space_plane(spaces, v_scales, spaces.host_v_scale, spaces.lent_v_scale, id, scale_page));
            if (tid < kGroupKeyTile) {
                const std::size_t r_at = row_at + tid;
                const bool ok          = tid < valid_rows;
                if constexpr (R::ScaleSlot == 4) {
                    cp_async_zfill<4>(raw_scales + tid * 4, scale_plane + (2 * r_at & ~std::size_t{3}), ok ? 4 : 0);
                } else if constexpr (R::ScaleSlot == 8) {
                    cp_async_zfill<8>(raw_scales + tid * 8, scale_plane + 8 * r_at, ok ? 8 : 0);
                } else {
                    cp_async_zfill<16>(raw_scales + tid * 16, scale_plane + 16 * r_at, ok ? 16 : 0);
                }
            }
            if constexpr (R::ScaleSlot == 4) {
                if (warp == 0) {
                    const unsigned odd = __ballot_sync(FullMask, lane < valid_rows && ((row_at + lane) & 1U) != 0U);
                    if (lane == 0) { meta_words[0] = odd; }
                }
            }
        }
    };

    auto expand_plane = [&]<Code C>(std::uint16_t* dst) -> float {
        if constexpr (!Quantized) {
            return 1.0f;
        } else if constexpr (C == Code::Nvfp4) {
            for (int i = tid; i < kGroupKeyTile * 32; i += Threads) {
                const int r          = i >> 5;
                const int c_idx      = i & 31;
                const int d          = c_idx << 3;
                const __half sc      = e4m3_to_half(raw_scales[r * 16 + (c_idx >> 1)]);
                const __half2 s2     = __halves2half2(sc, sc);
                const unsigned pack4 = reinterpret_cast<const unsigned*>(raw_codes + r * 128)[c_idx];
                uint4 decoded;
                decoded.x = e2m1x2_times(pack4 & 0xffu, s2);
                decoded.y = e2m1x2_times((pack4 >> 8) & 0xffu, s2);
                decoded.z = e2m1x2_times((pack4 >> 16) & 0xffu, s2);
                decoded.w = e2m1x2_times(pack4 >> 24, s2);
                store_vec(dst + r * kD + (d ^ ((r & 7) << 3)), decoded);
            }
            __syncthreads();
            return 1.0f;
        } else {
            using R               = Row<C>;
            const unsigned parity = (C == Code::Fp8) ? meta_words[0] : 0u;
            if (warp == 0) {
                float own_max = 0.0f;
                if constexpr (C == Code::Fp8) {
                    const __half s_h = reinterpret_cast<const __half*>(raw_scales)[2 * lane + ((parity >> lane) & 1u)];
                    own_max          = fabsf(__half2float(s_h));
                } else {
                    const __half* row_s = reinterpret_cast<const __half*>(raw_scales + lane * 8);
#pragma unroll
                    for (int g = 0; g < 4; ++g) { own_max = fmaxf(own_max, fabsf(__half2float(row_s[g]))); }
                }
                const float smax = warp_max(own_max, FullMask);
                if (lane == 0) {
                    int shift = 0;
                    while (shift < 14 && ldexpf(smax, -shift) > R::MaxScale) { ++shift; }
                    meta_words[1] = static_cast<unsigned>(shift);
                }
            }
            __syncthreads();
            const int shift  = static_cast<int>(meta_words[1]);
            const float down = ldexpf(1.0f, -shift);
            for (int i = tid; i < kGroupKeyTile * 32; i += Threads) {
                const int r     = i >> 5;
                const int c_idx = i & 31;
                const int d     = c_idx << 3;
                __half s_orig;
                if constexpr (C == Code::Fp8) {
                    s_orig = reinterpret_cast<const __half*>(raw_scales)[2 * r + ((parity >> r) & 1u)];
                } else {
                    s_orig = reinterpret_cast<const __half*>(raw_scales + r * 8)[c_idx >> 3];
                }
                const __half s_scaled = __float2half_rn(__half2float(s_orig) * down);
                const __half2 s2      = __halves2half2(s_scaled, s_scaled);
                const uint2 raw8      = reinterpret_cast<const uint2*>(raw_codes + r * 256)[c_idx];
                unsigned lo0, hi0, lo1, hi1;
                if constexpr (C == Code::Fp8) {
                    widen_e4m3(raw8.x, lo0, hi0);
                    widen_e4m3(raw8.y, lo1, hi1);
                } else {
                    widen_codes(raw8.x, lo0, hi0);
                    widen_codes(raw8.y, lo1, hi1);
                }
                const __half2 v0 = __hmul2(load_vec<__half2>(&lo0), s2);
                const __half2 v1 = __hmul2(load_vec<__half2>(&hi0), s2);
                const __half2 v2 = __hmul2(load_vec<__half2>(&lo1), s2);
                const __half2 v3 = __hmul2(load_vec<__half2>(&hi1), s2);
                store_vec(dst + r * kD + (d ^ ((r & 7) << 3)),
                          make_uint4(load_vec<unsigned>(&v0), load_vec<unsigned>(&v1), load_vec<unsigned>(&v2),
                                     load_vec<unsigned>(&v3)));
            }
            __syncthreads();
            return ldexpf(1.0f, shift);
        }
    };

    const int rowbase  = (warp / Parts) * 16;
    const int part     = warp % Parts;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const float scale_l2 = scale * Log2E;
    int q_pos[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int q_col = first + (rowbase + gid + 8 * j) / kGroupHeadsPerKv;
        q_pos[j]        = q_col < last ? positions[q_col] : -1;
    }

    float acc[OutCols][4]{};
    GroupedOnlineSoftmax state[2];

    if (end > 0) {
        stage_plane.template operator()<true>(0);
        cp_commit();
    }
    for (int k0 = 0; k0 < end; k0 += kGroupKeyTile) {
        cp_wait<0>();
        __syncthreads();
        const float k_up = expand_plane.template operator()<Shape::Key>(ks);
        stage_plane.template operator()<false>(k0);
        cp_commit();

        float score[kGroupKeyTile / 8][4]{};
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            unsigned qf[4];
            const int qr = rowbase + (lane & 7) + ((lane >> 3) & 1) * 8;
            const int qd = k * 16 + (lane >> 4) * 8;
            ldmatrix_x4(qf[0], qf[1], qf[2], qf[3], smem_addr(qs + qr * kD + (qd ^ ((qr & 7) << 3))));
#pragma unroll
            for (int n = 0; n < kGroupKeyTile / 8; ++n) {
                unsigned bf[2];
                const int r = n * 8 + (lane & 7);
                const int d = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(ks + r * kD + (d ^ ((r & 7) << 3))));
                qsa_mma<!Quantized>(score[n], qf, bf[0], bf[1]);
            }
        }
        if constexpr (Shape::Key == Code::Int8 || Shape::Key == Code::Fp8) {
#pragma unroll
            for (int n = 0; n < kGroupKeyTile / 8; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) { score[n][j] *= k_up; }
            }
        }

        cp_wait<0>();
        __syncthreads();
        const float v_up = expand_plane.template operator()<Shape::Value>(vs);
        if (k0 + kGroupKeyTile < end) {
            stage_plane.template operator()<true>(k0 + kGroupKeyTile);
            cp_commit();
        }

        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < kGroupKeyTile / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (k0 + n * 8 + lid * 2 + (j & 1) > q_pos[j >> 1]) { score[n][j] = -CUDART_INF_F; }
                maximum[j >> 1] = fmaxf(maximum[j >> 1], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], FullMask), scale_l2),
                                state[1].update(warp_max<4>(maximum[1], FullMask), scale_l2)};
        unsigned pf[kGroupKeyTile / 16][4];
        float tile_sum[2]{};
#pragma unroll
        for (int n = 0; n < kGroupKeyTile / 8; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_l2);
            const float p1 = state[0].probability(score[n][1], scale_l2);
            const float p2 = state[1].probability(score[n][2], scale_l2);
            const float p3 = state[1].probability(score[n][3], scale_l2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            if constexpr (Shape::Value == Code::Int8 || Shape::Value == Code::Fp8) {
                pf[n >> 1][(n & 1) * 2]     = pack_f16x2(p0 * v_up, p1 * v_up);
                pf[n >> 1][(n & 1) * 2 + 1] = pack_f16x2(p2 * v_up, p3 * v_up);
            } else {
                pf[n >> 1][(n & 1) * 2]     = pack_f16x2(p0, p1);
                pf[n >> 1][(n & 1) * 2 + 1] = pack_f16x2(p2, p3);
            }
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < OutCols; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[n][j] *= alpha[j >> 1]; }
#pragma unroll
            for (int k = 0; k < kGroupKeyTile / 16; ++k) {
                unsigned vf[2];
                const int r = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d = part * (kD / Parts) + n * 8;
                ldmatrix_x2_t(vf[0], vf[1], smem_addr(vs + r * kD + (d ^ ((r & 7) << 3))));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2], pf[k][3],
                        vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    state[0].finish();
    state[1].finish();
    float* result = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int row   = rowbase + gid + j * 8;
        const int q_col = first + row / kGroupHeadsPerKv;
        if (q_col >= last) { continue; }
        const int h        = row % kGroupHeadsPerKv;
        const int head_idx = kv_head * kGroupHeadsPerKv + h;
        const float inv    = state[j].sum > 0.0f ? (1.0f / state[j].sum) : 0.0f;
#pragma unroll
        for (int n = 0; n < OutCols; ++n) {
            const int d   = part * (kD / Parts) + n * 8 + lid * 2;
            const float x = acc[n][2 * j] * inv;
            const float y = acc[n][2 * j + 1] * inv;
            if constexpr (RotatedValue) {
                *reinterpret_cast<float2*>(result + row * kD + d) = make_float2(x, y);
            } else {
                const __nv_bfloat162 pair = __floats2bfloat162_rn(x, y);
                const std::int64_t g_idx  = static_cast<std::int64_t>(q_col) * gate_col_stride + head_idx * kD + d;
                *reinterpret_cast<__nv_bfloat162*>(out + (static_cast<std::size_t>(q_col) * heads + head_idx) * kD + d) =
                    apply_qsa_gate_pair(pair, gate, g_idx);
            }
        }
    }
    if constexpr (RotatedValue) {
        __syncthreads();
        for (int row = warp; row < kGroupRows; row += Warps) {
            const int q_col = first + row / kGroupHeadsPerKv;
            if (q_col >= last) { continue; }
            float vals[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { vals[i] = result[row * kD + lane * 8 + i]; }
            normalized_hadamard_d256_contiguous_inplace(vals, lane);
            const int head_idx = kv_head * kGroupHeadsPerKv + (row % kGroupHeadsPerKv);
            __nv_bfloat162 out_pairs[4];
#pragma unroll
            for (int p_i = 0; p_i < 4; ++p_i) {
                out_pairs[p_i] = __floats2bfloat162_rn(vals[2 * p_i], vals[2 * p_i + 1]);
            }
            if (gate != nullptr) {
                const std::int64_t g_base =
                    static_cast<std::int64_t>(q_col) * gate_col_stride + head_idx * kD + lane * 8;
#pragma unroll
                for (int p_i = 0; p_i < 4; ++p_i) {
                    out_pairs[p_i] = apply_qsa_gate_pair(out_pairs[p_i], gate, g_base + 2 * p_i);
                }
            }
            bf16* dst = out + (static_cast<std::size_t>(q_col) * heads + head_idx) * kD + lane * 8;
#pragma unroll
            for (int p_i = 0; p_i < 4; ++p_i) {
                *reinterpret_cast<__nv_bfloat162*>(dst + 2 * p_i) = out_pairs[p_i];
            }
        }
    }
}

template <KvCacheStorage Storage, int Stages, bool SplitD, bool SkipDenseGroups = false>
__global__ void __launch_bounds__(prompt_warps<Storage, SplitD>() * 32,
                                  prompt_launch_min_blocks<Storage, Stages, SplitD>())
    qsa_prompt_kernel(const bf16* __restrict__ q, int heads, int kv_heads, const void* __restrict__ k_pages,
                      const void* __restrict__ v_pages, const void* __restrict__ k_scales,
                      const void* __restrict__ v_scales, const std::int32_t* __restrict__ tables,
                      int table_stride, const std::int32_t* __restrict__ table_rows,
                      const std::int32_t* __restrict__ positions, int width, int ratio,
                      const std::int32_t* __restrict__ selected, const std::int32_t* __restrict__ counts,
                      int top_blocks, float scale, const bf16* __restrict__ gate, std::int64_t gate_col_stride,
                      bf16* __restrict__ out, QsaPageSpaces spaces, QsaVqWindow vq) {
    if constexpr (SkipDenseGroups) {
        if (dense_prefix_group_selected(static_cast<int>(blockIdx.x), width, static_cast<int>(gridDim.x), counts,
                                        positions)) {
            return;
        }
    }
    using Shape                 = PromptShape<Storage>;
    using S                     = Stage<Storage, SplitD>;
    using K                     = Row<Shape::Key>;
    using V                     = Row<Shape::Value>;
    constexpr int Rows          = kRows;
    constexpr int HalfBlocks    = SplitD ? 8 : 16;
    constexpr int WarpsKV       = SplitD ? 2 : Shape::Warps;
    constexpr int Warps         = prompt_warps<Storage, SplitD>();
    constexpr int Threads       = Warps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];

    const int t = static_cast<int>(blockIdx.x), kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int kv_w = SplitD ? (warp >> 1) : warp;
    const int d_w  = SplitD ? (warp & 1) : 0;
    const int gid = lane >> 2, lid = lane & 3;
    const int group = heads / kv_heads;
    const int p     = positions[t];
    const int count = counts[t];
    const int total = count < 0 ? p + 1 : count * ratio + (p + 1) % ratio;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    // Query rows (rotated into the keys' Hadamard domain where keys are rotated, then FP16 in the
    // key code's k-step order; BF16 keys: the stored values). Rows past the group are zero.
    unsigned char* q_s = smem;
    for (int r = warp; r < Rows; r += Warps) {
        if (r < group) {
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                v[k] = __bfloat162float(
                    q[(static_cast<std::size_t>(t) * heads + kv_head * group + r) * kD + lane + 32 * k]);
            }
            if constexpr (S::RotatedKeys) { normalized_hadamard_d256_inplace(v, lane); }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = lane + 32 * k;
                const int e = query_position<Shape::Key>(d);
                unsigned char* at = q_s + swizzled(r, kQRow, e >> 3) + 2 * (e & 7);
                if constexpr (Shape::Key == Code::Bf16) {
                    *reinterpret_cast<bf16*>(at) = __float2bfloat16_rn(v[k]);
                } else {
                    *reinterpret_cast<__half*>(at) = __float2half_rn(v[k]);
                }
            }
        } else {
            *reinterpret_cast<uint4*>(q_s + swizzled(r, kQRow, lane)) = uint4{0u, 0u, 0u, 0u};
        }
    }
    __syncthreads();

    // VQ2 codebook (vector-quantized storages): behind whichever shared-memory phase is larger.
    constexpr int kCodebookAt = prompt_smem_bytes<Storage, Stages, SplitD>() -
                                (Shape::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0);
    const auto* codebook = reinterpret_cast<const std::int8_t*>(smem + kCodebookAt);
    if constexpr (Shape::Vq > 0) {
        for (int i = tid; i < static_cast<int>(sizeof(kKVCacheVq2Codebook)) / 16; i += Threads) {
            reinterpret_cast<uint4*>(const_cast<std::int8_t*>(codebook))[i] =
                reinterpret_cast<const uint4*>(g_kv_cache_vq2_codebook)[i];
        }
        __syncthreads();
    }
    const int call_first  = positions[(t / width) * width];
    const int window_row  = vq.slots != nullptr ? vq.slots[t / width] : 0;
    float* pair_scores    = reinterpret_cast<float*>(smem + kQRow * Rows);
    unsigned char* stages = smem + kQRow * Rows + (SplitD ? WarpsKV * 1024 : 0) +
                            warp * (Stages * S::Bytes + S::Scratch);
    const int tiles       = (total + kTile - 1) / kTile;

    // The byte offset of tile i's value-scale slot. Consecutive tiles of one warp use different slots:
    // the single-stage route gathers a tile's value scales with its keys, so the next tile's are already
    // in flight while this tile's PV MMA still reads them.
    const auto value_scale_at = [&](int i) {
        return ((i / WarpsKV) & (S::ScaleSlots - 1)) * (kTile * S::VScaleSlot);
    };

    // Gathers tile i's `Part` (both halves, or one of them) into stage s and commits it as one group;
    // tokens past the column's list are zero-filled. Lane l locates row l % 16 once (its token, page
    // space and row in the planes); the lanes copying a row's chunks take its addresses by shuffle.
    // Splitting the gather into its own commit group is what lets the single-stage loop hide one half
    // behind the MMA phase that reads the other half.
    const auto issue = [&](int i, int s, auto part) {
        constexpr int Part = decltype(part)::value;
        static_assert(Part == kGatherBoth || Shape::Vq == 0,
                      "a vector-quantized stage resolves both halves together");
        unsigned char* k_s = stages + s * S::Bytes;
        [[maybe_unused]] unsigned char* v_s = k_s + S::Values;
        const int row      = lane & 15;
        const int index    = i * kTile + row;
        const bool valid   = index < total;
        const int token    = valid ? attended_token(index, count, list, ratio, p) : 0;
        const int id       = table[token >> kPagedKVPageShift];
        int page;
        const auto* k_plane =
            static_cast<const unsigned char*>(qsa_space_plane(spaces, k_pages, spaces.host_k, spaces.lent_k, id, page));
        [[maybe_unused]] const auto* v_plane =
            static_cast<const unsigned char*>(qsa_space_plane(spaces, v_pages, spaces.host_v, spaces.lent_v, id, page));
        const std::size_t row_at =
            static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + (token & kPagedKVPageMask);
        // Copies every row's `Bytes` from its lane's `own_row` (zero-filled unless `own_ok`), with the
        // 16-byte chunks swizzled for ldmatrix or in order.
        const auto copy_rows = [&](unsigned char* dst_rows, const void* own_row, bool own_ok, auto bytes, auto swizzle) {
            constexpr int Bytes  = decltype(bytes)::value;
            constexpr int Chunks = Bytes / 16;
            const auto own       = reinterpret_cast<unsigned long long>(own_row);
#pragma unroll
            for (int j = 0; j < kTile * Chunks / 32; ++j) {
                const int chunk = lane + 32 * j;
                const int r = chunk / Chunks, c = chunk % Chunks;
                const auto* src = reinterpret_cast<const unsigned char*>(__shfl_sync(FullMask, own, r));
                const bool ok   = __shfl_sync(FullMask, static_cast<int>(own_ok), r) != 0;
                unsigned char* dst = decltype(swizzle)::value ? dst_rows + swizzled(r, Bytes, c) : dst_rows + r * Bytes + 16 * c;
                cp_async_zfill<16, Cache::cg>(dst, src + 16 * c, ok ? 16 : 0);
            }
        };
        const auto rows = [&](unsigned char* dst_rows, const unsigned char* own_row, auto bytes) {
            copy_rows(dst_rows, own_row, valid, bytes, std::true_type{});
        };
        if constexpr (Shape::Vq > 0) {
            // Paged code rows; the exact rows (window slot or staged call row) of the keys this column
            // reads exactly, with their group scales and window tags; the code scales; token and mode.
            copy_rows(k_s + S::VqKeyCodes, k_plane + row_at * Shape::Vq, valid,
                      std::integral_constant<int, Shape::Vq>{}, std::false_type{});
            copy_rows(k_s + S::VqValueCodes, v_plane + row_at * kKVCacheVq2CodeBytes, valid,
                      std::integral_constant<int, kKVCacheVq2CodeBytes>{}, std::false_type{});
            const int mode = valid ? vq_read_mode(vq, token, p, call_first) : 0;
            std::int64_t exact = 0;
            if (mode == 2) { exact = vq_staged_row(vq, kv_head, token - call_first); }
            if (mode == 1) { exact = vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(token)); }
            const std::int8_t* ek = mode == 2 ? vq.staged_k_codes : vq.k_codes;
            const std::int8_t* ev = mode == 2 ? vq.staged_v_codes : vq.v_codes;
            copy_rows(k_s, mode != 0 ? static_cast<const void*>(ek + exact * kD) : k_plane, mode != 0,
                      std::integral_constant<int, kD>{}, std::true_type{});
            copy_rows(v_s, mode != 0 ? static_cast<const void*>(ev + exact * kD) : v_plane, mode != 0,
                      std::integral_constant<int, kD>{}, std::true_type{});
            const int role = lane >> 4;
            int scale_page;
            const auto* code_scales = static_cast<const unsigned char*>(
                role ? qsa_space_plane(spaces, v_scales, spaces.host_v_scale, spaces.lent_v_scale, id, scale_page)
                     : qsa_space_plane(spaces, k_scales, spaces.host_k_scale, spaces.lent_k_scale, id, scale_page));
            cp_async_zfill<4>(k_s + S::VqCodeScales + (2 * row + role) * 4, code_scales + (2 * row_at & ~std::size_t{3}),
                              valid ? 4 : 0);
            if (mode != 0) {
                const __half* groups = mode == 2 ? (role ? vq.staged_v_scales : vq.staged_k_scales)
                                                 : (role ? vq.v_scales : vq.k_scales);
                cp_async<8>(k_s + (role ? S::ValueScales : S::KeyScales) + row * 8, groups + exact * kKVWindowGroups);
            }
            if (mode == 1) { cp_async<4>(k_s + S::VqTags + (2 * row + role) * 4, vq.tags + exact * 2 + role); }
            if (role == 0) {
                reinterpret_cast<int*>(k_s + S::VqMeta)[2 * row]     = token;
                reinterpret_cast<int*>(k_s + S::VqMeta)[2 * row + 1] = mode;
            }
            const unsigned odd = __ballot_sync(FullMask, role == 0 && (token & 1) != 0);
            if (lane == 0) { *reinterpret_cast<unsigned*>(k_s + S::ParityWord) = odd; }
            cp_commit();
            return;
        }
        if constexpr ((Part & kGatherKey) != 0) {
            rows(k_s, k_plane + row_at * K::Bytes + d_w * S::KHalfBytes,
                 std::integral_constant<int, S::KHalfBytes>{});
        }
        if constexpr ((Part & kGatherValue) != 0) {
            rows(v_s, v_plane + row_at * V::Bytes + d_w * S::VHalfBytes,
                 std::integral_constant<int, S::VHalfBytes>{});
        }
        if constexpr ((Part & kGatherKey) != 0 && (K::ScaleSlot > 0 || V::ScaleSlot > 0)) {
            // Lanes 0-15 gather their row's key scales, lanes 16-31 its value scales; both travel with
            // the keys, and the parity word is the keys' token parity.
            const bool key    = lane < 16;
            const auto scales = [&](const void* pool, const void* host, const void* lent, auto orig_slot_t,
                                    auto slot_bytes_t, unsigned char* dst) {
                constexpr int OrigSlot  = decltype(orig_slot_t)::value;
                constexpr int SlotBytes = decltype(slot_bytes_t)::value;
                int scale_page;
                const auto* base = static_cast<const unsigned char*>(qsa_space_plane(spaces, pool, host, lent, id, scale_page));
                if constexpr (OrigSlot == 4) { // one FP16: its aligned word (needed by both D halves)
                    cp_async_zfill<4>(dst, base + (2 * row_at & ~std::size_t{3}), valid ? 4 : 0);
                } else if constexpr (SlotBytes == 4) {
                    cp_async_zfill<4>(dst, base + 8 * row_at + 4 * d_w, valid ? 4 : 0);
                } else if constexpr (SlotBytes == 8) {
                    cp_async_zfill<8>(dst, base + OrigSlot * row_at + 8 * d_w, valid ? 8 : 0);
                } else {
                    cp_async_zfill<16>(dst, base + 16 * row_at, valid ? 16 : 0);
                }
            };
            if (key) {
                if constexpr (K::ScaleSlot > 0) {
                    scales(k_scales, spaces.host_k_scale, spaces.lent_k_scale,
                           std::integral_constant<int, K::ScaleSlot>{}, std::integral_constant<int, S::KScaleSlot>{},
                           k_s + S::KeyScales + row * S::KScaleSlot);
                }
            } else {
                // The value scales travel with the key group: the exact power of two the 8-bit values
                // decode by comes from them, and the QK phase settles it before the value rows
                // themselves are waited for. Tile i's own slot keeps them clear of the next tile's.
                if constexpr (V::ScaleSlot > 0) {
                    scales(v_scales, spaces.host_v_scale, spaces.lent_v_scale,
                           std::integral_constant<int, V::ScaleSlot>{}, std::integral_constant<int, S::VScaleSlot>{},
                           k_s + S::ValueScales + value_scale_at(i) + row * S::VScaleSlot);
                }
            }
            if constexpr (S::Parity) {
                const unsigned odd = __ballot_sync(FullMask, key && (token & 1) != 0);
                if (lane == 0) { *reinterpret_cast<unsigned*>(k_s + S::ParityWord) = odd; }
            }
        }
        cp_commit();
    };

    // The A fragments of k step kb of the query rows. Base addresses for the 4 low chunk-pairs stay in
    // 4 registers across the tile loop; high steps add compile-time constant +128/+256/+384 offsets.
    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = a_rin + ((a_mat & 1) << 3);
    const unsigned q_base = smem_addr(q_s) + static_cast<unsigned>(a_row * kQRow + (SplitD ? 256 * d_w : 0));
    unsigned q_addr[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        q_addr[k] = q_base + static_cast<unsigned>(((2 * k + (a_mat >> 1)) ^ a_rin) << 4);
    }
    const auto query_fragment = [&](int kb_local, unsigned (&a)[4]) {
        ldmatrix_x4(a[0], a[1], a[2], a[3], q_addr[kb_local & 3] + static_cast<unsigned>(128 * (kb_local >> 2)));
    };

    // Output rows g (registers 0/1) and g + 8 (2/3) of 16 or 32 n8 tiles (`HalfBlocks` blocks).
    float o[HalfBlocks][2][4];
#pragma unroll
    for (int b = 0; b < HalfBlocks; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { o[b][h][i] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    // FP16 row scale of tile token `key` from its gathered word.
    const auto row_scale = [&](const unsigned char* k_s, unsigned parity, int scales_at, int key) {
        return reinterpret_cast<const __half*>(k_s + scales_at)[2 * key + ((parity >> key) & 1)];
    };

    // What the QK MMA and the online softmax leave for the PV MMA: the tile's probabilities as the FP16
    // A fragments of one k step (`pa`, scaled by the exact power of two `down`'s inverse where the
    // values are 8-bit codes) and the parity word as of this tile (the single-stage route refills that
    // word for the next tile while the values are being multiplied, so the PV MMA must not read it back
    // from shared memory).
    struct QkState {
        unsigned pa[4];
        __half2 down;
        unsigned parity;
        int scale_at; // this tile's value-scale slot, which the PV MMA reads after the next tile's landed
    };

    // QK MMA over the tile's keys and the online softmax over its 16 tokens (columns past the list are
    // masked), then the PV MMA's A fragments. Needs the key half of the stage plus the value scales,
    // which the key commit group gathers so that the decode rescale is settled before the values land.
    const auto score_tile = [&](int i, int s) {
        const unsigned char* k_s = stages + s * S::Bytes;
        QkState qk;
        qk.parity = 0;
        if constexpr (S::Parity) { qk.parity = *reinterpret_cast<const unsigned*>(k_s + S::ParityWord); }
        const unsigned parity = qk.parity;
        qk.scale_at           = V::ScaleSlot > 0 ? value_scale_at(i) : 0;
        float pr[2][4];
        float score[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { score[nt][j] = 0.0f; }
        }
        if constexpr (Shape::Key == Code::Int8 || Shape::Key == Code::Fp8) {
            constexpr int Groups = SplitD && K::Groups > 1 ? K::Groups / 2 : K::Groups;
            constexpr int Steps  = (SplitD ? 8 : 16) / Groups;
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                float acc[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { acc[nt][j] = 0.0f; }
                }
#pragma unroll
                for (int kk = 0; kk < Steps; ++kk) {
                    const int kb = g * Steps + kk;
                    unsigned a[4];
                    query_fragment(kb, a);
                    unsigned r0, r1;
                    ldmatrix_x2(r0, r1, smem_addr(k_s + swizzled(lane & 15, S::KHalfBytes, kb)));
                    unsigned lo, hi;
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r0, lo, hi);
                    } else {
                        widen_e4m3(r0, lo, hi);
                    }
                    qsa_mma<false>(acc[0], a, lo, hi);
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r1, lo, hi);
                    } else {
                        widen_e4m3(r1, lo, hi);
                    }
                    qsa_mma<false>(acc[1], a, lo, hi);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int key = nt * 8 + 2 * lid;
                    float ks0, ks1;
                    if constexpr (Shape::Key == Code::Int8) {
                        const __half* ks = reinterpret_cast<const __half*>(k_s + S::KeyScales);
                        ks0              = __half2float(ks[key * Groups + g]);
                        ks1              = __half2float(ks[(key + 1) * Groups + g]);
                    } else {
                        ks0 = __half2float(row_scale(k_s, parity, S::KeyScales, key));
                        ks1 = __half2float(row_scale(k_s, parity, S::KeyScales, key + 1));
                    }
                    score[nt][0] = __fmaf_rn(ks0, acc[nt][0], score[nt][0]);
                    score[nt][1] = __fmaf_rn(ks1, acc[nt][1], score[nt][1]);
                    score[nt][2] = __fmaf_rn(ks0, acc[nt][2], score[nt][2]);
                    score[nt][3] = __fmaf_rn(ks1, acc[nt][3], score[nt][3]);
                }
            }
        } else if constexpr (Shape::Key == Code::Nvfp4) {
            // Lane (gid, lid) decodes dimensions 32c + 8 lid .. + 7 of keys gid and gid + 8 with their
            // group scale 2c + lid / 2: exact FP16 values, no per-group partials.
            constexpr int Nvfp4Chunks = SplitD ? 4 : 8;
            unsigned ksc[2][2]{};
            if constexpr (SplitD) {
                const unsigned perm = (lid & 2) != 0 ? 0x7531u : 0x6420u;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint2 raw   = *reinterpret_cast<const uint2*>(k_s + S::KeyScales + (nt * 8 + gid) * 8);
                    const unsigned s4 = __byte_perm(raw.x, raw.y, perm);
                    ksc[nt][0]        = e4m3x2_to_f16x2(s4 & 0xffffu);
                    ksc[nt][1]        = e4m3x2_to_f16x2(s4 >> 16);
                }
            }
#pragma unroll
            for (int c = 0; c < Nvfp4Chunks; ++c) {
                unsigned r[2];
                ldmatrix_x2(r[0], r[1], smem_addr(k_s + swizzled(lane & 15, S::KHalfBytes, c)));
                unsigned b[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    __half2 s2;
                    if constexpr (SplitD) {
                        const unsigned s2_u = __byte_perm(ksc[nt][c >> 1], 0u, (c & 1) != 0 ? 0x3232u : 0x1010u);
                        s2                  = load_vec<__half2>(&s2_u);
                    } else {
                        const int key   = nt * 8 + gid;
                        const __half sc = e4m3_to_half(k_s[S::KeyScales + key * S::KScaleSlot + 2 * c + (lid >> 1)]);
                        s2              = __halves2half2(sc, sc);
                    }
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { b[nt][byte] = e2m1x2_times((r[nt] >> (8 * byte)) & 0xffu, s2); }
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    unsigned a[4];
                    query_fragment(2 * c + half, a);
                    qsa_mma<false>(score[0], a, b[0][2 * half], b[0][2 * half + 1]);
                    qsa_mma<false>(score[1], a, b[1][2 * half], b[1][2 * half + 1]);
                }
            }
        } else {
            const int b_mat = lane >> 3, b_rin = lane & 7;
            const int b_row = b_rin + ((b_mat >> 1) << 3);
            constexpr int Bf16Steps = SplitD ? 8 : 16;
#pragma unroll
            for (int kb = 0; kb < Bf16Steps; ++kb) {
                unsigned a[4];
                query_fragment(kb, a);
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(k_s + swizzled(b_row, S::KHalfBytes, 2 * kb + (b_mat & 1))));
                qsa_mma<true>(score[0], a, b[0], b[1]);
                qsa_mma<true>(score[1], a, b[2], b[3]);
            }
        }

        if constexpr (SplitD) {
            float* ps            = pair_scores + kv_w * 256;
            const auto pair_sync = [&]() {
                if (kv_w == 0) {
                    asm volatile("bar.sync 1, 64;" ::: "memory");
                } else {
                    asm volatile("bar.sync 2, 64;" ::: "memory");
                }
            };
            if (d_w == 1) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { ps[(nt * 4 + j) * 32 + lane] = score[nt][j]; }
                }
            }
            pair_sync();
            if (d_w == 0) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float sum              = score[nt][j] + ps[(nt * 4 + j) * 32 + lane];
                        score[nt][j]                 = sum;
                        ps[(nt * 4 + j) * 32 + lane] = sum;
                    }
                }
            }
            pair_sync();
            if (d_w == 1) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { score[nt][j] = ps[(nt * 4 + j) * 32 + lane]; }
                }
            }
        }

        // Online softmax over the tile's 16 tokens (columns past the list are masked).
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int index = i * kTile + nt * 8 + 2 * lid;
            if (index >= total) {
                score[nt][0] = -CUDART_INF_F;
                score[nt][2] = -CUDART_INF_F;
            }
            if (index + 1 >= total) {
                score[nt][1] = -CUDART_INF_F;
                score[nt][3] = -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        const float nm0    = fmaxf(m0, bm0);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
        m0                 = nm0;
        const float base0  = nm0 * scale_l2;
        float bl0          = 0.0f;
        const float nm1    = fmaxf(m1, bm1);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);
        m1                 = nm1;
        const float base1  = nm1 * scale_l2;
        float bl1          = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -base0));
            pr[nt][1] = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -base0));
            bl0 += pr[nt][0] + pr[nt][1];
            pr[nt][2] = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -base1));
            pr[nt][3] = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -base1));
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        // The output rescales are pure scaling: when no row's max moved this tile every alpha is
        // exp2_approx(0) = 1 exactly and the whole loop can be skipped (warp-uniform, bits unchanged).
        if (!__all_sync(FullMask, alpha0 == 1.0f && alpha1 == 1.0f)) {
#pragma unroll
            for (int b = 0; b < HalfBlocks; ++b) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    o[b][h][0] *= alpha0;
                    o[b][h][1] *= alpha0;
                    o[b][h][2] *= alpha1;
                    o[b][h][3] *= alpha1;
                }
            }
        }
        // The PV MMA's A fragments: this tile's probabilities as one FP16 k step. Where the values are
        // 8-bit codes the decoded values stay finite in FP16 by dividing a tile's scales by 2^shift and
        // multiplying the probabilities by it (both exact, so every product is unchanged); the shift
        // comes from the value scales and is settled here, before the value rows themselves are waited
        // for, so the PV MMA starts on its loads.
        if constexpr (S::EvenOddValues) {
            float own;
            if constexpr (Shape::Value == Code::Int8) {
                const __half* vs = reinterpret_cast<const __half*>(k_s + S::ValueScales + qk.scale_at);
                if constexpr (SplitD) {
                    if (lane < 16) {
                        const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
                        own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
                    } else {
                        own = 0.0f;
                    }
                } else {
                    const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
                    own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
                }
            } else {
                own = lane < 16
                          ? fabsf(__half2float(row_scale(k_s, parity, S::ValueScales + qk.scale_at, lane)))
                          : 0.0f;
            }
            const float vmax = warp_max(own, FullMask);
            int shift        = 0;
            while (shift < 14 && ldexpf(vmax, -shift) > V::MaxScale) { ++shift; }
            const float up = ldexpf(1.0f, shift);
            qk.down        = __float2half2_rn(ldexpf(1.0f, -shift));
            qk.pa[0]       = pack_f16x2(pr[0][0] * up, pr[0][1] * up);
            qk.pa[1]       = pack_f16x2(pr[0][2] * up, pr[0][3] * up);
            qk.pa[2]       = pack_f16x2(pr[1][0] * up, pr[1][1] * up);
            qk.pa[3]       = pack_f16x2(pr[1][2] * up, pr[1][3] * up);
        } else {
            qk.down  = __half2half2(__float2half(0.0f));
            qk.pa[0] = pack_f16x2(pr[0][0], pr[0][1]);
            qk.pa[1] = pack_f16x2(pr[0][2], pr[0][3]);
            qk.pa[2] = pack_f16x2(pr[1][0], pr[1][1]);
            qk.pa[3] = pack_f16x2(pr[1][2], pr[1][3]);
        }
        return qk;
    };

    // PV MMA over the tile's values with the A fragments of `qk`; needs the value rows and scales and
    // nothing else the key half holds.
    const auto value_tile = [&](int s, const QkState& qk) {
        const unsigned char* k_s = stages + s * S::Bytes;
        const unsigned char* v_s = k_s + S::Values;
        const unsigned parity    = qk.parity;
        const unsigned (&pa)[4]  = qk.pa;

        if constexpr (S::EvenOddValues) {
            constexpr int Groups = SplitD && V::Groups > 1 ? V::Groups / 2 : V::Groups;
            const __half2 down = qk.down;
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
            const auto value_scale = [&](int key, int g) {
                if constexpr (Shape::Value == Code::Int8) {
                    return reinterpret_cast<const __half*>(k_s + S::ValueScales + qk.scale_at)[key * Groups + g];
                } else {
                    return row_scale(k_s, parity, S::ValueScales + qk.scale_at, key);
                }
            };
#pragma unroll
            for (int db = 0; db < HalfBlocks; db += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(v_s + swizzled(v_row, S::VHalfBytes, db + (v_mat >> 1))));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g       = (db + h) * Groups / HalfBlocks;
                    const int key     = 2 * lid;
                    const __half2 s01 = __hmul2(__halves2half2(value_scale(key, g), value_scale(key + 1, g)), down);
                    const __half2 s89 = __hmul2(__halves2half2(value_scale(key + 8, g), value_scale(key + 9, g)), down);
                    unsigned e0, o0, e1, o1;
                    decode_v_pair<Shape::Value>(r[2 * h], s01, e0, o0);
                    decode_v_pair<Shape::Value>(r[2 * h + 1], s89, e1, o1);
                    float* even = o[db + h][0];
                    float* odd  = o[db + h][1];
                    qsa_mma<false>(even, pa, e0, e1);
                    qsa_mma<false>(odd, pa, o0, o1);
                }
            }
        } else {
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
            // P x V over FP16 rows of `row_bytes` holding dimensions 8 c0 .. 8 c1 - 1 (16-byte chunks).
            const auto pv = [&]<int C0, int C1>(const unsigned char* fp16_rows, int row_bytes) {
#pragma unroll
                for (int c = C0; c < C1; c += 2) {
                    unsigned r[4];
                    ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                  smem_addr(fp16_rows + swizzled(v_row, row_bytes, c - C0 + (v_mat >> 1))));
                    float* first  = o[c >> 1][0];
                    float* second = o[c >> 1][1];
                    qsa_mma<false>(first, pa, r[0], r[1]);
                    qsa_mma<false>(second, pa, r[2], r[3]);
                }
            };
            if constexpr (Shape::Value == Code::Nvfp4) {
                const int key  = 2 * lid;
                const int g_in = gid >> 2;
                constexpr int Nvfp4Chunks = SplitD ? 4 : 8;
                const unsigned vs_perm    = g_in != 0 ? 0x0031u : 0x0020u;
#pragma unroll
                for (int c = 0; c < Nvfp4Chunks; c += 2) {
                    unsigned r[4];
                    ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                  smem_addr(v_s + swizzled(v_row, S::VHalfBytes, c + (v_mat >> 1))));
                    const unsigned char* vs_c = k_s + S::ValueScales + qk.scale_at + 2 * c;
                    const auto load_v_pair    = [&](int row) {
                        const unsigned raw = *reinterpret_cast<const unsigned*>(vs_c + row * S::VScaleSlot);
                        return e4m3x2_to_f16x2(__byte_perm(raw, 0u, vs_perm));
                    };
                    const unsigned p0 = load_v_pair(key), p1 = load_v_pair(key + 1);
                    const unsigned p8 = load_v_pair(key + 8), p9 = load_v_pair(key + 9);
#pragma unroll
                    for (int hc = 0; hc < 2; ++hc) {
                        const int chunk     = c + hc;
                        const unsigned sel  = hc != 0 ? 0x3232u : 0x1010u;
                        const unsigned s0_u = __byte_perm(p0, 0u, sel);
                        const unsigned s1_u = __byte_perm(p1, 0u, sel);
                        const unsigned s8_u = __byte_perm(p8, 0u, sel);
                        const unsigned s9_u = __byte_perm(p9, 0u, sel);
                        unsigned b0[4], b1[4];
                        decode_v_quad_nvfp4(r[2 * hc], load_vec<__half2>(&s0_u), load_vec<__half2>(&s1_u), b0);
                        decode_v_quad_nvfp4(r[2 * hc + 1], load_vec<__half2>(&s8_u), load_vec<__half2>(&s9_u), b1);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            float* dst = o[2 * chunk + (u >> 1)][u & 1];
                            qsa_mma<false>(dst, pa, b0[u], b1[u]);
                        }
                    }
                }
            } else {
                constexpr int Fp16Chunks = SplitD ? 16 : 32;
                pv.template operator()<0, Fp16Chunks>(v_s, S::VHalfBytes);
            }
        }
    };

    // Vector-quantized storages: settle stage st's rows (exact rows kept, codes decoded).
    const auto expand = [&](int st) {
        if constexpr (Shape::Vq > 0) {
            // Lane l takes row l % 16 of role l / 16 (K, V): its exact row stays when the read is from
            // the call's staged row or the window slot's tag matches the stored codes; otherwise the
            // codes are decoded into the INT8 row and the code scale becomes every group's scale.
            unsigned char* k_s = stages + st * S::Bytes;
            const int row = lane & 15, role = lane >> 4;
            const int token = reinterpret_cast<const int*>(k_s + S::VqMeta)[2 * row];
            const int mode  = reinterpret_cast<const int*>(k_s + S::VqMeta)[2 * row + 1];
            const unsigned parity  = (*reinterpret_cast<const unsigned*>(k_s + S::ParityWord) >> row) & 1U;
            const __half code_scale =
                reinterpret_cast<const __half*>(k_s + S::VqCodeScales + (2 * row + role) * 4)[parity];
            const unsigned char* codes = role ? k_s + S::VqValueCodes + row * kKVCacheVq2CodeBytes
                                              : k_s + S::VqKeyCodes + row * Shape::Vq;
            bool keep = mode == 2;
            if (mode == 1) {
                const std::uint32_t bits = __half_as_ushort(code_scale);
                const std::uint32_t tag =
                    role == 0 && Shape::Vq == kKVCacheQ4CodeBytes
                        ? kv_window_tag<kKVCacheQ4CodeBytes / 4>(token, reinterpret_cast<const std::uint32_t*>(codes), bits)
                        : kv_window_tag<kKVCacheVq2CodeBytes / 4>(token, reinterpret_cast<const std::uint32_t*>(codes), bits);
                keep = static_cast<std::uint32_t>(reinterpret_cast<const std::int32_t*>(k_s + S::VqTags)[2 * row + role]) == tag;
            }
            if (!keep) {
                unsigned char* rows_base = role ? k_s + S::Values : k_s;
#pragma unroll 4
                for (int w = 0; w < 32; ++w) {
                    const uint2 values = role == 0 && Shape::Vq == kKVCacheQ4CodeBytes
                                             ? kv_cache_q4_decode_word(reinterpret_cast<const std::uint32_t*>(codes)[w])
                                             : kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(codes)[w], codebook);
                    *reinterpret_cast<uint2*>(rows_base + swizzled(row, kD, w >> 1) + 8 * (w & 1)) = values;
                }
                __half* groups = reinterpret_cast<__half*>(k_s + (role ? S::ValueScales : S::KeyScales) + row * 8);
#pragma unroll
                for (int g = 0; g < kKVWindowGroups; ++g) { groups[g] = code_scale; }
            }
            __syncwarp();
        }
    };
    // Splitting the gather pays where a tile's key row is at least one byte per dimension (16 scattered
    // 16-byte chunks): the gather is then long enough to hide behind the phase that does not read it.
    // With 128-byte rows (NVFP4 keys) the second addressing, commit-group and wait per tile measured
    // 3.5-5 % slower than the combined gather, so `Stage::SplitGather` keeps one group for it.
    if constexpr (Stages == 1 && S::SplitGather) {
        // One stage per warp: the key half and the value half are separate commit groups, so a
        // tile's values stream in behind its QK MMA and online softmax and the next tile's keys behind
        // the PV MMA. Each half of the stage is refilled only after the phase that reads it is done (the
        // `__syncwarp` barriers), and `cp_wait<1>` leaves exactly the one group the other phase still
        // needs in flight. The other warps (this CTA's and the other CTAs') hide what is left.
        if (kv_w < tiles) { issue(kv_w, 0, std::integral_constant<int, kGatherKey>{}); }
        for (int i = kv_w; i < tiles; i += WarpsKV) {
            issue(i, 0, std::integral_constant<int, kGatherValue>{});
            cp_wait<1>();  // this tile's keys are staged; its values may still be streaming
            __syncwarp();
            const QkState qk = score_tile(i, 0);
            __syncwarp();  // every lane is done with the key rows before the next tile's overwrite
            if (i + WarpsKV < tiles) {
                issue(i + WarpsKV, 0, std::integral_constant<int, kGatherKey>{});
                cp_wait<1>();  // this tile's values are staged; the next tile's keys may still stream
            } else {
                cp_wait<0>();
            }
            __syncwarp();
            value_tile(0, qk);
            __syncwarp();  // the value rows are read out before the next tile's overwrite
        }
    } else if constexpr (Stages == 1) {
        // The other warps hide each tile's gather, whose keys and values stay one commit group (see
        // `Stage::SplitGather`).
        for (int i = kv_w; i < tiles; i += WarpsKV) {
            issue(i, 0, std::integral_constant<int, kGatherBoth>{});
            cp_wait<0>();
            __syncwarp();
            expand(0);
            const QkState qk = score_tile(i, 0);
            value_tile(0, qk);
            __syncwarp();
        }
    } else {
        int s = 0;
        if (kv_w < tiles) { issue(kv_w, 0, std::integral_constant<int, kGatherBoth>{}); }
        for (int i = kv_w; i < tiles; i += WarpsKV) {
            if (i + WarpsKV < tiles) {
                issue(i + WarpsKV, s ^ 1, std::integral_constant<int, kGatherBoth>{});
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncwarp();
            expand(s);
            const QkState qk = score_tile(i, s);
            value_tile(s, qk);
            __syncwarp();
            s ^= 1;
        }
    }
    cp_wait<0>();

    // Merge the KV warps' rows in warp order: every KV pair publishes its (max, sum) per row, then the
    // KV pairs add their scaled unnormalized rows into one FP32 buffer in turn (both D halves writing
    // disjoint slices in parallel when SplitD). Rows overlays dead q_s at offset 0.
    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem); // [Rows][kD]
    float* stats = rows + Rows * kD;               // [WarpsKV][Rows][2]
    if (d_w == 0 && lid == 0) {
        stats[(kv_w * Rows + gid) * 2]         = m0;
        stats[(kv_w * Rows + gid) * 2 + 1]     = l0;
        stats[(kv_w * Rows + gid + 8) * 2]     = m1;
        stats[(kv_w * Rows + gid + 8) * 2 + 1] = l1;
    }
    __syncthreads();
    const auto row_max = [&](int row) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WarpsKV; ++w) { m = fmaxf(m, stats[(w * Rows + row) * 2]); }
        return m;
    };
    const float f0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - row_max(gid)) * scale_l2);
    const float f1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - row_max(gid + 8)) * scale_l2);
#pragma unroll 1
    for (int kw = 0; kw < WarpsKV; ++kw) {
        if (kv_w == kw) {
#pragma unroll
            for (int b = 0; b < HalfBlocks; ++b) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int upper = j >> 1;
                        const int row   = gid + 8 * upper;
                        const int d     = (SplitD ? 128 * d_w : 0) +
                                          (Shape::Value == Code::Nvfp4
                                               ? 32 * (b >> 1) + 8 * lid + 4 * (j & 1) + 2 * (b & 1) + h
                                               : (S::EvenOddValues ? 16 * b + 4 * lid + 2 * (j & 1) + h
                                                                   : 16 * b + 8 * h + 2 * lid + (j & 1)));
                        const bool live = (upper ? m1 : m0) != -CUDART_INF_F;
                        float& a        = rows[row * kD + d];
                        if (kw == 0) {
                            a = live ? __fmaf_rn(upper ? f1 : f0, o[b][h][j], 0.0f) : 0.0f;
                        } else if (live) {
                            a = __fmaf_rn(upper ? f1 : f0, o[b][h][j], a);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    for (int e = tid; e < group * kD; e += Threads) {
        const int row = e / kD, d = e % kD;
        const float m = row_max(row);
        float l       = 0.0f;
#pragma unroll
        for (int w = 0; w < WarpsKV; ++w) {
            const float mw = stats[(w * Rows + row) * 2];
            if (mw == -CUDART_INF_F) { continue; }
            l = __fmaf_rn(exp2_approx((mw - m) * scale_l2), stats[(w * Rows + row) * 2 + 1], l);
        }
        const float value = l > 0.0f ? rows[row * kD + d] / l : 0.0f;
        if constexpr (S::RotatedValues) {
            rows[row * kD + d] = value;
        } else {
            const int head_idx     = kv_head * group + row;
            const std::int64_t g_i = static_cast<std::int64_t>(t) * gate_col_stride + head_idx * kD + d;
            out[(static_cast<std::size_t>(t) * heads + head_idx) * kD + d] =
                apply_qsa_gate(__float2bfloat16_rn(value), gate, g_i);
        }
    }
    if constexpr (S::RotatedValues) {
        // Values are stored in the normalized Hadamard domain (its own inverse): rotate back.
        __syncthreads();
        for (int row = warp; row < group; row += Warps) {
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) { v[k] = rows[row * kD + lane + 32 * k]; }
            normalized_hadamard_d256_inplace(v, lane);
            const int head_idx = kv_head * group + row;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d            = lane + 32 * k;
                const std::int64_t g_i = static_cast<std::int64_t>(t) * gate_col_stride + head_idx * kD + d;
                out[(static_cast<std::size_t>(t) * heads + head_idx) * kD + d] =
                    apply_qsa_gate(__float2bfloat16_rn(v[k]), gate, g_i);
            }
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa prompt: ") + message); }
}

void resolve_gate(const Tensor* gate, int heads, int columns, const bf16*& gate_ptr, std::int64_t& gate_col_stride) {
    gate_ptr        = nullptr;
    gate_col_stride = 0;
    if (gate == nullptr || gate->data == nullptr) { return; }
    require(gate->dtype == DType::BF16 && gate->nb[0] == 2, "gate must be BF16 with contiguous inner elements");
    if (gate->ne[0] == heads * kD && gate->ne[2] == 1 && gate->ne[3] == 1 &&
        (gate->ne[1] == columns || (columns == 1 && gate->ne[1] == 1))) {
        gate_ptr        = static_cast<const bf16*>(gate->data);
        gate_col_stride = columns == 1 ? 0 : static_cast<std::int64_t>(gate->nb[1] / 2);
        return;
    }
    if (gate->is_contiguous() && gate->ne[0] == kD && gate->ne[1] == heads && gate->ne[2] == columns &&
        gate->ne[3] == 1) {
        gate_ptr        = static_cast<const bf16*>(gate->data);
        gate_col_stride = static_cast<std::int64_t>(heads) * kD;
        return;
    }
    require(false, "gate shape disagrees with attention output");
}

template <KvCacheStorage Storage>
void launch_grouped_prefix(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g,
                           float scale, const std::int32_t* counts, const bf16* gate_ptr,
                           std::int64_t gate_col_stride, Tensor& out, cudaStream_t stream) {
    constexpr int bytes = grouped_smem_bytes<Storage>();
    static_assert(bytes <= 99 * 1024, "grouped prefix CTA shared memory on sm_120/sm_121");
    static const cudaError_t configured = [&] {
        const cudaError_t e1 = cudaFuncSetAttribute(
            qsa_grouped_prefix_kernel<Storage>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e1 != cudaSuccess) { return e1; }
        return cudaFuncSetAttribute(qsa_grouped_prefix_kernel<Storage>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    }();
    require(configured == cudaSuccess, "cannot reserve grouped prefix shared memory");
    constexpr int Parts    = (Storage != KvCacheStorage::BFloat16) ? 2 : 1;
    constexpr int Threads  = (kGroupRows / 16) * Parts * 32;
    const int columns      = q.ne[2];
    const int groups_row   = (batch.width + kGroupQueries - 1) / kGroupQueries;
    const int total_groups = batch.batch * groups_row;
    const dim3 grid(static_cast<unsigned>(total_groups), static_cast<unsigned>(g.kv_heads));
    qsa_grouped_prefix_kernel<Storage><<<grid, Threads, bytes, stream>>>(
        static_cast<const bf16*>(q.data), g.heads, g.kv_heads, layer.kv.k_pages.data, layer.kv.v_pages.data,
        layer.kv.k_scale_pages.data, layer.kv.v_scale_pages.data,
        static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
        static_cast<const std::int32_t*>(batch.table_rows.data),
        static_cast<const std::int32_t*>(batch.positions.data), batch.width, columns, counts, scale, gate_ptr,
        gate_col_stride, static_cast<bf16*>(out.data), layer.spaces);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa grouped prefix attention: ") + cudaGetErrorString(error));
    }
}

template <KvCacheStorage Storage, bool SplitD, bool SkipDenseGroups = false>
void launch_split(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g, float scale,
                  const std::int32_t* selected, const std::int32_t* counts, const bf16* gate_ptr,
                  std::int64_t gate_col_stride, Tensor& out, cudaStream_t stream, const QsaVqWindow& vq,
                  int stages_choice) {
    const auto run = [&]<int Stages>() {
        constexpr int bytes = prompt_smem_bytes<Storage, Stages, SplitD>();
        static_assert(bytes <= 99 * 1024, "one CTA's shared memory on sm_120");
        static const cudaError_t configured = [&] {
            const cudaError_t e1 = cudaFuncSetAttribute(
                qsa_prompt_kernel<Storage, Stages, SplitD, SkipDenseGroups>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
            if (e1 != cudaSuccess) { return e1; }
            return cudaFuncSetAttribute(qsa_prompt_kernel<Storage, Stages, SplitD, SkipDenseGroups>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        }();
        require(configured == cudaSuccess, "cannot reserve the kernel's shared memory");
        const int columns = q.ne[2];
        const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(g.kv_heads));
        qsa_prompt_kernel<Storage, Stages, SplitD, SkipDenseGroups>
            <<<grid, prompt_warps<Storage, SplitD>() * 32, bytes, stream>>>(
                static_cast<const bf16*>(q.data), g.heads, g.kv_heads, layer.kv.k_pages.data, layer.kv.v_pages.data,
                layer.kv.k_scale_pages.data, layer.kv.v_scale_pages.data,
                static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
                static_cast<const std::int32_t*>(batch.table_rows.data),
                static_cast<const std::int32_t*>(batch.positions.data), batch.width, g.ratio, selected, counts,
                g.budget / g.ratio, scale, gate_ptr, gate_col_stride, static_cast<bf16*>(out.data), layer.spaces, vq);
    };
    if (stages_choice == 1) {
        run.template operator()<1>();
    } else {
        run.template operator()<2>();
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa prompt attention: ") + cudaGetErrorString(error)); }
}

template <KvCacheStorage Storage>
void launch(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g, float scale,
            const std::int32_t* selected, const std::int32_t* counts, const bf16* gate_ptr,
            std::int64_t gate_col_stride, Tensor& out, cudaStream_t stream, const QsaVqWindow& vq,
            bool allow_fast_paths) {
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
            throw std::runtime_error("qsa prompt: cannot query the multiprocessor count");
        }
        return count;
    }();
    const int forced = qsa_prompt_forced();
    const bool use_split_d = [&] {
        if constexpr (PromptShape<Storage>::Vq > 0) {
            return false;
        } else if (forced >= 0) {
            return forced != 0;
        } else {
            return qsa_prompt_split(Storage) && (Storage == KvCacheStorage::BFloat16 || q.ne[2] * g.kv_heads >= sms);
        }
    }();
    const auto pick_stages = [&]<bool SplitD>() {
        constexpr int blocks_1 = prompt_launch_min_blocks<Storage, 1, SplitD>();
        constexpr int blocks_2 = prompt_launch_min_blocks<Storage, 2, SplitD>();
        if constexpr (PromptShape<Storage>::Vq > 0) {
            if (forced == 0) { return 2; }
            if (forced == 1) { return 1; }
        }
        if constexpr (Storage == KvCacheStorage::Int8Group64 || Storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            return 2;
        }
        return (blocks_1 > blocks_2 && q.ne[2] * g.kv_heads > blocks_2 * sms) ? 1 : 2;
    };
    if constexpr (PromptShape<Storage>::Vq == 0) {
        const bool use_grouped = allow_fast_paths && forced < 0 && g.heads == 24 && g.kv_heads == 2 &&
                                 q.ne[2] >= 256 && batch.width >= kGroupQueries &&
                                 (kPagedKVPageSize % kGroupKeyTile) == 0;
        if (use_grouped) {
            launch_grouped_prefix<Storage>(q, layer, batch, g, scale, counts, gate_ptr, gate_col_stride, out, stream);
            if (use_split_d) {
                launch_split<Storage, true, true>(q, layer, batch, g, scale, selected, counts, gate_ptr,
                                                  gate_col_stride, out, stream, vq,
                                                  pick_stages.template operator()<true>());
            } else {
                launch_split<Storage, false, true>(q, layer, batch, g, scale, selected, counts, gate_ptr,
                                                   gate_col_stride, out, stream, vq,
                                                   pick_stages.template operator()<false>());
            }
            return;
        }
        if (use_split_d) {
            launch_split<Storage, true>(q, layer, batch, g, scale, selected, counts, gate_ptr, gate_col_stride, out,
                                        stream, vq, pick_stages.template operator()<true>());
            return;
        }
    }
    launch_split<Storage, false>(q, layer, batch, g, scale, selected, counts, gate_ptr, gate_col_stride, out, stream,
                                 vq, pick_stages.template operator()<false>());
}

} // namespace

bool qsa_prompt_supported(const QsaKVLayer& layer, const QsaGeometry& geometry) {
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::Vq2:
    case KvCacheStorage::Q4KeyVq2Value:
        return geometry.head_dim == kD && geometry.kv_heads > 0 && geometry.heads / geometry.kv_heads <= kRows;
    default:
        return false;
    }
}

void qsa_prompt_attention(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch,
                          const QsaGeometry& geometry, float scale, const std::int32_t* selected,
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream, const QsaVqWindow& vq,
                          const Tensor* gate, bool allow_fast_paths) {
    require(qsa_prompt_supported(layer, geometry), "the prompt route does not serve this layer");
    const bf16* gate_ptr         = nullptr;
    std::int64_t gate_col_stride = 0;
    resolve_gate(gate, geometry.heads, q.ne[2], gate_ptr, gate_col_stride);
    switch (layer.kv.storage) {
    case KvCacheStorage::Int8Group64:
        launch<KvCacheStorage::Int8Group64>(q, layer, batch, geometry, scale, selected, counts, gate_ptr,
                                            gate_col_stride, out, stream, vq, allow_fast_paths);
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch<KvCacheStorage::Fp8E4M3Row256>(q, layer, batch, geometry, scale, selected, counts, gate_ptr,
                                              gate_col_stride, out, stream, vq, allow_fast_paths);
        break;
    case KvCacheStorage::Nvfp4Group16:
        launch<KvCacheStorage::Nvfp4Group16>(q, layer, batch, geometry, scale, selected, counts, gate_ptr,
                                             gate_col_stride, out, stream, vq, allow_fast_paths);
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch<KvCacheStorage::Fp8KeyNvfp4Value>(q, layer, batch, geometry, scale, selected, counts, gate_ptr,
                                                 gate_col_stride, out, stream, vq, allow_fast_paths);
        break;
    case KvCacheStorage::Vq2:
        launch<KvCacheStorage::Vq2>(q, layer, batch, geometry, scale, selected, counts, gate_ptr, gate_col_stride,
                                    out, stream, vq, allow_fast_paths);
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        launch<KvCacheStorage::Q4KeyVq2Value>(q, layer, batch, geometry, scale, selected, counts, gate_ptr,
                                              gate_col_stride, out, stream, vq, allow_fast_paths);
        break;
    default:
        launch<KvCacheStorage::BFloat16>(q, layer, batch, geometry, scale, selected, counts, gate_ptr, gate_col_stride,
                                         out, stream, vq, allow_fast_paths);
        break;
    }
}

namespace {

// qsa_prompt_force's variant: -1 by storage.
std::atomic<int>& forced_variant() {
    static std::atomic<int> state{-1};
    return state;
}

} // namespace

bool qsa_prompt_split(KvCacheStorage storage) {
    const int forced = forced_variant().load(std::memory_order_relaxed);
    if (forced >= 0) { return forced != 0; }
    return storage != KvCacheStorage::Vq2 && storage != KvCacheStorage::Q4KeyVq2Value;
}

void qsa_prompt_force(int variant) {
    if (variant < -1 || variant > 1) { throw std::invalid_argument("qsa_prompt_force: -1, 0 or 1"); }
    forced_variant().store(variant, std::memory_order_relaxed);
}

int qsa_prompt_forced() { return forced_variant().load(std::memory_order_relaxed); }

} // namespace infernix::ops::detail
