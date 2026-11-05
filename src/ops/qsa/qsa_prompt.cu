// QSA prompt attention on Tensor Cores (see qsa_prompt.h).

#include "ops/qsa/qsa_prompt.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
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

// One warp stage: [K rows | V rows | K scales | V scales | parity word], 16 tokens each; rows keep
// their 16-byte chunks swizzled. The parity word's bit j is token j's parity (FP8 row scales).
template <KvCacheStorage Storage>
struct Stage {
    using Shape                     = PromptShape<Storage>;
    using K                         = Row<Shape::Key>;
    using V                         = Row<Shape::Value>;
    static constexpr bool Parity    = Shape::Key == Code::Fp8 || Shape::Value == Code::Fp8 || Shape::Vq > 0;
    static constexpr int Values     = kTile * K::Bytes;
    static constexpr int KeyScales  = Values + kTile * V::Bytes;
    static constexpr int ValueScales = KeyScales + kTile * K::ScaleSlot;
    // The value scales get two slots where the route gathers them with the keys: the QK phase settles
    // the 8-bit values' exact decode rescale from them, so they must land before the value rows, while
    // the next tile's are already being gathered as this tile's PV MMA reads them. A stage the combined
    // route fills (a vector-quantized stage, or a storage whose gather the split does not pay for) needs
    // one, and keeps its original footprint.
    static constexpr bool SplitGather = Shape::Vq == 0 && K::Bytes >= kD;
    static constexpr int ScaleSlots   = SplitGather ? 2 : 1;
    static constexpr int ParityWord   = ValueScales + ScaleSlots * kTile * V::ScaleSlot;
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

// Shared memory: the query rows, then each warp's `Stages` stages (and NVFP4 value tile), later the merge:
// one FP32 row buffer the warps accumulate into and every warp's (max, sum) per row; then the codebook.
template <KvCacheStorage Storage, int Stages>
__host__ __device__ constexpr int prompt_smem_bytes() {
    using S              = Stage<Storage>;
    constexpr int Warps  = PromptShape<Storage>::Warps;
    constexpr int stages = Warps * (Stages * S::Bytes + S::Scratch);
    constexpr int merge  = (kRows * kD + Warps * kRows * 2) * 4;
    constexpr int book   = PromptShape<Storage>::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0;
    return kQRow * kRows + (stages > merge ? stages : merge) + book;
}

// 16-byte chunk c of row r lives at chunk c ^ (r & 7): the eight rows an ldmatrix reads hit
// distinct banks.
__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
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
__device__ __forceinline__ void decode_v_quad_nvfp4(unsigned codes, __half s0, __half s1, unsigned (&out)[4]) {
    const __half2 s0_pair = __halves2half2(s0, s0);
    const __half2 s1_pair = __halves2half2(s1, s1);
    const unsigned w0_01  = e2m1x2_times(codes & 0xffu, s0_pair);
    const unsigned w1_01  = e2m1x2_times((codes >> 16) & 0xffu, s1_pair);
    out[0]                = __byte_perm(w0_01, w1_01, 0x5410);
    out[1]                = __byte_perm(w0_01, w1_01, 0x7632);
    const unsigned w0_23  = e2m1x2_times((codes >> 8) & 0xffu, s0_pair);
    const unsigned w1_23  = e2m1x2_times(codes >> 24, s1_pair);
    out[2]                = __byte_perm(w0_23, w1_23, 0x5410);
    out[3]                = __byte_perm(w0_23, w1_23, 0x7632);
}

template <KvCacheStorage Storage, int Stages>
__global__ void __launch_bounds__(PromptShape<Storage>::Warps * 32, Stages == 1 ? 2 : 1)
    qsa_prompt_kernel(const bf16* __restrict__ q, int heads, int kv_heads, const void* __restrict__ k_pages,
                      const void* __restrict__ v_pages, const void* __restrict__ k_scales,
                      const void* __restrict__ v_scales, const std::int32_t* __restrict__ tables,
                      int table_stride, const std::int32_t* __restrict__ table_rows,
                      const std::int32_t* __restrict__ positions, int width, int ratio,
                      const std::int32_t* __restrict__ selected, const std::int32_t* __restrict__ counts,
                      int top_blocks, float scale, bf16* __restrict__ out, QsaPageSpaces spaces, QsaVqWindow vq) {
    using Shape                 = PromptShape<Storage>;
    using S                     = Stage<Storage>;
    using K                     = Row<Shape::Key>;
    using V                     = Row<Shape::Value>;
    constexpr int Warps         = Shape::Warps;
    constexpr int Threads       = Warps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];

    const int t = static_cast<int>(blockIdx.x), kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
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
    for (int r = warp; r < kRows; r += Warps) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = r < group ? __bfloat162float(
                                   q[(static_cast<std::size_t>(t) * heads + kv_head * group + r) * kD + lane + 32 * k])
                             : 0.0f;
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
    }
    __syncthreads();

    // VQ2 codebook (vector-quantized storages): behind the stages and the merge rows.
    constexpr int kCodebookAt = prompt_smem_bytes<Storage, Stages>() -
                                (Shape::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0);
    const auto* codebook = reinterpret_cast<const std::int8_t*>(smem + kCodebookAt);
    if constexpr (Shape::Vq > 0) {
        for (int i = tid; i < static_cast<int>(sizeof(kKVCacheVq2Codebook)) / 16; i += Threads) {
            reinterpret_cast<uint4*>(const_cast<std::int8_t*>(codebook))[i] =
                reinterpret_cast<const uint4*>(g_kv_cache_vq2_codebook)[i];
        }
        __syncthreads();
    }
    const int call_first = positions[(t / width) * width];
    const int window_row = vq.slots != nullptr ? vq.slots[t / width] : 0;
    unsigned char* stages = smem + kQRow * kRows + warp * (Stages * S::Bytes + S::Scratch);
    const int tiles       = (total + kTile - 1) / kTile;

    // The byte offset of tile i's value-scale slot. Consecutive tiles of one warp use different slots:
    // the single-stage route gathers a tile's value scales with its keys, so the next tile's are already
    // in flight while this tile's PV MMA still reads them.
    const auto value_scale_at = [&](int i) {
        return ((i / Warps) & (S::ScaleSlots - 1)) * (kTile * V::ScaleSlot);
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
            rows(k_s, k_plane + row_at * K::Bytes, std::integral_constant<int, K::Bytes>{});
        }
        if constexpr ((Part & kGatherValue) != 0) {
            rows(v_s, v_plane + row_at * V::Bytes, std::integral_constant<int, V::Bytes>{});
        }
        if constexpr ((Part & kGatherKey) != 0 && (K::ScaleSlot > 0 || V::ScaleSlot > 0)) {
            // Lanes 0-15 gather their row's key scales, lanes 16-31 its value scales; both travel with
            // the keys, and the parity word is the keys' token parity.
            const bool key    = lane < 16;
            const auto scales = [&](const void* pool, const void* host, const void* lent, int slot_bytes,
                                    unsigned char* dst) {
                int scale_page;
                const auto* base = static_cast<const unsigned char*>(qsa_space_plane(spaces, pool, host, lent, id, scale_page));
                if (slot_bytes == 4) { // one FP16: its aligned word
                    cp_async_zfill<4>(dst, base + (2 * row_at & ~std::size_t{3}), valid ? 4 : 0);
                } else if (slot_bytes == 8) {
                    cp_async_zfill<8>(dst, base + 8 * row_at, valid ? 8 : 0);
                } else {
                    cp_async_zfill<16>(dst, base + 16 * row_at, valid ? 16 : 0);
                }
            };
            if (key) {
                if constexpr (K::ScaleSlot > 0) {
                    scales(k_scales, spaces.host_k_scale, spaces.lent_k_scale, K::ScaleSlot,
                           k_s + S::KeyScales + row * K::ScaleSlot);
                }
            } else {
                // The value scales travel with the key group: the exact power of two the 8-bit values
                // decode by comes from them, and the QK phase settles it before the value rows
                // themselves are waited for. Tile i's own slot keeps them clear of the next tile's.
                if constexpr (V::ScaleSlot > 0) {
                    scales(v_scales, spaces.host_v_scale, spaces.lent_v_scale, V::ScaleSlot,
                           k_s + S::ValueScales + value_scale_at(i) + row * V::ScaleSlot);
                }
            }
            if constexpr (S::Parity) {
                const unsigned odd = __ballot_sync(FullMask, key && (token & 1) != 0);
                if (lane == 0) { *reinterpret_cast<unsigned*>(k_s + S::ParityWord) = odd; }
            }
        }
        cp_commit();
    };

    // The A fragments of k step kb of the query rows.
    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = a_rin + ((a_mat & 1) << 3);
    const auto query_fragment = [&](int kb, unsigned (&a)[4]) {
        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(q_s + swizzled(a_row, kQRow, 2 * kb + (a_mat >> 1))));
    };

    // Output rows g (registers 0/1) and g + 8 (2/3) of 32 n8 tiles: 8-bit value tiles [db][0] and
    // [db][1] hold the even and odd dimensions of block db; FP16 tiles [db][h] dimensions
    // 16 db + 8 h .. + 7.
    float o[16][2][4];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
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
            constexpr int Groups = K::Groups, Steps = 16 / Groups;
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
                    ldmatrix_x2(r0, r1, smem_addr(k_s + swizzled(lane & 15, K::Bytes, kb)));
                    unsigned lo, hi;
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r0, lo, hi);
                    } else {
                        widen_e4m3(r0, lo, hi);
                    }
                    mma_f16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a[0], a[1], a[2], a[3], lo, hi);
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r1, lo, hi);
                    } else {
                        widen_e4m3(r1, lo, hi);
                    }
                    mma_f16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a[0], a[1], a[2], a[3], lo, hi);
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
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                unsigned r[2];
                ldmatrix_x2(r[0], r[1], smem_addr(k_s + swizzled(lane & 15, K::Bytes, c)));
                unsigned b[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int key   = nt * 8 + gid;
                    const __half sc = e4m3_to_half(k_s[S::KeyScales + key * K::ScaleSlot + 2 * c + (lid >> 1)]);
                    const __half2 s2 = __halves2half2(sc, sc);
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { b[nt][byte] = e2m1x2_times((r[nt] >> (8 * byte)) & 0xffu, s2); }
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    unsigned a[4];
                    query_fragment(2 * c + half, a);
                    mma_f16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1], a[2], a[3], b[0][2 * half],
                            b[0][2 * half + 1]);
                    mma_f16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1], a[2], a[3], b[1][2 * half],
                            b[1][2 * half + 1]);
                }
            }
        } else {
            const int b_mat = lane >> 3, b_rin = lane & 7;
            const int b_row = b_rin + ((b_mat >> 1) << 3);
#pragma unroll
            for (int kb = 0; kb < 16; ++kb) {
                unsigned a[4];
                query_fragment(kb, a);
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(k_s + swizzled(b_row, K::Bytes, 2 * kb + (b_mat & 1))));
                mma_bf16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }

        // Online softmax over the tile's 16 tokens (columns past the list are masked).
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int index = i * kTile + nt * 8 + 2 * lid;
            if (index >= total) { score[nt][0] = score[nt][2] = -CUDART_INF_F; }
            if (index + 1 >= total) { score[nt][1] = score[nt][3] = -CUDART_INF_F; }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        const float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);
        m0 = nm0;
        m1 = nm1;
        const float base0 = nm0 * scale_l2, base1 = nm1 * scale_l2;
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -base0));
            pr[nt][1] = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -base0));
            pr[nt][2] = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -base1));
            pr[nt][3] = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -base1));
            bl0 += pr[nt][0] + pr[nt][1];
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        // The 128 output rescales are pure scaling: when no row's max moved this tile every alpha is
        // exp2_approx(0) = 1 exactly and the whole loop can be skipped (warp-uniform, bits unchanged).
        if (!__all_sync(FullMask, alpha0 == 1.0f && alpha1 == 1.0f)) {
#pragma unroll
            for (int b = 0; b < 16; ++b) {
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
                const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
                own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
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
            constexpr int Groups = V::Groups;
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
            for (int db = 0; db < 16; db += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(v_s + swizzled(v_row, V::Bytes, db + (v_mat >> 1))));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g       = (db + h) * Groups / 16;
                    const int key     = 2 * lid;
                    const __half2 s01 = __hmul2(__halves2half2(value_scale(key, g), value_scale(key + 1, g)), down);
                    const __half2 s89 = __hmul2(__halves2half2(value_scale(key + 8, g), value_scale(key + 9, g)), down);
                    unsigned e0, o0, e1, o1;
                    decode_v_pair<Shape::Value>(r[2 * h], s01, e0, o0);
                    decode_v_pair<Shape::Value>(r[2 * h + 1], s89, e1, o1);
                    float* even = o[db + h][0];
                    float* odd  = o[db + h][1];
                    mma_f16(even[0], even[1], even[2], even[3], pa[0], pa[1], pa[2], pa[3], e0, e1);
                    mma_f16(odd[0], odd[1], odd[2], odd[3], pa[0], pa[1], pa[2], pa[3], o0, o1);
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
                    mma_f16(first[0], first[1], first[2], first[3], pa[0], pa[1], pa[2], pa[3], r[0], r[1]);
                    mma_f16(second[0], second[1], second[2], second[3], pa[0], pa[1], pa[2], pa[3], r[2], r[3]);
                }
            };
            if constexpr (Shape::Value == Code::Nvfp4) {
                const int key  = 2 * lid;
                const int g_in = gid >> 2;
#pragma unroll
                for (int c = 0; c < 8; c += 2) {
                    unsigned r[4];
                    ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                  smem_addr(v_s + swizzled(v_row, V::Bytes, c + (v_mat >> 1))));
#pragma unroll
                    for (int hc = 0; hc < 2; ++hc) {
                        const int chunk = c + hc;
                        const int g     = 2 * chunk + g_in;
                        const unsigned char* vs = k_s + S::ValueScales + qk.scale_at + g;
                        unsigned b0[4], b1[4];
                        decode_v_quad_nvfp4(r[2 * hc], e4m3_to_half(vs[key * V::ScaleSlot]),
                                            e4m3_to_half(vs[(key + 1) * V::ScaleSlot]), b0);
                        decode_v_quad_nvfp4(r[2 * hc + 1], e4m3_to_half(vs[(key + 8) * V::ScaleSlot]),
                                            e4m3_to_half(vs[(key + 9) * V::ScaleSlot]), b1);
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            float* dst = o[2 * chunk + (u >> 1)][u & 1];
                            mma_f16(dst[0], dst[1], dst[2], dst[3], pa[0], pa[1], pa[2], pa[3], b0[u], b1[u]);
                        }
                    }
                }
            } else {
                pv.template operator()<0, 32>(v_s, 2 * kD);
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
        // Two CTAs per SM, one stage: the key half and the value half are separate commit groups, so a
        // tile's values stream in behind its QK MMA and online softmax and the next tile's keys behind
        // the PV MMA. Each half of the stage is refilled only after the phase that reads it is done (the
        // `__syncwarp` barriers), and `cp_wait<1>` leaves exactly the one group the other phase still
        // needs in flight. The other warps (this CTA's and the other CTA's) hide what is left.
        if (warp < tiles) { issue(warp, 0, std::integral_constant<int, kGatherKey>{}); }
        for (int i = warp; i < tiles; i += Warps) {
            issue(i, 0, std::integral_constant<int, kGatherValue>{});
            cp_wait<1>();  // this tile's keys are staged; its values may still be streaming
            __syncwarp();
            const QkState qk = score_tile(i, 0);
            __syncwarp();  // every lane is done with the key rows before the next tile's overwrite
            if (i + Warps < tiles) {
                issue(i + Warps, 0, std::integral_constant<int, kGatherKey>{});
                cp_wait<1>();  // this tile's values are staged; the next tile's keys may still stream
            } else {
                cp_wait<0>();
            }
            __syncwarp();
            value_tile(0, qk);
            __syncwarp();  // the value rows are read out before the next tile's overwrite
        }
    } else if constexpr (Stages == 1) {
        // Two CTAs per SM: the other warps hide each tile's gather, whose keys and values stay one
        // commit group (see `Stage::SplitGather`).
        for (int i = warp; i < tiles; i += Warps) {
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
        if (warp < tiles) { issue(warp, 0, std::integral_constant<int, kGatherBoth>{}); }
        for (int i = warp; i < tiles; i += Warps) {
            if (i + Warps < tiles) {
                issue(i + Warps, s ^ 1, std::integral_constant<int, kGatherBoth>{});
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

    // Merge the warps' rows in warp order: every warp publishes its (max, sum) per row, then the warps
    // add their scaled unnormalized rows into one FP32 buffer in turn (fma per element in warp order,
    // skipping a warp without keys for the row), and l the same way.
    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem + kQRow * kRows); // [kRows][kD]
    float* stats = rows + kRows * kD;                               // [Warps][kRows][2]
    if (lid == 0) {
        stats[(warp * kRows + gid) * 2]         = m0;
        stats[(warp * kRows + gid) * 2 + 1]     = l0;
        stats[(warp * kRows + gid + 8) * 2]     = m1;
        stats[(warp * kRows + gid + 8) * 2 + 1] = l1;
    }
    __syncthreads();
    const auto row_max = [&](int row) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) { m = fmaxf(m, stats[(w * kRows + row) * 2]); }
        return m;
    };
    const float f0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - row_max(gid)) * scale_l2);
    const float f1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - row_max(gid + 8)) * scale_l2);
    for (int w = 0; w < Warps; ++w) {
        if (warp == w) {
#pragma unroll
            for (int b = 0; b < 16; ++b) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int upper = j >> 1;
                        const int row   = gid + 8 * upper;
                        const int d     = Shape::Value == Code::Nvfp4
                                              ? 32 * (b >> 1) + 8 * lid + 4 * (j & 1) + 2 * (b & 1) + h
                                              : (S::EvenOddValues ? 16 * b + 4 * lid + 2 * (j & 1) + h
                                                                  : 16 * b + 8 * h + 2 * lid + (j & 1));
                        const bool live = (upper ? m1 : m0) != -CUDART_INF_F;
                        float& a        = rows[row * kD + d];
                        if (w == 0) {
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
        for (int w = 0; w < Warps; ++w) {
            const float mw = stats[(w * kRows + row) * 2];
            if (mw == -CUDART_INF_F) { continue; }
            l = __fmaf_rn(exp2_approx((mw - m) * scale_l2), stats[(w * kRows + row) * 2 + 1], l);
        }
        const float value = l > 0.0f ? rows[row * kD + d] / l : 0.0f;
        if constexpr (S::RotatedValues) {
            rows[row * kD + d] = value;
        } else {
            out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + d] = __float2bfloat16_rn(value);
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
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + lane + 32 * k] =
                    __float2bfloat16_rn(v[k]);
            }
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa prompt: ") + message); }
}

template <KvCacheStorage Storage>
void launch(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g, float scale,
            const std::int32_t* selected, const std::int32_t* counts, Tensor& out, cudaStream_t stream,
            const QsaVqWindow& vq) {
    const auto run = [&]<int Stages>() {
        constexpr int bytes = prompt_smem_bytes<Storage, Stages>();
        static_assert(bytes <= 99 * 1024, "one CTA's shared memory on sm_120");
        static const cudaError_t configured = cudaFuncSetAttribute(
            qsa_prompt_kernel<Storage, Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        require(configured == cudaSuccess, "cannot reserve the kernel's shared memory");
        const int columns = q.ne[2];
        const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(g.kv_heads));
        qsa_prompt_kernel<Storage, Stages><<<grid, PromptShape<Storage>::Warps * 32, bytes, stream>>>(
            static_cast<const bf16*>(q.data), g.heads, g.kv_heads, layer.kv.k_pages.data, layer.kv.v_pages.data,
            layer.kv.k_scale_pages.data, layer.kv.v_scale_pages.data,
            static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
            static_cast<const std::int32_t*>(batch.table_rows.data),
            static_cast<const std::int32_t*>(batch.positions.data), batch.width, g.ratio, selected, counts,
            g.budget / g.ratio, scale, static_cast<bf16*>(out.data), layer.spaces, vq);
    };
    // One stage where two CTAs fit an SM and the call fills a wave: the second CTA's warps hide each
    // tile's gather better than a second stage does (26-41 % faster at 86-512 columns, design
    // §19.3.15). Narrower calls, and storages whose stages and value tile keep one CTA per SM, keep
    // two stages.
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
            throw std::runtime_error("qsa prompt: cannot query the multiprocessor count");
        }
        return count;
    }();
    // GB10: the split stage (upstream qsa_prompt_split) was removed with the one-stage gather
    // rewrite; the combined gather below is the only route.
    constexpr bool two_ctas = 2 * (prompt_smem_bytes<Storage, 1>() + 1024) <= 100 * 1024;
    if (two_ctas && q.ne[2] * g.kv_heads >= sms) {
        run.template operator()<1>();
    } else {
        run.template operator()<2>();
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa prompt attention: ") + cudaGetErrorString(error)); }
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
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream, const QsaVqWindow& vq) {
    require(qsa_prompt_supported(layer, geometry), "the prompt route does not serve this layer");
    switch (layer.kv.storage) {
    case KvCacheStorage::Int8Group64:
        launch<KvCacheStorage::Int8Group64>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch<KvCacheStorage::Fp8E4M3Row256>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Nvfp4Group16:
        launch<KvCacheStorage::Nvfp4Group16>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch<KvCacheStorage::Fp8KeyNvfp4Value>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Vq2:
        launch<KvCacheStorage::Vq2>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        launch<KvCacheStorage::Q4KeyVq2Value>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    default:
        launch<KvCacheStorage::BFloat16>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
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
    return storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp8E4M3Row256;
}

void qsa_prompt_force(int variant) {
    if (variant < -1 || variant > 1) { throw std::invalid_argument("qsa_prompt_force: -1, 0 or 1"); }
    forced_variant().store(variant, std::memory_order_relaxed);
}

int qsa_prompt_forced() { return forced_variant().load(std::memory_order_relaxed); }

} // namespace infernix::ops::detail
