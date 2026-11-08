#pragma once
#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_layout.h"

namespace infernix::ops::detail {
// The TMA route reads a whole [128 or 256 tokens, kNvfp4ScaleTileGroups groups] tile of
// activation scales per request. Row-major by token that tile is 16 bytes per row at the plane's
// row stride, so one stage would require a small request for each token. Writing the same bytes
// tile-contiguous makes the request wide. The byte order inside a tile is unchanged, so the shared
// image the MMA reads is identical.
//
// It addresses whole tiles, so a ragged token count is padded up to one and that padding is
// written with zeroes below. launch_nvfp4_a4_quantize checks that the plane was allocated
// for the padded count before this layout is selected.
template <class Geometry, Nvfp4ScaleLayout Layout>
__device__ __forceinline__ std::int64_t nvfp4_tiled_scale_offset(int token, int group) {
    constexpr int kGroupsPerTile = kNvfp4ScaleTileGroups;
    constexpr int kTilesPerPlane = Geometry::kGroupsPerRow / kGroupsPerTile;
    constexpr int kBlockTokens   = kNvfp4ScaleTileTokens<Layout>;
    const int token_tile         = token / kBlockTokens;
    const int group_tile         = group / kGroupsPerTile;
    const int tile               = token_tile * kTilesPerPlane + group_tile;
    return static_cast<std::int64_t>(tile) * kBlockTokens * kGroupsPerTile +
           static_cast<std::int64_t>(token - token_tile * kBlockTokens) * kGroupsPerTile +
           (group - group_tile * kGroupsPerTile);
}

// Four consecutive lanes (lane & 3 = 0..3) quantize four consecutive 16-element K-groups of the
// same token and same 16-group scale tile while keeping 16-byte BF16 input loads and 8-byte FP4
// code stores contiguous across the warp. In both RowMajor and Tiled128/Tiled256 layouts those
// four scale bytes occupy a 4-byte-aligned word, so packing them across the four lanes via warp
// shuffle emits a single coalesced uint32_t store per quad.
__device__ __forceinline__ void store_nvfp4_scale_quad(std::uint8_t* scales,
                                                       std::int64_t scale_offset,
                                                       std::uint8_t scale, int tid) {
    std::uint32_t word = static_cast<std::uint32_t>(scale);
    word |= __shfl_down_sync(0xFFFFFFFFU, word, 1, 4) << 8;
    word |= __shfl_down_sync(0xFFFFFFFFU, word, 2, 4) << 16;
    if ((tid & 3) == 0) {
        store_vec(scales + scale_offset, word);
    }
}

template <class Geometry, int Threads, Nvfp4ScaleLayout Layout, bool Reciprocal = false>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, std::int32_t written_tokens,
    float input_scale_divisor) {
    // A short kernel: it waits for the activation's producer (under CUDA Graph capture its launch
    // is a programmatic dependent) before reading it.
    pdl::enter();
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    static_assert((Geometry::kInputRows / 16) % 4 == 0);
    static_assert(Layout == Nvfp4ScaleLayout::RowMajor ||
                  (Geometry::kInputRows / 16) % kNvfp4ScaleTileGroups == 0);
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    const int tid               = static_cast<int>(threadIdx.x);
    const int task              = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    const int tasks             = tokens * kGroupsPerRow;
    if (task >= tasks) {
        // The tiled plane is addressed in whole tiles, so the launch covers the padding of the last
        // one. A padded token owns no input and no code byte, only the scale the consumer's tile
        // will read; zero it so it is defined rather than whatever the arena last held.
        // written_tokens is the extent the plane was allocated for and bounds this store; it equals
        // tokens for the row-major layout, which stops at the real count.
        if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
            const int pad_token = task / kGroupsPerRow;
            if (pad_token < written_tokens && (tid & 3) == 0) {
                const int pad_group = task - pad_token * kGroupsPerRow;
                store_vec(scales + nvfp4_tiled_scale_offset<Geometry, Layout>(pad_token, pad_group),
                          std::uint32_t{0});
            }
        }
        return;
    }

    const int token                   = task / kGroupsPerRow;
    const int group                   = task - token * kGroupsPerRow;
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16<Reciprocal>(
        input + static_cast<std::int64_t>(token) * Geometry::kInputRows + group * 16,
        input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    std::int64_t scale_offset = static_cast<std::int64_t>(token) * kGroupsPerRow + group;
    if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
        scale_offset = nvfp4_tiled_scale_offset<Geometry, Layout>(token, group);
    }
    store_nvfp4_scale_quad(scales, scale_offset, quantized.scale, tid);
}

// The A4 residual profile's planes (quantize_nvfp4_terms_k16), each laid out like the A4 plane: codes
// row-major, scales row-major or tiled (the TMA route, padding zeroed as for one plane).
template <class Geometry, int Threads, int Terms,
          Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::RowMajor>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_a4_residual_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, std::int32_t written_tokens,
    float input_scale_divisor, std::int64_t code_plane, std::int64_t scale_plane) {
    pdl::enter();
    static_assert((Geometry::kInputRows / 16) % 4 == 0);
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    const int tid               = static_cast<int>(threadIdx.x);
    const int task              = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    if (task >= tokens * kGroupsPerRow) {
        if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
            const int pad_token = task / kGroupsPerRow;
            if (pad_token < written_tokens && (tid & 3) == 0) {
                const int pad_group = task - pad_token * kGroupsPerRow;
                const std::int64_t scale_offset =
                    nvfp4_tiled_scale_offset<Geometry, Layout>(pad_token, pad_group);
#pragma unroll
                for (int term = 0; term < Terms; ++term) {
                    store_vec(scales + term * scale_plane + scale_offset, std::uint32_t{0});
                }
            }
        }
        return;
    }
    const int token = task / kGroupsPerRow;
    const int group = task - token * kGroupsPerRow;
    Nvfp4QuantizedK16 terms[Terms];
    quantize_nvfp4_terms_k16<Terms>(
        input + static_cast<std::int64_t>(token) * Geometry::kInputRows + group * 16,
        input_scale_divisor, terms);
    const std::int64_t code_offset =
        static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    std::int64_t scale_offset = static_cast<std::int64_t>(token) * kGroupsPerRow + group;
    if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
        scale_offset = nvfp4_tiled_scale_offset<Geometry, Layout>(token, group);
    }
#pragma unroll
    for (int term = 0; term < Terms; ++term) {
        store_vec(codes + term * code_plane + code_offset,
                  make_uint2(terms[term].codes_lo, terms[term].codes_hi));
        store_nvfp4_scale_quad(scales + term * scale_plane, scale_offset, terms[term].scale, tid);
    }
}

// Maximum 16-element groups per thread cached in registers between the amax pass and the quantize
// pass of nvfp4_a4_row_scaled_residual_quantize_kernel. Five groups (10 uint4 = 40 registers) covers
// K = 17408 (1,088 groups across 256 threads) within the 2-CTA/SM (128-register) budget so x is read
// once from DRAM without spilling.
inline constexpr int kNvfp4A4RowScaledCachedGroups = 5;

// Per-token row-scaled A4 residual quantizer (1 CTA per token): reduces amax across the token's K
// elements, sets row_scales[token] = amax / (6 * 448) so every 16-element group's first-term E4M3
// scale uses the full [2^-9, 448] dynamic range without clipping, and writes the `Terms` planes.
template <class Geometry, int Threads, int Terms,
          Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::RowMajor>
__global__ __launch_bounds__(
    Threads, ((Geometry::kInputRows / 16 + Threads - 1) / Threads <= 2) ? (1024 / Threads) : 2) void
nvfp4_a4_row_scaled_residual_quantize_kernel(const __nv_bfloat16* __restrict__ input,
                                             std::uint8_t* __restrict__ codes,
                                             std::uint8_t* __restrict__ scales,
                                             float* __restrict__ row_scales,
                                             std::int32_t tokens, std::int64_t code_plane,
                                             std::int64_t scale_plane) {
    static_assert(Threads % 32 == 0 && Threads <= 512);
    static_assert((Geometry::kInputRows / 16) % 4 == 0);
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    constexpr int kWarps        = Threads / 32;
    constexpr bool kExactGroups = (kGroupsPerRow % Threads) == 0;
    constexpr int kMaxGroups    = (kGroupsPerRow + Threads - 1) / Threads;
    constexpr bool kCacheGroups = kMaxGroups <= kNvfp4A4RowScaledCachedGroups;
    __shared__ float warp_maxima[kWarps];

    pdl::enter();
    const int total_blocks = static_cast<int>(gridDim.x);
    // GB10 (24 MiB L2, 1/4 of RTX 5090's 96 MiB L2): reverse CTA order past 256 tokens so the
    // last-executing CTAs leave token 0's quantized plane hot in L2 for the first TMA consumer band.
    const int token = total_blocks > 256 ? (total_blocks - 1 - static_cast<int>(blockIdx.x))
                                         : static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (token >= tokens) {
        if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
            for (int group = tid * 4; group < kGroupsPerRow; group += Threads * 4) {
                const std::int64_t scale_offset =
                    nvfp4_tiled_scale_offset<Geometry, Layout>(token, group);
#pragma unroll
                for (int term = 0; term < Terms; ++term) {
                    store_vec(scales + term * scale_plane + scale_offset, std::uint32_t{0});
                }
            }
        }
        return;
    }

    const __nv_bfloat16* row_in = input + static_cast<std::int64_t>(token) * Geometry::kInputRows;
    uint4 cached_p0[kCacheGroups ? kMaxGroups : 1];
    uint4 cached_p1[kCacheGroups ? kMaxGroups : 1];
    float amax = 0.0F;
#pragma unroll
    for (int iter = 0; iter < kMaxGroups; ++iter) {
        const int group = tid + iter * Threads;
        if (kExactGroups || group < kGroupsPerRow) {
            const uint4 p0 = load_vec<uint4>(row_in + group * 16);
            const uint4 p1 = load_vec<uint4>(row_in + group * 16 + 8);
            if constexpr (kCacheGroups) {
                cached_p0[iter] = p0;
                cached_p1[iter] = p1;
            }
            const std::uint32_t words[8]{p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const float2 v = bf16x2_bits_to_float2(words[pair]);
                const float a  = fabsf(v.x);
                const float b  = fabsf(v.y);
                amax           = (a > amax || a != a) ? a : amax;
                amax           = (b > amax || b != b) ? b : amax;
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFFU, amax, offset);
        amax              = (other > amax || other != other) ? other : amax;
    }
    if ((tid & 31) == 0) { warp_maxima[tid >> 5] = amax; }
    __syncthreads();
    float all = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const float m = warp_maxima[w];
        all           = (m > all || m != m) ? m : all;
    }
    const float rs = !isfinite(all) ? __int_as_float(0x7FC00000)
                                    : (all > 0.0F ? (all * (1.0F / (6.0F * 448.0F))) : 0.0F);
    if (tid == 0) { row_scales[token] = rs; }
    const float divisor = (rs > 0.0F && isfinite(rs)) ? (1.0F / rs) : 0.0F;
#pragma unroll
    for (int iter = 0; iter < kMaxGroups; ++iter) {
        const int group = tid + iter * Threads;
        if (kExactGroups || group < kGroupsPerRow) {
            Nvfp4QuantizedK16 terms[Terms];
            if constexpr (kCacheGroups) {
                quantize_nvfp4_terms_k16_packed<Terms, true>(cached_p0[iter], cached_p1[iter],
                                                             divisor, terms);
            } else {
                quantize_nvfp4_terms_k16<Terms, true>(row_in + group * 16, divisor, terms);
            }
            const std::int64_t code_offset =
                static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
            std::int64_t scale_offset = static_cast<std::int64_t>(token) * kGroupsPerRow + group;
            if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
                scale_offset = nvfp4_tiled_scale_offset<Geometry, Layout>(token, group);
            }
#pragma unroll
            for (int term = 0; term < Terms; ++term) {
                store_vec(codes + term * code_plane + code_offset,
                          make_uint2(terms[term].codes_lo, terms[term].codes_hi));
                store_nvfp4_scale_quad(scales + term * scale_plane, scale_offset,
                                       terms[term].scale, tid);
            }
        }
    }
}

} // namespace infernix::ops::detail
