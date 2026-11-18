#pragma once

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

__device__ __forceinline__ void kv_cache_load_bf16x8_contiguous(const __nv_bfloat16* row,
                                                                 int lane, float (&values)[8]) {
    const __nv_bfloat16* p = row + 8 * lane;
    if ((reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U) {
        const uint4 raw        = *reinterpret_cast<const uint4*>(p);
        const unsigned words[4]{raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[j] = __uint_as_float(((words[j / 2] >> (16 * (j % 2))) & 0xffffU) << 16);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) { values[j] = __bfloat162float(p[j]); }
    }
}

template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_nvfp4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token, int kv_head,
    int physical_page, int page_offset, int lane) {
    const int group = lane >> 1;
#pragma unroll 1
    for (int role = 0; role < 2; ++role) {
        const __nv_bfloat16* src =
            (role ? v : k) + kv_cache_nvfp4_src_index<Geometry>(kv_head, 0, token);
        float values[8];
        kv_cache_load_bf16x8_contiguous(src, lane, values);
        normalized_hadamard_d256_contiguous_inplace(values, lane);
        const auto quantized = kv_cache_nvfp4_quantize_group16_contiguous(values, lane);
        if ((lane & 1) == 0) {
            const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                physical_page, kv_head, group * kKVCacheNvfp4Group, page_offset);
            store_vec((role ? cache_v : cache_k) + code_offset,
                      make_uint2(quantized.codes_lo, quantized.codes_hi));
            (role ? scale_v : scale_k)[kv_cache_nvfp4_scale_index<Geometry>(
                physical_page, kv_head, group, page_offset)] = quantized.scale;
        }
    }
}

template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void kv_cache_append_full_nvfp4_kernel(const __nv_bfloat16* __restrict__ k,
                                           const __nv_bfloat16* __restrict__ v,
                                           const std::int32_t* __restrict__ positions,
                                           Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                           std::uint8_t* __restrict__ cache_v,
                                           std::uint8_t* __restrict__ scale_k,
                                           std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_nvfp4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                             kv_head, physical_page, position & kPagedKVPageMask,
                                             lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void kv_cache_append_full_nvfp4_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens        = metadata.valid_tokens(width);
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int kv_head       = static_cast<int>(blockIdx.y);
    const int base_position = positions[0];
    const int tile_position =
        (base_position / TokensPerTile + static_cast<int>(blockIdx.x)) * TokensPerTile;
    const int token_begin = max(0, tile_position - base_position);
    const int token_end   = min(tokens, tile_position + TokensPerTile - base_position);
    const int token       = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page  = lane == 0 ? block_table[tile_position >> kPagedKVPageShift] : 0;
    physical_page      = __shfl_sync(FullMask, physical_page, 0);
    const int position = base_position + token;
    kv_cache_append_full_nvfp4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                             kv_head, physical_page, position & kPagedKVPageMask,
                                             lane);
}

} // namespace infernix::ops
