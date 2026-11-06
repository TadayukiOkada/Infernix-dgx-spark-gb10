#pragma once

// NVFP4 codes multiplied by their raw E4M3 G16 scales are exactly representable
// in BF16. The global weight divisor is applied to the complete FP32 reduction.
// Activations remain the represented public BF16 inputs.

#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_sliced_k_mma_kernel(
    Nvfp4A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset) {
    pdl::enter_streaming();
    const auto* __restrict__ x             = operands.x;
    const auto* __restrict__ weight_codes  = operands.codes;
    const auto* __restrict__ weight_scales = operands.scales;
    const int kHidden                      = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kRowTiles      = Schedule::kRowTiles;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kBlockK        = Schedule::kBlockK;
    const int kGroups            = kHidden / kBlockK;
    constexpr int kBlockTokens   = Schedule::kBlockTokens;
    constexpr int kTokenMmas     = kBlockTokens / 8;
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kBlockTokens);
    static_assert((kWarps & 1) == 0);
    static_assert(kRowTiles == 1 || !RowPolicy::kPaired);

    union SharedStorage {
        struct {
            std::uint8_t codes[Schedule::kStages][kBlockRows][kBlockK / 2];
            __nv_bfloat16 activations[Schedule::kStages][kWarps][kBlockTokens * kTileK];
        } staging;

        float partial[kWarps * kRowTiles * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared =
        *reinterpret_cast<SharedStorage*>(nvfp4_shared_storage<Schedule::kSharedBytes>());
    auto& code_shared = shared.staging.codes;
    auto& x_shared    = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * (kBlockRows / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    // An output may claim a line of 32-bit words per row that this contraction fills before its
    // mainloop and hands back to its row consumer (see nvfp4_a16_gemv_kernel). Its global reads are
    // independent of the contraction, so issuing them here hides their memory round trip behind the
    // mainloop instead of serializing it in the single finishing warp's tail.
    using Prefetch = LinearRowPrefetch<Output>;
    constexpr int kPrefetchWords = RowPolicy::kPaired ? 0 : Prefetch::words;
    __shared__ std::uint32_t
        row_prefetch[kPrefetchWords == 0 ? 1 : kBlockRows * kPrefetchWords];

    if constexpr (kPrefetchWords != 0) {
        // One thread fills one row's whole line before the copies go out; the mainloop's first
        // barrier publishes them, so the fill costs no extra synchronization here. Measured (GB10,
        // 2026-11-01, T3): issuing it after the prologue instead - so the async copies are already in
        // flight when the fill stalls on its read - is 1.3 .. 2.4 us slower at every B = 3..8 the route
        // serves, because the fill then delays the mainloop's first group while the copies it was
        // supposed to overlap were already covered by the pipeline. The line covers the CTA's
        // contraction columns by their absolute index, which is what the row consumer below names when
        // it reads a column's window back.
        for (int local = tid; local < kBlockRows; local += Schedule::kThreads) {
            output.prefetch_row(row_prefetch + local * kPrefetchWords,
                                row_policy.weight_row(row0, local, operands.rows),
                                token_begin + live_columns);
        }
    }

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr bool kPadded          = Schedule::kActivationStage == Nvfp4ActivationStage::PaddedZero;
        const int stage_tokens          = kPadded ? kBlockTokens : live_columns;
        const int items                 = stage_tokens * (kTileK / 8);
        for (int item = lane; item < items; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[stage][warp][token * kTileK + nvfp4_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!kPadded || (!MaskedColumns && ActiveTokens == kBlockTokens)) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden +
                                     group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    constexpr int kCodeChunks  = kBlockK / 32;
    constexpr int kCodeSwizzle = kCodeChunks < 8 ? kCodeChunks - 1 : 7;
    const auto stage_codes     = [&](int stage, int group_k0) {
#pragma unroll
        for (int ri = 0; ri < Schedule::kRowsPerLoaderWarp; ++ri) {
            const int row    = warp * Schedule::kRowsPerLoaderWarp + ri;
            const int parent = row_policy.weight_row(row0, row, operands.rows);
            for (int chunk = lane; chunk < kCodeChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[stage][row][(chunk ^ (row & kCodeSwizzle)) * 16],
                    weight_codes + static_cast<std::int64_t>(parent) * (kHidden / 2) +
                        group_k0 / 2 + chunk * 16);
            }
        }
    };

    const int b_row                             = lane & 7;
    const int b_k_offset                        = ((lane >> 3) & 1) << 3;
    const int warp_k0                           = warp * kTileK;
    float accumulators[kRowTiles][kTokenMmas][4] = {};
    const std::uint32_t* scale_ptr_top[kRowTiles];
    const std::uint32_t* scale_ptr_bot[kRowTiles];
#pragma unroll
    for (int tile = 0; tile < kRowTiles; ++tile) {
        const int top        = tile * 16 + gid;
        const int parent_top = row_policy.weight_row(row0, top, operands.rows);
        const int parent_bot = row_policy.weight_row(row0, top + 8, operands.rows);
        scale_ptr_top[tile]  = reinterpret_cast<const std::uint32_t*>(
            weight_scales + nvfp4_scale_byte_offset(parent_top, warp_k0 / 16, kHidden));
        scale_ptr_bot[tile] = reinterpret_cast<const std::uint32_t*>(
            weight_scales + nvfp4_scale_byte_offset(parent_bot, warp_k0 / 16, kHidden));
    }

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
        std::uint32_t scale_top[kRowTiles];
        std::uint32_t scale_bot[kRowTiles];
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            scale_top[tile] = scale_ptr_top[tile][group_index * (kBlockK * 2)];
            scale_bot[tile] = scale_ptr_bot[tile][group_index * (kBlockK * 2)];
        }
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col = k_step * 16 + lid * 2;
            const int byte0    = (warp_k0 + code_col) / 2;
            const int byte_hi  = byte0 / 16;
            const int byte_lo  = byte0 & 15;
            const auto load_two_pairs = [&](int row, std::uint32_t scale_f16x2, unsigned& out0,
                                            unsigned& out1) {
                const int offset = (byte_hi ^ (row & kCodeSwizzle)) * 16 + byte_lo;
                nvfp4_scaled_two_pairs_bf16_f16scale(code_shared[stage][row][offset], code_shared[stage][row][offset + 4],
                                                     scale_f16x2, out0, out1);
            };
            unsigned a[kRowTiles][4];
#pragma unroll
            for (int tile = 0; tile < kRowTiles; ++tile) {
                const int top    = tile * 16 + gid;
                // The row pair's two E4M3 scales go through one cvt.rn.f16x2.e4m3x2.
                std::uint32_t mult_top, mult_bot;
                nvfp4_unpack_two_scales_f16x2(
                    static_cast<std::uint8_t>(scale_top[tile] >> (k_step * 8)),
                    static_cast<std::uint8_t>(scale_bot[tile] >> (k_step * 8)), mult_top, mult_bot);
                load_two_pairs(top, mult_top, a[tile][0], a[tile][2]);
                load_two_pairs(top + 8, mult_bot, a[tile][1], a[tile][3]);
            }
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int row = token_mma * 8 + b_row;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(
                        &x_shared[stage][warp][row * kTileK + nvfp4_a16_shared_col_64(
                                                                  row, k_step * 16 + b_k_offset)]));
#pragma unroll
                for (int tile = 0; tile < kRowTiles; ++tile) {
                    mma_bf16(accumulators[tile][token_mma][0], accumulators[tile][token_mma][1],
                             accumulators[tile][token_mma][2], accumulators[tile][token_mma][3],
                             a[tile][0], a[tile][1], a[tile][2], a[tile][3], b0, b1);
                }
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }

    __syncthreads();
    auto* partial            = shared.partial;
    const auto partial_index = [&](int split, int tile, int token_mma) {
        return (((split * kRowTiles + tile) * kTokenMmas + token_mma) * 32 + lane) * 4;
    };
    if ((warp & 1) != 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                store_vec(partial + partial_index(warp, tile, token_mma),
                          make_float4(accumulators[tile][token_mma][0],
                                      accumulators[tile][token_mma][1],
                                      accumulators[tile][token_mma][2],
                                      accumulators[tile][token_mma][3]));
            }
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const float4 partner =
                    load_vec<float4>(partial + partial_index(warp + 1, tile, token_mma));
                accumulators[tile][token_mma][0] += partner.x;
                accumulators[tile][token_mma][1] += partner.y;
                accumulators[tile][token_mma][2] += partner.z;
                accumulators[tile][token_mma][3] += partner.w;
                if (warp != 0) {
                    store_vec(partial + partial_index(warp, tile, token_mma),
                              make_float4(accumulators[tile][token_mma][0],
                                          accumulators[tile][token_mma][1],
                                          accumulators[tile][token_mma][2],
                                          accumulators[tile][token_mma][3]));
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        const auto destination =
            linear_output_tile<kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
        const float top_scale = operands.alpha, bottom_scale = operands.alpha;

#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            const int tile_row = tile * 16;
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                float4 sum = make_float4(accumulators[tile][token_mma][0],
                                         accumulators[tile][token_mma][1],
                                         accumulators[tile][token_mma][2],
                                         accumulators[tile][token_mma][3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value =
                        load_vec<float4>(partial + partial_index(split, tile, token_mma));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int local_token = token_mma * 8 + 2 * lid;
                const int token0      = token_begin + local_token;
                const int row_a       = row_policy.weight_row(row0, tile_row + gid, operands.rows);
                const int row_b = row_policy.weight_row(row0, tile_row + gid + 8, operands.rows);
                if constexpr (RowPolicy::kPaired) {
                    if (local_token < live_columns)
                        epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                            sum.z * bottom_scale);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                            sum.w * bottom_scale);
                } else if constexpr (kPrefetchWords != 0 && requires(const float (&vals)[2]) {
                                         epilogue.apply_row(destination, row_a, token0, vals, 2,
                                                            static_cast<const std::uint32_t*>(
                                                                nullptr));
                                     }) {
                    const int active = max(0, min(2, live_columns - local_token));
                    if (active > 0) {
                        const float values_a[2]{sum.x * top_scale, sum.y * top_scale};
                        const float values_b[2]{sum.z * bottom_scale, sum.w * bottom_scale};
                        epilogue.apply_row(destination, row_a, token0, values_a, active,
                                           row_prefetch + (tile_row + gid) * kPrefetchWords);
                        epilogue.apply_row(destination, row_b, token0, values_b, active,
                                           row_prefetch + (tile_row + gid + 8) * kPrefetchWords);
                    }
                } else if constexpr (requires(const float (&vals)[2]) {
                                         epilogue.apply_row(destination, row_a, token0, vals, 2);
                                     }) {
                    const int active = max(0, min(2, live_columns - local_token));
                    if (active > 0) {
                        const float values_a[2]{sum.x * top_scale, sum.y * top_scale};
                        const float values_b[2]{sum.z * bottom_scale, sum.w * bottom_scale};
                        epilogue.apply_row(destination, row_a, token0, values_a, active);
                        epilogue.apply_row(destination, row_b, token0, values_b, active);
                    }
                } else {
                    if (local_token < live_columns) {
                        destination.store(row_a, token0,
                                          epilogue.apply(row_a, token0, sum.x * top_scale));
                        destination.store(row_b, token0,
                                          epilogue.apply(row_b, token0, sum.z * bottom_scale));
                    }
                    if (local_token + 1 < live_columns) {
                        destination.store(row_a, token0 + 1,
                                          epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                        destination.store(row_b, token0 + 1,
                                          epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                    }
                }
            }
        }
    }
}

} // namespace infernix::ops::detail
