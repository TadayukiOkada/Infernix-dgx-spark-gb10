#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// The convolution that follows a materialized NVFP4 projection. Each CTA owns 32 channels (its
// lanes) and spreads the call's token tiles across its warps: each tile reloads its three history
// taps from the projected rows, so the tiles are independent and the warp count only has to cover
// them. The launcher sizes the CTA to the tile count (task T8, 2026-11-07): the retired 32-warp
// form ran one to eight working warps beside 24 to 31 idle ones across its registered widths.
template <int WarpsPerCta, int TokenTile, class Publish>
__global__ __launch_bounds__(WarpsPerCta * 32) void nvfp4_gdn_conv_post_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ conv_states, const std::int32_t* __restrict__ initial_slot,
    const std::int32_t* __restrict__ valid_columns, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t tokens,
    Publish publish) {
    static_assert((kGdnChannels % 32) == 0);

    struct Shared {
        float initial[3][32];
        float weight[4][32];
    };

    __shared__ Shared shared;

    // Consumer protocol: the projection is this grid's programmatic dependency; nothing above this
    // line reads it.
    infernix::pdl::enter();

    const int warp                     = static_cast<int>(threadIdx.x) >> 5;
    const int lane                     = static_cast<int>(threadIdx.x) & 31;
    const int row                      = static_cast<int>(blockIdx.x) * 32 + lane;
    constexpr std::int64_t kSlotStride = static_cast<std::int64_t>(kGdnChannels) * 3;
    std::int32_t valid                 = valid_columns == nullptr ? tokens : *valid_columns;
    valid                              = valid < 0 ? 0 : (valid > tokens ? tokens : valid);

    // The three history taps and the four weight taps (7 x 32 values) spread across the whole CTA:
    // value v = part * 32 + column, so consecutive threads read consecutive channels of one tap.
    constexpr int kThreads = WarpsPerCta * 32;
    const std::int64_t initial_base = static_cast<std::int64_t>(*initial_slot) * kSlotStride +
                                      static_cast<std::int64_t>(blockIdx.x) * 32;
#pragma unroll
    for (int value_index = static_cast<int>(threadIdx.x); value_index < 7 * 32;
         value_index += kThreads) {
        const int part   = value_index >> 5;
        const int column = value_index & 31;
        if (part < 3) {
            shared.initial[part][column] =
                __bfloat162float(conv_states[initial_base + static_cast<std::int64_t>(part) *
                                                 kGdnChannels + column]);
        } else {
            shared.weight[part - 3][column] =
                __bfloat162float(conv_weight[static_cast<std::int64_t>(part - 3) * kGdnChannels +
                                             static_cast<std::int64_t>(blockIdx.x) * 32 + column]);
        }
    }
    __syncthreads();

    const float w0 = shared.weight[0][lane];
    const float w1 = shared.weight[1][lane];
    const float w2 = shared.weight[2][lane];
    const float w3 = shared.weight[3][lane];

    __nv_bfloat16* output;
    std::int32_t output_rows;
    std::int32_t output_row;
    if (row < kGdnQueryRows) {
        output      = query;
        output_rows = kGdnQueryRows;
        output_row  = row;
    } else if (row < kGdnQueryRows + kGdnKeyRows) {
        output      = key;
        output_rows = kGdnKeyRows;
        output_row  = row - kGdnQueryRows;
    } else {
        output      = value;
        output_rows = kGdnValueRows;
        output_row  = row - kGdnQueryRows - kGdnKeyRows;
    }

    const int token_tiles = (tokens + TokenTile - 1) / TokenTile;
    for (int tile = warp; tile < token_tiles; tile += WarpsPerCta) {
        const int token0        = tile * TokenTile;
        const auto load_history = [&](int token) {
            if (token < 0) { return shared.initial[token + 3][lane]; }
            return __bfloat162float(
                projected[static_cast<std::int64_t>(token) * kGdnChannels + row]);
        };
        float s0 = load_history(token0 - 3);
        float s1 = load_history(token0 - 2);
        float s2 = load_history(token0 - 1);

#pragma unroll
        for (int local_token = 0; local_token < TokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token < tokens) {
                const std::int64_t projected_base = static_cast<std::int64_t>(token) * kGdnChannels;
                if (token >= valid) {
                    output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                        __float2bfloat16_rn(0.0F);
                    continue;
                }
                const float p = __bfloat162float(projected[projected_base + row]);
                float conv    = fmaf(w0, s0, 0.0F);
                conv          = fmaf(w1, s1, conv);
                conv          = fmaf(w2, s2, conv);
                conv          = fmaf(w3, p, conv);
                output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                    __float2bfloat16_rn(silu(conv));
                publish.publish(token, 0, row, s1, s2, p);
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }
    }
}

template <int WarpsPerCta, class Publish>
void launch_sized(const __nv_bfloat16* projected, const __nv_bfloat16* conv_weight,
                  const __nv_bfloat16* conv_states, const std::int32_t* initial_slot,
                  const std::int32_t* valid_columns, __nv_bfloat16* query, __nv_bfloat16* key,
                  __nv_bfloat16* value, std::int32_t tokens, Publish publish, cudaStream_t stream) {
    constexpr int kBlocks = kGdnChannels / 32;
    const infernix::pdl::LaunchConfig config{dim3(kBlocks), dim3(WarpsPerCta * 32), 0, stream};
    // Programmatic dependence on the projection that produced `projected`: its CTA drain overlaps
    // this grid's launch (only while a CUDA Graph captures the stream; eager launches serialize).
    CUDA_CHECK(infernix::pdl::launch_with(infernix::pdl::Dependency::Programmatic, config,
                                          nvfp4_gdn_conv_post_kernel<WarpsPerCta, 8, Publish>,
                                          projected, conv_weight, conv_states, initial_slot,
                                          valid_columns, query, key, value, tokens, publish));
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& conv_states,
            const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query, Tensor& key,
            Tensor& value, Publish publish, cudaStream_t stream) {
    const std::int32_t tokens = projected.ne[1];
    const std::int64_t tiles  = (tokens + 8 - 1) / 8;
    // One warp per token tile while four cover the call; eight through T = 64 (the A16 registration's
    // ceiling); the retired 32-warp form above that, where prefill chunks stream every tile anyway.
    const auto* projected_ptr   = static_cast<const __nv_bfloat16*>(projected.data);
    const auto* conv_weight_ptr = static_cast<const __nv_bfloat16*>(conv_weight.data);
    const auto* conv_states_ptr = static_cast<const __nv_bfloat16*>(conv_states.data);
    const auto* initial_ptr     = static_cast<const std::int32_t*>(initial_slot.data);
    const std::int32_t* valid_ptr =
        valid_columns.data == nullptr ? nullptr : static_cast<const std::int32_t*>(valid_columns.data);
    auto* query_ptr = static_cast<__nv_bfloat16*>(query.data);
    auto* key_ptr   = static_cast<__nv_bfloat16*>(key.data);
    auto* value_ptr = static_cast<__nv_bfloat16*>(value.data);
    if (tiles <= 1) {
        launch_sized<1>(projected_ptr, conv_weight_ptr, conv_states_ptr, initial_ptr, valid_ptr,
                        query_ptr, key_ptr, value_ptr, tokens, publish, stream);
    } else if (tiles <= 4) {
        launch_sized<4>(projected_ptr, conv_weight_ptr, conv_states_ptr, initial_ptr, valid_ptr,
                        query_ptr, key_ptr, value_ptr, tokens, publish, stream);
    } else if (tiles <= 8) {
        launch_sized<8>(projected_ptr, conv_weight_ptr, conv_states_ptr, initial_ptr, valid_ptr,
                        query_ptr, key_ptr, value_ptr, tokens, publish, stream);
    } else {
        launch_sized<32>(projected_ptr, conv_weight_ptr, conv_states_ptr, initial_ptr, valid_ptr,
                         query_ptr, key_ptr, value_ptr, tokens, publish, stream);
    }
}

} // namespace

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& valid_columns,
                                    const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                    Tensor& query, Tensor& key, Tensor& value,
                                    cudaStream_t stream) {
    launch(projected, conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
           SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                  static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                  kGdnChannels},
           stream);
}

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, cudaStream_t stream) {
    launch(conv_record, conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
           NoHistoryPublish{}, stream);
}

} // namespace infernix::ops::detail
