#pragma once

// Implements: include/infernix/ops/target_logprobs.h
// Match: contiguous BF16 or FP32 [physical_rows,C], I32 [T,C], and FP32 [T,C].
// Algorithm assumptions: one 256-thread CTA performs a single-pass online logsumexp per column.

#include "ops/common/warp.cuh"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kTargetLogprobsBlock = 256;

template <int BlockSize>
struct TargetLogprobsShared {
    float warp_scratch[BlockSize / kWarpSize];
    float scalar;
};

template <int BlockSize>
__device__ __forceinline__ float target_logprobs_block_max(float value,
                                                           TargetLogprobsShared<BlockSize>& shared) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { shared.warp_scratch[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? shared.warp_scratch[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { shared.scalar = value; }
    }
    __syncthreads();
    return shared.scalar;
}

// A column's maximum valid logit and log(sum(exp(logit - maximum))), returned to every thread.
struct TargetLogprobsColumn {
    float maximum;
    float log_sum;
};

// A logit as FP32 (exact for both storage types).
__device__ __forceinline__ float logprob_logit(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ __forceinline__ float logprob_logit(float value) { return value; }

template <int BlockSize, class Logit>
__device__ __forceinline__ TargetLogprobsColumn
target_logprobs_column(const Logit* values, std::int32_t valid_rows,
                       TargetLogprobsShared<BlockSize>& shared) {
    float local_max = -CUDART_INF_F;
    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = logprob_logit(values[row]);
        if (value > -CUDART_INF_F) {
            const float new_max = fmaxf(local_max, value);
            if (new_max > local_max) {
                local_sum *= expf(local_max - new_max);
            }
            local_max = new_max;
            local_sum += expf(value - local_max);
        }
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max, shared);

    if (maximum > -CUDART_INF_F) {
        local_sum *= expf(local_max - maximum);
    }
    const float sum = block_reduce_sum<BlockSize>(local_sum, shared.warp_scratch);
    if (threadIdx.x == 0) { shared.scalar = (maximum > -CUDART_INF_F && sum > 0.0f) ? logf(sum) : 0.0f; }
    __syncthreads();
    return {maximum, shared.scalar};
}

template <int BlockSize, class Logit>
__launch_bounds__(BlockSize) __global__
    void target_logprobs_kernel(const Logit* logits, const std::int32_t* target_ids,
                                float* output, std::int32_t valid_rows,
                                std::int32_t physical_rows, std::int32_t targets) {
    pdl::enter();
    __shared__ TargetLogprobsShared<BlockSize> shared;
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const Logit* values       = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn lse = target_logprobs_column<BlockSize>(values, valid_rows, shared);
    for (std::int32_t target = static_cast<std::int32_t>(threadIdx.x); target < targets;
         target += BlockSize) {
        const std::int64_t index = static_cast<std::int64_t>(column) * targets + target;
        output[index] = logprob_logit(values[target_ids[index]]) - lse.maximum - lse.log_sum;
    }
}

} // namespace infernix::ops
