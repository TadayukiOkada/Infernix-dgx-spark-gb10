#pragma once

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace infernix::ops {

struct LayerNormMoments {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ LayerNormMoments layer_norm_merge(LayerNormMoments a,
                                                             LayerNormMoments b) {
    if (b.count == 0) { return a; }
    if (a.count == 0) { return b; }
    const float delta = b.mean - a.mean;
    const int count   = a.count + b.count;
    const float ratio = static_cast<float>(b.count) / static_cast<float>(count);
    return LayerNormMoments{
        a.mean + delta * ratio,
        a.m2 + b.m2 +
            delta * delta *
                (static_cast<float>(a.count) * static_cast<float>(b.count) /
                 static_cast<float>(count)),
        count,
    };
}

__device__ __forceinline__ LayerNormMoments layer_norm_warp_reduce(LayerNormMoments value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const LayerNormMoments other{
            __shfl_down_sync(mask, value.mean, offset),
            __shfl_down_sync(mask, value.m2, offset),
            __shfl_down_sync(mask, value.count, offset),
        };
        value = layer_norm_merge(value, other);
    }
    return value;
}

__device__ __forceinline__ void layer_norm_update(LayerNormMoments& moments, float value) {
    ++moments.count;
    const float delta = value - moments.mean;
    moments.mean += delta / static_cast<float>(moments.count);
    moments.m2 += delta * (value - moments.mean);
}

template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                        const __nv_bfloat162* bias, __nv_bfloat162* out,
                                        std::int64_t rows, float eps) {
    pdl::enter();
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;

    constexpr int kMaxPairsPerThread = (pairs + Block - 1) / Block;
    __nv_bfloat162 cached_x[kMaxPairsPerThread];
    int count = 0;
    LayerNormMoments local{0.0f, 0.0f, 0};
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        const __nv_bfloat162 val = x[base + pair];
        cached_x[count++]        = val;
        const float2 values      = __bfloat1622float2(val);
        layer_norm_update(local, values.x);
        layer_norm_update(local, values.y);
    }
    local = layer_norm_warp_reduce(local);

    __shared__ LayerNormMoments warp_moments[warps];
    __shared__ float final_mean;
    __shared__ float final_inv;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { warp_moments[warp] = local; }
    __syncthreads();

    if (warp == 0) {
        LayerNormMoments block_value =
            lane < warps ? warp_moments[lane] : LayerNormMoments{0.0f, 0.0f, 0};
        block_value = layer_norm_warp_reduce(block_value);
        if (lane == 0) {
            final_mean = block_value.mean;
            final_inv  = rsqrtf(block_value.m2 / static_cast<float>(d) + eps);
        }
    }
    __syncthreads();

    count = 0;
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        const float2 values = __bfloat1622float2(cached_x[count++]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair] =
            __floats2bfloat162_rn((values.x - final_mean) * final_inv * scales.x + shifts.x,
                                  (values.y - final_mean) * final_inv * scales.y + shifts.y);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_warp_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                      const __nv_bfloat162* bias, __nv_bfloat162* out,
                                      std::int64_t rows, float eps) {
    pdl::enter();
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + warp;
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;

    constexpr int kMaxPairsPerLane = (pairs + 31) / 32; // 18 pairs
    __nv_bfloat162 cached_x[kMaxPairsPerLane];
    int count = 0;
    LayerNormMoments local{0.0f, 0.0f, 0};
    for (int pair = lane; pair < pairs; pair += 32) {
        const __nv_bfloat162 val = x[base + pair];
        cached_x[count++]        = val;
        const float2 values      = __bfloat1622float2(val);
        layer_norm_update(local, values.x);
        layer_norm_update(local, values.y);
    }
    local                   = layer_norm_warp_reduce(local);
    constexpr unsigned mask = 0xffffffffu;
    const float mean        = __shfl_sync(mask, local.mean, 0);
    const float m2          = __shfl_sync(mask, local.m2, 0);
    const float inv         = rsqrtf(m2 / static_cast<float>(d) + eps);

    count = 0;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(cached_x[count++]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair]    = __floats2bfloat162_rn((values.x - mean) * inv * scales.x + shifts.x,
                                                    (values.y - mean) * inv * scales.y + shifts.y);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_centered_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                          const __nv_bfloat162* bias, __nv_bfloat162* out,
                                          std::int64_t rows, float eps) {
    pdl::enter();
    constexpr int kPairs   = 576;
    constexpr int kItems   = (kPairs + Block - 1) / Block;
    const int lane         = static_cast<int>(threadIdx.x);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * kPairs;
    // Center around a represented input before reducing, including large constant rows.
    const float anchor = __low2float(x[base]);
    float2 values[kItems];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int pair = lane + i * Block;
        if (pair < kPairs) {
            const float2 v = __bfloat1622float2(x[base + pair]);
            values[i]      = {v.x - anchor, v.y - anchor};
            sum += values[i].x + values[i].y;
        }
    }
    __shared__ float sums[Block / kWarpSize];
    __shared__ float scalar;
    const auto reduce = [&](float value) {
        value = block_reduce_sum<Block>(value, sums);
        if (threadIdx.x == 0) { scalar = value; }
        __syncthreads();
        return scalar;
    };
    const float mean = reduce(sum) * (1.0f / 1152.0f);
    float squares    = 0.0f;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        if (lane + i * Block < kPairs) {
            values[i].x -= mean;
            values[i].y -= mean;
            squares = fmaf(values[i].x, values[i].x, squares);
            squares = fmaf(values[i].y, values[i].y, squares);
        }
    }
    const float inverse = rsqrtf(reduce(squares) * (1.0f / 1152.0f) + eps);
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int pair = lane + i * Block;
        if (pair < kPairs) {
            const float2 w   = __bfloat1622float2(weight[pair]);
            const float2 b   = __bfloat1622float2(bias[pair]);
            out[base + pair] = __floats2bfloat162_rn(values[i].x * inverse * w.x + b.x,
                                                     values[i].y * inverse * w.y + b.y);
        }
    }
}

template <int Block>
__global__ void layer_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                  const __nv_bfloat16* bias, __nv_bfloat16* out, std::int32_t d,
                                  std::int64_t rows, float eps) {
    pdl::enter();
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * d;

    LayerNormMoments local{0.0f, 0.0f, 0};
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = __bfloat162float(x[base + i]);
        ++local.count;
        const float delta = value - local.mean;
        local.mean += delta / static_cast<float>(local.count);
        local.m2 += delta * (value - local.mean);
    }

    local = layer_norm_warp_reduce(local);

    constexpr int kWarps = Block / 32;
    __shared__ LayerNormMoments warp_moments[kWarps > 0 ? kWarps : 1];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0 && warp < kWarps) { warp_moments[warp] = local; }
    __syncthreads();

    if (warp == 0) {
        LayerNormMoments block_value =
            lane < kWarps ? warp_moments[lane] : LayerNormMoments{0.0f, 0.0f, 0};
        block_value = layer_norm_warp_reduce(block_value);
        if (lane == 0) {
            warp_moments[0] = block_value;
        }
    }
    __syncthreads();

    const float mean = warp_moments[0].mean;
    const float inv  = rsqrtf(warp_moments[0].m2 / static_cast<float>(d) + eps);
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = (__bfloat162float(x[base + i]) - mean) * inv;
        out[base + i] =
            __float2bfloat16_rn(value * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}

} // namespace infernix::ops
