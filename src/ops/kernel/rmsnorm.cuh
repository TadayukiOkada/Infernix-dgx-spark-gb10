#pragma once

// infernix::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "core/pdl.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,        // x * inv * weight * SiLU(z)
    SigmoidGated, // x * inv * weight * sigmoid(z)
};

[[nodiscard]] __host__ __device__ constexpr bool rms_gated(RmsEpilogue epilogue) {
    return epilogue == RmsEpilogue::Gated || epilogue == RmsEpilogue::SigmoidGated;
}

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    if constexpr (Epilogue == RmsEpilogue::SigmoidGated) { value *= 1.0f / (1.0f + expf(-z)); }
    return value;
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
template <RmsEpilogue Epilogue, int Block, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out,
                                    std::int32_t input_d, std::int64_t rows, float eps) {
    pdl::enter();
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    __nv_bfloat162 weights[kMaxPairsPerLane];
    __nv_bfloat162 gates[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k] = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (rms_gated(Epilogue)) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (rms_gated(Epilogue)) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (rms_gated(Epilogue)) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (rms_gated(Epilogue)) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/infernix/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(kFullWarpMask, inv, 0);

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (rms_gated(Epilogue)) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out,
                                   std::int32_t input_d, std::int64_t rows, float eps) {
    pdl::enter();
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    // Neither the weight nor the gate depends on the sum of squares, so under Prefetch both
    // are issued in the same pass as x and the two trips to memory overlap instead of running
    // back to back across the reduction. They cost registers, so the launcher turns this off
    // once the grid is wide enough that occupancy is worth more than the overlap.
    __nv_bfloat162 weights[MaxPairsPerThread];
    __nv_bfloat162 gates[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            values[k]      = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (rms_gated(Epilogue)) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (rms_gated(Epilogue)) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (rms_gated(Epilogue)) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (rms_gated(Epilogue)) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/infernix/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    pdl::enter();
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (rms_gated(Epilogue)) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Implements: include/infernix/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain or offset epilogue, D=5120, sm_120a / sm_121a.
// Algorithm assumptions: exactly five BF16x2 values per thread; one 512-thread CTA owns one row.
// Hoisted weights and inputs in registers eliminate memory roundtrips; (1.0f / 5120.0f) replaces
// runtime division with single-cycle reciprocal multiplication.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d5120_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    pdl::enter();
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 2560;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const int pair2             = pair0 + kBlock * 2;
    const int pair3             = pair0 + kBlock * 3;
    const int pair4             = pair0 + kBlock * 4;

    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const __nv_bfloat162 value2 = x[row_base + pair2];
    const __nv_bfloat162 value3 = x[row_base + pair3];
    const __nv_bfloat162 value4 = x[row_base + pair4];

    const __nv_bfloat162 w_val0 = weight[pair0];
    const __nv_bfloat162 w_val1 = weight[pair1];
    const __nv_bfloat162 w_val2 = weight[pair2];
    const __nv_bfloat162 w_val3 = weight[pair3];
    const __nv_bfloat162 w_val4 = weight[pair4];

    const float2 x0 = __bfloat1622float2(value0);
    const float2 x1 = __bfloat1622float2(value1);
    const float2 x2 = __bfloat1622float2(value2);
    const float2 x3 = __bfloat1622float2(value3);
    const float2 x4 = __bfloat1622float2(value4);

    const float local_sum = x0.x * x0.x + x0.y * x0.y +
                            x1.x * x1.x + x1.y * x1.y +
                            x2.x * x2.x + x2.y * x2.y +
                            x3.x * x3.x + x3.y * x3.y +
                            x4.x * x4.x + x4.y * x4.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 5120.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = __bfloat1622float2(w_val0);
    const float2 w1 = __bfloat1622float2(w_val1);
    const float2 w2 = __bfloat1622float2(w_val2);
    const float2 w3 = __bfloat1622float2(w_val3);
    const float2 w4 = __bfloat1622float2(w_val4);

    float2 z0{0.0f, 0.0f}, z1{0.0f, 0.0f}, z2{0.0f, 0.0f}, z3{0.0f, 0.0f}, z4{0.0f, 0.0f};
    if constexpr (rms_gated(Epilogue)) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
        z2 = __bfloat1622float2(z[row_base + pair2]);
        z3 = __bfloat1622float2(z[row_base + pair3]);
        z4 = __bfloat1622float2(z[row_base + pair4]);
    }

    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
    out[row_base + pair2] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x2.x, inv, w2.x, z2.x),
                              rmsnorm_epilogue<Epilogue>(x2.y, inv, w2.y, z2.y));
    out[row_base + pair3] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x3.x, inv, w3.x, z3.x),
                              rmsnorm_epilogue<Epilogue>(x3.y, inv, w3.y, z3.y));
    out[row_base + pair4] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x4.x, inv, w4.x, z4.x),
                              rmsnorm_epilogue<Epilogue>(x4.y, inv, w4.y, z4.y));
}

__device__ __forceinline__ float rms_pair_sq(unsigned p) {
    const float a = __uint_as_float(p << 16);
    const float b = __uint_as_float(p & 0xffff0000u);
    return a * a + b * b;
}

__device__ __forceinline__ float rms_vec_sq(uint4 v) {
    return rms_pair_sq(v.x) + rms_pair_sq(v.y) + rms_pair_sq(v.z) + rms_pair_sq(v.w);
}

template <RmsEpilogue Epilogue>
__device__ __forceinline__ unsigned rms_pair_out(unsigned x, unsigned w, float inv) {
    const float x0    = __uint_as_float(x << 16);
    const float x1    = __uint_as_float(x & 0xffff0000u);
    const float w0    = __uint_as_float(w << 16);
    const float w1    = __uint_as_float(w & 0xffff0000u);
    const __nv_bfloat162 v = __floats2bfloat162_rn(
        rmsnorm_epilogue<Epilogue>(x0, inv, w0, 0.0f),
        rmsnorm_epilogue<Epilogue>(x1, inv, w1, 0.0f));
    return *reinterpret_cast<const unsigned*>(&v);
}

template <RmsEpilogue Epilogue>
__device__ __forceinline__ uint4 rms_vec_out(uint4 x, uint4 w, float inv) {
    return make_uint4(rms_pair_out<Epilogue>(x.x, w.x, inv), rms_pair_out<Epilogue>(x.y, w.y, inv),
                      rms_pair_out<Epilogue>(x.z, w.z, inv), rms_pair_out<Epilogue>(x.w, w.w, inv));
}

// Five independent row reductions for D=5120 decode (rows=1). Each CTA writes one 128-vector tile.
template <RmsEpilogue Epilogue>
__launch_bounds__(128) __global__
void rmsnorm_d5120_decode_vec16_kernel(const uint4* __restrict__ x, const uint4* __restrict__ w,
                                       uint4* __restrict__ y, float eps) {
    pdl::enter();
    const int t    = static_cast<int>(threadIdx.x);
    const int lane = t & 31;
    const int warp = t >> 5;
    const float a  = rms_vec_sq(x[t]) + rms_vec_sq(x[t + 256]);
    const float b  = rms_vec_sq(x[t + 128]) + rms_vec_sq(x[t + 384]);
    float sum      = a + b + rms_vec_sq(x[t + 512]);
    sum            = warp_reduce_sum(sum);
    __shared__ float partial[4];
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    float total = lane < 4 ? partial[lane] : 0.0f;
    total       = warp_reduce_sum(total);
    float inv   = lane == 0 ? rsqrtf(total * (1.0f / 5120.0f) + eps) : 0.0f;
    inv         = __shfl_sync(kFullWarpMask, inv, 0);
    const int j = static_cast<int>(blockIdx.x) * 128 + t;
    y[j]        = rms_vec_out<Epilogue>(x[j], w[j], inv);
}

// Verify: rows=16, D=5120. Two CTAs per row, 320 threads each (10 warps).
// Each thread loads exactly 2 uint4s (indices t and t+320), covering all 640.
// Writeback index j = part*320+t matches one of the loaded values (register cache).
// w[j] loaded before reduction to overlap L2 latency with shuffle steps.
template <RmsEpilogue Epilogue>
__launch_bounds__(320) __global__
void rmsnorm_d5120_verify_vec16_kernel(const uint4* __restrict__ x, const uint4* __restrict__ w,
                                       uint4* __restrict__ y, float eps) {
    pdl::enter();
    const int t       = static_cast<int>(threadIdx.x);
    const int lane    = t & 31;
    const int warp    = t >> 5;
    const size_t row  = static_cast<size_t>(blockIdx.x) >> 1;
    const int part    = static_cast<int>(blockIdx.x) & 1;
    const size_t base = row * 640;
    const uint4 v0    = x[base + t];
    const uint4 v1    = x[base + 320 + t];
    const int j       = part * 320 + t;
    const uint4 wv    = w[j];
    float sum         = rms_vec_sq(v0) + rms_vec_sq(v1);
    sum = warp_reduce_sum(sum);
    __shared__ float partial[10];
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    float total = lane < 10 ? partial[lane] : 0.0f;
    total       = warp_reduce_sum(total);
    float inv   = lane == 0 ? rsqrtf(total * (1.0f / 5120.0f) + eps) : 0.0f;
    inv         = __shfl_sync(kFullWarpMask, inv, 0);
    const uint4 xv = (part == 0) ? v0 : v1;
    y[base + j] = rms_vec_out<Epilogue>(xv, wv, inv);
}

template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
void rmsnorm_d5120_rows_vec16_kernel(const uint4* __restrict__ x, const uint4* __restrict__ w,
                                     uint4* __restrict__ y, float eps) {
    pdl::enter();
    const int t       = static_cast<int>(threadIdx.x);
    const int lane    = t & 31;
    const int warp    = t >> 5;
    const size_t base = static_cast<size_t>(blockIdx.x) * 640;
    const uint4 a     = x[base + t];
    const uint4 b     = x[base + 256 + t];
    uint4 c           = make_uint4(0, 0, 0, 0);
    if (t < 128) { c = x[base + 512 + t]; }
    float sum = rms_vec_sq(a) + rms_vec_sq(b);
    if (t < 128) { sum += rms_vec_sq(c); }
    sum = warp_reduce_sum(sum);
    __shared__ float partial[8];
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    float total = lane < 8 ? partial[lane] : 0.0f;
    total       = warp_reduce_sum(total);
    float inv   = lane == 0 ? rsqrtf(total * (1.0f / 5120.0f) + eps) : 0.0f;
    inv         = __shfl_sync(kFullWarpMask, inv, 0);
    y[base + t]       = rms_vec_out<Epilogue>(a, w[t], inv);
    y[base + 256 + t] = rms_vec_out<Epilogue>(b, w[256 + t], inv);
    if (t < 128) { y[base + 512 + t] = rms_vec_out<Epilogue>(c, w[512 + t], inv); }
}

template <RmsEpilogue Epilogue>
__launch_bounds__(320) __global__
void rmsnorm_d5120_prefill_2row_vec16_kernel(const uint4* __restrict__ x, const uint4* __restrict__ w,
                                             uint4* __restrict__ y, size_t rows, float eps) {
    pdl::enter();
    const int t        = static_cast<int>(threadIdx.x);
    const int lane     = t & 31;
    const int warp     = t >> 5;
    const size_t row0  = static_cast<size_t>(blockIdx.x) * 2;
    const size_t base0 = row0 * 640;
    const size_t base1 = base0 + 640;
    const bool has1    = (row0 + 1) < rows;

    const uint4 v0 = x[base0 + t];
    const uint4 v1 = x[base0 + 320 + t];
    uint4 u0 = make_uint4(0, 0, 0, 0);
    uint4 u1 = make_uint4(0, 0, 0, 0);
    if (has1) {
        u0 = x[base1 + t];
        u1 = x[base1 + 320 + t];
    }
    const uint4 w0 = w[t];
    const uint4 w1 = w[320 + t];

    float s0 = rms_vec_sq(v0) + rms_vec_sq(v1);
    float s1 = has1 ? (rms_vec_sq(u0) + rms_vec_sq(u1)) : 0.0f;

    s0 = warp_reduce_sum(s0);
    s1 = warp_reduce_sum(s1);

    __shared__ float p0[10];
    __shared__ float p1[10];
    if (lane == 0) {
        p0[warp] = s0;
        p1[warp] = s1;
    }
    __syncthreads();

    float t0 = p0[0];
    float t1 = p1[0];
#pragma unroll
    for (int i = 1; i < 10; ++i) {
        t0 += p0[i];
        t1 += p1[i];
    }
    const float inv0 = rsqrtf(t0 * (1.0f / 5120.0f) + eps);
    const float inv1 = rsqrtf(t1 * (1.0f / 5120.0f) + eps);

    y[base0 + t]       = rms_vec_out<Epilogue>(v0, w0, inv0);
    y[base0 + 320 + t] = rms_vec_out<Epilogue>(v1, w1, inv0);
    if (has1) {
        y[base1 + t]       = rms_vec_out<Epilogue>(u0, w0, inv1);
        y[base1 + 320 + t] = rms_vec_out<Epilogue>(u1, w1, inv1);
    }
}

// Rows the preloading instance of the generic kernel serves: wider than every fast route, at most
// kRmsPreloadMaxD (the Qwen4Exp MTP hidden_norm, d = 10240).
inline constexpr std::int32_t kRmsPreloadMinD = 8192;
inline constexpr std::int32_t kRmsPreloadMidD = 10240;
inline constexpr std::int32_t kRmsPreloadMaxD = 16384;

// The generic kernel's reduction: warp-shuffle sum reduction, then inverse RMS.
__device__ __forceinline__ float rmsnorm_generic_inv(float sum, float* scratch, std::int32_t d, float eps) {
    const float block_sum = block_reduce_sum<256>(sum, scratch);
    if (threadIdx.x == 0) { scratch[0] = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    return scratch[0];
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths. Thread i owns elements i, i + 256,
// ... and sums their squares in that order; the shared tree is fixed.
//
// Preload > 0 is the instance for kRmsPreloadMinD < d <= Preload * 256 (ungated epilogues only):
// a thread issues every load of its at most Preload elements of x and of the weight before its
// first sum, so a row costs one memory round trip instead of a chain of dependent ones. The
// arithmetic and its order are the same, so both instances return the same bits.
template <RmsEpilogue Epilogue, int Preload = 0>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    pdl::enter();
    static_assert(Preload >= 0 && (Preload == 0 || !rms_gated(Epilogue)),
                  "the preloading instance serves the ungated epilogues");
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    __shared__ float scratch[256];
    if constexpr (Preload > 0) {
        __nv_bfloat16 xs[Preload];
        __nv_bfloat16 ws[Preload];
#pragma unroll
        for (int j = 0; j < Preload; ++j) {
            const std::int64_t i = threadIdx.x + static_cast<std::int64_t>(j) * blockDim.x;
            if (i < static_cast<std::int64_t>(d)) {
                xs[j] = x[base + i];
                ws[j] = weight[i];
            }
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < Preload; ++j) {
            if (threadIdx.x + static_cast<std::int64_t>(j) * blockDim.x < static_cast<std::int64_t>(d)) {
                const float xv = __bfloat162float(xs[j]);
                sum += xv * xv;
            }
        }
        const float inv = rmsnorm_generic_inv(sum, scratch, d, eps);
#pragma unroll
        for (int j = 0; j < Preload; ++j) {
            const std::int64_t i = threadIdx.x + static_cast<std::int64_t>(j) * blockDim.x;
            if (i < static_cast<std::int64_t>(d)) {
                out[base + i] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(
                    __bfloat162float(xs[j]), inv, __bfloat162float(ws[j]), 0.0f));
            }
        }
    } else {
        float sum = 0.0f;
        for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
            const float xv = __bfloat162float(x[base + i]);
            sum += xv * xv;
        }
        const float inv = rmsnorm_generic_inv(sum, scratch, d, eps);
        for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
            const std::int64_t index = base + i;
            const float xv           = __bfloat162float(x[index]);
            const float wv           = __bfloat162float(weight[i]);
            float zv                 = 0.0f;
            if constexpr (rms_gated(Epilogue)) { zv = __bfloat162float(z[index]); }
            out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
        }
    }
}

} // namespace infernix::ops
