#include "infernix/ops/hyper_connection.h"

#include "ops/hyper_connection/hyper_connection_mix_fused.h"
#include "ops/hyper_connection/hyper_connection_up_collapse.h"

#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hyper_connection: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w]; }
    return total;
}

// One CTA per (stream, column). Each thread loads its elements d = tid, tid + 256, ... and their
// weights before any arithmetic (one memory round trip instead of two serial passes) and keeps the
// same per-thread order and block reduction.
constexpr int kNormPerThread = 16; // hidden <= 4096

__device__ __forceinline__ float norm_value(bf16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float norm_value(float w) { return w; }

// The norm weights are BF16, or FP32 (NVIDIA FP4-Local's trained norms).
template <class Norm = bf16, int PerThread = kNormPerThread, bool Full = false>
__global__ void __launch_bounds__(kThreads) norm_kernel(const bf16* __restrict__ residual,
                                                        const Norm* __restrict__ weight, int hidden,
                                                        int width, float eps, bf16* __restrict__ out) {
    __shared__ float scratch[kThreads / 32];
    const int s = blockIdx.x;
    const int t = gridDim.y > 1024 ? (static_cast<int>(gridDim.y) - 1 - static_cast<int>(blockIdx.y))
                                   : static_cast<int>(blockIdx.y);
    const bf16* x = residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    bf16* y       = out + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const Norm* w = weight + static_cast<std::size_t>(s) * hidden;
    bf16 xs[PerThread];
    Norm ws[PerThread];
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int d = threadIdx.x + i * kThreads;
        if (Full || d < hidden) {
            xs[i] = x[d];
            ws[i] = w[d];
        }
    }
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        if (Full || threadIdx.x + i * kThreads < hidden) {
            const float v = __bfloat162float(xs[i]);
            sum += v * v;
        }
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int d = threadIdx.x + i * kThreads;
        if (Full || d < hidden) {
            y[d] = __float2bfloat16_rn(__bfloat162float(xs[i]) * inv * (1.0F + norm_value(ws[i])));
        }
    }
}

template <class Norm = bf16, int PerThread = kNormPerThread, bool Full = false>
__global__ void __launch_bounds__(kThreads) inject_norm_kernel(const bf16* __restrict__ y_in,
                                                               const float* __restrict__ inject,
                                                               bf16* __restrict__ residual,
                                                               const Norm* __restrict__ weight, int hidden,
                                                               int streams, int width, float eps,
                                                               bf16* __restrict__ out) {
    pdl::enter();
    __shared__ float scratch[kThreads / 32];
    __shared__ float s_sc;
    const int s     = blockIdx.x;
    const int t     = gridDim.y > 1024 ? (static_cast<int>(gridDim.y) - 1 - static_cast<int>(blockIdx.y))
                                       : static_cast<int>(blockIdx.y);
    if (threadIdx.x == 0) {
        s_sc = inject[static_cast<std::size_t>(t) * streams + s];
    }
    __syncthreads();
    const float sc  = s_sc;
    const bf16* yin = y_in + static_cast<std::size_t>(t) * hidden;
    bf16* x         = residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    bf16* y         = out + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const Norm* w   = weight + static_cast<std::size_t>(s) * hidden;
    bf16 xs[PerThread];
    Norm ws[PerThread];
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int d = threadIdx.x + i * kThreads;
        if (Full || d < hidden) {
            const bf16 r = __float2bfloat16_rn(__bfloat162float(x[d]) + sc * __bfloat162float(yin[d]));
            x[d]         = r;
            xs[i]        = r;
            ws[i]        = w[d];
        }
    }
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        if (Full || threadIdx.x + i * kThreads < hidden) {
            const float v = __bfloat162float(xs[i]);
            sum += v * v;
        }
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int d = threadIdx.x + i * kThreads;
        if (Full || d < hidden) {
            y[d] = __float2bfloat16_rn(__bfloat162float(xs[i]) * inv * (1.0F + norm_value(ws[i])));
        }
    }
}

constexpr int kVec8Threads = 320; // 320 * 8 = 2560

template <class Norm>
__global__ void __launch_bounds__(kVec8Threads) norm_vec8_kernel(const bf16* __restrict__ residual,
                                                                 const Norm* __restrict__ weight, int hidden,
                                                                 int width, float eps, bf16* __restrict__ out) {
    __shared__ float scratch[kVec8Threads / 32];
    const int s   = blockIdx.x;
    const int t   = gridDim.y > 1024 ? (static_cast<int>(gridDim.y) - 1 - static_cast<int>(blockIdx.y))
                                     : static_cast<int>(blockIdx.y);
    const int d   = threadIdx.x * 8;
    const bf16* x = residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
    bf16* y       = out + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
    const Norm* w = weight + static_cast<std::size_t>(s) * hidden + d;

    const uint4 x_packed = *reinterpret_cast<const uint4*>(x);
    const bf16* xs       = reinterpret_cast<const bf16*>(&x_packed);
    float wf[8];
    if constexpr (sizeof(Norm) == 4) {
        const float4 w0 = *reinterpret_cast<const float4*>(w);
        const float4 w1 = *reinterpret_cast<const float4*>(w + 4);
        wf[0] = w0.x; wf[1] = w0.y; wf[2] = w0.z; wf[3] = w0.w;
        wf[4] = w1.x; wf[5] = w1.y; wf[6] = w1.z; wf[7] = w1.w;
    } else {
        const uint4 w_packed = *reinterpret_cast<const uint4*>(w);
        const bf16* ws       = reinterpret_cast<const bf16*>(&w_packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) { wf[j] = __bfloat162float(ws[j]); }
    }

    float sum = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = __bfloat162float(xs[j]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
    uint4 y_packed;
    bf16* ys = reinterpret_cast<bf16*>(&y_packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        ys[j] = __float2bfloat16_rn(__bfloat162float(xs[j]) * inv * (1.0F + wf[j]));
    }
    *reinterpret_cast<uint4*>(y) = y_packed;
}

template <class Norm>
__global__ void __launch_bounds__(kVec8Threads) inject_norm_vec8_kernel(const bf16* __restrict__ y_in,
                                                                        const float* __restrict__ inject,
                                                                        bf16* __restrict__ residual,
                                                                        const Norm* __restrict__ weight, int hidden,
                                                                        int streams, int width, float eps,
                                                                        bf16* __restrict__ out) {
    pdl::enter();
    __shared__ float scratch[kVec8Threads / 32];
    __shared__ float s_sc;
    const int s     = blockIdx.x;
    const int t     = gridDim.y > 1024 ? (static_cast<int>(gridDim.y) - 1 - static_cast<int>(blockIdx.y))
                                       : static_cast<int>(blockIdx.y);
    const int d     = threadIdx.x * 8;
    if (threadIdx.x == 0) {
        s_sc = inject[static_cast<std::size_t>(t) * streams + s];
    }
    __syncthreads();
    const float sc  = s_sc;
    const bf16* yin = y_in + static_cast<std::size_t>(t) * hidden + d;
    bf16* x         = residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
    bf16* y         = out + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
    const Norm* w   = weight + static_cast<std::size_t>(s) * hidden + d;

    const uint4 x_in_packed   = *reinterpret_cast<const uint4*>(x);
    const uint4 yin_in_packed = *reinterpret_cast<const uint4*>(yin);
    const bf16* x_in          = reinterpret_cast<const bf16*>(&x_in_packed);
    const bf16* yin_in        = reinterpret_cast<const bf16*>(&yin_in_packed);

    uint4 x_out_packed;
    bf16* xs = reinterpret_cast<bf16*>(&x_out_packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        xs[j] = __float2bfloat16_rn(__bfloat162float(x_in[j]) + sc * __bfloat162float(yin_in[j]));
    }
    *reinterpret_cast<uint4*>(x) = x_out_packed;

    float wf[8];
    if constexpr (sizeof(Norm) == 4) {
        const float4 w0 = *reinterpret_cast<const float4*>(w);
        const float4 w1 = *reinterpret_cast<const float4*>(w + 4);
        wf[0] = w0.x; wf[1] = w0.y; wf[2] = w0.z; wf[3] = w0.w;
        wf[4] = w1.x; wf[5] = w1.y; wf[6] = w1.z; wf[7] = w1.w;
    } else {
        const uint4 w_packed = *reinterpret_cast<const uint4*>(w);
        const bf16* ws       = reinterpret_cast<const bf16*>(&w_packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) { wf[j] = __bfloat162float(ws[j]); }
    }

    float sum = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = __bfloat162float(xs[j]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
    uint4 y_packed;
    bf16* ys = reinterpret_cast<bf16*>(&y_packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        ys[j] = __float2bfloat16_rn(__bfloat162float(xs[j]) * inv * (1.0F + wf[j]));
    }
    *reinterpret_cast<uint4*>(y) = y_packed;
}

__global__ void gates_kernel(const bf16* __restrict__ z, int rows, int rank, int streams, int columns,
                             bf16* __restrict__ mix, float* __restrict__ inject) {
    pdl::enter();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= rows || t >= columns) { return; }
    const float scale = 1.0F / static_cast<float>(streams);
    const float v     = __bfloat162float(z[static_cast<std::size_t>(t) * rows + i]) * scale;
    if (i < rank) {
        mix[static_cast<std::size_t>(t) * rank + i] = __float2bfloat16_rn(v * sigmoidf(v));
    } else if (inject != nullptr) {
        inject[static_cast<std::size_t>(t) * streams + (i - rank)] = 2.0F * sigmoidf(v);
    }
}

// The SiLU half of z in V-element register-cached vectors (one load, one store per 8 or 4 rows); the
// injection rows stay scalar. Elementwise, so the bits do not depend on V or the thread mapping.
template <int V>
__global__ void gates_vec_kernel(const bf16* __restrict__ z, int rows, int rank, int streams,
                                 bf16* __restrict__ mix, float* __restrict__ inject) {
    pdl::enter();
    const int item    = threadIdx.x;
    const int t       = blockIdx.y;
    const int vectors = rank / V;
    if (item < vectors) {
        unsigned bits[V / 2];
        if constexpr (V == 8) {
            const uint4 packed =
                *reinterpret_cast<const uint4*>(z + static_cast<std::size_t>(t) * rows + item * V);
            bits[0] = packed.x;
            bits[1] = packed.y;
            bits[2] = packed.z;
            bits[3] = packed.w;
        } else {
            const uint2 packed =
                *reinterpret_cast<const uint2*>(z + static_cast<std::size_t>(t) * rows + item * V);
            bits[0] = packed.x;
            bits[1] = packed.y;
        }
        unsigned out[V / 2];
        const float scale = 1.0F / static_cast<float>(streams);
#pragma unroll
        for (int g = 0; g < V / 2; ++g) {
            const __nv_bfloat162 in = *reinterpret_cast<const __nv_bfloat162*>(&bits[g]);
            const float v0          = __bfloat162float(in.x) * scale;
            const float v1          = __bfloat162float(in.y) * scale;
            const __nv_bfloat162 r  = __floats2bfloat162_rn(v0 * sigmoidf(v0), v1 * sigmoidf(v1));
            out[g]                  = *reinterpret_cast<const unsigned*>(&r);
        }
        if constexpr (V == 8) {
            const uint4 packed{out[0], out[1], out[2], out[3]};
            *reinterpret_cast<uint4*>(mix + static_cast<std::size_t>(t) * rank + item * V) = packed;
        } else {
            const uint2 packed{out[0], out[1]};
            *reinterpret_cast<uint2*>(mix + static_cast<std::size_t>(t) * rank + item * V) = packed;
        }
    } else if (inject != nullptr && item - vectors < streams) {
        const float scale = 1.0F / static_cast<float>(streams);
        const float v = __bfloat162float(z[static_cast<std::size_t>(t) * rows + rank + (item - vectors)]) * scale;
        inject[static_cast<std::size_t>(t) * streams + (item - vectors)] = 2.0F * sigmoidf(v);
    }
}

__global__ void collapse_kernel(const bf16* __restrict__ logits, const bf16* __restrict__ normalized,
                                int hidden, int streams, int columns, bf16* __restrict__ out) {
    pdl::enter();
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    float sum = 0.0F;
    for (int s = 0; s < streams; ++s) {
        const std::size_t i = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
        sum += sigmoidf(__bfloat162float(logits[i])) * __bfloat162float(normalized[i]);
    }
    const float inv_streams = 1.0F / static_cast<float>(streams);
    out[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum * inv_streams);
}

__global__ void inject_kernel(const bf16* __restrict__ y, const float* __restrict__ inject, int hidden,
                              int streams, int columns, bf16* __restrict__ residual) {
    // A programmatic dependent; its small grid lets the next mixer launch and fetch its weights at once.
    pdl::enter();
    const int t = blockIdx.y;
    if (t >= columns) { return; }
    __shared__ float s_inject[16];
    if (threadIdx.x < static_cast<unsigned>(streams) && threadIdx.x < 16U) {
        s_inject[threadIdx.x] = inject[static_cast<std::size_t>(t) * streams + threadIdx.x];
    }
    __syncthreads();
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) { return; }
    const float v           = __bfloat162float(y[static_cast<std::size_t>(t) * hidden + d]);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        const float inj = (s < 16) ? s_inject[s] : inject[static_cast<std::size_t>(t) * streams + s];
        bf16& r = residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d];
        r       = __float2bfloat16_rn(__bfloat162float(r) + inj * v);
    }
}

__global__ void expand_kernel(const bf16* __restrict__ x, int hidden, int streams, int columns,
                              bf16* __restrict__ residual) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const bf16 v            = x[static_cast<std::size_t>(t) * hidden + d];
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d] = v;
    }
}

__launch_bounds__(320) __global__ void collapse_vec8_kernel(const bf16* __restrict__ logits,
                                                            const bf16* __restrict__ normalized, int hidden,
                                                            int streams, int columns, bf16* __restrict__ out) {
    pdl::enter();
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    float sum[8]            = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int s = 0; s < streams; ++s) {
        const std::size_t i   = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
        const uint4 l_packed  = *reinterpret_cast<const uint4*>(logits + i);
        const uint4 n_packed  = *reinterpret_cast<const uint4*>(normalized + i);
        const bf16* l_bf16    = reinterpret_cast<const bf16*>(&l_packed);
        const bf16* n_bf16    = reinterpret_cast<const bf16*>(&n_packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum[j] += sigmoidf(__bfloat162float(l_bf16[j])) * __bfloat162float(n_bf16[j]);
        }
    }
    const float inv_streams = 1.0F / static_cast<float>(streams);
    uint4 out_packed;
    bf16* out_bf16 = reinterpret_cast<bf16*>(&out_packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) { out_bf16[j] = __float2bfloat16_rn(sum[j] * inv_streams); }
    *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(t) * hidden + d) = out_packed;
}

__launch_bounds__(320) __global__ void inject_vec8_kernel(const bf16* __restrict__ y, const float* __restrict__ inject,
                                                          int hidden, int streams, int columns,
                                                          bf16* __restrict__ residual) {
    pdl::enter();
    const int t = blockIdx.y;
    if (t >= columns) { return; }
    __shared__ float s_inject[16];
    if (threadIdx.x < static_cast<unsigned>(streams) && threadIdx.x < 16U) {
        s_inject[threadIdx.x] = inject[static_cast<std::size_t>(t) * streams + threadIdx.x];
    }
    __syncthreads();
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (d >= hidden) { return; }
    const uint4 y_packed = *reinterpret_cast<const uint4*>(y + static_cast<std::size_t>(t) * hidden + d);
    const bf16* y_bf16   = reinterpret_cast<const bf16*>(&y_packed);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(y_bf16[j]); }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        const float inj = (s < 16) ? s_inject[s] : inject[static_cast<std::size_t>(t) * streams + s];
        uint4* r_ptr    = reinterpret_cast<uint4*>(
            residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d);
        uint4 r_packed = *r_ptr;
        bf16* r        = reinterpret_cast<bf16*>(&r_packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) { r[j] = __float2bfloat16_rn(__bfloat162float(r[j]) + inj * v[j]); }
        *r_ptr = r_packed;
    }
}

__launch_bounds__(320) __global__ void expand_vec8_kernel(const bf16* __restrict__ x, int hidden, int streams,
                                                          int columns, bf16* __restrict__ residual) {
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const uint4 v           = *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * hidden + d);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        *reinterpret_cast<uint4*>(
            residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d) = v;
    }
}

bool aligned_16(const void* p) noexcept {
    return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U;
}

bool aligned_8(const void* p) noexcept {
    return (reinterpret_cast<std::uintptr_t>(p) & 7U) == 0U;
}

// out column t (row pitch `pitch` elements) = the mean of the streams, summed in stream order.
__global__ void stream_mean_kernel(const bf16* __restrict__ residual, int hidden, int streams, int columns,
                                   std::int64_t pitch, bf16* __restrict__ out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    const bf16* column      = residual + static_cast<std::size_t>(t) * width + d;
    float sum               = 0.0F;
    for (int s = 0; s < streams; ++s) { sum += __bfloat162float(column[static_cast<std::size_t>(s) * hidden]); }
    const float inv_streams = 1.0F / static_cast<float>(streams);
    out[static_cast<std::size_t>(t) * pitch + d] = __float2bfloat16_rn(sum * inv_streams);
}

__launch_bounds__(320) __global__ void stream_mean_vec8_kernel(const bf16* __restrict__ residual, int hidden,
                                                               int streams, int columns, std::int64_t pitch,
                                                               bf16* __restrict__ out) {
    const int d = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    const bf16* column      = residual + static_cast<std::size_t>(t) * width + d;
    float sum[8]            = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int s = 0; s < streams; ++s) {
        const uint4 r_packed = *reinterpret_cast<const uint4*>(column + static_cast<std::size_t>(s) * hidden);
        const bf16* r_vals   = reinterpret_cast<const bf16*>(&r_packed);
#pragma unroll
        for (int k = 0; k < 8; ++k) { sum[k] += __bfloat162float(r_vals[k]); }
    }
    const float inv_s = 1.0F / static_cast<float>(streams);
    uint4 out_packed;
    bf16* out_vals = reinterpret_cast<bf16*>(&out_packed);
#pragma unroll
    for (int k = 0; k < 8; ++k) { out_vals[k] = __float2bfloat16_rn(sum[k] * inv_s); }
    *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(t) * pitch + d) = out_packed;
}

// The composed route's steps (private since hyper_connection_mix).
void hyper_connection_norm(const Tensor& residual, const Tensor& weight, std::int32_t streams,
                           float eps, Tensor& out, cudaStream_t stream) {
    require(streams > 0 && contiguous_2d(residual, DType::BF16) && contiguous_2d(out, DType::BF16) &&
                (contiguous_2d(weight, DType::BF16) || contiguous_2d(weight, DType::FP32)),
            "norm requires contiguous BF16 tensors and a BF16 or FP32 weight");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % streams == 0 && (width / streams) % 8 == 0 && weight.ne[0] == width &&
                weight.ne[1] == 1 && out.ne[0] == width && out.ne[1] == columns && columns > 0,
            "norm shapes disagree");
    require(eps > 0, "norm epsilon must be positive");
    require(width / streams <= kThreads * kNormPerThread, "norm streams are wider than 4096");
    const int hidden = width / streams;
    const bool vec8  = hidden == kVec8Threads * 8 && aligned_16(residual.data) && aligned_16(weight.data) &&
                      aligned_16(out.data);
    if (weight.dtype == DType::FP32) {
        if (vec8) {
            norm_vec8_kernel<float><<<dim3(streams, columns), kVec8Threads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const float*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        } else if (hidden == kThreads * 10) {
            norm_kernel<float, 10, true><<<dim3(streams, columns), kThreads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const float*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        } else {
            norm_kernel<float><<<dim3(streams, columns), kThreads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const float*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        }
    } else {
        if (vec8) {
            norm_vec8_kernel<bf16><<<dim3(streams, columns), kVec8Threads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        } else if (hidden == kThreads * 10) {
            norm_kernel<bf16, 10, true><<<dim3(streams, columns), kThreads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        } else {
            norm_kernel<bf16><<<dim3(streams, columns), kThreads, 0, stream>>>(
                static_cast<const bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                hidden, width, eps, static_cast<bf16*>(out.data));
        }
    }
    check_launch("norm");
}

void hyper_connection_inject_norm(const Tensor& y, const Tensor& inject, Tensor& residual,
                                  const Tensor& weight, std::int32_t streams, float eps, Tensor& out,
                                  cudaStream_t stream) {
    require(streams > 0 && contiguous_2d(y, DType::BF16) && contiguous_2d(inject, DType::FP32) &&
                contiguous_2d(residual, DType::BF16) && contiguous_2d(out, DType::BF16) &&
                (contiguous_2d(weight, DType::BF16) || contiguous_2d(weight, DType::FP32)),
            "inject norm requires contiguous tensors and a BF16 or FP32 weight");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % streams == 0 && (width / streams) % 8 == 0 && weight.ne[0] == width &&
                weight.ne[1] == 1 && out.ne[0] == width && out.ne[1] == columns && columns > 0 &&
                y.ne[0] == width / streams && y.ne[1] == columns && inject.ne[0] == streams &&
                inject.ne[1] == columns,
            "inject norm shapes disagree");
    require(eps > 0, "norm epsilon must be positive");
    require(width / streams <= kThreads * kNormPerThread, "norm streams are wider than 4096");
    const int hidden = width / streams;
    const bool vec8  = hidden == kVec8Threads * 8 && aligned_16(y.data) && aligned_16(residual.data) &&
                      aligned_16(weight.data) && aligned_16(out.data);
    const pdl::Dependency dep = columns <= 16 ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized;
    const pdl::LaunchConfig cfg{dim3(streams, columns), dim3(vec8 ? kVec8Threads : kThreads), 0, stream};
    cudaError_t status = cudaSuccess;
    if (weight.dtype == DType::FP32) {
        if (vec8) {
            status = pdl::launch_with(dep, cfg, inject_norm_vec8_kernel<float>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const float*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        } else if (hidden == kThreads * 10) {
            status = pdl::launch_with(dep, cfg, inject_norm_kernel<float, 10, true>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const float*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        } else {
            status = pdl::launch_with(dep, cfg, inject_norm_kernel<float>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const float*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        }
    } else {
        if (vec8) {
            status = pdl::launch_with(dep, cfg, inject_norm_vec8_kernel<bf16>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        } else if (hidden == kThreads * 10) {
            status = pdl::launch_with(dep, cfg, inject_norm_kernel<bf16, 10, true>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        } else {
            status = pdl::launch_with(dep, cfg, inject_norm_kernel<bf16>,
                                      static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                                      static_cast<bf16*>(residual.data), static_cast<const bf16*>(weight.data),
                                      hidden, streams, width, eps, static_cast<bf16*>(out.data));
        }
    }
    if (status != cudaSuccess) { check_launch("inject norm"); }
}

void hyper_connection_gates(const Tensor& z, std::int32_t rank, std::int32_t streams, Tensor& mix,
                            Tensor* inject, cudaStream_t stream) {
    require(contiguous_2d(z, DType::BF16) && contiguous_2d(mix, DType::BF16) && rank > 0 && streams > 0,
            "gates require contiguous BF16 tensors");
    const int columns = z.ne[1];
    const int rows    = rank + (inject != nullptr ? streams : 0);
    require(z.ne[0] == rows && mix.ne[0] == rank && mix.ne[1] == columns && columns > 0,
            "gates shapes disagree");
    if (inject != nullptr) {
        require(contiguous_2d(*inject, DType::FP32) && inject->ne[0] == streams &&
                    inject->ne[1] == columns,
                "gates inject output must be FP32 [S, T]");
    }
    // z's column pitch is rank + streams, so Flash-Next's 324-row z admits 4-element vectors, not
    // 8, whenever the injection rows are present; 8-element vectors need the whole pitch 16-byte
    // aligned. One CTA per column covers rank / V + streams threads. The vector consumer serves the
    // verification widths (at most 16 columns, where multi-row verification lands): launched as a
    // programmatic dependent it overlaps the down projection's tail there, while from 17 columns the
    // scalar serialized form measured faster in both parts — the early-launched gates' CTAs hold SM
    // slots ahead of the down projection's later waves, and the one-CTA-per-column grid loses to the
    // wide scalar grid's latency at prefill widths (infernix_hyper_connection_bench, GB10,
    // interleaved: T = 32 +1.4% programmatic and +0.8% vector-serialized vs HEAD's scalar serialized).
    const bf16* zdata     = static_cast<const bf16*>(z.data);
    bf16* mixdata         = static_cast<bf16*>(mix.data);
    float* injectdata     = inject != nullptr ? static_cast<float*>(inject->data) : nullptr;
    const int inject_rows = inject != nullptr ? streams : 0;
    if (columns <= 16 && rank % 8 == 0 && (inject == nullptr || rows % 8 == 0) && aligned_16(z.data) &&
        aligned_16(mix.data) && rank / 8 + inject_rows <= 1024) {
        const dim3 block((rank / 8 + inject_rows + 31) / 32 * 32);
        if (pdl::launch_consumer({dim3(1, columns), block, 0, stream}, gates_vec_kernel<8>, zdata, rows, rank,
                                 streams, mixdata, injectdata) != cudaSuccess) {
            check_launch("gates");
        }
        return;
    }
    if (columns <= 16 && rank % 4 == 0 && (inject == nullptr || rows % 4 == 0) && aligned_8(z.data) &&
        aligned_8(mix.data) && rank / 4 + inject_rows <= 1024) {
        const dim3 block((rank / 4 + inject_rows + 31) / 32 * 32);
        if (pdl::launch_consumer({dim3(1, columns), block, 0, stream}, gates_vec_kernel<4>, zdata, rows, rank,
                                 streams, mixdata, injectdata) != cudaSuccess) {
            check_launch("gates");
        }
        return;
    }
    if (pdl::launch_with(pdl::Dependency::Serialized,
                         {dim3((rows + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream}, gates_kernel,
                         zdata, rows, rank, streams, columns, mixdata, injectdata) != cudaSuccess) {
        check_launch("gates");
    }
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The composed route's intermediates: Rn, z, m and u in BF16, then the projections' own workspace.
std::size_t composed_bytes(const Weight& down, const Weight& up, LinearPolicy policy, std::int32_t streams,
                           std::int32_t rank, std::int32_t max_tokens) {
    const auto width = static_cast<std::size_t>(up.n);
    const auto T     = static_cast<std::size_t>(max_tokens);
    (void)streams;
    return align256(width * T * 2) + align256(static_cast<std::size_t>(down.n) * T * 2) +
           align256(static_cast<std::size_t>(rank) * T * 2) + align256(width * T * 2) +
           std::max(linear_workspace_capacity_bytes(down.qtype, down.n, down.k, policy, 1, max_tokens),
                    linear_workspace_capacity_bytes(up.qtype, up.n, up.k, policy, 1, max_tokens));
}

} // namespace

namespace detail {

void hc_collapse(const Tensor& u, const Tensor& normalized, std::int32_t streams, Tensor& x, cudaStream_t stream) {
    require(contiguous_2d(u, DType::BF16) && contiguous_2d(normalized, DType::BF16) &&
                contiguous_2d(x, DType::BF16) && streams > 0,
            "collapse requires contiguous BF16 tensors");
    const int width = u.ne[0], columns = u.ne[1], hidden = x.ne[0];
    require(width == streams * hidden && normalized.ne[0] == width && normalized.ne[1] == columns &&
                x.ne[1] == columns && columns > 0,
            "collapse shapes disagree");
    // The same width rule as the gates: programmatic at the verification widths, serialized from 17
    // columns where the up projection's grid is multi-wave and the waiting consumer would hold its
    // later CTAs' SM slots (see hyper_connection_gates).
    const pdl::Dependency dependency =
        columns <= 16 ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized;
    if (hidden % 8 == 0 && aligned_16(u.data) && aligned_16(normalized.data) && aligned_16(x.data)) {
        const int vec_elems = hidden / 8;
        const int threads   = (vec_elems <= 320 && vec_elems % 32 == 0) ? vec_elems : kThreads;
        if (pdl::launch_with(dependency, {dim3((vec_elems + threads - 1) / threads, columns), dim3(threads), 0, stream},
                             collapse_vec8_kernel, static_cast<const bf16*>(u.data),
                             static_cast<const bf16*>(normalized.data), hidden, streams, columns,
                             static_cast<bf16*>(x.data)) != cudaSuccess) {
            check_launch("collapse");
        }
        return;
    }
    if (pdl::launch_with(dependency, {dim3((hidden + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream},
                         collapse_kernel, static_cast<const bf16*>(u.data),
                         static_cast<const bf16*>(normalized.data), hidden, streams, columns,
                         static_cast<bf16*>(x.data)) != cudaSuccess) {
        check_launch("collapse");
    }
}

} // namespace detail

void hyper_connection_mix(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                          const Weight& up, LinearPolicy policy, std::int32_t streams,
                          std::int32_t rank, float eps, Tensor& x, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(x, DType::BF16) &&
                (contiguous_2d(norm_weight, DType::BF16) || contiguous_2d(norm_weight, DType::FP32)) && streams > 0 &&
                rank > 0,
            "mix requires contiguous BF16 tensors and a BF16 or FP32 norm weight");
    const std::int32_t T = residual.ne[1], hidden = x.ne[0];
    require(T > 0 && x.ne[1] == T && residual.ne[0] == streams * hidden && norm_weight.ne[0] == residual.ne[0] &&
                down.k == residual.ne[0] && up.n == residual.ne[0] && up.k == rank &&
                down.n == rank + (inject != nullptr ? streams : 0),
            "mix shapes disagree");
    if (inject != nullptr) {
        require(contiguous_2d(*inject, DType::FP32) && inject->ne[0] == streams && inject->ne[1] == T,
                "mix inject output must be FP32 [S, T]");
    }
    require(eps > 0, "mix epsilon must be positive");
    auto scope = workspace.scope();
    if (detail::hc_mix_fused_supported(down, up, hidden, streams, rank, T)) {
        const DeviceSpan partials = workspace.alloc_bytes(detail::hc_mix_fused_workspace_bytes(down.n, streams, T));
        detail::hc_mix_fused(residual, norm_weight, down, up, streams, rank, eps, x, inject, partials.data, stream);
        return;
    }
    Tensor normalized = workspace.alloc(DType::BF16, {residual.ne[0], T});
    hyper_connection_norm(residual, norm_weight, streams, eps, normalized, stream);
    Tensor z = workspace.alloc(DType::BF16, {down.n, T});
    linear(normalized, down, z, policy, workspace, stream);
    Tensor m = workspace.alloc(DType::BF16, {rank, T});
    hyper_connection_gates(z, rank, streams, m, inject, stream);
    if (detail::hc_up_collapse_supported(up, hidden, streams, T)) {
        // Q8 W_up beyond the fused decode route: u stays in the MMA tile (bit-identical, §19.3.23).
        detail::hc_up_collapse(m, up, normalized, streams, x, stream);
        return;
    }
    Tensor u = workspace.alloc(DType::BF16, {up.n, T});
    linear(m, up, u, policy, workspace, stream);
    detail::hc_collapse(u, normalized, streams, x, stream);
}

void hyper_connection_inject_mix(const Tensor& y, const Tensor& pending_inject, Tensor& residual,
                                 const Tensor& norm_weight, const Weight& down, const Weight& up,
                                 LinearPolicy policy, std::int32_t streams, std::int32_t rank, float eps,
                                 Tensor& x, Tensor* inject, WorkspaceArena& workspace, cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(x, DType::BF16) &&
                (contiguous_2d(norm_weight, DType::BF16) || contiguous_2d(norm_weight, DType::FP32)) && streams > 0 &&
                rank > 0,
            "inject mix requires contiguous BF16 tensors and a BF16 or FP32 norm weight");
    const std::int32_t T = residual.ne[1], hidden = x.ne[0];
    require(T > 0 && x.ne[1] == T && residual.ne[0] == streams * hidden && norm_weight.ne[0] == residual.ne[0] &&
                down.k == residual.ne[0] && up.n == residual.ne[0] && up.k == rank &&
                down.n == rank + (inject != nullptr ? streams : 0),
            "inject mix shapes disagree");
    if (inject != nullptr) {
        require(contiguous_2d(*inject, DType::FP32) && inject->ne[0] == streams && inject->ne[1] == T,
                "inject mix output must be FP32 [S, T]");
    }
    require(eps > 0, "inject mix epsilon must be positive");
    require(inject == nullptr || inject->data != pending_inject.data,
            "inject mix: the mix's injects would overwrite the pending ones");
    auto scope = workspace.scope();
    if (detail::hc_mix_fused_supported(down, up, hidden, streams, rank, T)) {
        const DeviceSpan partials = workspace.alloc_bytes(detail::hc_mix_fused_workspace_bytes(down.n, streams, T));
        detail::hc_inject_mix_fused(y, pending_inject, residual, norm_weight, down, up, streams, rank, eps, x, inject,
                                    partials.data, stream);
        return;
    }
    Tensor normalized = workspace.alloc(DType::BF16, {residual.ne[0], T});
    hyper_connection_inject_norm(y, pending_inject, residual, norm_weight, streams, eps, normalized, stream);
    Tensor z = workspace.alloc(DType::BF16, {down.n, T});
    linear(normalized, down, z, policy, workspace, stream);
    Tensor m = workspace.alloc(DType::BF16, {rank, T});
    hyper_connection_gates(z, rank, streams, m, inject, stream);
    if (detail::hc_up_collapse_supported(up, hidden, streams, T)) {
        detail::hc_up_collapse(m, up, normalized, streams, x, stream);
        return;
    }
    Tensor u = workspace.alloc(DType::BF16, {up.n, T});
    linear(m, up, u, policy, workspace, stream);
    detail::hc_collapse(u, normalized, streams, x, stream);
}

std::size_t hyper_connection_mix_workspace_capacity_bytes(const Weight& down, const Weight& up, LinearPolicy policy,
                                                          std::int32_t streams, std::int32_t rank,
                                                          std::int32_t max_tokens) {
    require(max_tokens > 0 && streams > 0 && rank > 0, "mix workspace needs positive T, streams and rank");
    std::size_t bytes = composed_bytes(down, up, policy, streams, rank, max_tokens);
    for (std::int32_t T = 1; T <= std::min(max_tokens, detail::kHcMixFusedMaxColumns); ++T) {
        bytes = std::max(bytes, detail::hc_mix_fused_workspace_bytes(down.n, streams, T) + 256);
    }
    return bytes;
}

void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream) {
    require(contiguous_2d(y, DType::BF16) && contiguous_2d(inject, DType::FP32) &&
                contiguous_2d(residual, DType::BF16),
            "inject requires contiguous tensors");
    const int hidden = y.ne[0], columns = y.ne[1], streams = inject.ne[0];
    require(inject.ne[1] == columns && residual.ne[0] == streams * hidden && residual.ne[1] == columns &&
                columns > 0,
            "inject shapes disagree");
    if (hidden % 8 == 0 && aligned_16(y.data) && aligned_16(residual.data)) {
        const int vec_elems = hidden / 8;
        const int threads   = (vec_elems <= 320 && vec_elems % 32 == 0) ? vec_elems : kThreads;
        if (pdl::launch_consumer({dim3((vec_elems + threads - 1) / threads, columns), dim3(threads), 0, stream},
                                 inject_vec8_kernel, static_cast<const bf16*>(y.data),
                                 static_cast<const float*>(inject.data), hidden, streams, columns,
                                 static_cast<bf16*>(residual.data)) != cudaSuccess) {
            check_launch("inject");
        }
        return;
    }
    if (pdl::launch_consumer({dim3((hidden + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream},
                             inject_kernel, static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data),
                             hidden, streams, columns, static_cast<bf16*>(residual.data)) != cudaSuccess) {
        check_launch("inject");
    }
}

void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream) {
    require(contiguous_2d(x, DType::BF16) && contiguous_2d(residual, DType::BF16) && streams > 0,
            "expand requires contiguous BF16 tensors");
    const int hidden = x.ne[0], columns = x.ne[1];
    require(residual.ne[0] == streams * hidden && residual.ne[1] == columns && columns > 0,
            "expand shapes disagree");
    if (hidden % 8 == 0 && aligned_16(x.data) && aligned_16(residual.data)) {
        const int vec_elems = hidden / 8;
        const int threads   = (vec_elems <= 320 && vec_elems % 32 == 0) ? vec_elems : kThreads;
        expand_vec8_kernel<<<dim3((vec_elems + threads - 1) / threads, columns), threads, 0, stream>>>(
            static_cast<const bf16*>(x.data), hidden, streams, columns, static_cast<bf16*>(residual.data));
        check_launch("expand");
        return;
    }
    expand_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), hidden, streams, columns, static_cast<bf16*>(residual.data));
    check_launch("expand");
}

void hyper_connection_stream_mean(const Tensor& residual, std::int32_t streams, Tensor& out,
                                  cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && streams > 0 && out.data != nullptr &&
                out.dtype == DType::BF16 && out.ne[2] == 1 && out.ne[3] == 1 && out.nb[0] == 2 &&
                out.nb[1] % 2 == 0 && out.nb[1] >= 2LL * out.ne[0],
            "stream mean requires a contiguous BF16 residual and BF16 output columns");
    const int hidden        = out.ne[0], columns = out.ne[1];
    const std::int64_t pitch = out.nb[1] / 2;
    require(residual.ne[0] == streams * hidden && residual.ne[1] == columns && columns > 0,
            "stream mean shapes disagree");
    if (hidden % 8 == 0 && pitch % 8 == 0 && aligned_16(residual.data) && aligned_16(out.data)) {
        const int vec_elems = hidden / 8;
        const int threads   = (vec_elems <= 320 && vec_elems % 32 == 0) ? vec_elems : kThreads;
        stream_mean_vec8_kernel<<<dim3((vec_elems + threads - 1) / threads, columns), threads, 0, stream>>>(
            static_cast<const bf16*>(residual.data), hidden, streams, columns, pitch,
            static_cast<bf16*>(out.data));
        check_launch("stream mean");
        return;
    }
    stream_mean_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), hidden, streams, columns, pitch,
        static_cast<bf16*>(out.data));
    check_launch("stream mean");
}

} // namespace infernix::ops
