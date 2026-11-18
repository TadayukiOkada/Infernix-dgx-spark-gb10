#pragma once

// infernix::ops — silu_mul kernel: out = silu(gate) * up, elementwise.
// silu(x) = x / (1 + e^-x), computed exactly in fp32 (NOT a polynomial fit).
// Vectorized over bf16 pairs; included only by its launcher. See
// docs/op-development.md §6 (no math approximation).

#include "ops/common/math.cuh"
#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kSiluAndMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 silu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = silu(__low2float(g)) * __low2float(u);
    const float r1 = silu(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__launch_bounds__(256) __global__
void silu_and_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                __nv_bfloat16* out, std::int64_t n) {
    pdl::enter();
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16(silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

__launch_bounds__(256) __global__
void silu_and_mul_strided_input_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    std::int32_t ne0, std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1,
    std::int64_t gnb2, std::int64_t gnb3, std::int64_t unb0, std::int64_t unb1, std::int64_t unb2,
    std::int64_t unb3) {
    pdl::enter();
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i]        = __float2bfloat16(silu(__bfloat162float(gv)) * __bfloat162float(uv));
    }
}

__launch_bounds__(256) __global__
    void silu_and_mul_dim0_split_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                        __nv_bfloat16* out, std::int32_t rows,
                                        std::int64_t gate_col_stride, std::int64_t up_col_stride) {
    pdl::enter();
    const int col            = blockIdx.y;
    const std::int64_t pair0 = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) *
                               kSiluAndMulPairsPerThread;
    const std::int64_t row_pairs = rows / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate + col * gate_col_stride);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up + col * up_col_stride);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out + col * static_cast<std::int64_t>(rows));

    for (int p = 0; p < kSiluAndMulPairsPerThread; ++p) {
        const std::int64_t j = pair0 + p;
        if (j < row_pairs) { out2[j] = silu_mul_pair(gate2[j], up2[j]); }
    }
}

__device__ __forceinline__ float swiglu_fast_value(float g, float u) {
    return (g / (1.0f + expf(-g))) * u;
}

__device__ __forceinline__ unsigned int swiglu_fast_word(unsigned int g, unsigned int u) {
    const float g0 = __uint_as_float(g << 16);
    const float g1 = __uint_as_float(g & 0xffff0000u);
    const float u0 = __uint_as_float(u << 16);
    const float u1 = __uint_as_float(u & 0xffff0000u);
    const float r0 = swiglu_fast_value(g0, u0);
    const float r1 = swiglu_fast_value(g1, u1);
    unsigned int packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(r0), "f"(r1));
    return packed;
}

__device__ __forceinline__ uint4 swiglu_fast_vector(uint4 g, uint4 u) {
    uint4 r;
    r.x = swiglu_fast_word(g.x, u.x);
    r.y = swiglu_fast_word(g.y, u.y);
    r.z = swiglu_fast_word(g.z, u.z);
    r.w = swiglu_fast_word(g.w, u.w);
    return r;
}

__device__ __forceinline__ uint4 swiglu_fast_vector_staged(uint4 g, uint4 u) {
    const unsigned int gw[4] = {g.x, g.y, g.z, g.w};
    const unsigned int uw[4] = {u.x, u.y, u.z, u.w};
    float gv[8], uv[8], den[8], result[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        gv[2 * j]     = __uint_as_float(gw[j] << 16);
        gv[2 * j + 1] = __uint_as_float(gw[j] & 0xffff0000u);
        uv[2 * j]     = __uint_as_float(uw[j] << 16);
        uv[2 * j + 1] = __uint_as_float(uw[j] & 0xffff0000u);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { den[j] = 1.0f + expf(-gv[j]); }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        result[j] = (gv[j] / den[j]) * uv[j];
    }
    uint4 r;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r.x) : "f"(result[0]), "f"(result[1]));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r.y) : "f"(result[2]), "f"(result[3]));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r.z) : "f"(result[4]), "f"(result[5]));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r.w) : "f"(result[6]), "f"(result[7]));
    return r;
}

template <unsigned int Block>
__launch_bounds__(Block) __global__
void silu_and_mul_exact_kernel(const __nv_bfloat16* __restrict__ gate,
                               const __nv_bfloat16* __restrict__ up,
                               __nv_bfloat16* __restrict__ out) {
    pdl::enter();
    const unsigned int i = blockIdx.x * Block + threadIdx.x;
    const uint4 g        = reinterpret_cast<const uint4*>(gate)[i];
    const uint4 u        = reinterpret_cast<const uint4*>(up)[i];
    reinterpret_cast<uint4*>(out)[i] = swiglu_fast_vector(g, u);
}

__launch_bounds__(128) __global__
void silu_and_mul_prefill_staged_kernel(const __nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up,
                                        __nv_bfloat16* __restrict__ out) {
    pdl::enter();
    const unsigned int i = blockIdx.x * 128u + threadIdx.x;
    const uint4 g        = reinterpret_cast<const uint4*>(gate)[i];
    const uint4 u        = reinterpret_cast<const uint4*>(up)[i];
    reinterpret_cast<uint4*>(out)[i] = swiglu_fast_vector_staged(g, u);
}

__launch_bounds__(256) __global__
void silu_and_mul_uint4_kernel(const uint4* __restrict__ gate,
                               const uint4* __restrict__ up,
                               uint4* __restrict__ out,
                               std::int64_t num_vectors) {
    pdl::enter();
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = tid; i < num_vectors; i += stride) {
        out[i] = swiglu_fast_vector(gate[i], up[i]);
    }
}

__launch_bounds__(256) __global__
    void silu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                             std::int64_t n) {
    pdl::enter();
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSiluAndMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kSiluAndMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 u0 = up2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1 = gate2[j + 1];
            const __nv_bfloat162 u1 = up2[j + 1];
            const __nv_bfloat162 g2 = gate2[j + 2];
            const __nv_bfloat162 u2 = up2[j + 2];
            const __nv_bfloat162 g3 = gate2[j + 3];
            const __nv_bfloat162 u3 = up2[j + 3];
            out2[j]                 = silu_mul_pair(g0, u0);
            out2[j + 1]             = silu_mul_pair(g1, u1);
            out2[j + 2]             = silu_mul_pair(g2, u2);
            out2[j + 3]             = silu_mul_pair(g3, u3);
        } else {
            out2[j] = silu_mul_pair(g0, u0);
            if (j + 1 < n2) { out2[j + 1] = silu_mul_pair(gate2[j + 1], up2[j + 1]); }
            if (j + 2 < n2) { out2[j + 2] = silu_mul_pair(gate2[j + 2], up2[j + 2]); }
        }
    }

    if (tid == 0) {
        if ((n & 1) != 0) {
            const std::int64_t i = n - 1;
            out[i] = __float2bfloat16(silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
        }
    }
}

} // namespace infernix::ops
