#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace infernix::ops::detail {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ __half2 decode_nvfp4_e2m1x2_f16(std::uint8_t storage) {
    union {
        unsigned bits;
        __half2 h2;
    } out;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.f16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(out.bits)
        : "r"(static_cast<unsigned>(storage)));
    return out.h2;
}

__device__ __forceinline__ __half2 decode_nvfp4_e4m3_f16x2(std::uint8_t storage) {
    union {
        unsigned bits;
        __half2 h2;
    } out;
    const auto scale_pair =
        static_cast<std::uint16_t>(storage | (static_cast<unsigned>(storage) << 8));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out.bits) : "h"(scale_pair));
    return out.h2;
}

__device__ __forceinline__ std::uint32_t nvfp4_e4m3_f16x2_bits(std::uint8_t storage) {
    union {
        __half2 h2;
        std::uint32_t bits;
    } out{decode_nvfp4_e4m3_f16x2(storage)};
    return out.bits;
}

// One cvt.rn.f16x2.e4m3x2 decodes two E4M3 scale bytes, then each half is broadcast back into a
// full f16x2 multiplier. A widened MMA A-fragment register shares its scale with a sibling
// register, so a caller that widens a weight-row pair per k-step spends one conversion
// instruction per two registers instead of one per register.
__device__ __forceinline__ void nvfp4_unpack_two_scales_f16x2(std::uint8_t s0, std::uint8_t s1,
                                                               std::uint32_t& s0_f16x2,
                                                               std::uint32_t& s1_f16x2) {
    const std::uint16_t scale_pair =
        static_cast<std::uint16_t>(s0) | (static_cast<std::uint16_t>(s1) << 8);
    std::uint32_t both;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(both) : "h"(scale_pair));
    s0_f16x2 = __byte_perm(both, both, 0x1010);
    s1_f16x2 = __byte_perm(both, both, 0x3232);
}

// The widening of nvfp4_scaled_pair_bf16 with the E4M3 -> f16x2 conversion hoisted to the caller:
// `scale_f16x2` comes from nvfp4_e4m3_f16x2_bits or nvfp4_unpack_two_scales_f16x2. Bit-identical to
// the raw-scale form because E2M1 times a finite E4M3 scale is exact in FP16, so only the final
// BF16 pack rounds.
__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16_f16scale(std::uint8_t code,
                                                                    std::uint32_t scale_f16x2) {
    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;
    const float2 product =
        __half22float2(__hmul2(decode_nvfp4_e2m1x2_f16(code),
                               *reinterpret_cast<const __half2*>(&scale_f16x2)));
    result.pair = __floats2bfloat162_rn(product.x, product.y);
    return result.bits;
}

// The two-pair form of nvfp4_scaled_pair_bf16_f16scale: one multiplier for both output registers.
__device__ __forceinline__ void nvfp4_scaled_two_pairs_bf16_f16scale(std::uint8_t code0,
                                                                     std::uint8_t code1,
                                                                     std::uint32_t scale_f16x2,
                                                                     unsigned& out0, unsigned& out1) {
    unsigned h0, h1;
    const unsigned packed_pair = static_cast<unsigned>(code0) | (static_cast<unsigned>(code1) << 8);
    asm("{\n"
        ".reg .b8 b0, b1;\n"
        "mov.b32 {b0, b1, _, _}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "}\n"
        : "=r"(h0), "=r"(h1)
        : "r"(packed_pair));
    const auto mult = *reinterpret_cast<const __half2*>(&scale_f16x2);
    const float2 prod0 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&h0), mult));
    const float2 prod1 = __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&h1), mult));
    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } r0, r1;
    r0.pair = __floats2bfloat162_rn(prod0.x, prod0.y);
    r1.pair = __floats2bfloat162_rn(prod1.x, prod1.y);
    out0    = r0.bits;
    out1    = r1.bits;
}

__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16(std::uint8_t code, std::uint8_t scale) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    // PTX 9.2 exposes native pair widening. E2M1 times a finite E4M3 scale needs
    // at most six significand bits, so the BF16 multiplication is exact.
    unsigned values, multiplier, result;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(values)
        : "r"(static_cast<unsigned>(code)));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
    return result;
#else
    // E2M1 times a finite E4M3 scale has at most 6 significand bits and exponent in [-10, +11],
    // so the FP16 pair multiplication is exact before rounding to BF16.
    return nvfp4_scaled_pair_bf16_f16scale(code, nvfp4_e4m3_f16x2_bits(scale));
#endif
}

__device__ __forceinline__ void nvfp4_scaled_two_pairs_bf16(std::uint8_t code0, std::uint8_t code1,
                                                            std::uint8_t scale, unsigned& out0,
                                                            unsigned& out1) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    unsigned values0, values1, multiplier;
    const unsigned packed_pair = static_cast<unsigned>(code0) | (static_cast<unsigned>(code1) << 8);
    asm("{\n"
        ".reg .b8 b0, b1;\n"
        "mov.b32 {b0, b1, _, _}, %2;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        "}\n"
        : "=r"(values0), "=r"(values1)
        : "r"(packed_pair));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out0) : "r"(values0), "r"(multiplier));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out1) : "r"(values1), "r"(multiplier));
#else
    nvfp4_scaled_two_pairs_bf16_f16scale(code0, code1, nvfp4_e4m3_f16x2_bits(scale), out0, out1);
#endif
}

__device__ __forceinline__ uint4 nvfp4_scaled_oct_bf16(std::uint32_t codes4, std::uint8_t scale) {
    const __half2 mult = decode_nvfp4_e4m3_f16x2(scale);
    unsigned h0, h1, h2, h3;
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
        : "r"(codes4));
    const auto pack_pair = [&](unsigned raw) {
        union {
            unsigned bits;
            __half2 h2;
        } in{raw};
        const float2 prod = __half22float2(__hmul2(in.h2, mult));
        union {
            __nv_bfloat162 pair;
            unsigned bits;
        } res;
        res.pair = __floats2bfloat162_rn(prod.x, prod.y);
        return res.bits;
    };
    return make_uint4(pack_pair(h0), pack_pair(h1), pack_pair(h2), pack_pair(h3));
}

__device__ __forceinline__ int nvfp4_a16_shared_col_64(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ std::int64_t nvfp4_scale_byte_offset(int row, int group, int k) {
    return static_cast<std::int64_t>((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (group & 3);
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}

// Reciprocal replaces the sixteen per-value divisions with one reciprocal plus multiplies.
// Measured faster on latency-bound small grids (linear MMA route, T < 1024) and measurably
// slower on the throughput-bound TMA route, so callers select it explicitly; the default keeps
// historical behavior. The two formulations are algebraically identical and differ only in
// floating-point rounding (~1 ulp); the resulting output error stays within the A4 allowance.
template <bool Reciprocal = false>
__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
    if constexpr (Reciprocal) {
        const float gain = input_scale_divisor * __fdiv_rn(1.0F, decoded_scale);
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            values[pair].x = values[pair].x * gain;
            values[pair].y = values[pair].y * gain;
        }
    } else {
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
            values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
        }
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

// The A4 residual profile: one K16 group as a sum of NVFP4 terms. The remainder starts at the
// divisor-scaled input x * input_scale_divisor; each term encodes it like quantize_nvfp4_k16 (an
// E4M3 scale of max/6, E2M1 codes of value/scale), and the next term encodes what that term's
// decoded code * scale left. Term 0 is therefore bit-identical to quantize_nvfp4_k16, and every term
// shares its divisor, so the MMA sums the terms' products into one accumulator under the same alpha.
template <int Terms, bool Reciprocal = false>
__device__ __forceinline__ void
quantize_nvfp4_terms_k16_packed(const uint4 packed0, const uint4 packed1, float input_scale_divisor,
                                Nvfp4QuantizedK16 (&terms)[Terms]) {
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 remainder[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float2 value = bf16x2_bits_to_float2(represented[pair]);
        remainder[pair]    = make_float2(__fmul_rn(value.x, input_scale_divisor),
                                         __fmul_rn(value.y, input_scale_divisor));
    }

#pragma unroll
    for (int term = 0; term < Terms; ++term) {
        float max_abs = 0.0F;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            max_abs = fmaxf(max_abs, fabsf(remainder[pair].x));
            max_abs = fmaxf(max_abs, fabsf(remainder[pair].y));
        }
        Nvfp4QuantizedK16 quantized{};
        const float unencoded =
            Reciprocal ? (max_abs * (1.0F / 6.0F)) : __fdiv_rn(max_abs, 6.0F);
        quantized.scale = __nv_cvt_float_to_fp8(unencoded, __NV_SATFINITE, __NV_E4M3);
        terms[term]     = quantized;
        // A zero scale encodes zero; the remainder carries on to the next term unchanged.
        if (quantized.scale == 0) { continue; }

        const float scale = decode_nvfp4_e4m3(quantized.scale);
        float2 quotients[8];
        if constexpr (Reciprocal) {
            const float inv_scale = 1.0F / scale;
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                quotients[pair] = make_float2(remainder[pair].x * inv_scale,
                                              remainder[pair].y * inv_scale);
            }
        } else {
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                quotients[pair] = make_float2(__fdiv_rn(remainder[pair].x, scale),
                                              __fdiv_rn(remainder[pair].y, scale));
            }
        }
        pack_nvfp4_e2m1x16(quotients, terms[term].codes_lo, terms[term].codes_hi);
        if (term + 1 == Terms) { break; }
        // Code * scale is exact in FP32 (two and four significant bits), so each remainder rounds
        // once, in the subtraction.
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const std::uint32_t word = pair < 4 ? terms[term].codes_lo : terms[term].codes_hi;
            const float2 code =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> (8 * (pair & 3))));
            remainder[pair].x = __fsub_rn(remainder[pair].x, __fmul_rn(code.x, scale));
            remainder[pair].y = __fsub_rn(remainder[pair].y, __fmul_rn(code.y, scale));
        }
    }
}

template <int Terms, bool Reciprocal = false>
__device__ __forceinline__ void quantize_nvfp4_terms_k16(const __nv_bfloat16* source,
                                                         float input_scale_divisor,
                                                         Nvfp4QuantizedK16 (&terms)[Terms]) {
    const uint4 packed0 = load_vec<uint4>(source);
    const uint4 packed1 = load_vec<uint4>(source + 8);
    quantize_nvfp4_terms_k16_packed<Terms, Reciprocal>(packed0, packed1, input_scale_divisor,
                                                       terms);
}

} // namespace infernix::ops::detail
