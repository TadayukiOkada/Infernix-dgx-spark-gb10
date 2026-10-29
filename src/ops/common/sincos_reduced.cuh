#pragma once

// Shared FP32 RoPE coefficient arithmetic: the FP64 one-turn reduction plus the libdevice
// `sincosf` fast path without its Payne-Hanek slowpath. `sincosf` inlines
// `__internal_trig_reduction_slowpathf` whenever it cannot statically prove |x| < 105615.0f, and
// the slowpath's 7-word staging array forces a 32 B local-memory frame (STACK) into every kernel
// that evaluates a single coefficient. RoPE angles are reduced to |r| <= pi in FP64 before the
// trigonometric evaluation, so the slowpath is unreachable there. `sincos_reduced` reproduces the
// libdevice fast path (three-term Cody-Waite reduction and minimax polynomials, CUDA 13
// `__nv_sincosf`) bit for bit for every |x| < 105615.0f, so reduced routes keep their bits and
// unreduced routes gain the FP64 reduction the YaRN routes already use.

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// One-turn FP64 range reduction shared by every RoPE route: |result| <= pi.
__device__ __forceinline__ float rope_reduced_angle(double angle) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    return static_cast<float>(angle - nearbyint(angle * kInvTwoPi) * kTwoPi);
}

// Bit-exact libdevice `sincosf` fast path for |x| < 105615.0f (verified exhaustively against
// `sincosf` over all finite bit patterns in [-105615, 105615]).
__device__ __forceinline__ void sincos_reduced(float x, float* sine, float* cosine) {
    int quadrant;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(quadrant)
        : "f"(__fmul_rn(x, __uint_as_float(0x3F22F983u)))); // x * 2/pi
    float turns;
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(turns) : "r"(quadrant));
    const float reduced =
        __fmaf_rn(turns, __uint_as_float(0xA7C234C5u),
                  __fmaf_rn(turns, __uint_as_float(0xB3A22168u),
                            __fmaf_rn(turns, __uint_as_float(0xBFC90FDAu), x)));
    const float z = __fmul_rn(reduced, reduced);
    // cos polynomial in z; sin polynomial applied to z*reduced.
    const float cos_poly = __fmaf_rn(
        __fmaf_rn(
            __fmaf_rn(__fmaf_rn(z, __uint_as_float(0x37CBAC00u), __uint_as_float(0xBAB607EDu)), z,
                      __uint_as_float(0x3D2AAABBu)),
            z, __uint_as_float(0xBEFFFFFFu)),
        z, __uint_as_float(0x3F800000u));
    const float zr = __fmaf_rn(z, reduced, __uint_as_float(0x00000000u));
    const float sin_poly = __fmaf_rn(
        __fmaf_rn(__fmaf_rn(z, __uint_as_float(0xB94D4153u), __uint_as_float(0x3C0885E4u)), z,
                  __uint_as_float(0xBE2AAAA8u)),
        zr, reduced);
    float sine_value   = (quadrant & 1) != 0 ? cos_poly : sin_poly;
    float cosine_value = (quadrant & 1) != 0 ? sin_poly : cos_poly;
    if ((quadrant & 2) != 0) { sine_value = -sine_value; }
    if (((quadrant + 1) & 2) != 0) { cosine_value = -cosine_value; }
    *sine   = sine_value;
    *cosine = cosine_value;
}

// The coefficient entry shared by every RoPE/QSA site: the FP64 angle of the exact product,
// reduced to one turn, evaluated on the frame-free bit-exact fast path.
__device__ __forceinline__ void sincos_turns(double angle, float* sine, float* cosine) {
    sincos_reduced(rope_reduced_angle(angle), sine, cosine);
}

} // namespace infernix::ops
