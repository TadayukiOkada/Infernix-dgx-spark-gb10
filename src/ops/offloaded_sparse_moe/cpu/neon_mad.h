#pragma once

// AArch64 building blocks of the exact CPU expert routes (ggml_expert.cpp, fp8_expert.cpp,
// w4a4_expert.cpp, w4a16_expert.cpp): per-function ISA targets and the unsigned-weight x
// signed-activation multiply-add of each ISA level. Both levels are exact in int32 lanes, so
// every variant returns the same integer as the scalar path and as the other level.
//
// AArch64 has no CPUID: FEAT_I8MM is reported by the kernel in AT_HWCAP2 (HWCAP2_I8MM), which
// neon_features() in w4a4_expert.cpp reads; that replaces the x86 CPUID/xgetbv probing.
//
// The unbias-and-multiply-add scheme of the A4 routes needs unsigned weights times signed
// activations, exactly as x86's vpdpbusd provides it. Baseline AArch64 gets that from
// vmull/smlal pairs; with FEAT_I8MM, usdot does four products per lane in one instruction.

#include <cstdint>

#include <arm_neon.h>

#define INFERNIX_CPU_NEON 1
#define INFERNIX_TARGET(x) __attribute__((target(x)))

// FEAT_I8MM lives in arch=armv8.6-a: no earlier -a level and no bare "+i8mm" or "+dotprod" enables
// the usdot intrinsic. The target has to be on the helper AND on every kernel that calls it, because
// GCC refuses to inline an i8mm callee into a baseline caller.
#define INFERNIX_I8MM_TARGET INFERNIX_TARGET("arch=armv8.6-a+i8mm")

#if defined(INFERNIX_CPU_NEON)

namespace infernix::ops::offloaded_moe::neon {

// Baseline AArch64 (FEAT_ASIMD only): unsigned weight x signed activation widened to int16 and
// accumulated as smlal pairs. 16 elements per call into one int32x4 accumulator.
struct AsimdMad {
    static inline int32x4_t mad(int32x4_t acc, uint8x16_t w, int8x16_t x) {
        const int16x8_t wl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(w)));
        const int16x8_t wh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(w)));
        const int16x8_t xl = vmovl_s8(vget_low_s8(x));
        const int16x8_t xh = vmovl_s8(vget_high_s8(x));
        acc                = vmlal_s16(acc, vget_low_s16(wl), vget_low_s16(xl));
        acc                = vmlal_high_s16(acc, wl, xl);
        acc                = vmlal_s16(acc, vget_low_s16(wh), vget_low_s16(xh));
        return vmlal_high_s16(acc, wh, xh);
    }
};

// FEAT_I8MM (HWCAP2_I8MM): unsigned x signed dot product, four products per lane.
struct I8mmMad {
    INFERNIX_I8MM_TARGET static inline int32x4_t mad(int32x4_t acc, uint8x16_t w, int8x16_t x) {
        return vusdotq_s32(acc, w, x);
    }
};

// All four lanes of `v` summed as one integer (the caller keeps sums below 2^31).
inline std::int32_t hsum(int32x4_t v) { return vaddvq_s32(v); }

} // namespace infernix::ops::offloaded_moe::neon

#endif // INFERNIX_CPU_NEON
