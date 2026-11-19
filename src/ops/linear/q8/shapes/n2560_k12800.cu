// GB10 (sm_121a) tuned dispatch, 2026-10-30: bench/ops/q8_route_sweep.cu over T = 1..64 (sweep
// record /tmp q8sweep_2560x12800, shipped 2026-10-30) against the pre-change runtime-shape
// fallback (predicated SIMT for T <= 8, mma_r32_t64 tiles beyond). The DFlash2 companion's
// feature_projection (Q8_G32_FP16 [2560, 12800], 33.2 MB, K = 12800 reduction) pays one DRAM pass
// over the weight at every width (156..168 us reached); the fallback's T = 9..64 tiles idled half
// the SMs (264..403 us, 20..48% slower than the measured argmin) and its T = 1..5 SIMT calls paid
// 8.2..19.6% more.
//
// The exact table's T = 1..32 band is the shape's fixed sliced-K family (q8_invariant_families):
// the sweep's per-width bit_exact gates cannot certify a cross-schedule column contract (the
// SIMT variants simt_r1_t4_w8 / r8_t5 / r8_t8 / r8_t8 reproduce the fallback's bits at their own
// widths yet differ from each other by a few tail outputs across widths — the shipped
// column-invariance test at 8 columns caught exactly that), and one family across the contracted
// widths is the pattern every other Flash-Next Q8 shape uses.

#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {

// The exact table: the fixed 8-K-warp family (capacity 8/16/32 rungs) through T = 32, then the
// sweep's 64-row x 16-warp K128 MMA tiles (211..214 us over T = 33..48, where the fallback paid
// 265..270) and the T = 48-row tile at T = 49..64 (194..199 us). The fallback's T = 12 cliff of
// 403 us is inside the family band. The T > 64 tail mirrors the [2560, 6144] shape's MMA ladder:
// unmeasured here, it replaces the narrower generic tiles that lost 20%+ at every measured width.
Q8Launch select_q8_n2560_k12800(std::int32_t tokens) {
    if (tokens <= 16) return select_q8_n2560_k12800_invariant(tokens);
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w8_s1_m2;
    if (tokens <= 48) return launch_q8_a16_sliced_r32_t48_w4_s1;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t32_k128_a1;
    if (tokens <= 128) return launch_q8_a16_mma_r64x16_t128_k128_a2;
    if (tokens <= 192) return launch_q8_a16_mma_r64x32_t96_k128_a1;
    if (tokens <= 448) return launch_q8_a16_mma_r64_t128;
    // T1 2026-10-31: the 128x128 Grouped-8 tile now runs to the top (production: 72.7 vs
    // 59.6 TFLOP/s at T = 1024, 75.8..76.2 vs 60.1..60.5 at T = 4096 over the r64x32 a2
    // tile). The 257..448 band moves to r64_t128: the 2026-10-31 sweep measured it ~9%
    // under r128x64 at T = 257/385 (20 M-tiles x 2 N-tiles under-fills 48 SMs x 2 CTAs),
    // with the sweep's crossover interpolated at ~450.
    return launch_q8_a16_mma_r128x64_t128_k64_a1;
}

// The q8-sliced-k fast path (LinearRoutes::Q8SlicedK): the T-dependent winners. The swept T = 1
// SIMT (155.5 us) beats the family rung's 162..164; the capacity-16 8-K-warp tile holds
// 162..166 us over T = 2..16; the capacity-32 4-K-warp tile pays 182..184 us over T = 17..32;
// beyond T = 32 the exact MMA bands already sit at the sweep's argmin, so the fast table
// delegates to them.
Q8Launch select_q8_n2560_k12800_fast(std::int32_t tokens) {
    if (tokens <= 1) return launch_q8_a16_simt_r1_t4_w8;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w4_s2;
    return select_q8_n2560_k12800(tokens);
}

} // namespace infernix::ops::detail
