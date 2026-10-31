// [10240, 5120] Q8G32 prefill shape (GB10 tuning, 2026-10-31).
//
// This shape previously fell through to the runtime-shape generic table (SIMT pair, then
// mma_r32_t64/t96/t128 tiles), which streamed the 52 MB weight per CTA-row without 128-wide
// L2 reuse: measured 34.6..36.9 TFLOP/s at T = 4096 against the 80+ TFLOP/s the 128-row x
// 128-token tile class sustains at this K on the [6144, 5120] and [17408, 5120] tables.
// The ladder below mirrors the [5120, 4608] / [6144, 5120] GB10 tables: the sliced-K pair to
// T = 32 (K = 5120 = 20 x 256 = 10 x 512 divides every rung's K block), K128 MMA tiles to
// T = 128, and from T = 256 the 128x128 Grouped-8 tile.

#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {

Q8Launch select_q8_n10240_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 48) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 64) return launch_q8_a16_mma_r64x32_t64_k128_a1;
    if (tokens <= 96) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 255) return launch_q8_a16_mma_r64_t128;
    // T1 2026-10-31: the 128x128 Grouped-8 tile fills the SMs from two token-tiles at this N
    // (80 M-tiles x 2) and measured 49.1 vs 35.7 TFLOP/s over the old generic r32_t128
    // fallback at T = 257, 82.6 at T = 4096.
    return launch_q8_a16_mma_r128x64_t128_k64_a1;
}

} // namespace infernix::ops::detail
