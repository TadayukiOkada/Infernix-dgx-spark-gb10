#include "ops/linear/q5/q5_shapes.h"

namespace infernix::ops::detail {
Q5Launch select_q5_n5120_k6144(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on GB10.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k6144;
    if (tokens <= 2) return launch_q5_a16_direct_r2_t4_w4_g4_b4;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 12) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q5_a16_mma_r64_t64_k128_s1_a1;
    if (tokens <= 80) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 96) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    if (tokens <= 128) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 160) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 192) return launch_q5_a16_mma_r64_t64_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
