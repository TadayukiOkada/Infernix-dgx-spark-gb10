#include "ops/linear/q5/q5_shapes.h"

namespace infernix::ops::detail {
Q5Launch select_q5_n6144_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on GB10.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w4_k5120;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w4_k5120;
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w4_g4_b4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 48) return launch_q5_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 64) return launch_q5_a16_mma_r64_t64_k128_s1_a1;
    if (tokens <= 80) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 96) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    if (tokens <= 112) return launch_q5_a16_mma_r32_t128_k128_s2_a1;
    if (tokens <= 128) return launch_q5_a16_mma_r64_t128_k128_s1_a1;
    if (tokens <= 160) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 192) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
