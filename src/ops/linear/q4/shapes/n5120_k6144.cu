#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n5120_k6144(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on GB10.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_k6144;
    if (tokens <= 2) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t64_k128_s2_a1;
}
} // namespace infernix::ops::detail
