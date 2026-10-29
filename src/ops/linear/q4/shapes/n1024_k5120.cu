#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n1024_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on GB10.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 2) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 48) return launch_q4_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 128) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2;
    if (tokens <= 768) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
