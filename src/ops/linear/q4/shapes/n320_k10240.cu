#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n320_k10240(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 896) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 1152) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 1440) return launch_q4_a16_mma_r64_t80;
    return launch_q4_a16_mma_r64_t96;
}
} // namespace infernix::ops::detail
