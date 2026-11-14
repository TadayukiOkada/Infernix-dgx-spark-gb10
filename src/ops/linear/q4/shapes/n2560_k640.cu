#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n2560_k640(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 256) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
}
} // namespace infernix::ops::detail
