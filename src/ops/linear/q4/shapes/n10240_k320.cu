#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n10240_k320(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 48) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t64_k128_s2_a1;
    return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
}
} // namespace infernix::ops::detail
