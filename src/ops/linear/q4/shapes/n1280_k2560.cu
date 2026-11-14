#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
Q4Launch select_q4_n1280_k2560(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 1) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 256) return launch_q4_a16_sliced_r32_t32_w2_s2;
    return launch_q4_a16_mma_r64_t80;
}
} // namespace infernix::ops::detail
