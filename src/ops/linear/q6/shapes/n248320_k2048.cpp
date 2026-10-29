#include "ops/linear/q6/q6_shapes.h"

namespace infernix::ops::detail {

Q6Launch select_q6_n248320_k2048(std::int32_t tokens) {
    // Capacity routes come from complete-Op cold Graph comparisons on GB10.
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 12) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 24) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 56) return launch_q6_a16_mma_r32_t128_k128_s2_a1;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r32_t128_k128_s2_a1;
    return launch_q6_a16_mma_r64_t128_k128_s1_a1;
}

} // namespace infernix::ops::detail
