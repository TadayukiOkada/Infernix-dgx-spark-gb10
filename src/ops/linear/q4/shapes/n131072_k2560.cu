#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
// GB10 (sm_121a, 48 SMs): Qwen3.8-Flash-Next's proposal head ([131072, 2560], --lm-head-draft).
// Uses the GB10-selected dynamic-K families of the proposal heads (n131072_k3072.cu / n131072_k5120.cu).
Q4Launch select_q4_n131072_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 24) return launch_q4_a16_sliced_r32_t24_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 72) return launch_q4_a16_mma_r64_t72;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
