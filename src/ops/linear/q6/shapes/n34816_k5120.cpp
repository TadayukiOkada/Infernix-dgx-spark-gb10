#include "ops/linear/q6/q6_shapes.h"

namespace infernix::ops::detail {

// GB10 (sm_121a, 48 SMs): The fused gate_up of a Q6 Qwen3.8-27B MLP (NInfer PR #284). It shares
// K=5120 with the output head (n248320_k5120.cpp), so it takes that shape's GB10 capacity routes.
Q6Launch select_q6_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 2) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 3) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 16) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 24) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 40) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 56) return launch_q6_a16_mma_r32_t128_k128_s2_a1;
    if (tokens <= 64) return launch_q6_a16_mma_r32_t64_k128_s2_a1;
    if (tokens <= 112) return launch_q6_a16_mma_r32_t128_k128_s2_a1;
    if (tokens <= 256) return launch_q6_a16_mma_r64_t128_k128_s1_a1;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
