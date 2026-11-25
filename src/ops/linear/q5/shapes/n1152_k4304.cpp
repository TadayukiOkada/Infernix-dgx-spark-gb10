#include "ops/linear/q5/q5_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {
Q5Launch select_q5_n1152_k4304(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on GB10.
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 160) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 256) return launch_q5_a16_mma_r64_t64_k128_s1_a1;
    if (tokens <= 384) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 512) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 1024) return launch_q5_a16_mma_r64_t80_k128_s1_a1;
    if (tokens <= 1536) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
