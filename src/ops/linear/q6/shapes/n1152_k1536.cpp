#include "ops/linear/q6/q6_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {

Q6Launch select_q6_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q6 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 4) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 8) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 12) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 24) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 48) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 112) return launch_q6_a16_mma_r32_t64_k128_s2_a1;
    if (tokens <= 128) return launch_q6_a16_mma_r32_t128_k64_s2_a2;
    if (tokens <= 160) return launch_q6_a16_mma_r32_t64_k128_s2_a1;
    if (tokens <= 192) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 256) return launch_q6_a16_mma_r32_t128_k64_s2_a2;
    if (tokens <= 384) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 512) return launch_q6_a16_mma_r64_t128;
    if (tokens <= 768) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 1024) return launch_q6_a16_mma_r64_t128;
    if (tokens <= 1536) return launch_q6_a16_mma_r64_t96;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
