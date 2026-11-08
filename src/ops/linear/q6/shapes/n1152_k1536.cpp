// GB10 (sm_121a, 48 SMs): Qwen3.5 vision patch merger / projection [1152, 1536].
// Sized for 48-SM wave quantization (72 row tiles at R=16 = 1.5 waves/slice; 36 row tiles at R=32 =
// 0.75 waves/tile; 18 row tiles at R=64 = 0.375 waves/tile), eliminating the 170-SM RTX 5090
// oscillations between r64_t96 and r64_t128.
#include "ops/linear/q6/q6_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {

Q6Launch select_q6_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q6 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q6_a16_mma_r32_t64_k128_s2_a1;
    if (tokens <= 256) return launch_q6_a16_mma_r32_t128_k64_s2_a2;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
