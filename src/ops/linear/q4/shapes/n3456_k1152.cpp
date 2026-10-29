#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {
Q4Launch select_q4_n3456_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    // Retuned on GB10 (sm_121, 48 SMs, 32 MiB cold-L2 flush): use sliced_r32_t32_w2_s2 across
    // 4..320 (faster even at T=4..16: 12.29 vs 14.18 us, and eliminates the +24.0% T=100..128 and
    // +30.2% T=196..256 mma_r32_t32_k128 islands) and mma_r32_t128_k64_s2_a2 (108 row tiles = 2.25
    // waves per token tile) for all T > 320 (eliminating the +101.3% cliff at T=388 and +20.0%
    // cliff at T=516).
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w2_s2;
    return launch_q4_a16_mma_r32_t128_k64_s2_a2;
}
} // namespace infernix::ops::detail
