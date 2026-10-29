#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {
Q4Launch select_q4_n4304_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    // Retuned on GB10 (sm_121, 48 SMs, 32 MiB cold-L2 flush): keep sliced_r16_t16_w4_s2 for a
    // single 16-token tile (T <= 16) and extend sliced_r32_t32_w2_s2 across 20..448 (eliminating
    // the +25.2% cliffs at T=20 and T=68..96 and the +15.6% cliff at T=388).
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 448) return launch_q4_a16_sliced_r32_t32_w2_s2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
