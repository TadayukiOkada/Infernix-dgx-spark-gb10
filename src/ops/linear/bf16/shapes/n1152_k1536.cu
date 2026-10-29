#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision patch embedding, T = raw patches. Verified on GB10 (sm_121, 48 SMs).
Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 32)
        return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>>; // tma_r64t32k64s3 (1.33x faster at T=96)
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
}
} // namespace infernix::ops::detail
