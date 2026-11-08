#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision attention output, T = raw patches. Verified on GB10 (sm_121, 48 SMs).
Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 64)
        return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>>; // tma_r64t32k64s3 (16.2 vs 18.3 us)
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3 (20.1 vs 22.3 us)
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>;
}
} // namespace infernix::ops::detail
