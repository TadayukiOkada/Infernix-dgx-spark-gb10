#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision merger fc1, T = merged tokens. Verified on GB10 (sm_121, 48 SMs, 240–244 GB/s at small T).
Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 4608>>; // sliced_r32t16w4
    if (tokens <= 128)
        return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3 (1.82x faster at T=17..32)
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3 (2.81x faster at T=256)
    return launch_bf16_tma_mma<
        Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4608>>;
}
} // namespace infernix::ops::detail
