#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision MLP fc2, T = raw patches; 4304 inputs end in a partial K tile. Verified on GB10 (sm_121,
// 48 SMs, 218 GB/s at T=8 and 56 TFLOP/s at large T).
Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 4304>>; // tail_r64t32s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 4304>>; // tail_r64t64s3 (1.14x faster at T=256)
    return launch_bf16_tma_mma<
        Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4304>>;
}
} // namespace infernix::ops::detail
