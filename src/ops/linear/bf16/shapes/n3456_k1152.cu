#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision fused QKV, T = raw patches. Retuned on GB10 (sm_121, 48 SMs, 24 MiB L2): use the 3-stage
// 128x128 tile for 513..1024 and > 1152 to avoid the 2-stage 64x128 L2/register dip at 769..1024
// and > 3072.
Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 1152>>; // tail_r64t32s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 1152>>; // tma_r64t64k64s3
    if (tokens <= 1024)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 1152>>; // tma_r128t128k64s3_w64x32
    if (tokens <= 1152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 1152>>; // tail_r64t128s2
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 1152>>; // tma_r128t128k64s3_w64x32_rows
}
} // namespace infernix::ops::detail
