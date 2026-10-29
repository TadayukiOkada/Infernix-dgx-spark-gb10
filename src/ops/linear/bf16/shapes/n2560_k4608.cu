#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision merger fc2 for a 2560-wide text model (Qwen3.8-Flash-Next), T = merged tokens. Retuned on
// GB10 (sm_121, 48 SMs, 24 MiB L2): keep tma_r64t64k64s3 across 129..512 and use the Grouped-8
// 128x128 schedule (84–88 TFLOP/s, 90% of GB10 BF16 TC peak) for all T > 512, eliminating the
// RowFast/TokenFast L2-thrashing cliffs at 257..512 and 1025..2304.
Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 4608>>; // sliced_r32t16w4
    if (tokens <= 64)
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 4608>>; // mma_r32t32k256s3
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 4608>>; // tail_r64t32s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 4608>>; // tma_r64t64k64s3
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4608>>; // tail_r128t128s3_w64x32_group8
}
} // namespace infernix::ops::detail
