#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision MLP fc1, T = raw patches; 4304 rows end in a partial row tile. Verified on GB10 (sm_121,
// 48 SMs, 228–237 GB/s at small T and 50–60 TFLOP/s at large T).
Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 1152>>; // tail_r64t32s3
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 1152>>; // tail_r64t64s3 (1.08x faster at T=256)
}
} // namespace infernix::ops::detail
