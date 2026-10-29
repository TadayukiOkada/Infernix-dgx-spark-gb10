#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision merger fc2 for a 5120-wide text model (Qwen3.8 27B), T = merged tokens. Verified on GB10
// (sm_121, 48 SMs, 241–246.7 GB/s at small T and 61–62 TFLOP/s at large T).
Bf16Launch select_bf16_n5120_k4608(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 128)
        return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3 (1.91x faster at T=17..32)
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    return launch_bf16_tma_mma<
        Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4608>>;
}
} // namespace infernix::ops::detail
