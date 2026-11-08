// GB10 (sm_121a, 48 SMs): Qwen3.5 MoE router gate [256, 5120].
#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
    if (tokens <= 24)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>,
            pdl::Dependency::Programmatic>;
    if (tokens <= 128)
        return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3 (up to 1.40x faster)
    if (tokens <= 512) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 5120>>;
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 5120>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2Rows, 5120>>;
}
} // namespace infernix::ops::detail
