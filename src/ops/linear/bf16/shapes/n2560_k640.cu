#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next: the shared expert's down projection (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T).
Bf16Launch select_bf16_n2560_k640(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens); // skinny GEMV (equal to registered over T = 1..8)
    if (tokens <= 24) return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3
    if (tokens <= 96) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2
    if (tokens <= 3072) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>; // tma_r128t128k64s3_w64x32
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>; // tma_r128t128k64s3_w64x32_group8
}
} // namespace infernix::ops::detail
