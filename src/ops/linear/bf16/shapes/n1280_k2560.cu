#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next: the shared expert's gate and up projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T).
Bf16Launch select_bf16_n1280_k2560(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens); // skinny GEMV (equal to registered over T = 1..8)
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 64) return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3 (1.13-1.26x at 48..64, faster from T=17 on GB10)
    if (tokens <= 96) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3 (1.06x at 96)
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>>; // tma_r64t64k128s2 (1.17-1.26x at 100..128)
    if (tokens <= 192) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>; // tail_r128t64s4_rows (1.10x at 192)
    if (tokens <= 255) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4 (1.19x at 255)
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>; // tma_r128t64k64s4_rows (1.19x at 256)
    if (tokens <= 384) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>; // tma_r128t128k64s3_w32x64_rows (1.40x at 384)
    if (tokens <= 512) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>; // tma_r128t128k64s3_w64x32_group8 (1.24x at 512)
    if (tokens <= 768) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2 (HEAD wins 768)
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>; // tma_r128t128k64s3_w64x32_group8 (1.08-1.37x at 1024..2048)
    if (tokens <= 3072) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w32x64_rows (1.28x at 3072)
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tma_r128t128k64s3_w64x32_rows (HEAD wins 4096..8192)
}
} // namespace infernix::ops::detail
