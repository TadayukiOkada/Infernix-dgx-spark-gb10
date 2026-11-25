#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    10240>;
template <int BlockTokens>
using Simt = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2,
                        BlockTokens>,
    10240>;
using S1 = Simt<2>;
using S2 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 10240>>;
using S3 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>, 10240>>;
using S4 =
    Bf16RowTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 6>, 10240>>;
using S5 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>>;
using S6 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 8, 4, 1>, 10240>>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_simt<S1>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 512) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace

// Qwen3.8-Flash-Next: the hyper-connection mixers' down and inject projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T).
Bf16Launch select_bf16_n324_k10240(std::int32_t tokens) {
    if (tokens <= 1) return ninfer_route(1);
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 64) return ninfer_route(tokens); // 9..64 ninfer fastest on GB10
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3 (1.10-1.52x at 96..128)
    if (tokens <= 384) return ninfer_route(tokens); // 192..384 ninfer fastest on GB10
    if (tokens <= 512) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3 (1.07x at 512)
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>; // tail_r128t64s4_rows (1.75-1.79x at 768..1024)
    if (tokens <= 1536) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w64x32_rows (1.71x at 1536)
    if (tokens <= 3072) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w32x64_rows (1.04-1.63x at 2048..3072)
    if (tokens <= 4096) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w64x32_rows (3.38x at 4096)
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w32x64_rows (1.25x at 8192)
}
} // namespace infernix::ops::detail
