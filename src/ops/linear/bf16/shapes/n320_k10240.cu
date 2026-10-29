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
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 10240>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>, 10240>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 3>, 10240>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 80, 64, 16, 8, 5, 1>, 10240>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_simt<S1>;
    if (tokens <= 48) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 512) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace

// Qwen3.8-Flash-Next: the final mixer's down projection (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T; this shape only ever sees
// the head's verification columns, T <= 64).
Bf16Launch select_bf16_n320_k10240(std::int32_t tokens) {
    if (tokens == 1) return ninfer_route(1);
    if (tokens <= 4) return launch_bf16_simt<Simt<2>>;
    if (tokens <= 8) return launch_bf16_simt<Simt<4>>;
    if (tokens <= 16) return ninfer_route(tokens); // 9..16 ninfer fastest on GB10
    if (tokens <= 64) return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3 (1.11-1.43x at 24..64)
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>;
    if (tokens <= 384) return ninfer_route(tokens);
    if (tokens <= 512) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>;
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>;
    if (tokens <= 1536) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>;
    if (tokens <= 3072) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>;
    if (tokens <= 4096) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>;
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>;
}
} // namespace infernix::ops::detail
