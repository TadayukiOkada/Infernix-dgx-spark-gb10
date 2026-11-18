#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    6144>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 6144>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 8, 64, 2>, 6144>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 16, 4, 64, 2>, 6144>;
using S4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    6144>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 16, 2, 2>, 6144>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 2, 2>, 6144>;
using S7 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::Grouped, 4>, 6144>;
using S8 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    if (tokens <= 128) return launch_bf16_tma_mma<S6>;
    if (tokens <= 512) return launch_bf16_tma_mma<S7>;
    return launch_bf16_tma_mma<S8>;
}
} // namespace

// Qwen3.8-Flash-Next: the GDN and QSA output projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T).
Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 8) return ninfer_route(8); // decode 133.0-140.5 us vs skinny 135.3-141.4 us over T = 1..8 (GB10)
    if (tokens <= 16) return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3 (1.5x)
    if (tokens <= 48) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>>; // tma_r64t64k128s2
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>; // tma_r128t64k128s2_rows
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>; // tail_r64t128s2_rows
    if (tokens <= 384) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4
    if (tokens <= 1536) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 3>, 6144>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 3>, 6144>>;
}
} // namespace infernix::ops::detail
