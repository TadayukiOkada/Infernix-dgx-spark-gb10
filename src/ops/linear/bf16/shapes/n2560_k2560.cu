#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 2, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 3>, 2560>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>, 2560>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 24, 4, 64, 2>, 2560>;
using S4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 8, 3, 1>, 2560>;
using S6 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 128, 32, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S7 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1>, 2560>;
using S8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>;
using S9 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 2, Bf16MmaRaster::RowFast, 1>, 2560>;
} // namespace

Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens) {
    // T = 1 takes the T <= 8 route below (one column-invariant route per decode width; NInfer used a GEMV).
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 96) return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3 (1.19x faster at T=32..48)
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>>;
    if (tokens <= 512) return launch_bf16_tma_mma<S7>;
    if (tokens <= 1024) return launch_bf16_tma_mma<S8>;
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>;
}
} // namespace infernix::ops::detail
