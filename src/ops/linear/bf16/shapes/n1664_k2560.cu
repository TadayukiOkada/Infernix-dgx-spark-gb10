// GB10 (sm_121a, 48 SMs): Qwen3.8-Flash-Next QSA indexer projection [1664, 2560].
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
namespace {
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 3>, 2560>;
using S2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S3 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>, 2560>;
using S4 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>, 2560>;
using S8 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 2560>;
} // namespace

Bf16Launch select_bf16_n1664_k2560(std::int32_t tokens) {
    // T = 1 takes the T <= 8 route below (one column-invariant route per decode width; NInfer used a GEMV).
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 32) return launch_bf16_mma<S2>;
    if (tokens <= 96) return launch_bf16_tma_mma<S3>;
    if (tokens <= 512) return launch_bf16_tma_mma<S4>;
    return launch_bf16_tma_mma<S8>;
}
} // namespace infernix::ops::detail
