// GB10 (sm_121a, 48 SMs): Qwen3.8-Flash-Next vocabulary head [248320, 2560].
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
namespace {
using S1     = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 5, 128, 2>, 2560>;
using S2     = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 24, 4, 64, 2>, 2560>;
using S3     = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>, 2560>;
using S4     = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 32, 16, 2, 1>, 2560>;
using S5     = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>, 2560>;
using S6     = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>;
using Medium = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 2560>;
using Large  = Bf16ScheduleInstance<
     Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 3>, 2560>;
} // namespace

Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens) {
    // T = 1 takes the T <= 8 route below (one column-invariant route per decode width).
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_tma_mma<S4, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 128) return launch_bf16_tma_mma<S6>;
    if (tokens <= 256) return launch_bf16_tma_mma<Medium>;
    return launch_bf16_tma_mma<Large, Bf16MmaStreamingWarpStoreEpilogue>;
}
} // namespace infernix::ops::detail
