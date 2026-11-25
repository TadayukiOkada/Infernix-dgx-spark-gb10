#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
namespace {
using S0 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    320>>;
using S1 = Bf16A16MmaSchedule<32, 8, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S2 = Bf16A16MmaSchedule<64, 16, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S3 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 320>;
using Hot = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 2, 2>, 320>;
using Prefill64 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>, 320>;
using Prefill128 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>, 320>;

Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_mma<S1>;
    if (tokens <= 64) return launch_bf16_tma_mma<S3, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 128) return launch_bf16_tma_mma<Hot, Bf16MmaWarpStoreEpilogue>;
    const auto token_tiles64 = (static_cast<std::int64_t>(tokens) + 63) / 64;
    if (tokens <= 448 || (tokens <= 896 && token_tiles64 % 2 != 0))
        return launch_bf16_tma_mma<Prefill64>;
    return launch_bf16_tma_mma<Prefill128>;
}
} // namespace

// Qwen3.8-Flash-Next: the hyper-connection mixers' up projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text).
Bf16Launch select_bf16_n10240_k320(std::int32_t tokens) {
    if (tokens <= 8) return ninfer_route(8);
    return ninfer_route(tokens);
}
} // namespace infernix::ops::detail
