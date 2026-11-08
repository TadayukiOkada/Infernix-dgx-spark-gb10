#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
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
using S4 = Bf16A16MmaSchedule<64, 64, 64, 16, 16, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S5 =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 3, Bf16MmaRaster::Grouped, 8>,
                         320>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 16) return launch_bf16_mma<S1>;
    if (tokens <= 48) return launch_bf16_mma<S2>;
    if (tokens <= 64) return launch_bf16_tma_mma<S3>;
    if (tokens <= 128) return launch_bf16_mma<S4>;
    return launch_bf16_tma_mma<S5>;
}
} // namespace

// Qwen3.8-Flash-Next: the hyper-connection mixers' up projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, GB10 sm_121a,
// warmup 8 / repeat 40, 2 % tolerance, seams at the last measured T).
Bf16Launch select_bf16_n10240_k320(std::int32_t tokens) {
    if (tokens <= 8) return ninfer_route(8); // decode 48.2 us vs skinny 57.4 us over T = 1..8 (GB10)
    if (tokens <= 64) return ninfer_route(tokens); // 9..64 ninfer fastest on GB10
    if (tokens <= 96) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3 (1.07x at 96)
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3 (1.04-1.06x at 100..128)
    if (tokens <= 255) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3 (1.05-1.31x at 192..255)
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3 (1.09x at 256)
    if (tokens <= 384) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3 (1.03x at 384)
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2 (1.05x at 512; HEAD wins 768..2048)
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>;
}
} // namespace infernix::ops::detail
