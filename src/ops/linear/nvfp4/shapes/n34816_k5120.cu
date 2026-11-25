// GB10 (sm_121a) tuned dispatch, 2026-10-30: bench/ops/nvfp4_route_sweep.cu (A16) over
// T = 1..512 against the pre-change table, which carried the RTX 5090 fit. The 32-row (RowTiles 2)
// sliced-K tiles win the decode and verification band (-8..-10% at T = 3 and T = 17..32), the
// 32-row K128 tile takes T = 49..64 (-7%) and T = 97..128 (-8.6%), and the 128-row K64 tile
// replaces the RTX 5090 t256 tail over the whole prefill range (T = 192..384: -17..-37%).
// The A4 route is unchanged (it dominates every width it admits on this shape).
// GB10 2026-10-31: Nvfp4A4Pair is registered (T >= 65) so --routes fast serves the 2-term residual
// route; the A16 table above remains the fallback for A16Only artifacts.
// GB10 2026-10-31 (T5): the T = 17..32 verification bands were still the 2026-10-30 fit and lost
// 9..17% to the 2-stage / RowTiles-2 sliced-K tiles (491.9 -> 446.8 at T = 20, 550.1 -> 457.9 at
// T = 32; 84.2% -> 92.7% and 76.0% -> 91.3% of the 246 GB/s READ reference). Those two tiles are
// selected in the runtime-K form: the same Nvfp4ScheduleInstance<Schedule, 5120> bake the engine
// normally uses measures 525.8us for <24,4,1,2> against 449.0us with K read from the operands,
// because baking K lets ptxas fully unroll the K-group loop and it schedules worse at this tile.
// T = 33..64 is unchanged: the 32-row K128 tile that looks 5.4% ahead at T = 48 in one run is
// 0.2% behind in the other, so the 2026-10-30 fit stands.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<34816, 5120>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R64  = Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2, 2>>;
    if (tokens <= 24)
        return nvfp4_linear_a16_sliced_k_dyn<Geometry, Nvfp4SlicedInstance<24, 4, 2, 2>>;
    if (tokens <= 32)
        return nvfp4_linear_a16_sliced_k_dyn<Geometry, Nvfp4SlicedInstance<32, 4, 1, 2>>;
    if (tokens <= 48)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>;
    if (tokens <= 160)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 80, 128, 32, 16, 1, 2>>;
    if (tokens <= 448)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 257) return nvfp4_a4_tma_route<>();
    if (tokens >= 129)
        return nvfp4_a4_tma_route<Nvfp4ScaleLayout::Tiled128>();
    if (tokens <= 24) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T64R64>();
    return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }

} // namespace

// GB10 2026-11-01 (T2): pair gate 65 -> 48 and the 64-row pair tile. Measured against the A16
// table the public route runs in the same process (cold 32 MiB L2, 15 repeats, minimum over 2
// passes) the 64-row panel is ahead across the band: 496.0/495.1/499.9/502.4 us at
// T = 33/40/47/56 against 506.0/508.8/510.4/510.8 us for A16, and 518.4/527.3 us at T = 96/128
// against 657.4/696.6 us. The 128-row panel is 2..3% behind it throughout. The A16 table still
// wins at T = 2..32, where its GEMV and sliced-K tiles read the weights once per CTA.
const Nvfp4LinearShape kNvfp4N34816K5120{
    34816, 5120, launch_a16, launch_nvfp4_a4<select_a4>, uses_a4,
    launch_nvfp4_a4_pair_r64, 33};
} // namespace infernix::ops::detail
