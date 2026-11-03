// GB10 (sm_121a) tuned dispatch, 2026-10-30: bench/ops/nvfp4_route_sweep.cu (A16) over
// T = 1..512 against the pre-change table, which carried the RTX 5090 fit. The 32-row (RowTiles 2)
// sliced-K tiles and the K64 64-row MMA tile are the GB10 winners: -13..-21% over the DFlash2 and
// speculative verification widths T = 17..32, -26% at T = 192, 2..5% elsewhere. The A4 admission
// (T >= 17) is unchanged; the A16 table serves the A16-only and pair-eligible routes.
// GB10 2026-10-31 (T5): T = 17..32 was still 18..26% off the sweep argmin (114.9 -> 86.7us at
// T = 20, 122.1 -> 90.0 at T = 32; 64.2% -> 85.1% and 61.3% -> 83.2% of the 246 GB/s READ
// reference). The same two sliced-K tiles win, taken in the runtime-K form, which measures 24%
// faster than the K-baked twin the table used (see nvfp4_linear_a16_sliced_k_dyn). T = 33..64 is
// unchanged at ~69%: no A16 tile reaches 85% there, and this shape admits A4 from T = 17.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<5120, 6144>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using T16R64  = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>;
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2, 2>>;
    if (tokens <= 20)
        return nvfp4_linear_a16_sliced_k_dyn<Geometry, Nvfp4SlicedInstance<24, 4, 1, 2>>;
    if (tokens <= 32)
        return nvfp4_linear_a16_sliced_k_dyn<Geometry, Nvfp4SlicedInstance<32, 4, 1, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>;
    if (tokens <= 256)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    // GB10 2026-11-03: the prefill token tile is now picked per width by the 48-SM wave-fill rule
    // (nvfp4_a4_tma_prefill_route). Measured here: T = 257-384 and 640-896 run 4.8-12.8% ahead on
    // the 128-token tile (mean 6.5%); from 960 up the 17.3 MB weight re-read costs more than the
    // wave saves, up to 13.9% behind at 1536.
    if (tokens >= 257) return nvfp4_a4_tma_prefill_route<Geometry>(tokens);
    if (tokens <= 16) return nvfp4_a4_mma_route<Geometry, T16R64>();
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }
} // namespace

// GB10 2026-11-01 (T2): pair gate 65 -> 33, measured against the A16 table the public route runs
// in the same process (cold 32 MiB L2, 15 repeats, minimum over 2 passes). The pair leads the
// whole band: T = 33/40/47/56 is 97.5/98.7/100.1/100.3 us against 107.8/109.8/110.1/111.7 us, and
// T = 96/128 is 102.0/103.5 us against 126.0/148.9 us.
const Nvfp4LinearShape kNvfp4N5120K6144{
    5120, 6144, launch_a16, launch_nvfp4_a4<select_a4>, uses_a4, launch_nvfp4_a4_pair, 33};
} // namespace infernix::ops::detail
