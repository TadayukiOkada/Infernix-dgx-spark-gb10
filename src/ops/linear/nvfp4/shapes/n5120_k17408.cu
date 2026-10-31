// GB10 (sm_121a) tuned dispatch, 2026-10-30: bench/ops/nvfp4_route_sweep.cu (A16) over
// T = 1..512 against the pre-change table, which carried the RTX 5090 fit. The 32-row (RowTiles 2)
// sliced-K tiles widen the DRAM-streaming decode: -6..-9% at T = 2..16, -19..-36% at the
// verification widths T = 17..32 (the old T = 25..32 tile cost 391 us where the GB10 tile pays
// 249), and the T >= 129 tail (T = 384: 1606 -> 900 us, T = 512: 1564 -> 1222) replaces the
// RTX 5090 t256 tile, which lost 22..44% at every prefill width. The A4 admission (T >= 8) is
// unchanged; the A16 table serves the A16-only and pair-eligible routes.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<5120, 17408>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using T16R64  = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 2, 2>;
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R64  = Nvfp4A4MmaSchedule<64, 64, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<24, 4, 1, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 32, 16, 1, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 128, 32, 16, 2, 1>>;
    if (tokens <= 192)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    if (tokens <= 256)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 257) return nvfp4_a4_tma_route<>();
    if (tokens <= 16) return nvfp4_a4_mma_route<Geometry, T16R64>();
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 48) return nvfp4_a4_mma_route<Geometry, T64R64>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }
} // namespace

const Nvfp4LinearShape kNvfp4N5120K17408{
    5120, 17408, launch_a16, launch_nvfp4_a4<select_a4>, uses_a4, launch_nvfp4_a4_pair, 65};
} // namespace infernix::ops::detail
