// GB10 (sm_121a) tuned dispatch, 2026-10-30: bench/ops/q8_route_sweep.cu over T = 1..64 (sweep
// record /tmp q8sweep_1280x2560, shipped 2026-10-30). The SIMT pair left T = 3..48 on 160 CTAs
// iterating K = 2560 serially (the shape serves 49x per forward pass): the sliced-K MMA family
// tiles hold ~16 us through T = 16 and ~21..33 us through T = 48, 28..68% faster at every width.
// The exact table delegates T = 1..48 to the shape's fixed sliced-K family: the sweep's per-width
// bit_exact gates cannot certify a cross-schedule column contract (each candidate was only
// checked at its own width against the pre-change route, and the SIMT, 4-K-warp and grouped
// families reduce a column differently), and the [1280,2560] 48-column invariance contract needs
// one bit family across the whole range. The T-dependent 4-K-warp winners the family cannot
// serve bit-invariantly stay on the fast route.

#include "ops/linear/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/q8/q8_a16_tma_k32.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {
namespace {
using TmaHot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using TmaBulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;
using TmaK32     = Q8A16TmaK32Schedule<8, 2560>;

template <class Schedule, Q8Launch Fallback>
void tma_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != 2560 || out.dtype != DType::BF16 || weight.n % 64 != 0) {
        Fallback(x, weight, out, stream);
        return;
    }
    launch_q8_a16_tma_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}

void tma_k32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != 2560 || out.dtype != DType::BF16 || weight.n % 128 != 0) {
        launch_q8_a16_mma_r128x64_t128_k64_a1(x, weight, out, stream);
        return;
    }
    launch_q8_a16_tma_k32<TmaK32>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}
} // namespace

// The exact table: the fixed 8-K-warp family (capacity 8/16/32 rungs) through T = 48 — the width
// the column-invariance contract covers — then the T3 (2026-10-31) sweep argmin through T = 128,
// and Q8 TMA MMA beyond (T = 129..192 TmaHot96, T = 193..512 TmaBulk128, T > 512 TmaK32).
Q8Launch select_q8_n1280_k2560(std::int32_t tokens) {
    if (tokens <= 48) return select_q8_n1280_k2560_invariant(tokens);
    if (tokens <= 96) return launch_q8_a16_sliced_r32_t48_w4_s1;
    if (tokens <= 128) return launch_q8_a16_mma_r32_t64_w16x32_k128_a1;
    if (tokens <= 192) return tma_mma<TmaHot96, launch_q8_a16_mma_r64x32_t64_k128_a1>;
    if (tokens <= 512) return tma_mma<TmaBulk128, launch_q8_a16_mma_r64_t128>;
    return tma_k32;
}

// The q8-sliced-k fast path (LinearRoutes::Q8SlicedK): the T-dependent sliced winners. Beyond
// T = 16 the 4-K-warp tiles beat the family (19.2..19.4 us flat over T = 17..32, 25..28 us over
// T = 33..64, where the swept sliced candidates pay 27..32); beyond T = 64 it falls back to the
// exact MMA tail.
Q8Launch select_q8_n1280_k2560_fast(std::int32_t tokens) {
    if (tokens <= 1) return select_q8_n1280_k2560_invariant(1);
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 32) return launch_q8_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 44) return launch_q8_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q8_a16_sliced_r32_t16_w4_s2;
    return select_q8_n1280_k2560(tokens);
}

} // namespace infernix::ops::detail
