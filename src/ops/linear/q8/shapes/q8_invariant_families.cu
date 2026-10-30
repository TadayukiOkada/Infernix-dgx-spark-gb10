// Column-invariant Q8 routes for the Qwen3.8-Flash-Next shapes: one fixed sliced-K MMA family per
// shape, run at token capacity 8, 16 or 32 by call width. Each schedule runs KWarps K-warps over
// the same K-slice order and reduces the splits in a fixed tree, and the MMA's token columns are
// independent, so a column's bits do not depend on how many columns share the call (the same
// invariant the vocabulary head's family provides). Shapes whose K is not a multiple of
// KWarps * 64 use the runtime-K variant of the schedule; shapes whose rows are not a multiple of
// the 16-row MMA tile (N = 324) cannot take this route at all and keep the SIMT pair.
//
// The schedules are the GB10 retune of 2026-10-29: 8 K-warps (16 where the 20-CTA row grid of
// [320, 10240] needs the wider slice) at the measured Stages/MinBlocksPerSm per shape, replacing
// the SIMT pair, whose 8-column calls doubled beyond T = 8 and lost 15..40% of the bandwidth at
// T = 5..8.

#include "ops/linear/q8/q8_instance_launch.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {
namespace {

using Access = Q8ScaleAccess;
using Stage  = Q8ActivationStage;

// One rung of a family: BlockTokens-wide, runtime K (a static ladder pins K through the geometry
// when launch_q8_a16_sliced applies with_problem).
template <int BlockTokens, int KWarps, int Stages, int MinBlocksPerSm,
          Q8ScaleAccess ScaleAccess = Access::Shared, int RowTiles = 1>
using Rung = Q8A16SlicedKMmaSchedule<BlockTokens, KWarps, Stages, MinBlocksPerSm, ScaleAccess, Cache::ca,
                                     Cache::cg, Stage::ActiveOnly, 0, BlockTokens, false, RowTiles>;

// Static-K ladder: launch_q8_a16_sliced pins K to the geometry's K through with_problem.
template <class Geometry, class R8, class R16, class R32>
Q8Launch static_ladder(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, R8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, R16>;
    return launch_q8_a16_sliced<Geometry, 32, R32>;
}

// The same ladder for runtime-K schedules, which must not pass through with_problem (it would
// pin a static K the operands cannot match).
template <class Geometry, class Schedule>
void runtime_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream, {}, pdl::Dependency::Programmatic);
}

template <class Geometry, class R8, class R16, class R32>
Q8Launch runtime_ladder(std::int32_t tokens) {
    if (tokens <= 8) return runtime_launch<Geometry, R8>;
    if (tokens <= 16) return runtime_launch<Geometry, R16>;
    return runtime_launch<Geometry, R32>;
}

// The runtime-K geometry carries only the output width: the schedule takes K from the operands, so
// there is no static K to pin through with_problem.
struct G10240x320 {
    static constexpr std::int32_t kOutputRows = 10240;
};
struct G320x10240 {
    static constexpr std::int32_t kInputRows = 10240, kOutputRows = 320;
};
struct G2560x640 {
    static constexpr std::int32_t kInputRows = 640, kOutputRows = 2560;
};
struct G1280x2560 {
    static constexpr std::int32_t kInputRows = 2560, kOutputRows = 1280;
};
struct G2560x2560 {
    static constexpr std::int32_t kInputRows = 2560, kOutputRows = 2560;
};
struct G12800x2560 {
    static constexpr std::int32_t kInputRows = 2560, kOutputRows = 12800;
};
struct G13952x2560 {
    static constexpr std::int32_t kInputRows = 2560, kOutputRows = 13952;
};
struct G16384x2560 {
    static constexpr std::int32_t kInputRows = 2560, kOutputRows = 16384;
};
struct G2560x12800 {
    static constexpr std::int32_t kInputRows = 12800, kOutputRows = 2560;
};

} // namespace

// K = 320 admits no static K-warps split (320 is no multiple of 128, 256 or 512): the runtime-K
// KWarps-8 family, capped by each rung's shared-memory fit per SM (4/3/2 blocks at T = 8/16/32;
// measured best at every T, 197..203 GB/s across T = 1..16).
Q8Launch select_q8_n10240_k320_invariant(std::int32_t tokens) {
    return runtime_ladder<G10240x320, Rung<8, 8, 1, 4>, Rung<16, 8, 1, 3>, Rung<32, 8, 1, 2>>(tokens);
}

// N = 320 is twenty 16-row CTA tiles: the 16-K-warp family (a 1024-K group per call round) keeps
// the 20-CTA grid's load balanced (83 KiB shared staging fits one CTA/SM at T = 32), holding
// T = 1 at the SIMT pair's bandwidth and gaining 22..55% at T = 5..8.
Q8Launch select_q8_n320_k10240_invariant(std::int32_t tokens) {
    return static_ladder<G320x10240, Rung<8, 16, 1, 2>, Rung<16, 16, 1, 2>, Rung<32, 16, 1, 1>>(tokens);
}

// K = 640: the static 2-K-warp family (five 128-K groups) over 160 CTAs; the wider K-splits only
// idle warps here. 144..152 GB/s flat across T = 1..16.
Q8Launch select_q8_n2560_k640_invariant(std::int32_t tokens) {
    return static_ladder<G2560x640, Rung<8, 2, 1, 2>, Rung<16, 2, 1, 2>, Rung<32, 2, 1, 2>>(tokens);
}

// The 2560-K families: 8 K-warps over five 512-K groups. Direct scale reads measure best at
// N = 1280 (80 CTAs, little scale reuse); two pipeline stages per block measure best from
// N = 2560 up, where the CTAs hide each other's weight stream.
Q8Launch select_q8_n1280_k2560_invariant(std::int32_t tokens) {
    return static_ladder<G1280x2560, Rung<8, 8, 1, 2, Access::Direct>, Rung<16, 8, 1, 2, Access::Direct>,
                         Rung<32, 8, 1, 2, Access::Direct>>(tokens);
}

Q8Launch select_q8_n2560_k2560_invariant(std::int32_t tokens) {
    return static_ladder<G2560x2560, Rung<8, 8, 2, 1>, Rung<16, 8, 2, 1>, Rung<32, 8, 2, 1>>(tokens);
}
Q8Launch select_q8_n12800_k2560_invariant(std::int32_t tokens) {
    return static_ladder<G12800x2560, Rung<8, 8, 2, 1>, Rung<16, 8, 2, 1>, Rung<32, 8, 2, 1>>(tokens);
}
Q8Launch select_q8_n13952_k2560_invariant(std::int32_t tokens) {
    return static_ladder<G13952x2560, Rung<8, 8, 2, 1>, Rung<16, 8, 2, 1>, Rung<32, 8, 2, 1>>(tokens);
}
Q8Launch select_q8_n16384_k2560_invariant(std::int32_t tokens) {
    return static_ladder<G16384x2560, Rung<8, 8, 2, 1>, Rung<16, 8, 2, 1>, Rung<32, 8, 2, 1>>(tokens);
}

// DFlash2's feature_projection (added 2026-10-30 with the shape's dispatch entry): the [2560, 6144]
// 2-stage pattern over the 12800-K reduction; the sweep's width-1..8 sliced winners run at
// 160..168 us here, 15..20% under the SIMT pair's 187..215.
Q8Launch select_q8_n2560_k12800_invariant(std::int32_t tokens) {
    return static_ladder<G2560x12800, Rung<8, 8, 2, 1>, Rung<16, 8, 2, 1>, Rung<32, 8, 2, 1>>(tokens);
}

} // namespace infernix::ops::detail
