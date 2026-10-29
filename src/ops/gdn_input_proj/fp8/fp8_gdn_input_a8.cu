#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Fp8GdnInputScheduleId schedule;
};

// GB10 (48 SM, sm_121a) A8 GEMM band table for the 27B GDN input projection, retuned by direct
// candidate sweep. All six schedules slice tokens internally and validate only row/K tiles
// (kBlockRows in {32,128,256} all divide 16384; kBlockK divides 5120), so they are structurally
// legal at any token count >= 17 (the A8 route floor set by resolve_route) and produce
// byte-identical output; the choice between them is purely a speed decision.
//
// On GB10 the old (5090-inherited) table is badly mis-tuned. Direct argmin sweep (cold eager,
// warmup 25-30 / repeat 200-300) shows:
//   * A8T64R128K256 dominates the whole decode band 17..256 (HEAD wrongly used T32R32K128 at
//     17..32 and TMA64/TMA192 at 65..192, and Bulk at 193..256 -- each 4-9% slower than this tile).
//   * TMA64x128 wins 257..448 (crossover from the tile pinned at 256/258; crossover to TMA192 at
//     448/450). HEAD used Bulk here (193..384) and MidBulk (385..448): Bulk loses 10-12%.
//   * TMA192x128 with Grouped-8 L2 raster wins from 449 up, keeping activation tiles resident in
//     GB10's 24 MiB L2 cache at large T (4096..8192) without Bulk's register spills.
constexpr std::array<RouteSpec, 3> kRoutes{{
    {17, 256, Fp8GdnInputScheduleId::A8T64R128K256},
    {257, 448, Fp8GdnInputScheduleId::A8Tma64x128},
    {449, kAnyCols, Fp8GdnInputScheduleId::A8Tma192x128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 17;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "fp8 A8 GDN input routes must be exact and closed from T=17");

bool supported_shape(const Fp8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qkv_rows == 10240 && problem.z_rows == 6144 &&
           problem.parent_rows == 16384 && problem.padded_k == 5120;
}

Fp8GdnInputScheduleId resolve_a8_route(std::int32_t tokens) {
    for (const RouteSpec& route : kRoutes) {
        if (tokens >= route.first && tokens <= route.last) { return route.schedule; }
    }
    throw std::logic_error("fp8 A8 GDN input: token count has no covering route");
}

template <class Schedule>
void launch_one(const Fp8A8Operands& operands, const Fp8GdnInputOutput& output,
                float* partials, cudaStream_t stream) {
    using S = Fp8ScheduleInstance<Schedule, 5120>;
    if constexpr (S::kTmaSwizzle)
        launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, partials);
    else
        launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
}

void launch_a8_schedule(Fp8GdnInputScheduleId schedule, const Tensor& x, const Weight& weight,
                        Tensor& qkv, Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    switch (schedule) {
    case Fp8GdnInputScheduleId::A8T32R32K128:
        return launch_one<Fp8A8T32R32K128>(operands, output, workspace.partials, stream);
    case Fp8GdnInputScheduleId::A8T64R128K256:
        return launch_one<Fp8A8T64R128K256>(operands, output, workspace.partials, stream);
    case Fp8GdnInputScheduleId::A8Tma64x128:
        return launch_one<Tma64x128>(operands, output, workspace.partials, stream);
    case Fp8GdnInputScheduleId::A8Tma192x128:
        return launch_one<Tma192x128>(operands, output, workspace.partials, stream);
    case Fp8GdnInputScheduleId::A8MidBulk:
        return launch_one<MidBulk>(operands, output, workspace.partials, stream);
    case Fp8GdnInputScheduleId::A8Bulk:
        return launch_one<Bulk>(operands, output, workspace.partials, stream);
    }
    throw std::logic_error("fp8 A8 GDN input: unknown schedule");
}

} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}

std::size_t fp8_gdn_input_candidate_partial_capacity_bytes() {
    // Largest partial plane any forced candidate may need: Bulk's 128x256 split-K tiles.
    constexpr std::size_t mid = MidBulk::kPartialBytes;
    constexpr std::size_t bulk = Bulk::kPartialBytes;
    return mid > bulk ? mid : bulk;
}

const char* fp8_gdn_input_schedule_name(Fp8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Fp8GdnInputScheduleId::A8T32R32K128:
        return "gdn_input_proj.fp8.a8.t32.r32.k128";
    case Fp8GdnInputScheduleId::A8T64R128K256:
        return "gdn_input_proj.fp8.a8.t64.r128.k256";
    case Fp8GdnInputScheduleId::A8Tma64x128:
        return "gdn_input_proj.fp8.a8.tma64x128";
    case Fp8GdnInputScheduleId::A8Tma192x128:
        return "gdn_input_proj.fp8.a8.tma192x128";
    case Fp8GdnInputScheduleId::A8MidBulk:
        return "gdn_input_proj.fp8.a8.midbulk";
    case Fp8GdnInputScheduleId::A8Bulk:
        return "gdn_input_proj.fp8.a8.bulk";
    }
    return "gdn_input_proj.fp8.a8.unknown";
}

bool fp8_gdn_input_candidate_is_legal(Fp8GdnInputScheduleId schedule,
                                      const Fp8GdnInputProblem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    // Every A8 schedule slices tokens internally and validates only row/K tiles, so it is legal at
    // any token count; the sweep measures each one everywhere to find the true per-token argmin.
    (void)schedule;
    return true;
}

void fp8_gdn_input_execute_candidate(Fp8GdnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& qkv, Tensor& z,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    if (!fp8_gdn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("fp8 A8 GDN input: candidate schedule is not legal for problem");
    }
    auto scope = workspace.scope();
    // Always reserve the largest split-K partial plane so a forced MidBulk/Bulk has aligned partials
    // regardless of token count (production sizing keys on tokens, candidates key on the schedule).
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_gdn_input_candidate_partial_capacity_bytes());
    launch_a8_schedule(schedule, x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, 5120>>(
            fp8_a8_operands(weight, workspace, x.ne[1]), output, LinearIdentityEpilogue{}, stream);
    };
    // 32x128 tiles at 65-96 run 87-91 against 91-99 us for 64x64 (RTX 5090).
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 32) return launch.template operator()<Fp8A8ResidualT32R64>();
    if (tokens <= 64) return launch.template operator()<Fp8A8ResidualT64R64>();
    if (tokens <= 96) return launch.template operator()<Fp8A8ResidualT32R128>();
    launch.template operator()<Fp8A8ResidualT64R128>();
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8_schedule(resolve_a8_route(x.ne[1]), x, weight, qkv, z, workspace, stream);
}
} // namespace infernix::ops::detail
