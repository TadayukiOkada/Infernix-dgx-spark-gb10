#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1, Fp8MmaRaster::Grouped, 8>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Fp8AttnInputScheduleId schedule;
};

// GB10 (48 SM, sm_121a) A8 GEMM band table for the 27B attention input projection, retuned by direct
// candidate sweep. All six schedules slice tokens internally and validate only row/K tiles, so each is
// structurally legal at any token count >= 1 and produces byte-identical output; the choice is purely speed.
// HEAD's table (inherited from RTX 5090 tuning) engages Bulk far too early and misplaces the mid TMA tiles.
// Direct argmin sweep (cold eager, warmup 25-30 / repeat 200-300) shows:
//   * t32r32 wins or ties the whole decode band 1..96; t32r128 never wins by more than noise (~1us) and is
//     dropped from production routing. Crossover to tma64x128 pinned at 96/97.
//   * tma64x128 wins 97..384 (HEAD wrongly used tma64x256 at 129..192 and tma96x256 at 193..288 -- each 3-13%
//     slower than this tile). Crossover to tma96x256 pinned at 384/385.
//   * tma96x256 (TokenFast) wins 385..2560 while FP8 activations fit inside GB10's 24 MiB L2 cache
//     (e.g. T=1024: 857us / 175.4 TFLOP/s; T=2048: 1638us / 183.6 TFLOP/s).
//   * tma64x256 with Grouped-8 L2 raster wins from 2561 up once FP8 activations exceed half of L2
//     (T=3840..8192: 155-159 TFLOP/s vs 98-141 TFLOP/s for TokenFast).
// Bulk is now unreachable in production routing: it is never the argmin at any token count up to 12288.
constexpr std::array<RouteSpec, 4> kRoutes{{
    {1, 96, Fp8AttnInputScheduleId::A8T32R32K128},
    {97, 384, Fp8AttnInputScheduleId::A8Tma64x128},
    {385, 2560, Fp8AttnInputScheduleId::A8Tma96x256},
    {2561, kAnyCols, Fp8AttnInputScheduleId::A8Tma64x256},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "fp8 A8 attention input routes must be exact and closed from T=1");

bool supported_shape(const Fp8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.parent_rows == 14336 && problem.padded_k == 5120;
}

Fp8AttnInputScheduleId resolve_a8_route(std::int32_t tokens) {
    for (const RouteSpec& route : kRoutes) {
        if (tokens >= route.first && tokens <= route.last) { return route.schedule; }
    }
    throw std::logic_error("fp8 A8 attention input: token count has no covering route");
}

template <class Schedule>
void launch_one(const Fp8A8Operands& operands, const Fp8AttentionInputOutput& output,
                float* partials, cudaStream_t stream) {
    using S = Fp8ScheduleInstance<Schedule, 5120>;
    if constexpr (S::kTmaSwizzle)
        launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, partials);
    else
        launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, {},
                             pdl::Dependency::Programmatic);
}

void launch_a8_schedule(Fp8AttnInputScheduleId schedule, const Tensor& x, const Weight& weight,
                        Tensor& q, Tensor& gate, Tensor& key, Tensor& value, Fp8A8Workspace workspace,
                        cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(value.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    switch (schedule) {
    case Fp8AttnInputScheduleId::A8T32R32K128:
        return launch_one<Fp8A8T32R32K128>(operands, output, workspace.partials, stream);
    case Fp8AttnInputScheduleId::A8T32R128K128:
        return launch_one<Fp8A8T32R128K128>(operands, output, workspace.partials, stream);
    case Fp8AttnInputScheduleId::A8Tma64x128:
        return launch_one<Tma64x128>(operands, output, workspace.partials, stream);
    case Fp8AttnInputScheduleId::A8Tma64x256:
        return launch_one<Tma64x256>(operands, output, workspace.partials, stream);
    case Fp8AttnInputScheduleId::A8Tma96x256:
        return launch_one<Tma96x256>(operands, output, workspace.partials, stream);
    case Fp8AttnInputScheduleId::A8Bulk:
        return launch_one<Bulk>(operands, output, workspace.partials, stream);
    }
    throw std::logic_error("fp8 A8 attention input: unknown schedule");
}

} // namespace

void fp8_attn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(value.data)};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a8_mma<Fp8ScheduleInstance<Schedule, 5120>>(
            fp8_a8_operands(weight, workspace, x.ne[1]), output, LinearIdentityEpilogue{}, stream,
            {}, pdl::Dependency::Programmatic);
    };
    // 32-token tiles to 32 columns, 32x128 tiles at 65-104 (58-62 against 78-83 us for 32x64 at
    // 65-96, 89-91 against 105 for 64x128 at 97-104), and 64x128 tiles otherwise (RTX 5090).
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 32) return launch.template operator()<Fp8A8ResidualT32R64>();
    if (tokens > 64 && tokens <= 104) return launch.template operator()<Fp8A8ResidualT32R128>();
    launch.template operator()<Fp8A8ResidualT64R128>();
}

std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    // No production route reaches a split-K schedule any more (Bulk is unreachable, all four bands are
    // plain TMA/MMA tiles), so production needs no partial plane. Candidates force Bulk via
    // candidate_partial_capacity_bytes() independently of this sizing.
    (void)max_tokens;
    return 0;
}

std::size_t fp8_attn_input_candidate_partial_capacity_bytes() {
    // Largest partial plane any forced candidate may need: Bulk's 128x256 split-K tiles.
    return Bulk::kPartialBytes;
}

const char* fp8_attn_input_schedule_name(Fp8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Fp8AttnInputScheduleId::A8T32R32K128:
        return "attn_input_proj.fp8.a8.t32.r32.k128";
    case Fp8AttnInputScheduleId::A8T32R128K128:
        return "attn_input_proj.fp8.a8.t32.r128.k128";
    case Fp8AttnInputScheduleId::A8Tma64x128:
        return "attn_input_proj.fp8.a8.tma64x128";
    case Fp8AttnInputScheduleId::A8Tma64x256:
        return "attn_input_proj.fp8.a8.tma64x256";
    case Fp8AttnInputScheduleId::A8Tma96x256:
        return "attn_input_proj.fp8.a8.tma96x256";
    case Fp8AttnInputScheduleId::A8Bulk:
        return "attn_input_proj.fp8.a8.bulk";
    }
    return "attn_input_proj.fp8.a8.unknown";
}

bool fp8_attn_input_candidate_is_legal(Fp8AttnInputScheduleId schedule,
                                       const Fp8AttnInputProblem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    // Every A8 schedule slices tokens internally and validates only row/K tiles, so it is legal at any
    // token count; the sweep measures each one everywhere to find the true per-token argmin.
    (void)schedule;
    return true;
}

void fp8_attn_input_execute_candidate(Fp8AttnInputScheduleId schedule, const Tensor& x,
                                      const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                      Tensor& v, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                      x.ne[1]};
    if (!fp8_attn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("fp8 A8 attention input: candidate schedule is not legal for problem");
    }
    auto scope = workspace.scope();
    // Always reserve the largest split-K partial plane so a forced Bulk has aligned partials regardless
    // of token count (production sizing keys on tokens, candidates key on the schedule).
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_attn_input_candidate_partial_capacity_bytes());
    launch_a8_schedule(schedule, x, weight, q, gate, k, v, scratch, stream);
}

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& key, Tensor& value, Fp8A8Workspace workspace,
                              cudaStream_t stream) {
    launch_a8_schedule(resolve_a8_route(x.ne[1]), x, weight, q, gate, key, value, workspace, stream);
}
} // namespace infernix::ops::detail
