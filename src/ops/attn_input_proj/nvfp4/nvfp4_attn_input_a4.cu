#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"

#include <cuda_bf16.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using Geometry = Nvfp4N14336K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Nvfp4AttnInputA4ScheduleId schedule;
};

// GB10 (48 SM, sm_121a) A4 GEMM band table, retuned by direct candidate sweep. All seven tile/TMA
// schedules slice tokens internally and validate only row/K tiles and scale layout (no token floor),
// producing byte-identical output at every token count, so the choice between them is purely speed.
// HEAD's table was inherited from RTX 5090 tuning and is badly mis-tuned here: the 32-wide tiles
// collapse hard past their tile width (M32N64 jumps ~187us->~349us at T=33; M32N128 runs ~530us
// through its old 65..96 band), M64N128 runs ~570us through 129..192, and the resident/pipelined
// tiles run ~400-820us across 193..511. On GB10 the TMA lane (which validates no token floor and
// whose scale plane is already allocated over whole tiles) dominates from 129 up -- exactly the same
// conservatism that gated the GDN projection at >=512. Crossovers pinned at high repeat (250-300):
// 32/33, 64/65, 128/129, 384/385 (tma128 -> tma256; the isolated 576/832 reversals are <3% noise).
// Bankable wins vs the old table: T=33..64 ~46%, T=65..96 ~63%, T=129..192 ~61%, T=193..384 ~42-58%,
// T=385..511 ~65%. M32N128 and M128N128Resident are no longer reachable in production routing.
constexpr std::array<RouteSpec, 5> kRoutes{{
    {1, 32, Nvfp4AttnInputA4ScheduleId::M32N64},
    {33, 64, Nvfp4AttnInputA4ScheduleId::M64N128},
    {65, 128, Nvfp4AttnInputA4ScheduleId::M128N128Pipelined},
    {129, 384, Nvfp4AttnInputA4ScheduleId::Tma128},
    {385, kAnyCols, Nvfp4AttnInputA4ScheduleId::Tma256},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "nvfp4 A4 attention input routes must be exact and closed");

bool supported_shape(const Nvfp4AttnInputA4Problem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.parent_rows == 14336 && problem.padded_k == 5120;
}

Nvfp4AttnInputA4ScheduleId resolve_a4_route(std::int32_t tokens) {
    for (const RouteSpec& route : kRoutes) {
        if (tokens >= route.first && tokens <= route.last) { return route.schedule; }
    }
    throw std::logic_error("nvfp4 A4 attention input: token count has no covering route");
}

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Nvfp4AttentionInputOutput{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        LinearIdentityEpilogue{}, stream);
}

void launch_a4_schedule(Nvfp4AttnInputA4ScheduleId schedule, const Tensor& x, const Weight& weight,
                        Tensor& q, Tensor& gate, Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                        std::int32_t tokens, cudaStream_t stream) {
    const auto layout = schedule == Nvfp4AttnInputA4ScheduleId::Tma128
                            ? Nvfp4ScaleLayout::Tiled128
                            : schedule == Nvfp4AttnInputA4ScheduleId::Tma256
                                  ? Nvfp4ScaleLayout::Tiled256
                                  : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (schedule == Nvfp4AttnInputA4ScheduleId::Tma128 ||
        schedule == Nvfp4AttnInputA4ScheduleId::Tma256) {
        launch_nvfp4_a4_tma_attention(
            nvfp4_a4_operands(weight, workspace, tokens, layout),
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data), stream);
        return;
    }
    switch (schedule) {
    case Nvfp4AttnInputA4ScheduleId::M32N64:
        launch_gemm<M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
        return;
    case Nvfp4AttnInputA4ScheduleId::M32N128:
        launch_gemm<M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
        return;
    case Nvfp4AttnInputA4ScheduleId::M128N128Pipelined:
        launch_gemm<M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
        return;
    case Nvfp4AttnInputA4ScheduleId::M64N128:
        launch_gemm<M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
        return;
    case Nvfp4AttnInputA4ScheduleId::M128N128Resident:
        launch_gemm<M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
        return;
    default:
        break;
    }
    throw std::logic_error("nvfp4 A4 attention input: unknown schedule");
}

} // namespace

const char* nvfp4_attn_input_a4_schedule_name(Nvfp4AttnInputA4ScheduleId schedule) noexcept {
    switch (schedule) {
    case Nvfp4AttnInputA4ScheduleId::M32N64:
        return "attn_input_proj.nvfp4.a4.m32.n64";
    case Nvfp4AttnInputA4ScheduleId::M32N128:
        return "attn_input_proj.nvfp4.a4.m32.n128";
    case Nvfp4AttnInputA4ScheduleId::M64N128:
        return "attn_input_proj.nvfp4.a4.m64.n128";
    case Nvfp4AttnInputA4ScheduleId::M128N128Pipelined:
        return "attn_input_proj.nvfp4.a4.m128.n128.pipelined";
    case Nvfp4AttnInputA4ScheduleId::M128N128Resident:
        return "attn_input_proj.nvfp4.a4.m128.n128.resident";
    case Nvfp4AttnInputA4ScheduleId::Tma128:
        return "attn_input_proj.nvfp4.a4.tma128";
    case Nvfp4AttnInputA4ScheduleId::Tma256:
        return "attn_input_proj.nvfp4.a4.tma256";
    }
    return "attn_input_proj.nvfp4.a4.unknown";
}

bool nvfp4_attn_input_a4_candidate_is_legal(Nvfp4AttnInputA4ScheduleId schedule,
                                            const Nvfp4AttnInputA4Problem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    (void)schedule;
    // Every tile schedule slices tokens internally and is numerically valid at any token count. The
    // TMA paths validate only row/K tiles and scale layout (no token floor) and their scale plane is
    // allocated over whole tiles, so they are legal at any token count; the sweep finds the speed
    // crossover.
    return true;
}

void nvfp4_attn_input_a4_execute_candidate(Nvfp4AttnInputA4ScheduleId schedule, const Tensor& x,
                                           const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                           Tensor& v, WorkspaceArena& workspace, cudaStream_t stream) {
    const Nvfp4AttnInputA4Problem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                          x.ne[1]};
    if (!nvfp4_attn_input_a4_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument(
            "nvfp4 A4 attention input: candidate schedule is not legal for problem");
    }
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    launch_a4_schedule(schedule, x, weight, q, gate, k, v, scratch, x.ne[1], stream);
}

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_a4_schedule(resolve_a4_route(tokens), x, weight, q, gate, k, v, workspace, tokens,
                       stream);
}

} // namespace infernix::ops::detail
