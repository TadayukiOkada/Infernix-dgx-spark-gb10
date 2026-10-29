#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using Geometry = Nvfp4N16384K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Nvfp4GdnInputA4ScheduleId schedule;
};

// GB10 (48 SM, sm_121a) A4 GEMM band table, retuned by direct candidate sweep. All five tile/TMA
// schedules produce byte-identical output at every token count (verified against an FP64 reference:
// forcing M32N64, M128N128Resident, M128N128Pipelined and Tma all yield the same values, and the
// canonical NVFP4 A4 linear suite passes this exact {16384,5120} geometry across ragged widths
// 129..511), so the choice between them is purely a speed decision, not a correctness one.
// On the RTX 5090 the M32N64/M32N128 tiles held the whole decode band, but on GB10 they collapse
// hard: M32N64 jumps from ~214us at T=32 to ~411us at T=33, and M32N128 runs ~620us through its old
// 65..96 band where the pipelined tile holds ~220us. M64N128 is the fastest tile in 33..64 (it then
// collapses past 72); the pipelined tile is the best tile schedule in 65..128 and matches the
// resident tile there within noise. The pipelined 128-wide tile is fastest through 128
// (~220us) but spikes at 129 where a nearly-empty second tile appears (~447us); the TMA lane (its
// 256-wide full-tile path) holds ~262us at 129 and only ~335us at 511, so it dominates from 129 up.
// TMA's structural legality has no token floor (it validates only row/K tiles and scale layout, and
// its workspace plane is already allocated over whole 256-token tiles), so the >=512 gate that HEAD
// inherited from the 5090 tuning was pure conservatism, not a limit. Bankable wins vs the old table:
// T=33..64 ~48%, T=65..128 ~65%, T=129..511 ~40-64%. Crossovers pinned at high repeat (150-200):
// 32/33, 64/65, 128/129.
constexpr std::array<RouteSpec, 4> kRoutes{{
    {1, 32, Nvfp4GdnInputA4ScheduleId::M32N64},
    {33, 64, Nvfp4GdnInputA4ScheduleId::M64N128},
    {65, 128, Nvfp4GdnInputA4ScheduleId::M128N128Pipelined},
    {129, kAnyCols, Nvfp4GdnInputA4ScheduleId::Tma},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "nvfp4 A4 GDN input routes must be exact and closed");

bool supported_shape(const Nvfp4GdnInputA4Problem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qkv_rows == 10240 && problem.z_rows == 6144 &&
           problem.parent_rows == 16384 && problem.padded_k == 5120;
}

Nvfp4GdnInputA4ScheduleId resolve_a4_route(std::int32_t tokens) {
    for (const RouteSpec& route : kRoutes) {
        if (tokens >= route.first && tokens <= route.last) { return route.schedule; }
    }
    throw std::logic_error("nvfp4 A4 GDN input: token count has no covering route");
}

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

void launch_a4_schedule(Nvfp4GdnInputA4ScheduleId schedule, const Tensor& x, const Weight& weight,
                        Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace, std::int32_t tokens,
                        cudaStream_t stream) {
    const auto layout = schedule == Nvfp4GdnInputA4ScheduleId::Tma ? Nvfp4ScaleLayout::Tiled256
                                                                    : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    switch (schedule) {
    case Nvfp4GdnInputA4ScheduleId::Tma:
        launch_nvfp4_a4_tma_gdn(nvfp4_a4_operands(weight, workspace, tokens, layout),
                                static_cast<__nv_bfloat16*>(qkv.data),
                                static_cast<__nv_bfloat16*>(z.data), stream);
        return;
    case Nvfp4GdnInputA4ScheduleId::M32N64:
        launch_gemm<M32N64>(weight, qkv, z, workspace, tokens, stream);
        return;
    case Nvfp4GdnInputA4ScheduleId::M32N128:
        launch_gemm<M32N128>(weight, qkv, z, workspace, tokens, stream);
        return;
    case Nvfp4GdnInputA4ScheduleId::M128N128Pipelined:
        launch_gemm<M128N128Pipelined>(weight, qkv, z, workspace, tokens, stream);
        return;
    case Nvfp4GdnInputA4ScheduleId::M64N128:
        launch_gemm<M64N128>(weight, qkv, z, workspace, tokens, stream);
        return;
    case Nvfp4GdnInputA4ScheduleId::M128N128Resident:
        launch_gemm<M128N128Resident>(weight, qkv, z, workspace, tokens, stream);
        return;
    }
    throw std::logic_error("nvfp4 A4 GDN input: unknown schedule");
}

} // namespace

const char* nvfp4_gdn_input_a4_schedule_name(Nvfp4GdnInputA4ScheduleId schedule) noexcept {
    switch (schedule) {
    case Nvfp4GdnInputA4ScheduleId::M32N64:
        return "gdn_input_proj.nvfp4.a4.m32.n64";
    case Nvfp4GdnInputA4ScheduleId::M32N128:
        return "gdn_input_proj.nvfp4.a4.m32.n128";
    case Nvfp4GdnInputA4ScheduleId::M128N128Pipelined:
        return "gdn_input_proj.nvfp4.a4.m128.n128.pipelined";
    case Nvfp4GdnInputA4ScheduleId::M64N128:
        return "gdn_input_proj.nvfp4.a4.m64.n128";
    case Nvfp4GdnInputA4ScheduleId::M128N128Resident:
        return "gdn_input_proj.nvfp4.a4.m128.n128.resident";
    case Nvfp4GdnInputA4ScheduleId::Tma:
        return "gdn_input_proj.nvfp4.a4.tma";
    }
    return "gdn_input_proj.nvfp4.a4.unknown";
}

bool nvfp4_gdn_input_a4_admits(const Nvfp4GdnInputA4Problem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

bool nvfp4_gdn_input_a4_candidate_is_legal(Nvfp4GdnInputA4ScheduleId schedule,
                                           const Nvfp4GdnInputA4Problem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    // Every tile schedule slices tokens internally and is numerically valid at any token count. The
    // TMA path is structurally legal at any token count (it validates only row/K tiles and scale
    // layout, and its scale plane is allocated over whole 256-token tiles); it becomes the fastest
    // lane from 129 up, so candidates only measure it from there.
    if (schedule == Nvfp4GdnInputA4ScheduleId::Tma) { return problem.cols >= 129; }
    return true;
}

void nvfp4_gdn_input_a4_execute_candidate(Nvfp4GdnInputA4ScheduleId schedule, const Tensor& x,
                                          const Weight& weight, Tensor& qkv, Tensor& z,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    const Nvfp4GdnInputA4Problem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                         x.ne[1]};
    if (!nvfp4_gdn_input_a4_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("nvfp4 A4 GDN input: candidate schedule is not legal for problem");
    }
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    launch_a4_schedule(schedule, x, weight, qkv, z, scratch, x.ne[1], stream);
}

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_a4_schedule(resolve_a4_route(tokens), x, weight, qkv, z, workspace, tokens, stream);
}

} // namespace infernix::ops::detail
