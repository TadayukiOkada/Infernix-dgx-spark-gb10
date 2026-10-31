#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"

#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8GdnInputScheduleId schedule;
};

// The split-K direct MMA holds the whole 2..96 decode band and the plain MMA tiles take 97 columns
// and above. GB10 (48 SM, sm_121a) re-measured 2026-10-19, cold cache, 60 repeats:
//   65 cols  split-K 151 us, R64C128 143 us
//   80 cols  split-K 166 us, R64C128 143 us
//   96 cols  split-K 190 us, R64C128 143 us
// A lower crossover is genuinely faster, but R64C128 fails the A16 reduction criterion at 96
// aggregate columns: rel-L2 3.28e-3 against the 3.15e-3 limit (2.59e-3 for split-K). Both are
// valid FP32 accumulation orders; the wider tile reduces over a different K split and loses
// accuracy the Op contract does not allow. Accuracy binds before speed, so the crossover stays
// at 96. Re-tuning this band requires a numerically-equal R64C128, not a route change.
//
// 2026-10-31 T8: above 128 columns the 128x128 tile wins outright (2675 us against 2903 at
// T = 4096, 76.9 against 70.9 TFLOP/s cold), because 96 row tiles let a row-group's activation slab
// stay resident in L2 across the group. It shares R64C128's K order (BlockK = 64, no K split), so it
// is bit-identical to it and inherits the same 96-column accuracy bound; the 97..128 band keeps the
// 64-row tile, where it is still 6-11% ahead.
constexpr std::array<RouteSpec, 4> kRoutes{{
    {1, 1, Q8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8GdnInputScheduleId::SplitKMmaDirect},
    {97, 128, Q8GdnInputScheduleId::MmaR64C128},
    {129, kAnyCols, Q8GdnInputScheduleId::MmaR128C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q8 GDN input routes must be exact and closed");

bool supported_shape(const Q8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

const char* q8_gdn_input_schedule_name(Q8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        return "gdn_input_proj.q8.decode.r8.direct.k2048.split2";
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        return "gdn_input_proj.q8.mma.splitk.direct.k2048";
    case Q8GdnInputScheduleId::MmaR64C128:
        return "gdn_input_proj.q8.mma.r64.c128.split2";
    case Q8GdnInputScheduleId::MmaR128C128:
        return "gdn_input_proj.q8.mma.r128.c128.split2";
    case Q8GdnInputScheduleId::ColumnInvariant:
        return "gdn_input_proj.q8.mma.splitk.column_invariant";
    }
    return "gdn_input_proj.q8.unknown";
}

const char* q8_gdn_input_conv_schedule_name(Q8GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8GdnInputConvScheduleId::DecodeFused:
        return "gdn_input_proj_conv.q8.decode.fused";
    case Q8GdnInputConvScheduleId::SplitKMmaFused:
        return "gdn_input_proj_conv.q8.mma.splitk.fused";
    case Q8GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q8.materialized";
    }
    return "gdn_input_proj_conv_snapshot.q8.unknown";
}

bool q8_gdn_input_admits(const Q8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8GdnInputPlan q8_gdn_input_resolve_plan(const Q8GdnInputProblem& problem,
                                          bool column_invariant) {
    if (!q8_gdn_input_admits(problem)) {
        throw std::invalid_argument("Q8 GDN input: exact problem or column count is not admitted");
    }
    if (column_invariant) { return {Q8GdnInputScheduleId::ColumnInvariant}; }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 GDN input: admitted problem has no covering route");
}

Q8GdnInputConvPlan q8_gdn_input_conv_resolve_plan(const Q8GdnInputProblem& problem,
                                                  std::int32_t batch_size, bool column_invariant) {
    if (!q8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1 || column_invariant) { return {Q8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {Q8GdnInputConvScheduleId::DecodeFused}; }
    if (problem.cols <= 16) { return {Q8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {Q8GdnInputConvScheduleId::Materialized};
}

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream, bool column_invariant) {
    const Q8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const Q8GdnInputPlan plan = q8_gdn_input_resolve_plan(problem, column_invariant);
    switch (plan.schedule) {
    case Q8GdnInputScheduleId::ColumnInvariant:
        q8_gdn_input_column_invariant_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::DecodeR8Direct:
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR64C128:
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR128C128:
        q8_gdn_input_mma_r128_c128_launch(x, weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q8 GDN input: unknown schedule");
}

bool q8_gdn_input_candidate_is_legal(Q8GdnInputScheduleId schedule,
                                     const Q8GdnInputProblem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    switch (schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        return problem.cols == 1;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        return problem.cols >= 2 && problem.cols <= 96;
    case Q8GdnInputScheduleId::MmaR64C128:
        return true;
    case Q8GdnInputScheduleId::MmaR128C128:
        return true;
    }
    return false;
}

void q8_gdn_input_execute_candidate(Q8GdnInputScheduleId schedule, const Tensor& x,
                                    const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    const Q8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    if (!q8_gdn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("Q8 GDN input: candidate schedule is not legal for problem");
    }
    switch (schedule) {
    case Q8GdnInputScheduleId::DecodeR8Direct:
        q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::SplitKMmaDirect:
        q8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR64C128:
        q8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    case Q8GdnInputScheduleId::MmaR128C128:
        q8_gdn_input_mma_r128_c128_launch(x, weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q8 GDN input: unknown candidate schedule");
}

} // namespace infernix::ops::detail
