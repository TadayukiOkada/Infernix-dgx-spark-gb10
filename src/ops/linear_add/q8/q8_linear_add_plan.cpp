#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8LinearAddScheduleId schedule;
};

// GB10 (sm_121a, 48 SMs) route table, re-fitted from a complete cold-cache candidate sweep of every
// admitted tile across the token-width range (bench/ops/q8_linear_add_bench, rows=2048, K=4096, warmup
// 8 / repeat 40). The previous 5-band table was fitted on the RTX 5090 (170 SMs); its MediumSplitK band
// (cols 49-128) and single MmaR32C128 band (cols 129-640) under-fill 48 SMs and lose 14-47% in the
// prefill band. The exact-T split-K band (cols 2-48) was already optimal on GB10 (0% gap) and is
// unchanged. All MMA tiles accumulate each output over K in the same order, so swapping them is
// bit-identical; only speed differs.
//
// 2026-10-31 T8 re-fit at the two ends of the range. At col 1 the r16 GEMV reads the weight once at
// 212.7 GB/s against the r8x4-column SIMT's 198.2 (the SIMT's four-column reuse buys nothing when a
// single token leaves nothing to reuse), so decode now takes the same GEMV family K = 6144 uses. From
// col 2560 the 128-row/64-column tile beats the 64x128 incumbent by 7% (1001 us against 1075 at
// T = 4096, 68.6 against 64.4 TFLOP/s): with 16 row tiles the wider row tile halves the activation
// re-stream while 64 columns keep 1024 CTAs, which is what fills 48 SMs at two CTAs each. The
// 128x128 tile the task proposed is 27% slower here (54.0 TFLOP/s) — see the GB10 changelog.
constexpr std::array<RouteSpec, 20> kK4096Routes{{
    {1, 1, Q8LinearAddScheduleId::DecodeR16},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 64, Q8LinearAddScheduleId::MmaR16C32K128},
    {65, 195, Q8LinearAddScheduleId::MmaR32C64K128},
    {196, 259, Q8LinearAddScheduleId::MmaR48C128},
    {260, 291, Q8LinearAddScheduleId::MmaR64C96},
    {292, 391, Q8LinearAddScheduleId::MmaR64C128},
    {392, 495, Q8LinearAddScheduleId::MmaR128C80},
    {496, 607, Q8LinearAddScheduleId::MmaR64C96},
    {608, 639, Q8LinearAddScheduleId::MmaR64C112},
    {640, 671, Q8LinearAddScheduleId::MmaR64C128},
    {672, 703, Q8LinearAddScheduleId::MmaR64C112},
    {704, 799, Q8LinearAddScheduleId::MmaR64C128},
    {800, 831, Q8LinearAddScheduleId::MmaR128C80},
    {832, 863, Q8LinearAddScheduleId::MmaR64C128},
    {864, 895, Q8LinearAddScheduleId::MmaR64C96},
    {896, 927, Q8LinearAddScheduleId::MmaR64C128},
    {928, 991, Q8LinearAddScheduleId::MmaR128C80},
    {992, 2559, Q8LinearAddScheduleId::MmaR64C128},
    {2560, kAnyCols, Q8LinearAddScheduleId::MmaR128C64},
}};

// GB10 (sm_121a, 48 SMs) route table, re-fitted from a complete cold-cache candidate sweep of every
// admitted tile across the token-width range (bench/ops/q8_linear_add_bench, rows=2048, K=6144, warmup
// 8 / repeat 40). The previous bands were fitted on the RTX 5090 (170 SMs); its small row-tiles and
// MediumSplitK under-fill 48 SMs in the prefill band, costing 14-47% there. Decode (col 1) and the
// exact-T split-K band (cols 2-48) were already optimal on GB10 (0% gap) and are unchanged. Wider
// MMA tiles (r48/r64/r128) win the prefill band on 48 SMs. All MMA tiles accumulate each output over
// K in the same order, so swapping them is bit-identical; only speed differs.
//
// 2026-10-31 T8: from col 1792 the 128-row/64-column tile beats the 64x128 incumbent by 7-10%
// (1509 us against 1620 at T = 4096, 68.3 against 63.6 TFLOP/s); it still loses below 1536, where the
// 64-row tile's 32 row tiles are what keeps enough CTAs in flight.
constexpr std::array<RouteSpec, 18> kK6144Routes{{
    {1, 1, Q8LinearAddScheduleId::DecodeR16},
    {2, 48, Q8LinearAddScheduleId::SplitKMmaExactT},
    {49, 64, Q8LinearAddScheduleId::MmaR16C32K128},
    {65, 96, Q8LinearAddScheduleId::MmaR32C32K128},
    {97, 195, Q8LinearAddScheduleId::MmaR32C64K128},
    {196, 259, Q8LinearAddScheduleId::MmaR48C128},
    {260, 291, Q8LinearAddScheduleId::MmaR64C96},
    {292, 343, Q8LinearAddScheduleId::MmaR64C112},
    {344, 391, Q8LinearAddScheduleId::MmaR64C128},
    {392, 495, Q8LinearAddScheduleId::MmaR128C80},
    {496, 591, Q8LinearAddScheduleId::MmaR64C96},
    {592, 703, Q8LinearAddScheduleId::MmaR64C112},
    {704, 767, Q8LinearAddScheduleId::MmaR128C64},
    {768, 959, Q8LinearAddScheduleId::MmaR64C128},
    {960, 975, Q8LinearAddScheduleId::MmaR128C80},
    {976, 1008, Q8LinearAddScheduleId::MmaR64C112},
    {1009, 1791, Q8LinearAddScheduleId::MmaR64C128},
    {1792, kAnyCols, Q8LinearAddScheduleId::MmaR128C64},
}};

constexpr std::array<RouteSpec, 2> kN5120Routes{{
    {1, 64, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {65, kAnyCols, Q8LinearAddScheduleId::GroupedSplitK},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK4096Routes) && routes_are_closed(kK6144Routes) &&
                  routes_are_closed(kN5120Routes),
              "Q8 LinearAdd routes must be exact, contiguous, and closed");

std::int32_t schedule_rows(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 8;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR32C128:
        return 32;
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR48C128:
        return 48;
    case Q8LinearAddScheduleId::MmaR64C96:
    case Q8LinearAddScheduleId::MmaR64C112:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 64;
    case Q8LinearAddScheduleId::MmaR128C64:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
    case Q8LinearAddScheduleId::ColumnInvariant:
    case Q8LinearAddScheduleId::MmaR16C32K128:
    case Q8LinearAddScheduleId::MmaR32C32K128:
    case Q8LinearAddScheduleId::MmaR32C64K128:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule);

std::int32_t schedule_cols(Q8LinearAddScheduleId schedule) {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        break;
    case Q8LinearAddScheduleId::SimtR8C4:
        return 4;
    case Q8LinearAddScheduleId::MmaR32C64:
    case Q8LinearAddScheduleId::MmaR48C64:
    case Q8LinearAddScheduleId::MmaR128C64:
        return 64;
    case Q8LinearAddScheduleId::MmaR32C80:
    case Q8LinearAddScheduleId::MmaR128C80:
        return 80;
    case Q8LinearAddScheduleId::MmaR32C96:
    case Q8LinearAddScheduleId::MmaR48C96:
    case Q8LinearAddScheduleId::MmaR64C96:
        return 96;
    case Q8LinearAddScheduleId::MmaR48C112:
    case Q8LinearAddScheduleId::MmaR64C112:
        return 112;
    case Q8LinearAddScheduleId::MmaR32C128:
    case Q8LinearAddScheduleId::MmaR48C128:
    case Q8LinearAddScheduleId::MmaR64C128:
        return 128;
    case Q8LinearAddScheduleId::SplitKMmaExactT:
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
    case Q8LinearAddScheduleId::GroupedSplitK:
    case Q8LinearAddScheduleId::ColumnInvariant:
    case Q8LinearAddScheduleId::MmaR16C32K128:
    case Q8LinearAddScheduleId::MmaR32C32K128:
    case Q8LinearAddScheduleId::MmaR32C64K128:
        break;
    }
    throw std::logic_error("q8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* q8_linear_add_schedule_name(Q8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8LinearAddScheduleId::DecodeR16:
        return "linear_add.q8.decode.r16.residual";
    case Q8LinearAddScheduleId::GroupedSplitK:
        return "linear_add.q8.grouped_splitk.residual";
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        return "linear_add.q8.splitk.mma.capacity.residual";
    case Q8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.q8.splitk8.mma.r16.exact_t.residual";
    case Q8LinearAddScheduleId::SimtR8C4:
        return "linear_add.q8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case Q8LinearAddScheduleId::MmaR32C64:
        return "linear_add.q8.mma.r32.c64.residual";
    case Q8LinearAddScheduleId::MmaR32C80:
        return "linear_add.q8.mma.r32.c80.residual";
    case Q8LinearAddScheduleId::MmaR32C96:
        return "linear_add.q8.mma.r32.c96.residual";
    case Q8LinearAddScheduleId::MmaR32C128:
        return "linear_add.q8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR48C64:
        return "linear_add.q8.mma.r48.c64.residual";
    case Q8LinearAddScheduleId::MmaR48C96:
        return "linear_add.q8.mma.r48.c96.residual";
    case Q8LinearAddScheduleId::MmaR48C112:
        return "linear_add.q8.mma.r48.c112.residual";
    case Q8LinearAddScheduleId::MmaR48C128:
        return "linear_add.q8.mma.r48.c128.residual";
    case Q8LinearAddScheduleId::MmaR64C96:
        return "linear_add.q8.mma.r64.c96.residual";
    case Q8LinearAddScheduleId::MmaR64C112:
        return "linear_add.q8.mma.r64.c112.residual";
    case Q8LinearAddScheduleId::MmaR64C128:
        return "linear_add.q8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case Q8LinearAddScheduleId::MmaR128C64:
        return "linear_add.q8.mma.r128.c64.residual";
    case Q8LinearAddScheduleId::MmaR128C80:
        return "linear_add.q8.mma.r128.c80.residual";
    case Q8LinearAddScheduleId::MmaR16C32K128:
        return "linear_add.q8.mma.r16.c32.k128.residual";
    case Q8LinearAddScheduleId::MmaR32C32K128:
        return "linear_add.q8.mma.r32.c32.k128.residual";
    case Q8LinearAddScheduleId::MmaR32C64K128:
        return "linear_add.q8.mma.r32.c64.k128.residual";
    case Q8LinearAddScheduleId::ColumnInvariant:
        return "linear_add.q8.splitk16.mma.column_invariant.residual";
    }
    return "linear_add.q8.unknown";
}

bool q8_linear_add_schedule_uses_mma(Q8LinearAddScheduleId schedule) noexcept {
    return schedule != Q8LinearAddScheduleId::DecodeR16 &&
           schedule != Q8LinearAddScheduleId::SimtR8C4;
}

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept {
    const bool shape = (problem.rows == 2048 && (problem.k == 4096 || problem.k == 6144)) ||
                       (problem.rows == 5120 && (problem.k == 6144 || problem.k == 17408));
    return shape && problem.padded_k == problem.k && problem.cols >= 1;
}

Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem,
                                           bool column_invariant) {
    if (!q8_linear_add_admits(problem)) {
        throw std::invalid_argument("q8 linear_add: exact problem or column count is not admitted");
    }
    if (column_invariant) {
        if (problem.rows != 2048 || problem.k != 4096) {
            throw std::invalid_argument("q8 linear_add: the problem registers no column-invariant route");
        }
        return {Q8LinearAddScheduleId::ColumnInvariant};
    }
    const auto resolve_from = [&](const auto& routes) -> Q8LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("q8 linear_add: admitted problem has no covering route");
    };
    if (problem.rows == 5120) { return resolve_from(kN5120Routes); }
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
}

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream, bool column_invariant) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearAddPlan resolved = q8_linear_add_resolve_plan(problem, column_invariant);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == Q8LinearAddScheduleId::ColumnInvariant) {
        q8_linear_add_column_invariant_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::GroupedSplitK) {
        q8_linear_add_grouped_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaCapacity) {
        q8_linear_add_splitk_capacity_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::DecodeR16) {
        q8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::SplitKMmaExactT) {
        q8_linear_add_splitk_mma_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MmaR16C32K128) {
        q8_linear_add_mma_r16_c32_k128_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MmaR32C32K128) {
        q8_linear_add_mma_r32_c32_k128_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == Q8LinearAddScheduleId::MmaR32C64K128) {
        q8_linear_add_mma_r32_c64_k128_launch(x, w, residual_out, stream);
        return;
    }
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case Q8LinearAddScheduleId::DecodeR16:
                break;
            case Q8LinearAddScheduleId::SimtR8C4:
                q8_linear_add_simt_r8_c4_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C64:
                q8_linear_add_mma_r32_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C80:
                q8_linear_add_mma_r32_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C96:
                q8_linear_add_mma_r32_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR32C128:
                q8_linear_add_mma_r32_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C64:
                q8_linear_add_mma_r48_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C96:
                q8_linear_add_mma_r48_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C112:
                q8_linear_add_mma_r48_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR48C128:
                q8_linear_add_mma_r48_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C96:
                q8_linear_add_mma_r64_c96_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C112:
                q8_linear_add_mma_r64_c112_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR64C128:
                q8_linear_add_mma_r64_c128_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C64:
                q8_linear_add_mma_r128_c64_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::MmaR128C80:
                q8_linear_add_mma_r128_c80_launch(x_slice, w, residual_slice, stream);
                return;
            case Q8LinearAddScheduleId::GroupedSplitK:
            case Q8LinearAddScheduleId::SplitKMmaCapacity:
            case Q8LinearAddScheduleId::SplitKMmaExactT:
            case Q8LinearAddScheduleId::ColumnInvariant:
            case Q8LinearAddScheduleId::MmaR16C32K128:
            case Q8LinearAddScheduleId::MmaR32C32K128:
            case Q8LinearAddScheduleId::MmaR32C64K128:
                break;
            }
            throw std::logic_error("q8 linear_add: unknown tiled schedule");
        });
}

void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream, bool column_invariant) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_add_execute_plan(q8_linear_add_resolve_plan(problem, column_invariant), x, w,
                               residual_out, stream, column_invariant);
}

} // namespace infernix::ops::detail
