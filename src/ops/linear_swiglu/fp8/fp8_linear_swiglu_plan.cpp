#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Fp8LinearSwiGluRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A8,
    A8Residual,
};

Fp8LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    if (column_invariant(policy)) { return Fp8LinearSwiGluRoute::ColumnInvariantA16; }
    if (policy == LinearPolicy::A16Only || allows_a4_residual(policy)) {
        return Fp8LinearSwiGluRoute::A16;
    }
    if (allows_a8_residual(policy)) {
        // Two E4M3 terms stream the same bytes as A16 below 17 columns and run slower there
        // (111 against 107 us at 16, RTX 5090).
        return tokens >= 17 ? Fp8LinearSwiGluRoute::A8Residual : Fp8LinearSwiGluRoute::A16;
    }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    // GB10 (sm_121a, 48 SMs) crossover, re-measured with a cold-cache A16-vs-A8 sweep across token
    // widths (bench/ops/fp8_linear_swiglu_bench, warmup 12 / repeat 60). A16 wins the small-token band
    // (decode + MTP verification rounds, T=2-32) by 4-6%; A8 dominates from 34 up (e.g. T=1024:
    // 2242 us A8 vs 8106 us A16). The previous threshold (5) was fitted on the RTX 5090 (170 SMs) and
    // switched to A8 ~7x too early on 48 SMs, costing 4-6% across the 5-32 band. Keeping T<34 on A16
    // also preserves the dedicated A16 decode/small-T kernels for the graph-captured decode path.
    return tokens >= 34 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) return fp8_linear_swiglu_decode_launch(x, weight, out, stream);
    if (x.ne[1] <= 4) return fp8_linear_swiglu_small_t_launch(x, weight, out, stream);
    fp8_linear_swiglu_matrix_launch(x, weight, out, stream);
}

} // namespace

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    // Reserve scratch exactly when the dispatcher would take an A8 path for this interval's widest
    // token count. This must track resolve_route; a hardcoded A8 threshold here desyncs from the
    // router once the crossover moves, over-reserving scratch for the A16 small-token band and
    // breaking the query/execute workspace invariant.
    const auto max_route = resolve_route(policy, max_tokens);
    if (max_route == Fp8LinearSwiGluRoute::A8Residual) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N34816K5120::kInputRows, 0, true);
    }
    const bool interval_uses_a8 = max_route == Fp8LinearSwiGluRoute::A8;
    return interval_uses_a8 ? fp8_a8_workspace_capacity_bytes(
                                  max_tokens, Fp8N34816K5120::kInputRows,
                                  fp8_linear_swiglu_partial_capacity_bytes(max_tokens))
                            : 0;
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (weight.qtype == QType::FP8_E4M3FN_ROW_F32) {
        // The 122B shared expert: 16-bit activations under every policy.
        if (column_invariant(policy)) { return fp8_linear_swiglu_n2048_invariant_launch(x, weight, out, stream); }
        return fp8_linear_swiglu_n2048_a16_launch(x, weight, out, stream);
    }
    const Fp8LinearSwiGluRoute route = resolve_route(policy, x.ne[1]);
    if (route == Fp8LinearSwiGluRoute::ColumnInvariantA16) {
        fp8_linear_swiglu_invariant_launch(x, weight, out, stream);
        return;
    }
    if (route == Fp8LinearSwiGluRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (route == Fp8LinearSwiGluRoute::A8Residual) {
        fp8_linear_swiglu_a8_residual_launch(x, weight, out, workspace, stream);
        return;
    }
    fp8_linear_swiglu_a8_launch(x, weight, out, workspace, stream);
}

} // namespace infernix::ops::detail
