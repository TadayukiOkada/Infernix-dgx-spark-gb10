#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A4,
    A4Residual,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (column_invariant(policy)) { return Nvfp4LinearAddRoute::ColumnInvariantA16; }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_residual(policy)) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (allows_a4_residual(policy)) {
        // From 5 tokens: in warm single-request DFlash2 verification rounds of 5-8 columns (RTX
        // 5090, 2026-10-28) the two-term routes of the MLP saved 0.7-1.2 ms per round against A16,
        // and greedy agreement with the lossless reference stayed at A16's (98.7 % against 98.4 %).
        return tokens >= 5 ? Nvfp4LinearAddRoute::A4Residual : Nvfp4LinearAddRoute::A16;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    // GB10 (sm_121a, 48 SMs): re-measured with a cold-cache A16-vs-A4 sweep (bench/ops/nvfp4_linear_add_bench,
    // N=5120, warmup 12 / repeat 60). A4 wins from 8 tokens up at both K=6144 (81.7 us vs 86.9 us A16 at T=8,
    // 330.8 vs 1008.6 at T=1024) and K=17408. The threshold (8) carried over from the RTX 5090 sweep still
    // matches GB10, so the value is unchanged; the crossover was re-confirmed on 48 SMs rather than assumed.
    const std::int32_t first_a4 = 8;
    return tokens >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
}


} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    switch (resolve_route(output_rows, input_rows, policy, max_tokens)) {
    case Nvfp4LinearAddRoute::A4:
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows);
    case Nvfp4LinearAddRoute::A4Residual:
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows, kNvfp4A4ResidualTerms);
    case Nvfp4LinearAddRoute::ColumnInvariantA16:
    case Nvfp4LinearAddRoute::A16:
        break;
    }
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream, bool wide_down_tile) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::ColumnInvariantA16) {
        nvfp4_linear_add_invariant_launch(x, weight, residual, stream);
        return;
    }
    if (route == Nvfp4LinearAddRoute::A16) {
        // The faster 8 K-warp tiles change the reduction order: their fast path's route flag only.
        nvfp4_linear_add_a16_launch(x, weight, residual, wide_down_tile, stream);
        return;
    }
    auto scope = workspace.scope();
    if (route == Nvfp4LinearAddRoute::A4Residual) {
        nvfp4_linear_add_a4_residual_launch(
            x, weight, residual,
            allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k, kNvfp4A4ResidualTerms),
            stream);
        return;
    }
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}

} // namespace infernix::ops::detail
