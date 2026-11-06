#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A4,
};

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (column_invariant(policy)) { return Nvfp4GdnInputRoute::ColumnInvariantA16; }
    // The residual profiles are registered for the NVFP4 MLP and FP8 projections only; an A4
    // residual Use falls back to this projection's one-term A4 from prefill widths (A8 residual
    // has no 4-bit activation permission and stays on A16).
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_residual(policy)) {
        return Nvfp4GdnInputRoute::A16;
    }
    if (allows_a4(policy) || allows_a4_residual(policy)) {
        // GB10 (sm_121a, 48 SMs), 2026-10-31 (T7): the A16 dequant tiles stream this projection's
        // 42 MiB parent faster than the A4 lane until its quantizer is amortised. Cold-cache sweep
        // (infernix_gdn_input_proj_bench, warmup 3 / repeat 15): T=1 202.6 us A16 against 212.8
        // A4, T=4 207.7 against 213.8, T=8 208.8 against 213.8, T=16 211.8 against 215.9. Above
        // 16 the A16 tile table collapses (237.4 at T=17, 366.7 at T=31) while A4 holds ~216-219,
        // so the hand-over is exactly at 17.
        return tokens <= 16 ? Nvfp4GdnInputRoute::A16 : Nvfp4GdnInputRoute::A4;
    }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}


} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N16384K5120::kInputRows)
               : 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    const Nvfp4GdnInputRoute route = resolve_route(policy, x.ne[1]);
    if (route == Nvfp4GdnInputRoute::ColumnInvariantA16) {
        nvfp4_gdn_input_invariant_launch(x, weight, qkv, z, stream);
        return;
    }
    if (route == Nvfp4GdnInputRoute::A16) {
        nvfp4_gdn_input_a16_launch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 gdn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace infernix::ops::detail
