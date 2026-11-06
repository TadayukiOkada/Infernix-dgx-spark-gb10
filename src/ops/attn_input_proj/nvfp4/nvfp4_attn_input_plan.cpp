#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (column_invariant(policy)) { return Nvfp4AttnInputRoute::ColumnInvariantA16; }
    // The residual profiles are registered for the NVFP4 MLP and FP8 projections only; an A4
    // residual Use falls back to this projection's one-term A4 from prefill widths (A8 residual
    // has no 4-bit activation permission and stays on A16).
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_residual(policy)) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy) && !allows_a4_residual(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    // GB10 (sm_121a, 48 SMs), 2026-11-06 (T7): the RowTiles-2 16-token sliced-K A16 tile (see
    // nvfp4_attn_input_a16.cu) beats the fused A4 lane across the whole MTP/DFlash2 verification
    // band. Cold-cache interleaved A/B on this fused [14336,5120] op (forced A16 vs the A4 lane,
    // warmup 5 / repeat 30): T=4 179.2 vs 185.0, T=6 179.0 vs 186.5, T=12 183.2 vs 185.2,
    // T=16 180.0 vs 186.2 us; the hand-over is 17, where A16's tile table collapses
    // (T=17 194.4, T=24 196.3 against A4's 186.2/186.2). Staying on A16 through T=16 also keeps
    // the draft (T=1) and verify (T=5..16) widths on the same 16-bit activation semantics.
    return tokens >= 17 ? Nvfp4AttnInputRoute::A4 : Nvfp4AttnInputRoute::A16;
}


} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    const Nvfp4AttnInputRoute route = resolve_route(policy, x.ne[1]);
    if (route == Nvfp4AttnInputRoute::ColumnInvariantA16) {
        nvfp4_attn_input_invariant_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (route == Nvfp4AttnInputRoute::A16) {
        nvfp4_attn_input_a16_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 attn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace infernix::ops::detail
