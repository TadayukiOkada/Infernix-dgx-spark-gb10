#include "core/weight.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {

bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept {
    return tokens >= kSparseMoeSmallTMin && tokens <= kSparseMoeSmallTMax;
}

std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t tokens) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: tokens must be in [2,46]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_small_t_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

SparseMoeSmallTPlan resolve_sparse_moe_small_t_plan(std::int32_t tokens, QType routed_gate_up,
                                                    QType routed_down, bool column_invariant,
                                                    const SparseMoeHints& hints) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
    const bool main_profile =
        routed_gate_up == QType::Q4_G64_FP16 &&
        (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16);
    const bool mtp_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    if (!main_profile && !mtp_profile) {
        throw std::invalid_argument("sparse_moe small-T: unsupported routed codec profile");
    }

    SparseMoeSmallTPlan plan{tokens, sparse_moe_small_t_workspace_bytes(tokens)};
    plan.column_invariant = column_invariant;
    // The next consumer's warm at T = 1's limit (8 MiB).
    if (hints.next_weight_prefetch != nullptr) {
        constexpr std::size_t kNextWeightPrefetchLimit = std::size_t{8} << 20;
        plan.next_weight_prefetch       = hints.next_weight_prefetch;
        plan.next_weight_prefetch_bytes = std::min(hints.next_weight_prefetch_bytes, kNextWeightPrefetchLimit);
    }
    if (mtp_profile) {
        // GB10 retune (2026-10-14, cold-eager candidate sweep over all 9 D3xD4 lanes,
        // warmup10 repeat60, seed 20260718, distribution trace-like). The HEAD cutoffs here
        // (Paths1 only <=5, Rows4 above 8) were inherited from RTX 5090 tuning and misroute on
        // GB10: paths1.rows1 is the argmin at EVERY token in [2,46] except a single 2 % island at
        // T=13 (paths1.rows4), with runner-up gaps of 1.0-4.1 %. Routing T>=9 to paths9.rows4 (the
        // old HEAD choice) costs 4-6 % across the whole upper band and 26 % at T=10. On GB10 the
        // 48-SM part stays latency-bound far longer, so the single-path/one-row lane dominates the
        // entire small-T domain. Committing Paths1.rows1 uniformly; the T=13 island is within the
        // 2-3 % noise floor and not reproduced as a band, so it is ignored.
        plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths1;
        plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows1;
        return plan;
    }

    // GB10 retune (2026-10-14, same cold-eager sweep). HEAD forced Paths3 for the whole main
    // profile and picked Rows purely by a 5090-tuned crossover. On GB10 the measured lane medians
    // show paths9.rows2 beating HEAD's paths3.rows4 on EVERY token from T=18..33 for both Q5 and
    // Q6 (gains 1.7-2.8 %, consistent across all 16 tokens - a structural win, not noise), because
    // the wider 9-path fan-out amortises the gather once the grid is large enough that Rows4's
    // extra L1 re-read no longer buys parallelism. Below that, HEAD's paths3.rows2 lane stays the
    // argmin (or within noise) down to T=3, so the low band is left unchanged. In the top band
    // (T>=34) paths9.rows4 only edges paths9.rows2 by sub-noise islands, so Rows2 is kept.
    //
    // Rows is only a grid split: (kHidden / Rows, tokens). Rows2 runs twice the blocks, so it has
    // twice the parallelism to cover memory latency while the grid is small, and it re-reads the
    // same per-expert activations in every block (+63 % L1 traffic, L2 flat). The 9-path lane wins
    // once that re-read cost stops buying coverage - which on GB10 happens earlier (T>=18) than on
    // the higher-clocked 5090 lane HEAD was tuned for.
    if (routed_down == QType::Q5_G64_FP16) {
        // Q5@T=2 is a clear outlier: paths9.rows1 beats HEAD's paths3.rows1 by 17.5 % (31 us gap,
        // far above the noise floor), so it gets its own exception. Q6@T=2 keeps paths3.rows1.
        if (tokens == 2) {
            plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths9;
            plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows1;
        } else if (tokens <= 17) {
            plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
            plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows2;
        } else {
            plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths9;
            plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows2;
        }
    } else {
        if (tokens <= 17) {
            plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
            plan.d4_schedule = tokens <= 2 ? SparseMoeSmallTD4Schedule::Rows1
                                           : SparseMoeSmallTD4Schedule::Rows2;
        } else {
            plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths9;
            plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows2;
        }
    }
    return plan;
}

bool sparse_moe_small_t_candidate_is_legal(std::int32_t tokens, QType routed_gate_up,
                                           QType routed_down, SparseMoeSmallTD3Schedule d3_schedule,
                                           SparseMoeSmallTD4Schedule d4_schedule) noexcept {
    if (!sparse_moe_uses_small_t(tokens)) { return false; }
    const bool main_profile =
        routed_gate_up == QType::Q4_G64_FP16 &&
        (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16);
    const bool mtp_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    if (!main_profile && !mtp_profile) { return false; }
    // Every D3 path width (1/3/9) and every D4 row width (1/2/4) is instantiated for
    // both codec families, so any pair is a legal candidate within the small-T domain.
    (void)d3_schedule;
    (void)d4_schedule;
    return true;
}

void sparse_moe_small_t_execute_candidate(const Tensor& x, const SparseMoeWeights& weights,
                                          Tensor& destination, std::int32_t tokens,
                                          SparseMoeSmallTD3Schedule d3_schedule,
                                          SparseMoeSmallTD4Schedule d4_schedule,
                                          const SparseMoeSmallTWorkspace& workspace,
                                          cudaStream_t stream) {
    SparseMoeSmallTPlan plan{tokens, sparse_moe_small_t_workspace_bytes(tokens), d3_schedule,
                             d4_schedule};
    sparse_moe_small_t_launch(x, weights, destination, plan, workspace, stream);
}

} // namespace infernix::ops::detail
