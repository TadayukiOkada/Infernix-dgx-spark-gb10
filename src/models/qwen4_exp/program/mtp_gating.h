#pragma once

// The host side of a Qwen4Exp MTP draft round: which of a drafted chain the round verifies, how many
// draft steps the device actually ran, and whether an n-gram copy proposal is worth more per round
// time than the neural chain. Pure policy over the drafter's stored log-probabilities and the lane's
// learned per-position acceptances, so it is testable without a model.
//
// The continuation gate and the verify cut are two different decisions, and the token the gate stops
// on is not verified. The device chain (core/decode_graph.h's launch_graph_gate_product) stops after
// step j when the cumulative product of steps 0..j's binned rates falls below the round's threshold,
// so step j ran and its token is sunk; verifying it looked free, since one more column of a verify
// round costs a fraction of a plain round on GB10 (the [16384, 2560] NVFP4 GEMV is 104.3 us at T = 1
// against 106.3 us at T = 4). Measured on the W4Dense NVFP4 artifact it is not (2026-11-01, T6): the
// cut-position draft is accepted ~1-2% of the time, not the 12-26% its bin claims — the gate cuts
// exactly where the drafter's probability is overconfident — while every cut round pays that column's
// share of the verify path and every kept-0 round stops decaying to a plain one. Four interleaved
// ABBA passes with retention on: cold decode -2.1% (tg128) to -3.6% (tg256), warm -1.7%/-2.8%; before
// the cut-position outcome was excluded from the draft-length learner, -9.1%. The cut therefore drops
// the step it lands on, as before; the executed count below still reports what the round paid for.

#include "core/decode_graph.h"

#include <algorithm>
#include <cstdint>
#include <span>

namespace infernix::models::qwen4_exp {

// The cumulative cut of an argmax MTP chain (candidates §7 D8; Basalt draft_cum.hpp): the chain keeps
// draft j while the product over i <= j of the observed acceptance rate of draft i's probability bin
// stays at or above the request's threshold. The bins are Basalt's, as log-probability edges (the
// drafter's stored values, compared without exp, so host and device classify alike:
// core/decode_graph.h). The threshold starts at kMtpCumulativeBase and falls toward 0.25 as the
// request's drafts are accepted more often than the table predicts (observed / expected accepted,
// decayed 0.97 per round, mapped linearly over 0.9..1.1). Draft-only: verification and outputs are
// unchanged. It is the cut of `--draft-tokens auto` (Dense8 MTP auto, ABBA x5: conv 1.0529x, 5/5, worst 1.0296;
// struct 1.0009x); a fixed draft length keeps the --spec-min-p cut (MTP-3 under this cut: conv 1.0423x, struct
// 0.9919x, worse in every pair; 2026-10-28).
inline constexpr GraphGateBins kMtpCumulativeBins{
    {-1.20397280F, -0.693147181F, -0.356674944F, -0.223143551F, -0.105360516F, -0.0512932944F, -0.0100503359F},
    {0.12F, 0.26F, 0.43F, 0.51F, 0.59F, 0.66F, 0.79F, 0.96F}};
inline constexpr float kMtpCumulativeBase = 0.35F;

struct MtpCumulative {
    float threshold = kMtpCumulativeBase;
    double observed = 0.0, expected = 0.0;

    void reset() { *this = MtpCumulative{}; }
    // One verified chain: `offered` drafts with these log-probabilities, `accepted` of them taken.
    void observe(const float* logprobs, std::int32_t offered, std::int32_t accepted) {
        if (offered <= 0) { return; }
        constexpr double kDecay = 0.97, kLowRatio = 0.9, kHighRatio = 1.1;
        constexpr float kFloor  = 0.25F;
        double chance = 1.0, sum = 0.0;
        for (std::int32_t j = 0; j < offered; ++j) {
            chance *= static_cast<double>(graph_gate_rate(logprobs[j], kMtpCumulativeBins));
            sum += chance;
        }
        observed = observed * kDecay + static_cast<double>(accepted);
        expected = expected * kDecay + sum;
        if (expected < 0.5) { return; }
        const double t = std::clamp((observed / expected - kLowRatio) / (kHighRatio - kLowRatio), 0.0, 1.0);
        threshold      = kMtpCumulativeBase + static_cast<float>(t) * (kFloor - kMtpCumulativeBase);
    }
};

// A draft round's readout: the drafts the round verifies and the draft steps the device ran. They
// differ whenever a gated chain stopped early — the cut step ran but is not verified (header note),
// and the drafts beyond `executed` were never written and must not be read.
struct MtpDraftCut {
    std::int32_t kept     = 0;
    std::int32_t executed = 0;
};

// The cumulative cut of one row's chain: `steps` drafts at logprobs[j * stride], cut at the first j
// whose product over i <= j falls below `threshold` — the device gate's comparison and its binary32
// multiplications in the same order, so host and device agree on every chain. `gated` says whether the
// device ran the chain under that gate (mtp_draft_gated): there step j ran exactly when every earlier
// gate passed, so the chain's last executed step is the one it was cut on; an ungated chain ran every
// step and only the host cuts. The cut step itself is not verified even though its cost is sunk: see
// the header note for the GB10 measurement.
[[nodiscard]] inline MtpDraftCut mtp_cumulative_cut(const float* logprobs, std::int32_t stride, std::int32_t steps,
                                                    float threshold, const GraphGateBins& bins, bool gated) noexcept {
    MtpDraftCut cut{steps, steps};
    for (std::int32_t j = 0; j < steps; ++j) {
        const float product = graph_gate_product(logprobs, stride, j, bins);
        if (product >= threshold) { continue; }
        cut.executed = gated ? j + 1 : steps;
        cut.kept     = j;
        break;
    }
    return cut;
}

// The fixed-length cut of one row's chain (--spec-min-p): cut at the first draft below
// `min_logprob`, which is also the device gate's comparison. There the threshold is an explicit floor
// on what a round may verify, not only on what it may keep drafting; the executed count is still
// reported, as the round paid for the cut step either way.
[[nodiscard]] inline MtpDraftCut mtp_min_logprob_cut(const float* logprobs, std::int32_t stride, std::int32_t steps,
                                                     float min_logprob, bool gated) noexcept {
    MtpDraftCut cut{steps, steps};
    for (std::int32_t j = 0; j < steps; ++j) {
        if (logprobs[j * stride] >= min_logprob) { continue; }
        cut.executed = gated ? j + 1 : steps;
        cut.kept     = j;
        break;
    }
    return cut;
}

// Expected tokens a round commits by verifying `count` drafts whose per-position acceptance is
// `accept`: the target's own token, then each draft whose whole prefix was accepted.
[[nodiscard]] inline double expected_copy_tokens(std::int32_t count, double accept) noexcept {
    double reach = 1.0, tokens = 1.0;
    for (std::int32_t j = 0; j < count; ++j) {
        reach *= accept;
        tokens += reach;
    }
    return tokens;
}

// The same for a neural chain: `drafts` drafts with the lane's learned per-position acceptances
// accept[0..drafts).
[[nodiscard]] inline double expected_chain_tokens(std::span<const double> accept, std::int32_t drafts) noexcept {
    double reach = 1.0, tokens = 1.0;
    const auto limit = std::min(drafts, static_cast<std::int32_t>(accept.size()));
    for (std::int32_t j = 0; j < limit; ++j) {
        reach *= accept[static_cast<std::size_t>(j)];
        tokens += reach;
    }
    return tokens;
}

// Whether a round verifies an n-gram copy rather than a neural chain: expected tokens per round time,
// the copy's columns costing `copy_cost` and the chain's drafts `chain_cost` plain rounds each (a copy
// needs no draft step). The chain may be shorter than the drafts the lane drafted — after the round's
// cut, only its kept prefix still counts.
[[nodiscard]] inline bool copy_beats_chain(std::int32_t copies, double copy_accept, double copy_cost,
                                           std::span<const double> chain_accept, std::int32_t chain_drafts,
                                           double chain_cost) noexcept {
    const double copy_score =
        expected_copy_tokens(copies, copy_accept) / (1.0 + copy_cost * static_cast<double>(copies));
    const double chain_score =
        expected_chain_tokens(chain_accept, chain_drafts) / (1.0 + chain_cost * static_cast<double>(chain_drafts));
    return copy_score > chain_score;
}

} // namespace infernix::models::qwen4_exp
