// The MTP draft round's host policy (models/qwen4_exp/program/mtp_gating.h): the cumulative cut and
// the executed-draft-step count it reports (what a gated chain paid for), and the copy-versus-chain
// choice that lets a round the cut truncated still verify the n-gram proposal it declined before the
// drafter ran. Every expectation is checked against a direct evaluation of the same rule, and the cut
// against a simulation of the device chain (core/decode_graph.h's launch_graph_gate_product), which
// must run exactly the steps the host believes were paid for. The cut step is never verified, even
// though a gated chain ran it — retaining it was measured net-negative on GB10 (mtp_gating.h), so
// the kept count stopping at the cut is itself a regression guard.

#include "models/qwen4_exp/program/mtp_gating.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::GraphGateBins;
using infernix::graph_gate_product;
using infernix::graph_gate_rate;
using infernix::models::qwen4_exp::copy_beats_chain;
using infernix::models::qwen4_exp::expected_chain_tokens;
using infernix::models::qwen4_exp::expected_copy_tokens;
using infernix::models::qwen4_exp::kMtpCumulativeBins;
using infernix::models::qwen4_exp::kMtpCumulativeBase;
using infernix::models::qwen4_exp::kMtpCumulativeFloor;
using infernix::models::qwen4_exp::kMtpSampledCumulativeBins;
using infernix::models::qwen4_exp::kMtpSampledCumulativeBase;
using infernix::models::qwen4_exp::kMtpSampledCumulativeFloor;
using infernix::models::qwen4_exp::MtpCumulative;
using infernix::models::qwen4_exp::MtpDraftCut;
using infernix::models::qwen4_exp::mtp_cumulative_cut;
using infernix::models::qwen4_exp::mtp_min_logprob_cut;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

void require_cut(const MtpDraftCut& cut, std::int32_t kept, std::int32_t executed, const std::string& what) {
    require(cut.kept == kept && cut.executed == executed,
            what + " (kept " + std::to_string(cut.kept) + ", executed " + std::to_string(cut.executed) +
                "; expected " + std::to_string(kept) + ", " + std::to_string(executed) + ")");
}

// One log-probability per bin of the shipped table; the rates the tests quote are those bins' rates.
constexpr float kRateFloorLogprob = -3.00F;  //  0.12
constexpr float kRateMidLogprob   = -0.30F;  //  0.51
constexpr float kRateGoodLogprob  = -0.10F;  //  0.66
constexpr float kRateTopLogprob   = -0.005F; //  0.96

// The chain the device runs: body j runs, and body j + 1 only while steps 0..j's product clears
// `threshold` (launch_graph_gate_product clears the conditional handle below it). Step i's value sits
// at values[i * stride], the draft round's row pitch.
std::int32_t simulated_device_steps(const float* values, std::int32_t stride, std::int32_t steps, float threshold,
                                    const GraphGateBins& bins) {
    std::int32_t ran = 0;
    for (std::int32_t j = 0; j < steps; ++j) {
        ++ran;
        if (graph_gate_product(values, stride, j, bins) < threshold) { break; }
    }
    return ran;
}

// The first j whose chain product falls below `threshold`, or `steps` when none does.
std::int32_t brute_cut(const float* values, std::int32_t stride, std::int32_t steps, float threshold,
                       const GraphGateBins& bins) {
    for (std::int32_t j = 0; j < steps; ++j) {
        if (graph_gate_product(values, stride, j, bins) < threshold) { return j; }
    }
    return steps;
}

// Expected tokens of verifying `count` positions that each survive with probability `p`.
double brute_copy_tokens(std::int32_t count, double p) {
    double tokens = 1.0, reach = 1.0;
    for (std::int32_t j = 0; j < count; ++j) {
        reach *= p;
        tokens += reach;
    }
    return tokens;
}

struct Rng {
    std::uint32_t state = 12345U;
    float next_logprob() {
        state = state * 1664525U + 1013904223U;
        return -6.0F + 6.5F * (static_cast<float>(state >> 8) / static_cast<float>(1U << 24));
    }
    double next_unit() {
        state = state * 1664525U + 1013904223U;
        return static_cast<double>(state >> 8) / static_cast<double>(1U << 24);
    }
};

// The shipped table, read from the header: the rates the tests below quote really are its rates.
void check_table() {
    require(graph_gate_rate(kRateFloorLogprob, kMtpCumulativeBins) == 0.12F, "the cheapest bin is 0.12");
    require(graph_gate_rate(kRateMidLogprob, kMtpCumulativeBins) == 0.51F, "the middle bin is 0.51");
    require(graph_gate_rate(kRateGoodLogprob, kMtpCumulativeBins) == 0.66F, "the good bin is 0.66");
    require(graph_gate_rate(kRateTopLogprob, kMtpCumulativeBins) == 0.96F, "the top bin is 0.96");
    require(kMtpCumulativeBase == 0.40F, "the continuation threshold starts at 0.40");
}

// Where the continuation cut lands, and which steps a gated chain paid for. The cut step is dropped
// even though it ran: its acceptance is conditioned on the chain being cut there and was measured
// net-negative to verify (mtp_gating.h) — kept stopping at the cut is the regression guard.
void check_continuation_cut() {
    // 0.66, then 0.66 * 0.66 = 0.4356, then 0.2875: the gate stops the chain on step 2, which ran but
    // is not verified, so the round verifies steps 0..1.
    const std::vector<float> chain{kRateGoodLogprob, kRateGoodLogprob, kRateGoodLogprob, kRateGoodLogprob};
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, true), 2, 3,
                "a gated chain verifies up to but not including its cut step");
    // The same chain on an ungated round (multi-row, single-step or sampled): every drafted step ran,
    // so the executed count is the drafted count even though the host cuts at the same place.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 2, 4,
                "an ungated chain ran every step it drafted");
    // The request-adapted threshold (the 0.25 floor) moves the cut later: 0.66 to the 4th is 0.19, so
    // the chain is cut on step 3 and verifies three drafts.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.25F, kMtpCumulativeBins, true), 3, 4,
                "the adapted threshold cuts one step later");
    // A threshold above every product cuts at step 0, which always ran and is never verified.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.9F, kMtpCumulativeBins, true), 0, 1,
                "a chain cut on its first step verifies nothing");
    require_cut(mtp_cumulative_cut(chain.data(), 1, 4, 0.9F, kMtpCumulativeBins, false), 0, 4,
                "an ungated chain cut on its first step ran them all");
    // No cut at all: everything drafted is verified and ran.
    const std::vector<float> sure{kRateTopLogprob, kRateTopLogprob, kRateTopLogprob, kRateTopLogprob};
    require_cut(mtp_cumulative_cut(sure.data(), 1, 4, kMtpCumulativeBase, kMtpCumulativeBins, true), 4, 4,
                "an uncut chain keeps every draft");
    // A collapsed prefix cuts just as surely: 0.51, then 0.51 * 0.12 = 0.0612.
    const std::vector<float> collapsed{kRateMidLogprob, kRateFloorLogprob};
    require_cut(mtp_cumulative_cut(collapsed.data(), 1, 2, kMtpCumulativeBase, kMtpCumulativeBins, true), 1, 2,
                "a chain cut on a collapsed prefix keeps only the step before it");
    // An empty round drafts nothing.
    require_cut(mtp_cumulative_cut(chain.data(), 1, 0, kMtpCumulativeBase, kMtpCumulativeBins, true), 0, 0,
                "no steps, no drafts");
}

// The fixed --spec-min-p cut: an explicit floor on what a round may verify; the executed count still
// follows whether the device gated the chain.
void check_min_logprob_cut() {
    const std::vector<float> chain{-0.1F, -0.2F, -0.9F, -0.1F};
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -0.5F, true), 2, 3, "the fixed cut drops the floor draft");
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -0.5F, false), 2, 4, "an ungated fixed cut ran every step");
    require_cut(mtp_min_logprob_cut(chain.data(), 1, 4, -1.0F, true), 4, 4, "a floor below every draft cuts nothing");
    const std::vector<float> all_below{-0.6F, -0.6F};
    require_cut(mtp_min_logprob_cut(all_below.data(), 1, 2, -0.5F, true), 0, 1, "a floor above every draft cuts all");
}

// The rows of a multi-row round read their own drafts through the round's stride.
void check_strided_rows() {
    const std::vector<float> logprobs{// step-major: [row0 step0, row1 step0, row0 step1, ...]
                                      kRateGoodLogprob, kRateTopLogprob, kRateGoodLogprob, kRateTopLogprob,
                                      kRateGoodLogprob, kRateTopLogprob, kRateGoodLogprob, kRateTopLogprob};
    require_cut(mtp_cumulative_cut(logprobs.data(), 2, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 2, 4,
                "row 0 reads its own chain");
    require_cut(mtp_cumulative_cut(logprobs.data() + 1, 2, 4, kMtpCumulativeBase, kMtpCumulativeBins, false), 4, 4,
                "row 1 reads its own chain");
}

// The cut against a direct evaluation and against the device chain, over random chains and thresholds.
void check_against_device_chain() {
    Rng rng;
    for (int trial = 0; trial < 20000; ++trial) {
        const std::int32_t steps  = 1 + trial % 8;
        const std::int32_t rows   = 1 + trial % 3;
        std::vector<float> logprobs(static_cast<std::size_t>(steps) * rows);
        for (float& v : logprobs) { v = rng.next_logprob(); }
        const float threshold = 0.05F + 0.9F * static_cast<float>(rng.next_unit());
        const bool gated      = trial % 2 == 0;
        for (std::int32_t row = 0; row < rows; ++row) {
            const float* values       = logprobs.data() + row;
            const std::int32_t stride = rows;
            const auto cut            = mtp_cumulative_cut(values, stride, steps, threshold, kMtpCumulativeBins, gated);
            const std::string at      = "trial " + std::to_string(trial) + " row " + std::to_string(row);
            const std::int32_t first  = brute_cut(values, stride, steps, threshold, kMtpCumulativeBins);
            require(cut.kept >= 0 && cut.kept <= cut.executed && cut.executed <= steps, at + ": bounds");
            require(cut.kept == first, at + ": the cut drops its own step — nothing past the cut is verified");
            require(cut.executed == (gated ? std::min(first + 1, steps) : steps), at + ": the executed count");
            // Only a one-row round runs under the device gate (mtp_draft_gated); there the host's
            // executed count must be exactly the steps its conditional chain ran.
            if (gated && rows == 1) {
                require(cut.executed == simulated_device_steps(values, stride, steps, threshold, kMtpCumulativeBins),
                        at + ": the device chain ran that many steps");
            }
        }
    }
}

// A truncated round still verifies a copy: the chain side of the comparison is the kept prefix, so a
// row the cut left with nothing takes any copy whose own acceptance beats its column cost.
void check_copy_fallback() {
    const std::array<double, 8> good{0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95};
    const std::array<double, 8> poor{0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30};
    constexpr double kCopyCost = 0.12, kChainCost = 0.28;

    // A row the cut left with no draft: its chain is worth the one target token, so a copy of three at
    // 0.75 (2.73 tokens for 1.36 rounds) is verified instead of the round decaying to a plain one.
    require(copy_beats_chain(3, 0.75, kCopyCost, good, 0, kChainCost), "an empty chain loses to a good copy");
    require(copy_beats_chain(1, 0.75, kCopyCost, good, 0, kChainCost), "even a one-token copy beats no draft");
    // Break-even: one copy at exactly its column cost earns nothing over the plain round.
    require(!copy_beats_chain(1, kCopyCost, kCopyCost, good, 0, kChainCost), "a copy at its column cost is not taken");
    require(!copy_beats_chain(4, 0.05, kCopyCost, good, 0, kChainCost), "a copy nobody accepts is not taken");
    // A chain the cut left intact still wins against a poor copy, and a good copy still wins against a
    // chain whose own acceptance is poor.
    require(!copy_beats_chain(2, 0.4, kCopyCost, good, 4, kChainCost), "a whole good chain beats a poor copy");
    require(copy_beats_chain(4, 0.9, kCopyCost, poor, 4, kChainCost), "a good copy beats a poor chain");
    // The post-cut comparison is the pre-drafter one over the kept prefix: as the cut keeps more of a
    // good chain, the round stops preferring the same five-token copy at 0.7.
    for (std::int32_t drafts = 0; drafts <= 8; ++drafts) {
        const bool take = copy_beats_chain(5, 0.7, kCopyCost, good, drafts, kChainCost);
        require(take == (drafts <= 2), "a five-token copy wins only while the chain is short (drafts " +
                                           std::to_string(drafts) + ")");
    }
    // A copy longer than the round's whole chain is taken whenever it earns more per round time, which
    // is the same rule before the drafter ran and after its cut.
    require(copy_beats_chain(8, 0.9, kCopyCost, good, 2, kChainCost), "a long good copy beats a short chain");
    require(!copy_beats_chain(8, 0.2, kCopyCost, good, 2, kChainCost), "a long poor copy does not");

    // Acceptance-floored best_copy_length: below the floor, no copy drafts are admitted; above the
    // floor, length is capped so expected acceptance stays >= min_accept while high-confidence
    // copies still reach the full window.
    using infernix::models::qwen4_exp::best_copy_length;
    require(best_copy_length(31, 0.55, kCopyCost, 0.60) == 0, "below min_accept floor yields zero drafts");
    const std::int32_t mid = best_copy_length(31, 0.72, kCopyCost, 0.60);
    require(mid >= 1 && mid <= 3, "moderate prior probes short instead of drafting 10+ tokens");
    const std::int32_t high = best_copy_length(31, 0.985, kCopyCost, 0.60);
    require(high >= 24, "confirmed high-confidence copy still expands to a wide window");
    require(best_copy_length(31, 0.995, kCopyCost, 0.60) == 31,
            "near-exact copy reaches the full 31-token window");
}

// The two expected-token helpers against a direct evaluation.
void check_expected_tokens() {
    Rng rng;
    for (int trial = 0; trial < 2000; ++trial) {
        const std::int32_t count = trial % 9;
        const double p           = rng.next_unit();
        require(std::abs(expected_copy_tokens(count, p) - brute_copy_tokens(count, p)) < 1e-12, "copy tokens");
        std::array<double, 8> accept{};
        for (double& a : accept) { a = rng.next_unit(); }
        double expect = 1.0, reach = 1.0;
        for (std::int32_t j = 0; j < count; ++j) {
            reach *= accept[static_cast<std::size_t>(j)];
            expect += reach;
        }
        require(std::abs(expected_chain_tokens(std::span<const double>(accept), count) - expect) < 1e-12, "chain tokens");
        // A chain longer than the learned positions counts only the positions it has.
        require(std::abs(expected_chain_tokens(std::span<const double>(accept), 12) -
                         expected_chain_tokens(std::span<const double>(accept.data(), 8), 8)) < 1e-12,
                "a chain counts only the acceptances it has");
    }
}

void check_sampled_cumulative_table() {
    for (std::size_t i = 0; i < std::size(kMtpSampledCumulativeBins.edges); ++i) {
        if (i > 0) {
            require(kMtpSampledCumulativeBins.edges[i - 1] < kMtpSampledCumulativeBins.edges[i],
                    "sampled cumulative bin edges must strictly increase");
        }
        require(kMtpSampledCumulativeBins.rates[i] < kMtpSampledCumulativeBins.rates[i + 1],
                "sampled cumulative bin rates must strictly increase");
    }
    for (const float r : kMtpSampledCumulativeBins.rates) {
        require(r > 0.0F && r <= 1.0F, "sampled cumulative bin rates must lie in (0, 1]");
    }
    require(kMtpSampledCumulativeFloor < kMtpSampledCumulativeBase,
            "sampled cumulative floor must sit below base");
    MtpCumulative cum;
    cum.reset(kMtpSampledCumulativeBase);
    require(cum.threshold == kMtpSampledCumulativeBase, "reset sets sampled base threshold");
    // Feed perfect acceptance across moderate-confidence chains -> threshold lowers smoothly toward floor.
    const std::array<float, 4> mod_lp{-0.8F, -0.8F, -0.8F, -0.8F};
    for (int round = 0; round < 64; ++round) {
        cum.observe(mod_lp.data(), 4, 4, kMtpSampledCumulativeBins, kMtpSampledCumulativeBase,
                    kMtpSampledCumulativeFloor);
    }
    require(cum.threshold >= kMtpSampledCumulativeFloor && cum.threshold < kMtpSampledCumulativeBase,
            "high observed/expected acceptance lowers sampled threshold toward floor");
    // Check host/device cut simulation agreement over kMtpSampledCumulativeBins.
    Rng rng;
    for (int trial = 0; trial < 5000; ++trial) {
        const std::int32_t steps = 1 + trial % 8;
        std::vector<float> lps(static_cast<std::size_t>(steps));
        for (float& v : lps) { v = rng.next_logprob(); }
        const float threshold = kMtpSampledCumulativeFloor +
                                0.5F * static_cast<float>(rng.next_unit());
        const bool gated = (trial % 2) == 0;
        const auto cut   = mtp_cumulative_cut(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins, gated);
        require(cut.kept == brute_cut(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins),
                "sampled cumulative kept matches brute cut");
        if (gated) {
            require(cut.executed == simulated_device_steps(lps.data(), 1, steps, threshold, kMtpSampledCumulativeBins),
                    "sampled cumulative executed matches device gate simulation");
        }
    }
}

} // namespace

int main() {
    try {
        check_table();
        check_continuation_cut();
        check_min_logprob_cut();
        check_strided_rows();
        check_against_device_chain();
        check_copy_fallback();
        check_expected_tokens();
        check_sampled_cumulative_table();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("mtp round gating checks passed\n");
    return 0;
}
