// speculative_sample_sparse_drafts (include/infernix/ops/speculative_round.h): one thread per column.

#include "ops/launcher/speculative_round.h"

#include "core/device.h"
#include "infernix/ops/speculative_round.h"
#include "ops/kernel/sampling_device.cuh"

namespace infernix::ops::detail {
namespace {

constexpr int kCandidates = kSparseSpeculativeCandidates;

__global__ void speculative_sample_sparse_drafts_kernel(const std::int32_t* rows, const float* logprobs,
                                                        const std::int32_t* token_ids,
                                                        const SamplingConfig* configs,
                                                        const std::int32_t* positions, unsigned int draw,
                                                        const std::int32_t* prior, int prior_steps, int columns,
                                                        int count, std::int32_t* drafts, std::int32_t* candidates,
                                                        float* proposal_q, float* confidence) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= columns) { return; }
    const SamplingConfig cfg = configs[t];
    const std::int32_t* row  = rows + t * count;
    const float* lp          = logprobs + t * count;
    std::int32_t* cand       = candidates + t * kCandidates;
    float* q                 = proposal_q + t * kCandidates;
    for (int j = 0; j < count; ++j) {
        cand[j] = token_ids != nullptr ? token_ids[row[j]] : row[j];
        q[j]    = 0.0F;
    }
    // Unused slots: the smallest ids not already candidates.
    int next = 0;
    for (int j = count; j < kCandidates; ++j) {
        bool taken = true;
        while (taken) {
            taken = false;
            for (int i = 0; i < j; ++i) { taken |= cand[i] == next; }
            if (taken) { ++next; }
        }
        cand[j] = next++;
        q[j]    = 0.0F;
    }
    const bool penalized =
        (cfg.token_counts != nullptr || prior_steps > 0) &&
        (cfg.presence_penalty != 0.0F || cfg.frequency_penalty != 0.0F);
    float top_lp = lp[0];
    if (penalized) {
        const float lp0     = lp[0];
        float orig_prob_sum = 0.0F;
        float pen_prob_sum  = 0.0F;
        for (int j = 0; j < count; ++j) {
            const int id    = cand[j];
            std::int32_t c  = (cfg.token_counts != nullptr && id >= 0) ? cfg.token_counts[id] : 0;
            for (int p = 0; p < prior_steps; ++p) {
                if (prior[p * columns + t] == id) { ++c; }
            }
            const float adj =
                lp[j] - (c > 0 ? cfg.presence_penalty : 0.0F) - static_cast<float>(c) * cfg.frequency_penalty;
            q[j] = adj;
            orig_prob_sum += __expf(lp[j] - lp0);
            pen_prob_sum += __expf(adj - lp0);
        }
        // Order (q[0..count-1], cand[0..count-1]) in-place by descending adjusted score, lower id breaking ties.
        for (int i = 1; i < count; ++i) {
            const float s        = q[i];
            const std::int32_t c = cand[i];
            int k                = i;
            while (k > 0 && (s > q[k - 1] || (s == q[k - 1] && c < cand[k - 1]))) {
                q[k]    = q[k - 1];
                cand[k] = cand[k - 1];
                --k;
            }
            q[k]    = s;
            cand[k] = c;
        }
        const float tail_scaled = fmaxf(0.0F, __expf(-lp0) - orig_prob_sum);
        const float log_z_ratio = __logf(fmaxf(pen_prob_sum + tail_scaled, 1e-30F)) + lp0;
        top_lp                  = fminf(0.0F, q[0] - log_z_ratio);
    }
    if (confidence != nullptr) { confidence[t] = top_lp; }
    if (!(cfg.temperature > 0.0F)) {
        q[0] = 1.0F;
        for (int j = 1; j < count; ++j) { q[j] = 0.0F; }
        drafts[t] = cand[0];
        return;
    }
    const int ranks = cfg.top_k > 0 && cfg.top_k < count ? cfg.top_k : count;
    // The weights are recomputed per pass (at most 16 each) rather than held in a local array; s0 is
    // kept in a register because writing q[0] overwrites the penalized slot before j > 0 is computed.
    const float inverse = 1.0F / cfg.temperature;
    const float s0      = penalized ? q[0] : lp[0];
    const auto score    = [&](int j) { return penalized ? q[j] : lp[j]; };
    const auto weight   = [&](int j) { return __expf((score(j) - s0) * inverse); };
    float total         = 0.0F;
    for (int j = 0; j < ranks; ++j) { total += weight(j); }
    const float floor  = cfg.min_p > 0.0F ? cfg.min_p : -1.0F; // weight(0) == 1
    const float target = cfg.top_p * total;
    float mass         = 0.0F;
    int support        = 0;
    for (int j = 0; j < ranks; ++j) {
        const float w = weight(j);
        if (w < floor) { break; }
        mass += w;
        support = j + 1;
        if (cfg.top_p < 1.0F && mass >= target) { break; }
    }
    if (support < 1) {
        support = 1;
        mass    = weight(0);
    }
    for (int j = 0; j < support; ++j) { q[j] = weight(j) / mass; }
    for (int j = support; j < count; ++j) { q[j] = 0.0F; }
    const float u = sampling_uniform(cfg.seed, positions[t], kSamplePurposeDFlash2Proposal, draw);
    // A draw past the rounded cumulative mass falls back to the last candidate with q > 0: a tail
    // weight that underflowed has q = 0 and is never drawn (verification would reject it).
    int pick         = 0;
    float cumulative = 0.0F;
    for (int j = 0; j < support; ++j) {
        if (!(q[j] > 0.0F)) { continue; }
        cumulative += q[j];
        pick = j;
        if (u < cumulative) { break; }
    }
    drafts[t] = cand[pick];
}

} // namespace

void speculative_sample_sparse_drafts_launch(const Tensor& rows, const Tensor& logprobs, const std::int32_t* token_ids,
                                             const SamplingConfig* configs, const Tensor& positions, std::uint32_t draw, Tensor& drafts,
                                             Tensor& candidate_ids, Tensor& proposal_q, Tensor& confidence,
                                             const Tensor& prior_drafts, cudaStream_t stream) {
    const int columns          = static_cast<int>(drafts.numel());
    const int count            = rows.ne[0];
    const auto* prior          = prior_drafts.data != nullptr ? static_cast<const std::int32_t*>(prior_drafts.data) : nullptr;
    const int prior_steps      = prior_drafts.data != nullptr ? prior_drafts.ne[1] : 0;
    speculative_sample_sparse_drafts_kernel<<<1, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(rows.data), static_cast<const float*>(logprobs.data), token_ids, configs,
        static_cast<const std::int32_t*>(positions.data), draw, prior, prior_steps, columns, count,
        static_cast<std::int32_t*>(drafts.data), static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(proposal_q.data), static_cast<float*>(confidence.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
