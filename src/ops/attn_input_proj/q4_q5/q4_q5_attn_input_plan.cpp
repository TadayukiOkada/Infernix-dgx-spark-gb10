#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        return "attn_input_proj.q4_q5.mixed.r32.c32.s2";
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        return "attn_input_proj.q4_q5.mixed.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        return "attn_input_proj.q4_q5.pair.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        return "attn_input_proj.q4_q5.mixed.r64.c128.s2";
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        return "attn_input_proj.q4_q5.pair.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::ColumnInvariant:
        return "attn_input_proj.q4_q5.sliced.column_invariant";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem,
                                                bool column_invariant) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }
    if (column_invariant) return {Q4Q5AttnInputScheduleId::ColumnInvariant};

    // GB10 (sm_121a, 48 SM) retune: candidate sweeps show mixed.r64.c128.s2 dominates the entire
    // 65+ band, and mixed.r32.c64.s3 owns 13-64. The 5090-inherited pair.r32.c64.s3 (65-104) and
    // pair.r32.c64.s4 (129-192) bands are severely mis-routed (~36% / ~20% penalties); mixed.r32.c32.s2
    // never wins a sustained band. Those lanes stay legal in the candidate API but are dropped here.
    if (problem.cols <= 12) return {Q4Q5AttnInputScheduleId::ParentSplitFixed};
    if (problem.cols <= 64) return {Q4Q5AttnInputScheduleId::MixedR32C64S3};
    return {Q4Q5AttnInputScheduleId::MixedR64C128S2};
}

bool q4_q5_attn_input_candidate_is_legal(Q4Q5AttnInputScheduleId schedule,
                                         const Q4Q5AttnInputProblem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    // The small-T (parent-split fixed) kernel is hard-bound to T in [1,12]; every MMA lane slices
    // arbitrary column counts and is legal at any cols >= 1.
    if (schedule == Q4Q5AttnInputScheduleId::ParentSplitFixed) { return problem.cols <= 12; }
    return true;
}

void q4_q5_attn_input_execute_candidate(Q4Q5AttnInputScheduleId schedule, const Tensor& x,
                                        const Weight& query_key_weight,
                                        const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                        Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    if (!q4_q5_attn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("Q4/Q5 attention input: candidate schedule not legal for shape");
    }
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        q4_q5_attn_input_mixed_r32_c32_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                  v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown candidate schedule");
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream, bool column_invariant) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem, column_invariant);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        q4_q5_attn_input_mixed_r32_c32_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                  k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::ColumnInvariant:
        q4_q5_attn_input_column_invariant_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                 k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream, bool column_invariant) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem, column_invariant);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream, column_invariant);
}

} // namespace infernix::ops::detail
