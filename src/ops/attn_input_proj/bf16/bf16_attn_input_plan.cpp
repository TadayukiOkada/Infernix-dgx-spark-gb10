#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kBf16TargetHidden     = 5120;
constexpr std::int32_t kBf16TargetQueryRows  = 6144;
constexpr std::int32_t kBf16TargetKvRows     = 1024;
constexpr std::int32_t kBf16TargetParentRows = 14336;

bool bf16_supported_shape(const Bf16AttnInputProblem& problem) noexcept {
    return problem.input_rows == kBf16TargetHidden && problem.query_rows == kBf16TargetQueryRows &&
           problem.kv_rows == kBf16TargetKvRows && problem.parent_rows == kBf16TargetParentRows &&
           problem.padded_k == kBf16TargetHidden;
}

} // namespace

const char* bf16_attn_input_schedule_name(Bf16AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16AttnInputScheduleId::Decode:            return "attn_input_proj.bf16.decode";
    case Bf16AttnInputScheduleId::SimtSmall:         return "attn_input_proj.bf16.simt.small";
    case Bf16AttnInputScheduleId::SlicedR32T16W4:    return "attn_input_proj.bf16.sliced.r32.t16.w4";
    case Bf16AttnInputScheduleId::MmaR32T32K128S3:   return "attn_input_proj.bf16.mma.r32.t32.k128.s3";
    case Bf16AttnInputScheduleId::TmaR64T64K64S3:    return "attn_input_proj.bf16.tma.r64.t64.k64.s3";
    case Bf16AttnInputScheduleId::MmaR64T32K64S3:    return "attn_input_proj.bf16.mma.r64.t32.k64.s3";
    case Bf16AttnInputScheduleId::TmaR64T128K64S2:   return "attn_input_proj.bf16.tma.r64.t128.k64.s2";
    }
    return "attn_input_proj.bf16.unknown";
}

bool bf16_attn_input_candidate_is_legal(Bf16AttnInputScheduleId schedule,
                                        const Bf16AttnInputProblem& problem) noexcept {
    if (!bf16_supported_shape(problem) || problem.cols < 1) { return false; }
    const std::int32_t cols = problem.cols;
    switch (schedule) {
    case Bf16AttnInputScheduleId::Decode:    return cols == 1;
    case Bf16AttnInputScheduleId::SimtSmall: return cols >= 2 && cols <= 4;
    default:                                 return cols >= 2;
    }
}

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream, bool column_invariant) {
    if (column_invariant) {
        bf16_attn_input_column_invariant_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] == 1) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
}

} // namespace infernix::ops::detail
