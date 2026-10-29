#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <stdexcept>

namespace infernix::ops::detail {

void bf16_attn_input_execute_candidate(Bf16AttnInputScheduleId schedule, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    const Bf16AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                       x.ne[1]};
    if (!bf16_attn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("bf16 attention input: candidate schedule not legal for shape");
    }
    if (schedule == Bf16AttnInputScheduleId::Decode) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (schedule == Bf16AttnInputScheduleId::SimtSmall) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}};
    const auto p = bf16_a16_operands(x, weight);
    switch (schedule) {
    case Bf16AttnInputScheduleId::SlicedR32T16W4:
        launch_bf16_a16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
        return;
    case Bf16AttnInputScheduleId::MmaR32T32K128S3:
        launch_bf16_a16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
        return;
    case Bf16AttnInputScheduleId::TmaR64T64K64S3:
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
        return;
    case Bf16AttnInputScheduleId::MmaR64T32K64S3:
        launch_bf16_a16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
        return;
    case Bf16AttnInputScheduleId::TmaR64T128K64S2:
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
        return;
    default:
        throw std::logic_error("bf16 attention input: unknown candidate schedule");
    }
}

} // namespace infernix::ops::detail
