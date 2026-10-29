#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace infernix::ops::detail {
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}};

    const auto p   = bf16_a16_operands(x, weight);
    const auto mma = [&]<class S>() {
        launch_bf16_a16_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                           stream);
    };
    const auto tma = [&]<class S>() {
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                               stream);
    };
    // GB10 (sm_121a, 48 SM) retune: candidate sweep + edge sweeps show a single
    // mid-tile lane (mma.r32.t32.k128.s3) dominates a much wider band than the
    // 5090-inherited cutoffs assumed. mma.r64.t32.k64.s3 never wins and is dropped.
    if (p.tokens <= 16)
        launch_bf16_a16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
    else if (p.tokens <= 146)
        mma.template operator()<Bf16A16MmaR32T32K128S3>();
    else if (p.tokens <= 320)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
}
} // namespace infernix::ops::detail
