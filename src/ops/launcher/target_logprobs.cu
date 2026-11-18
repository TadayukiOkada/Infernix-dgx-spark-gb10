// Implements: include/infernix/ops/target_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/target_logprobs.h"

#include "core/device.h"
#include "ops/kernel/target_logprobs.cuh"

namespace infernix::ops::detail {

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, cudaStream_t stream) {
    const auto columns    = static_cast<unsigned int>(logits.ne[1]);
    const bool full_vocab = logits.ne[0] == 248320 && valid_rows == 248077;
    const auto launch     = [&](const auto* values) {
        if (full_vocab) {
            target_logprobs_kernel<1024><<<columns, 1024, 0, stream>>>(
                values, static_cast<const std::int32_t*>(target_ids.data),
                static_cast<float*>(output.data), valid_rows, logits.ne[0], target_ids.ne[0]);
            return;
        }
        target_logprobs_kernel<kTargetLogprobsBlock><<<columns, kTargetLogprobsBlock, 0, stream>>>(
            values, static_cast<const std::int32_t*>(target_ids.data), static_cast<float*>(output.data),
            valid_rows, logits.ne[0], target_ids.ne[0]);
    };
    if (logits.dtype == DType::FP32) {
        launch(static_cast<const float*>(logits.data));
    } else {
        launch(static_cast<const __nv_bfloat16*>(logits.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
