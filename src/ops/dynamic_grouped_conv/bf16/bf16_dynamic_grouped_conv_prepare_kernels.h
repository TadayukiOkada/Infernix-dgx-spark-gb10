#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infernix::ops::detail {

struct DynamicConvFusedOutputs {
    const __nv_bfloat16* normalized;
    const __nv_bfloat16* base;
    __nv_bfloat16* prepared;
    __nv_bfloat16* finish;
    int width;
};

void bf16_dynamic_grouped_conv_prepare_fused_sliced_launch(const Tensor& normalized,
                                                           const Tensor& base,
                                                           const Weight& projection,
                                                           Tensor& prepared, Tensor& finish,
                                                           cudaStream_t stream);

void bf16_dynamic_grouped_conv_prepare_fused_tma_launch(const Tensor& normalized,
                                                        const Tensor& base,
                                                        const Weight& projection, Tensor& prepared,
                                                        Tensor& finish, cudaStream_t stream);

} // namespace infernix::ops::detail
