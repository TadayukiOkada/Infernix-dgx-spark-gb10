#pragma once
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <type_traits>

namespace infernix::ops::detail {
template <class Schedule>
void launch_bf16_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_bf16_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_simt<Schedule>(bf16_a16_operands(x, w),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                   LinearIdentityEpilogue{}, stream);
}

template <class Schedule,
          class Epilogue = std::conditional_t<bf16_predicated_rows<Schedule>,
                                              LinearIdentityEpilogue, Bf16MmaWarpStoreEpilogue>>
void launch_bf16_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_mma<Schedule>(bf16_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  Epilogue{}, stream);
}

template <class Schedule, pdl::Dependency Dependency = pdl::Dependency::Serialized,
          class Epilogue = std::conditional_t<bf16_predicated_rows<Schedule>,
                                              LinearIdentityEpilogue, Bf16MmaWarpStoreEpilogue>>
void launch_bf16_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_sliced_k_mma<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Epilogue{}, stream, Dependency);
}

template <class Schedule,
          class Epilogue = std::conditional_t<bf16_predicated_rows<Schedule>,
                                              Bf16TmaRowVectorEpilogue, Bf16TmaVectorEpilogue>>
void launch_bf16_tma_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      Epilogue{}, stream);
}

template <class Schedule,
          class Epilogue =
              std::conditional_t<bf16_predicated_rows<Schedule>, Bf16TmaRowVectorEpilogue,
                                 Bf16MmaCtaWarpStoreEpilogue<>>>
void launch_bf16_tma_register_b(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    if (out.dtype == DType::FP32) {
        select_bf16_general_launch(x.ne[1])(x, w, out, stream);
        return;
    }
    launch_bf16_a16_tma_register_b<Schedule>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Epilogue{}, stream);
}
} // namespace infernix::ops::detail
