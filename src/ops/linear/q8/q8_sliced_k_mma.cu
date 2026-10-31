#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_launch.h"

namespace infernix::ops::detail {
void launch_q8_a16_sliced_r16_t16_w8_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T16W8S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_sliced_r16_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T16W4S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T32W4S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_sliced_r16_t8_w16_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR16T8W16S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR32T16W4S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_sliced_r32_t16_w8_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<q8_instances::SlicedR32T16W8S2>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

// The T2 (2026-10-31) decode/verify tiles, registered the way q8_mma.cu registers its MMA tiles.
#define INFERNIX_Q8_SLICED_LAUNCHER(Name, Schedule)                                                \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        launch_q8_a16_sliced_k_mma<q8_instances::Schedule>(                                        \
            q8_linear_operands(x, w),                                                              \
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},                    \
            LinearIdentityEpilogue{}, stream);                                                     \
    }

INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t8_w8_s2_m3, SlicedR16T8W8S2M3)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t8_w16_s1, SlicedR16T8W16S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t16_w16_s1, SlicedR16T16W16S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t16_w16_s1_m2, SlicedR16T16W16S1M2)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t16_w8_s2_m1, SlicedR16T16W8S2M1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t32_w4_s1, SlicedR16T32W4S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t32_w8_s1_m2, SlicedR16T32W8S1M2)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t40_w16_s1, SlicedR16T40W16S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r32_t48_w4_s1, SlicedR32T48W4S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t16_w4_s1, SlicedR16T16W4S1)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t48_w4_s2, SlicedR16T48W4S2)
INFERNIX_Q8_SLICED_LAUNCHER(launch_q8_a16_sliced_r16_t48_w4_s1, SlicedR16T48W4S1)
} // namespace infernix::ops::detail
