#include "ops/linear/q5/q5_instance_launch.cuh"

namespace infernix::ops::detail {
void launch_q5_a16_mma_r32_t32_k128_s2_a2(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T32K128S2A2>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t64_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T64K128S2A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t128_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T128K128S2A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t64_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T64K128S1A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t64_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T64K128S2A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t80_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T80K128S1A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t96_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T96K128S1A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t128_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T128K128S1A1>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t16_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T16K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t32_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T32K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t48_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T48K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r32_t64_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T64K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t32_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T32K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t48_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T48K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t64_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T64K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t96_k64(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T96K64>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t128_k64_m2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T128K64M2>(x, w, out, stream);
}
} // namespace infernix::ops::detail
