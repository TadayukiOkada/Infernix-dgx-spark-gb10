#include "ops/linear/q4/q4_instance_launch.cuh"

namespace infernix::ops::detail {

void launch_q4_a16_mma_r64_t48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T48K64Wr16Wt16S2A2B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T72K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T80K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T96K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t112(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T112K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t120(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T120K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T128K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t32_k128_s2_a2(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T32K128S2A2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t64_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T64K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t64_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T64K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t128_k64_s2_a2(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T128K64S2A2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T32K64Wr16Wt16S3A3B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T64K64Wr16Wt32S2A2B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T64K64Wr16Wt32S3A3B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T64K64Wr32Wt16S2A2B2>(x, w, out, stream);
}

} // namespace infernix::ops::detail
