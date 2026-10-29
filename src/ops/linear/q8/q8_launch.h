#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace infernix::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_gemv_r4_w1_k16384(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t5(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t6(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r1_t4_w8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r1_t8_w8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t16(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t24(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t32(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t48(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r48_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t32(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t48(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r96_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128_t80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x16_t48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t96_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x16_t128_k128_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x16_t256_k64_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t64_k64_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t128_k64_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128x32_t64_k64_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128x64_t128_k64_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128x32_t128_k64_a2(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_simt_r4_t4_w2_g16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r4_t5_w2_g16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r4_t8_w2_g16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w8_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t32_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t8_w16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r32_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r32_t16_w8_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace infernix::ops::detail
