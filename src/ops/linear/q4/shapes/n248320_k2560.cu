#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
namespace {

using Batch8  = Q4A16SlicedKMmaSchedule<16, 8, 8, 1, Cache::cg, Cache::ca, 2, 2560, 8>;
using Batch16 = Q4A16SlicedKMmaSchedule<16, 16, 8, 1, Cache::cg, Cache::ca, 2, 2560, 16>;

void launch_batch8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] == 2560)
        launch_q4_a16_sliced_instance<Batch8>(x, w, out, stream);
    else
        launch_q4_a16_sliced_r32_t8_w4_s2(x, w, out, stream);
}

void launch_batch16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] == 2560)
        launch_q4_a16_sliced_instance<Batch16>(x, w, out, stream);
    else
        launch_q4_a16_sliced_r32_t16_w4_s1(x, w, out, stream);
}

} // namespace

Q4Launch select_q4_n248320_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_batch8;
    if (tokens <= 16) return launch_batch16;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
