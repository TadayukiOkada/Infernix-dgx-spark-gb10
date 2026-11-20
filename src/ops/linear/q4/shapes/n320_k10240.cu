#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
namespace {
using SlicedK10240T8 = Q4A16SlicedKMmaSchedule<16, 8, 8, 1, Cache::cg, Cache::ca, 2, 10240, 8>;
using MmaR64T128B2   = Q4A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q4MmaFragmentPipeline::Serial,
                                        Cache::cg, Cache::cg, Q4ScaleLoad::Pair32, 2>;

void launch_t8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] == 10240)
        launch_q4_a16_sliced_instance<SlicedK10240T8>(x, w, out, stream);
    else
        launch_q4_a16_sliced_r16_t8_w4_s2(x, w, out, stream);
}
} // namespace

Q4Launch select_q4_n320_k10240(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 8) return launch_t8;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 896) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 1152) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 1440) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 1824) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 3072) return launch_q4_a16_mma_instance<MmaR64T128B2>;
    return launch_q4_a16_mma_r64_t112;
}
} // namespace infernix::ops::detail
