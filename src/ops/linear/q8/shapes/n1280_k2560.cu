#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next shared expert gate/up: the register-streamed route up to 48 columns but 4 (SIMT there),
// the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 4, MaxCols, ColsPerPass, Q8StreamX::Global, 2>;
// route-sweep 2026-10-24 (bench/ops/q8_route_sweep.cu, bit_exact=1 && t_invariant=1 vs production):
// the sliced-K winners replicate the swept template args exactly, including StaticK=0, TokenCapacity
// and RowTiles, so the shipped instantiations are precisely the ones the sweep verified and timed.
using SlicedK_T8W8S1M1RT1 = Q8A16SlicedKMmaSchedule<8, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                    Cache::cg, Q8ActivationStage::ActiveOnly, 0, 8,
                                                    false, 1>;
using SlicedK_T8W8S2M1RT2 = Q8A16SlicedKMmaSchedule<8, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                    Cache::cg, Q8ActivationStage::ActiveOnly, 0, 8,
                                                    false, 2>;
using SlicedK_T16W8S1M1RT1 = Q8A16SlicedKMmaSchedule<16, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                     Cache::cg, Q8ActivationStage::ActiveOnly, 0, 16,
                                                     false, 1>;

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n1280_k2560(std::int32_t tokens) {
    if (tokens <= 2) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 48) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 64) return launch_q8_a16_mma_r64x32_t64_k128_a1;
    if (tokens <= 96) return launch_q8_a16_mma_r64_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
