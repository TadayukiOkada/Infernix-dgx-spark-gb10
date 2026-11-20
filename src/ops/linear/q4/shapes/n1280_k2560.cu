#include "ops/linear/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/q4/q4_a16_tma_prefetch.cuh"
#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
namespace {
using Tma128  = Q4A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 2560, 1, 8>;
using Bulk256 = Q4A16TmaPrefetchSchedule<64, 256, 64, 32, 1, 2560, 1, 8, 4, 1, 64, 1>;

template <class Schedule, Q4Launch Fallback>
void launch_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] != 2560 || w.n % Schedule::kBlockRows != 0 || out.dtype != DType::BF16) {
        Fallback(x, w, out, stream);
        return;
    }
    launch_q4_a16_tma_prefetch<Schedule>(
        q4_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}
} // namespace

Q4Launch select_q4_n1280_k2560(std::int32_t tokens) {
    // Selected with complete-Op cold-L2 route sweeps on GB10.
    if (tokens <= 1) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 256) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 512) return launch_q4_a16_mma_r64_t80;
    return launch_tma<Tma128, launch_q4_a16_mma_r64_t80>;
}
} // namespace infernix::ops::detail
