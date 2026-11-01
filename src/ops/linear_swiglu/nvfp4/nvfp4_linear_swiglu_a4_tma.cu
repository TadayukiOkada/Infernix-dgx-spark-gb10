#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace infernix::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B>, 5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}

void launch_nvfp4_linear_swiglu_a4_residual_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                                cudaStream_t stream) {
    // The 128-token tile with a 128-row weight panel (three 128-K stages, one CTA per SM) is the
    // shape the standalone pair route's grid heuristic also picks for this row count. Within the
    // candidate harness (bench/ops/nvfp4_linear_swiglu_bench.cu --variants) the 64-row panel is
    // 1.4% ahead while the call fits one token tile and 26% behind from the second tile on, so the
    // band is served by the one instantiation. That harness's absolute numbers are not the
    // production route's, and its per-schedule ordering is not either: on the shipped route this
    // lane loses to the cp.async ladder below 97 columns (498.6 against 445.3 us at 5) even though
    // the harness ranks it ahead there, which is why the entry was set from the route binary.
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms>,
        5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace infernix::ops::detail
