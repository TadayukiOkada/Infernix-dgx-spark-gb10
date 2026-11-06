#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

namespace infernix::ops::detail {
namespace {
template <int K>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4A4TmaWeightCodeL2_128B>, K>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows},
                               LinearResidualAddEpilogue{{y, p.rows}}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream) {
    if (p.k == 6144)
        launch<6144>(p, residual, stream);
    else if (p.k == 17408)
        launch<17408>(p, residual, stream);
    else
        throw std::invalid_argument("NVFP4 TMA linear_add: unsupported K");
}

namespace {
// The 1-term GEMM on a 128-token tile instead of the 256-token one: the same 128-row weight panel
// and the same K order, with 4 stages to keep the producer ahead at 1 CTA/SM (the shape the
// standalone narrow prefill entry runs; see nvfp4_a4_prefer_narrow_token_tile for the widths).
template <int K>
void launch_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 4, 1, kNvfp4A4TmaWeightCodeL2_128B>, K>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows},
                               LinearResidualAddEpilogue{{y, p.rows}}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear_add_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                           cudaStream_t stream) {
    if (p.k == 6144)
        launch_narrow<6144>(p, residual, stream);
    else if (p.k == 17408)
        launch_narrow<17408>(p, residual, stream);
    else
        throw std::invalid_argument("NVFP4 narrow-token TMA linear_add: unsupported K");
}

namespace {
// The 128-token tile with a 128-row weight panel, three 128-K stages, one CTA per SM -- the shape
// the standalone pair route's grid heuristic picks for this row count and the one that wins every
// round width this lane serves on GB10 (cold weight, quantize included): 103 against 126 us at 65
// columns and 5.2 against 14.1 ms at 4,096 for K=17408. The 64-row panel doubles a 40-CTA grid
// into a second wave without filling the 48 SMs and is 8..16% behind from 129 columns up; the
// 256-token `Tiled256` tile loses its third pipeline stage to the two activation planes and is
// 10..18% behind at both K.
template <int K>
void launch_residual(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 128>,
        K>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows},
                               LinearResidualAddEpilogue{{y, p.rows}}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear_add_residual(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                             cudaStream_t stream) {
    if (p.k == 6144)
        launch_residual<6144>(p, residual, stream);
    else if (p.k == 17408)
        launch_residual<17408>(p, residual, stream);
    else
        throw std::invalid_argument("NVFP4 TMA linear_add residual: unsupported K");
}
} // namespace infernix::ops::detail
