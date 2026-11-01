#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

using Geometry = Nvfp4N34816K5120;
// GB10 (sm_121a, 48 SMs) cold-weight sweeps over every A4 schedule candidate, quantize included.
// These replace the earlier RTX 5090 fit. Column tiles amortize gate/up decode over the complete
// speculative block; within a band the neighbours are within measurement noise, but the boundaries
// below are separated by >10% (the resident schedules fall off a cliff once the wave spills past
// what they can hide), so those boundaries are what matter.
using M32N64S3 = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>;
using M64N64   = Nvfp4A4MmaSchedule<64, 64, 256, 4, 2, 2, 2>;
using M128N128 = Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128  = Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4A4Workspace workspace, int tokens,
                 cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a4_mma<S>(nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
                           SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{},
                           pdl::Dependency::Programmatic);
}

template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4A4Workspace scratch =
        allocate_nvfp4_a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Schedule>(weight, out, scratch, x.ne[1], stream);
}

// The A4 residual tiles: the same 64-row weight tiles at two CTAs per SM, each column's activations
// in `Terms` planes. Narrow rounds take 16-token tiles so the extra planes cost no staging.
template <int Terms>
using ResidualM16N64 = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM32N64 = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM64N128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM96N128 = Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1, Cache::cg, Cache::cg, Terms>;

// The first round width the warp-specialized TMA route owns on GB10 (48 SMs), and it now has no
// upper cap. Measured on the production route (cold L2, quantizer included, min of two alternating
// rounds of the HEAD and candidate binaries over the same process protocol):
//   * up to 96 columns the cp.async ladder streams the 89 MiB gate/up panel better than a single
//     128-token TMA tile does -- 445.3 against 498.6 us at 5, 448.2 against 505.6 at 16, 461.7
//     against 517.0 at 64 and 470.8 against 523.1 at 96 -- so the TMA entry stays at 97 instead of
//     following the column-invariant routes down into the decode band;
//   * at 97 and 128 the two lanes are level (527.2/528.2 and 527.6/529.3 us, TMA/ladder) and TMA
//     keeps the band it already had;
//   * above 128 the ladder falls off a cliff, because one 64-token tile per round re-reads the
//     whole panel: 1368.9 against 557.9 us at 129 and 29376 against 8098 us at 4,096. The RTX 5090
//     fit capped TMA at 128 columns and sent everything above it back to `ResidualM64N128`, which
//     cost the prefill band a factor of 3.6.
constexpr std::int32_t kResidualTmaTokens = 97;

template <int Terms>
void launch_residual(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4A4Workspace scratch =
        allocate_nvfp4_a4_workspace(workspace, x.ne[1], Geometry::kInputRows, Terms);
    if (Terms == kNvfp4A4ResidualTerms && x.ne[1] >= kResidualTmaTokens) {
        launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled128, stream);
        launch_nvfp4_linear_swiglu_a4_residual_tma(
            nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled128),
            static_cast<__nv_bfloat16*>(out.data), stream);
        return;
    }
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 16) {
        launch_gemm<ResidualM16N64<Terms>>(weight, out, scratch, tokens, stream);
    } else if (tokens <= 32) {
        launch_gemm<ResidualM32N64<Terms>>(weight, out, scratch, tokens, stream);
    } else if (tokens <= 64) {
        launch_gemm<ResidualM64N128<Terms>>(weight, out, scratch, tokens, stream);
    } else {
        launch_gemm<ResidualM96N128<Terms>>(weight, out, scratch, tokens, stream);
    }
}

} // namespace

void nvfp4_linear_swiglu_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                            WorkspaceArena& workspace, cudaStream_t stream) {
    launch_residual<kNvfp4A4ResidualTerms>(x, weight, out, workspace, stream);
}

void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= 32) {
        launch<M32N64S3>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M64N64::kBlockTokens) {
        launch<M64N64>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockTokens) {
        launch<M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<M128N128>(x, weight, out, workspace, stream);
    }
}

} // namespace infernix::ops::detail
