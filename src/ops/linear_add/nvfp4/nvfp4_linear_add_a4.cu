#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/common/epilogue.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

// GB10 (sm_121a, 48 SMs) cold-weight sweeps over every A4 schedule candidate, quantize included.
// These replace the earlier RTX 5090 fit, which mis-picked the mid- and high-token bands here.
// Up to ~32 tokens the N=5120 projections are one wave of CTAs, so their time is the weight stream
// per CTA; a third stage keeps more of it in flight (K=17408 221-232 us, K=6144 85-90 us).
using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>;
// On GB10 the K=17408 down projection prefers the 32x32/512-K-per-stage shape out to 32 tokens
// (221-232 us vs 230-240 for M32N64), a much wider window than the RTX 5090 needed.
using M32N32K512        = Nvfp4A4MmaSchedule<32, 32, 512, 2, 2, 3, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
// Pipelined wins the 49..128 band on GB10 (K6144 91-99 us, K17408 244-267 us); above it the
// warp-specialized TMA route owns everything (see uses_tma).
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
// GB10 (cold-weight candidates, quantize included) puts the TMA/cp.async crossover at 129, not the
// RTX 5090 fit's 288: TMA is 113.7 against 140.4 us resident and 133.8 pipelined at 129, 127.0
// against 155.6 at 256 and 140.3 against 200.8 at 287 for K=6144; 314.8 against 473.1 at 129,
// 340.0 against 511.0 at 256 and 371.7 against 730.2 at 287 for K=17408. The resident tile that
// held 129..287 under the old boundary ran 34..49% behind.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 129; }

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream,
        {}, pdl::Dependency::Programmatic);
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    // Band boundaries are the per-token cold-weight winners on GB10 (see the schedule notes above);
    // 129 tokens up never reaches here -- `uses_tma` sends it to the TMA route. Within each band the
    // neighbours are within measurement noise; the boundaries themselves are separated by >5%, so
    // they are the ones that matter.
    if constexpr (Geometry::kInputRows == 17408) {
        if (tokens <= 32) {
            launch_gemm<Geometry, M32N32K512>(weight, residual, workspace, tokens, stream);
        } else if (tokens <= 48) {
            launch_gemm<Geometry, M32N128>(weight, residual, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, stream);
        }
    } else {
        if (tokens <= 16) {
            launch_gemm<Geometry, M32N128>(weight, residual, workspace, tokens, stream);
        } else if (tokens <= 48) {
            launch_gemm<Geometry, M64N128>(weight, residual, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, stream);
        }
    }
}

// The A4 residual tiles, each column's activations in `Terms` planes. At K=17408 narrow rounds
// stream 512 K per stage over 160 CTAs, as the A4 route does at up to 8 tokens.
template <int Terms>
using ResidualM16N32K512 = Nvfp4A4MmaSchedule<16, 32, 512, 1, 2, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM16N64 = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM32N64 = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM64N128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1, Cache::cg, Cache::cg, Terms>;

// From 65 columns the 2-term TMA lane owns the band on GB10, quantize included: 103 against
// 126-129 us at 65 and 111 against 133-137 at 128 for K=6144, and 290 against 440-487 at 65 and
// 5.2 against 14.1 ms at 4,096 for K=17408. The cp.async tiles only stay ahead inside one
// 16-token tile, where their extra planes cost no staging; the RTX 5090 fit kept 64x64 to 128 and
// 64x128 above it, which left the whole prefill band 2.7x behind the standalone pair route.
constexpr bool uses_residual_tma(std::int32_t tokens) { return tokens >= 65; }

// Up to 32 columns the 32x64 tile keeps one token tile's worth of CTAs (the 64x128 tile halves
// them); from 33 the 64-column tile's second token tile re-streams the whole weight panel from
// LPDDR5X, and the wider 64x128 tile that fits one tile wins: 245.8 against 462.8 us at 33, 248.8
// against 471.0 at 48 and 254.0 against 477.2 at 64 for K=17408; 92.2 against 109.0, 95.2 against
// 120.1 and 95.2 against 110.6 for K=6144 (GB10 cold-weight candidates, quantize included). The
// earlier RTX 5090 fit kept 32x64 out to 64, which cost this band 1.86x.
template <class Geometry, int Terms>
void launch_residual_problem(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                             std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 16) {
        if constexpr (Geometry::kInputRows == 17408) {
            launch_gemm<Geometry, ResidualM16N32K512<Terms>>(weight, residual, workspace, tokens,
                                                             stream);
        } else {
            launch_gemm<Geometry, ResidualM16N64<Terms>>(weight, residual, workspace, tokens,
                                                         stream);
        }
    } else if (tokens <= 32) {
        launch_gemm<Geometry, ResidualM32N64<Terms>>(weight, residual, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, ResidualM64N128<Terms>>(weight, residual, workspace, tokens, stream);
    }
}

template <class Geometry>
void launch_residual_terms(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                           std::int32_t tokens, cudaStream_t stream) {
    if (workspace.terms != kNvfp4A4ResidualTerms) {
        throw std::invalid_argument("nvfp4 linear_add A4 residual: unexpected activation planes");
    }
    if (uses_residual_tma(tokens)) {
        launch_nvfp4_a4_tma_linear_add_residual(
            nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::Tiled128),
            static_cast<__nv_bfloat16*>(residual.data), stream);
        return;
    }
    launch_residual_problem<Geometry, kNvfp4A4ResidualTerms>(weight, residual, workspace, tokens,
                                                             stream);
}

} // namespace

void nvfp4_linear_add_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                         Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // The quantizer writes the plane the GEMM below reads, so both come from `uses_residual_tma`.
    const auto layout = uses_residual_tma(tokens) ? Nvfp4ScaleLayout::Tiled128
                                                  : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_residual_terms<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_residual_terms<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const bool tma = uses_tma(tokens);
    // The shared 48-SM wave-fill rule picks the 128-token tile where the 256-token grid leaves a
    // short last wave; the quantizer writes the scale layout the chosen tile reads.
    const bool narrow = tma && nvfp4_a4_prefer_narrow_token_tile(weight.n, weight.k, tokens);
    const auto layout = !tma ? Nvfp4ScaleLayout::RowMajor
                              : narrow ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (tma) {
        const Nvfp4A4Operands operands = nvfp4_a4_operands(weight, workspace, tokens, layout);
        __nv_bfloat16* y = static_cast<__nv_bfloat16*>(residual.data);
        if (narrow) {
            launch_nvfp4_a4_tma_linear_add_narrow(operands, y, stream);
        } else {
            launch_nvfp4_a4_tma_linear_add(operands, y, stream);
        }
        return;
    }
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_problem<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_problem<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace infernix::ops::detail
