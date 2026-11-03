#pragma once
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace infernix::ops::detail {
using Nvfp4A4Launch = void (*)(const Weight&, Tensor&, Nvfp4A4Workspace, std::int32_t,
                               cudaStream_t);

// A selected A4 route: the GEMM to run, the activation-scale layout that GEMM reads, and the
// quantizer quotient formulation. They travel together because the quantizer writes the plane
// before the GEMM runs and the two must agree; a shape selects a route rather than selecting
// the parts separately. Reciprocal quotient is measured faster on the latency-bound MMA route
// and slower on the throughput-bound TMA route, so each route states its own choice.
struct Nvfp4A4Route {
    Nvfp4A4Launch launch;
    Nvfp4ScaleLayout scales;
    bool reciprocal_quantize;
};

template <class Geometry, class Schedule>
void nvfp4_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Full = false>
void nvfp4_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Full>;
    launch_nvfp4_a16_simt<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

// The bare-schedule (runtime-K) form of nvfp4_linear_a16_sliced_k. Baking K lets ptxas unroll the
// K-group loop, which helps the narrow tiles and costs the wide-token ones dearly: at [34816,5120]
// the baked <24,4,1,2> measures 525.8us against 449.0us for the same schedule with K read from the
// operands, and at [5120,17408] the baked <16,8,2,2> measures 287.0us against 228.2us (GB10, cold
// L2). A shape table therefore picks per band; the full-MMA tiles have no such gap, so only the
// sliced-K launch has a runtime-K form.
template <class Geometry, class Schedule>
void nvfp4_linear_a16_sliced_k_dyn(const Tensor& x, const Weight& w, Tensor& y,
                                   cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Schedule>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a4_mma(const Weight& w, Tensor& y, Nvfp4A4Workspace workspace, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(w, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n}, LinearIdentityEpilogue{},
        stream, {}, pdl::Dependency::Programmatic);
}

template <Nvfp4ScaleLayout Layout>
void nvfp4_linear_a4_tma(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_tma_linear(nvfp4_a4_operands(weight, scratch, tokens, Layout),
                               static_cast<__nv_bfloat16*>(out.data), stream);
}

// The 1-term TMA GEMM: the shape picks only the activation-scale tile, and the launcher owns the
// instantiated schedules for the registered Ks (see nvfp4_a4_tma.cu).
template <Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::Tiled256>
constexpr Nvfp4A4Route nvfp4_a4_tma_route() {
    return {nvfp4_linear_a4_tma<Layout>, Layout, false};
}

template <Nvfp4ScaleLayout Layout>
void nvfp4_linear_a4_tma_narrow(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch,
                                int tokens, cudaStream_t stream) {
    launch_nvfp4_a4_tma_linear_narrow(nvfp4_a4_operands(weight, scratch, tokens, Layout),
                                      static_cast<__nv_bfloat16*>(out.data), stream);
}

constexpr Nvfp4A4Route nvfp4_a4_tma_narrow_route() {
    return {nvfp4_linear_a4_tma_narrow<Nvfp4ScaleLayout::Tiled128>, Nvfp4ScaleLayout::Tiled128,
            false};
}

// Which 1-term A4 prefill token tile to run, on the 48 SMs of GB10 (both tiles hold one CTA/SM).
// A 256-token tile packs twice the work of a 128-token tile into one CTA, so it wins while the grid
// fills whole waves; when ceil(rows/128) * ceil(T/256) leaves a short last wave, or the last token
// tile is only partly real, the 128-token tile fills the machine better. Wave fill alone is not
// enough: halving the token tile doubles how many times each weight panel is streamed, so the win
// stops once that extra traffic exceeds what the 24 MiB L2 absorbs. Measured on GB10 over 161
// shape/width points (bench/ops/nvfp4_route_sweep.cu, cold 48 MiB L2, 15 repeats): the two
// conditions below pick the faster tile at 19 of 20 held-out points, average 6.5% ahead across the
// widths they select (worst +1.0%), and the widths they decline are up to 40% behind.
constexpr bool nvfp4_a4_prefer_narrow_token_tile(std::int32_t rows, std::int32_t k,
                                                 std::int32_t tokens) {
    constexpr std::int32_t kWaveCtas = 48;
    const std::int32_t row_tiles    = rows / 128;
    const std::int32_t wide_tiles   = (tokens + 255) / 256;
    const std::int32_t narrow_tiles = (tokens + 127) / 128;
    // Whole waves times the token work one CTA owns, in units of one 128-token tile.
    const std::int64_t wide_cost =
        ((static_cast<std::int64_t>(row_tiles) * wide_tiles + kWaveCtas - 1) / kWaveCtas) * 2;
    const std::int64_t narrow_cost =
        (static_cast<std::int64_t>(row_tiles) * narrow_tiles + kWaveCtas - 1) / kWaveCtas;
    if (narrow_cost * 8 > 7 * wide_cost) return false;
    // Weight bytes = codes (k/2) + scales (k/16) per row. The narrow grid re-reads them
    // (narrow_tiles - wide_tiles) panel-times more; the measured boundary sits between 51 MiB,
    // where the tile is still 5% ahead, and 66 MiB, where it is 8% behind.
    const std::int64_t weight_bytes = static_cast<std::int64_t>(rows) * k * 9 / 16;
    return weight_bytes * (narrow_tiles - wide_tiles) <= (52LL << 20);
}

// The 1-term TMA prefill route for a shape, with the token tile chosen by that rule.
template <class Geometry>
constexpr Nvfp4A4Route nvfp4_a4_tma_prefill_route(std::int32_t tokens) {
    return nvfp4_a4_prefer_narrow_token_tile(Geometry::kOutputRows, Geometry::kInputRows, tokens)
               ? nvfp4_a4_tma_narrow_route()
               : nvfp4_a4_tma_route<>();
}

template <class Geometry, class Schedule>
constexpr Nvfp4A4Route nvfp4_a4_mma_route() {
    return {nvfp4_linear_a4_mma<Geometry, Schedule>, Nvfp4ScaleLayout::RowMajor, true};
}

template <Nvfp4A4Route (*Select)(std::int32_t)>
void launch_nvfp4_a4(const Tensor& x, const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch,
                     cudaStream_t stream) {
    const Nvfp4A4Route route = Select(x.ne[1]);
    launch_nvfp4_a4_quantize(x, weight, scratch, route.scales, stream,
                             route.reciprocal_quantize);
    route.launch(weight, out, scratch, x.ne[1], stream);
}

inline void launch_nvfp4_a4_pair(const Tensor& x, const Weight& weight, Tensor& out,
                                 Nvfp4A4Workspace scratch, cudaStream_t stream) {
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled128, stream);
    launch_nvfp4_a4_residual_tma_linear(
        nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled128),
        static_cast<__nv_bfloat16*>(out.data), stream);
}

// The same 2-term residual pair with the 64-row weight panel a shape table measured ahead of the
// grid-size heuristic's choice, over the widths that measurement covers. The panel halves the
// weight tile each CTA owns and doubles the grid, which pays while the grid is small enough to
// leave SMs idle in its last wave. From the prefill width up, thousands of CTAs run and the halved
// panel reads its weight tile twice as often, where it measures 10..30% behind on every shape;
// those widths are delegated to the public heuristic, which is the route they ran before.
inline void launch_nvfp4_a4_pair_r64(const Tensor& x, const Weight& weight, Tensor& out,
                                     Nvfp4A4Workspace scratch, cudaStream_t stream) {
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled128, stream);
    const Nvfp4A4Operands operands =
        nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled128);
    __nv_bfloat16* y = static_cast<__nv_bfloat16*>(out.data);
    if (x.ne[1] <= 128) {
        launch_nvfp4_a4_residual_tma_linear_r64(operands, y, stream);
        return;
    }
    launch_nvfp4_a4_residual_tma_linear(operands, y, stream);
}
} // namespace infernix::ops::detail
