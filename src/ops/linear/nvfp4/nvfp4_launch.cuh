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
} // namespace infernix::ops::detail
