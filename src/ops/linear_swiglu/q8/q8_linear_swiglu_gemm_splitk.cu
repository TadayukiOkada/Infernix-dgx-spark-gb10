#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_instances.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;
constexpr int kFirstExactT  = 2;
constexpr int kLastExactT   = 48;

template <int TileCols, Q8ScaleAccess ScaleAccess>
void launch_exact_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Geometry  = Q8LinearGeometry<2 * kIntermediate, kHidden>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    using Schedule  = std::conditional_t<
         (TileCols <= 32), Q8SlicedKRuntimeDefault<TileCols, ScaleAccess>,
         Q8A16SlicedKMmaSchedule<TileCols, 4, 1, 3, ScaleAccess, Cache::cg, Cache::cg,
                                 Q8ActivationStage::RuntimeActive>>;
    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileCols, true>,
        RowPolicy>(q8_linear_operands(x, w), ignored_output, epilogue, stream);
}

} // namespace

void q8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactT || cols > kLastExactT) {
        throw std::invalid_argument("Q8 LinearSwiGLU exact split-K requires T=2..48");
    }
    if (cols <= 4) {
        launch_exact_schedule<8, Q8ScaleAccess::Direct>(x, w, out, stream);
    } else if (cols <= 8) {
        launch_exact_schedule<8, Q8ScaleAccess::Shared>(x, w, out, stream);
    } else if (cols <= 16) {
        launch_exact_schedule<16, Q8ScaleAccess::Shared>(x, w, out, stream);
    } else if (cols <= 24) {
        launch_exact_schedule<24, Q8ScaleAccess::Shared>(x, w, out, stream);
    } else if (cols <= 32) {
        launch_exact_schedule<32, Q8ScaleAccess::Shared>(x, w, out, stream);
    } else if (cols <= 40) {
        launch_exact_schedule<40, Q8ScaleAccess::Shared>(x, w, out, stream);
    } else {
        launch_exact_schedule<48, Q8ScaleAccess::Shared>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
