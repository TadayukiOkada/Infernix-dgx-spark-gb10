#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

namespace infernix::ops::detail {
namespace {
template <int K, CUtensorMapL2promotion Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE,
          int Tiled128Rows = 128, int Tiled128Stages = 2>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<128, Tiled128Stages, (Tiled128Rows == 64 ? 2 : 1), Promotion, 1,
                                  Tiled128Rows>,
            K>>(p, y, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 64, 3>(p, output, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 64, 3>(p, output, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 64, 3>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch<6144, CU_TENSOR_MAP_L2_PROMOTION_NONE, 128, 2>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<17408, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 128, 2>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 TMA linear: unsupported geometry");
}

namespace {
template <int K, CUtensorMapL2promotion Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B>
void launch_residual(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    // When 128-row tiles launch at most 24 CTAs (half of GB10's 48 SMs), 64-row tiles double
    // the CTA count while staying within a single 48-SM wave; when ctas_128 is 32..47, keeping
    // 128-row tiles finishes in 1 wave instead of spilling 64..94 CTAs into a 2nd wave.
    const int ctas_128 = (p.rows / 128) * ((p.tokens + 127) / 128);
    if (ctas_128 <= 24) {
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotion, kNvfp4A4ResidualTerms, 64>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
        return;
    }
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotion, kNvfp4A4ResidualTerms, 128>, K>>(
        p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_residual_tma_linear(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                         cudaStream_t stream) {
    switch (p.k) {
    case 2560:
        launch_residual<2560, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case 5120:
        launch_residual<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case 6144:
        launch_residual<6144, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case 17408:
        launch_residual<17408, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 residual TMA linear: unsupported K");
}
} // namespace infernix::ops::detail
