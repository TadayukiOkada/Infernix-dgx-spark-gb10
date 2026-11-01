#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

namespace infernix::ops::detail {
namespace {
// The 1-term A4 GEMM instances the public route runs, one per registered K. A 256-token tile owns
// the prefill band; the 128-token tile takes the 129..256 band, where a 256-token tile would spend
// half its M rows on padding. MinBlocksPerSm follows from the row count: the 64-row tile keeps two
// CTAs resident per SM, the 128-row tile needs the whole register file for one.
template <int K, int Tiled128Rows, int Tiled128Stages,
          Nvfp4A4TmaPromotions Promotions = kNvfp4A4TmaWeightCodeL2_128B>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<128, Tiled128Stages, (Tiled128Rows == 64 ? 2 : 1), Promotions, 1,
                                  Tiled128Rows>,
            K>>(p, y, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotions>, K>>(p, y, LinearIdentityEpilogue{},
                                                              stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                cudaStream_t stream) {
    switch (p.k) {
    case 2560:
        launch<2560, 64, 3>(p, output, stream);
        return;
    case 5120:
        launch<5120, 64, 3>(p, output, stream);
        return;
    case 6144:
        launch<6144, 64, 3>(p, output, stream);
        return;
    case 17408:
        launch<17408, 64, 3>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 TMA linear: unsupported geometry");
}

namespace {
template <int K, Nvfp4A4TmaPromotions Promotions = kNvfp4A4TmaWeightCodeL2_128B>
void launch_residual(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    // Support explicit Tiled256 residual scale layout (2 stages, 64-row weight panel to fit GB10's
    // 99 KiB shared memory and 168-register file with 2 activation planes per stage); the default
    // Tiled128 path runs 3 stages at 128x128 (or 128x64 for small grids).
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled256) {
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<256, 2, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
        return;
    }
    // When 128-row tiles launch at most 24 CTAs (half of GB10's 48 SMs), 64-row tiles double
    // the CTA count while staying within a single 48-SM wave; when ctas_128 is 32..47, keeping
    // 128-row tiles finishes in 1 wave instead of spilling 64..94 CTAs into a 2nd wave.
    const int ctas_128 = (p.rows / 128) * ((p.tokens + 127) / 128);
    if (ctas_128 <= 24) {
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
        return;
    }
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotions, kNvfp4A4ResidualTerms, 128>, K>>(
        p, y, LinearIdentityEpilogue{}, stream);
}

// The shape table's 64-row pair panel, over the same K set the public pair route serves. The
// stage/occupancy pair is the one that fits the 99 KiB shared-memory limit with two activation
// planes per stage and won the A/B recorded in the shape tables.
template <int K, Nvfp4A4TmaPromotions Promotions = kNvfp4A4TmaWeightCodeL2_128B>
void launch_residual_r64(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
        p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_residual_tma_linear(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                         cudaStream_t stream) {
    switch (p.k) {
    case 2560:
        launch_residual<2560>(p, output, stream);
        return;
    case 5120:
        launch_residual<5120>(p, output, stream);
        return;
    case 6144:
        launch_residual<6144>(p, output, stream);
        return;
    case 17408:
        launch_residual<17408>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 residual TMA linear: unsupported K");
}

void launch_nvfp4_a4_residual_tma_linear_r64(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                             cudaStream_t stream) {
    switch (p.k) {
    case 2560:
        launch_residual_r64<2560>(p, output, stream);
        return;
    case 5120:
        launch_residual_r64<5120>(p, output, stream);
        return;
    case 6144:
        launch_residual_r64<6144>(p, output, stream);
        return;
    case 17408:
        launch_residual_r64<17408>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 residual TMA linear: unsupported K");
}
} // namespace infernix::ops::detail
