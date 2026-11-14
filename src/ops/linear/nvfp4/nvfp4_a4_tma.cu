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
            K>>(p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
                pdl::Dependency::Programmatic);
    else
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotions>, K>>(
            p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
            pdl::Dependency::Programmatic);
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

// The 1-term prefill GEMM on a 128-token tile instead of the 256-token one: the same 128-row weight
// panel and the same K order, with 4 stages to keep the producer ahead at 1 CTA/SM. Halving the
// token tile doubles the grid, which pays exactly where the 256-token grid leaves a short last wave
// across the 48 SMs or pads its last token tile -- see nvfp4_a4_prefer_narrow_token_tile, which
// decides per shape and width. The route that reaches this entry always carries the Tiled128 scale
// layout, which is what selects the 128-token schedule in `launch`.
void launch_nvfp4_a4_tma_linear_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    switch (p.k) {
    case 2560:
        launch<2560, 128, 4>(p, output, stream);
        return;
    case 5120:
        launch<5120, 128, 4>(p, output, stream);
        return;
    case 6144:
        launch<6144, 128, 4>(p, output, stream);
        return;
    case 17408:
        launch<17408, 128, 4>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 narrow-token TMA linear: unsupported geometry");
}

namespace {
inline constexpr Nvfp4A4TmaPromotions kTmaAllL2_256B{
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B};

template <int K, Nvfp4A4TmaPromotions Promotions = kNvfp4A4TmaWeightCodeL2_128B>
void launch_residual(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    // Support explicit Tiled256 residual scale layout (2 stages, 64-row weight panel to fit GB10's
    // 99 KiB shared memory and 168-register file with 2 activation planes per stage); the default
    // Tiled128 path runs 3 stages at 128x128 (or 128x64 for small grids).
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled256) {
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<256, 2, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
            p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
            pdl::Dependency::Programmatic);
        return;
    }
    // When 128-row tiles launch at most 24 CTAs (half of GB10's 48 SMs), 64-row tiles double
    // the CTA count while staying within a single 48-SM wave; when ctas_128 is 32..47, keeping
    // 128-row tiles finishes in 1 wave instead of spilling 64..94 CTAs into a 2nd wave.
    // For N = 2560 at T = 257..384, ctas_128 = 60 leaves a 25% tail wave; 64-row tiles double the
    // grid to 120 CTAs (2.5 waves of 48) eliminating the wave-tail efficiency cliff.
    const int ctas_128 = (p.rows / 128) * ((p.tokens + 127) / 128);
    if (ctas_128 <= 24 || (p.rows == 2560 && p.tokens > 256 && p.tokens <= 384)) {
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<128, 4, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
            p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
            pdl::Dependency::Programmatic);
        return;
    }
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotions, kNvfp4A4ResidualTerms, 128>, K>>(
        p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
        pdl::Dependency::Programmatic);
}

// The shape table's 64-row pair panel, over the same K set the public pair route serves. The
// stage/occupancy pair is the one that fits the 99 KiB shared-memory limit with two activation
// planes per stage and won the A/B recorded in the shape tables.
template <int K, Nvfp4A4TmaPromotions Promotions = kNvfp4A4TmaWeightCodeL2_128B>
void launch_residual_r64(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, Promotions, kNvfp4A4ResidualTerms, 64>, K>>(
        p, y, LinearIdentityEpilogue{}, stream, Nvfp4IdentityRows{},
        pdl::Dependency::Programmatic);
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
        launch_residual<6144, kTmaAllL2_256B>(p, output, stream);
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
        launch_residual_r64<6144, kTmaAllL2_256B>(p, output, stream);
        return;
    case 17408:
        launch_residual_r64<17408>(p, output, stream);
        return;
    }
    throw std::invalid_argument("NVFP4 residual TMA linear: unsupported K");
}
} // namespace infernix::ops::detail
