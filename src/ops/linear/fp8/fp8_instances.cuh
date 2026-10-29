#pragma once
#include "ops/linear/fp8/fp8_schedule.cuh"

namespace infernix::ops::detail {
using Fp8A8T64R128K128 =
    Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R64K128 = Fp8A8MmaSchedule<16, 64, 128, 1, 2, 2, 4, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T16R128K128 =
    Fp8A8MmaSchedule<16, 128, 128, 1, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R32K128 = Fp8A8MmaSchedule<32, 32, 128, 1, 2, 3, 3, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R64K128 = Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T32R128K128 =
    Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128 = Fp8A8MmaSchedule<64, 64, 128, 2, 2, 3, 2, Cache::cg, Cache::cg,
                                         Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R64K128S2 =
    Fp8A8MmaSchedule<64, 64, 128, 2, 2, 2, 3, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Fp8A8T64R128K256 =
    Fp8A8MmaSchedule<64, 128, 256, 2, 4, 2, 1, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
// The A8 residual profile's tiles: two activation planes per stage.
using Fp8A8ResidualT32R64 =
    Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1, 2>;
using Fp8A8ResidualT32R128 =
    Fp8A8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1, 2>;
using Fp8A8ResidualT64R64 =
    Fp8A8MmaSchedule<64, 64, 128, 2, 2, 2, 2, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1, 2>;
using Fp8A8ResidualT64R128 =
    Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                     Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::Grouped, 8, 2>;

using Fp8A16TmaT32R64          = Fp8A16TmaMmaSchedule<32, 64, 1, 4, 6, 1>;
using Fp8A16TmaT32R64S3        = Fp8A16TmaMmaSchedule<32, 64, 1, 4, 3, 2>;
using Fp8A16TmaT32R128         = Fp8A16TmaMmaSchedule<32, 128, 1, 8, 4, 1>;
using Fp8A16TmaT64R128         = Fp8A16TmaMmaSchedule<64, 128, 2, 4, 3, 1>;

using Fp8A16RingT32R64   = Fp8A16TmaT32R64;
using Fp8A16RingT32R64S3 = Fp8A16TmaT32R64S3;
using Fp8A16RingT32R128  = Fp8A16TmaT32R128;
using Fp8A16RingT64R128  = Fp8A16TmaT64R128;
using Fp8A16TmaT64R128G8       = Fp8A16TmaMmaSchedule<64, 128, 2, 4, 3, 1, 128, Fp8MmaRaster::Grouped, 8>;
using Fp8A16TmaT32R128S4B2K64  = Fp8A16TmaMmaSchedule<32, 128, 1, 4, 4, 2, 64>;
using Fp8A16TmaT64R128S3B2K64  = Fp8A16TmaMmaSchedule<64, 128, 2, 4, 3, 2, 64>;
using Fp8A16TmaT96R128S3K64    = Fp8A16TmaMmaSchedule<96, 128, 2, 4, 3, 1, 64>;
using Fp8A16TmaT128R128S2B2K64 = Fp8A16TmaMmaSchedule<128, 128, 2, 4, 2, 2, 64>;
using Fp8A16TmaT64R256RF       = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1, 128, Fp8MmaRaster::RowFast>;
using Fp8A16TmaT128R128RF      = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1, 128, Fp8MmaRaster::RowFast>;
using Fp8A16TmaT128R128G8      = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1, 128, Fp8MmaRaster::Grouped, 8>;
using Fp8A16TmaT64R256G8       = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1, 128, Fp8MmaRaster::Grouped, 8>;

using Fp8A16RingT32R64   = Fp8A16RingMmaSchedule<32, 64, 1, 4, 6, 1>;
using Fp8A16RingT32R64S3 = Fp8A16RingMmaSchedule<32, 64, 1, 4, 3, 2>;
using Fp8A16RingT32R128  = Fp8A16RingMmaSchedule<32, 128, 1, 8, 4, 1>;
using Fp8A16RingT64R128  = Fp8A16RingMmaSchedule<64, 128, 2, 4, 3, 1>;

template <int Tokens, int Warps, int Stages, int RowTiles = 1>
inline constexpr int fp8_sliced_min_blocks_per_sm() {
    constexpr int kSharedBytes =
        Fp8A16SlicedKMmaSchedule<Warps, Tokens, 1, Cache::ca, Cache::cg,
                                 Fp8ActivationStage::ActiveOnly, Stages, RowTiles>::kSharedBytes;
    if constexpr (Warps <= 4 && kSharedBytes <= 99 * 1024 / 4) {
        return 4;
    } else if constexpr (Warps <= 8 && Tokens <= 8 && kSharedBytes <= 99 * 1024 / 3) {
        return 3;
    } else if constexpr (Warps <= 4 && kSharedBytes <= 99 * 1024 / 3) {
        return 3;
    } else if constexpr (kSharedBytes <= 99 * 1024 / 2) {
        return 2;
    } else {
        return 1;
    }
}

template <int Tokens, int Warps, int Stages>
using Fp8SlicedInstance =
    Fp8A16SlicedKMmaSchedule<Warps, Tokens, fp8_sliced_min_blocks_per_sm<Tokens, Warps, Stages, 1>(),
                             Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, Stages>;
template <int Tokens, int Warps, int Stages>
using Fp8SlicedR2Instance =
    Fp8A16SlicedKMmaSchedule<Warps, Tokens, fp8_sliced_min_blocks_per_sm<Tokens, Warps, Stages, 2>(),
                             Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly, Stages, 2>;
} // namespace infernix::ops::detail
