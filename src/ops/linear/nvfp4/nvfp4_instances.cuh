#pragma once
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"

namespace infernix::ops::detail {
template <int Tokens, int Warps, int Stages, int RowTiles = 1>
inline constexpr int nvfp4_sliced_min_blocks_per_sm() {
    constexpr int kSharedBytes =
        Nvfp4A16SlicedKMmaSchedule<Warps, Tokens, 1, Cache::ca, Cache::cg,
                                   Nvfp4ActivationStage::ActiveOnly, Stages, RowTiles>::kSharedBytes;
    if constexpr (RowTiles == 1 && Warps <= 4 && kSharedBytes <= 99 * 1024 / 4) {
        return 4;
    } else if constexpr (RowTiles == 1 && Warps <= 8 && Tokens <= 8 && kSharedBytes <= 99 * 1024 / 3) {
        return 3;
    } else if constexpr (RowTiles == 1 && Warps <= 4 && kSharedBytes <= 99 * 1024 / 3) {
        return 3;
    } else if constexpr (kSharedBytes <= 99 * 1024 / 2) {
        return 2;
    } else {
        return 1;
    }
}

template <int Tokens, int Warps, int Stages, int RowTiles = 1>
using Nvfp4SlicedInstance =
    Nvfp4A16SlicedKMmaSchedule<Warps, Tokens,
                               nvfp4_sliced_min_blocks_per_sm<Tokens, Warps, Stages, RowTiles>(),
                               Cache::ca, Cache::cg, Nvfp4ActivationStage::ActiveOnly, Stages,
                               RowTiles>;
} // namespace infernix::ops::detail
