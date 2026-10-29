#pragma once
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_schedule.cuh"

namespace infernix::ops::detail {
template <int TileTokens, int ActiveTokens>
using Q8SlicedKDefault = Q8A16SlicedKMmaSchedule<
    TileTokens, 8, 1, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? Q8ScaleAccess::Shared : Q8ScaleAccess::Direct)>;

template <int TileTokens, Q8ScaleAccess ScaleAccess>
using Q8SlicedKRuntimeDefault = Q8A16SlicedKMmaSchedule<
    TileTokens, 8, 1, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    ScaleAccess, Cache::ca, Cache::cg, Q8ActivationStage::RuntimeActive>;

namespace q8_instances {
using MmaR32T16  = Q8A16MmaSchedule<32, 16, 64, 16, 8, 2, 4>;
using MmaR32T24  = Q8A16MmaSchedule<32, 24, 64, 16, 8, 2, 4>;
using MmaR32T32  = Q8A16MmaSchedule<32, 32, 64, 16, 16, 2, 4>;
using MmaR32T48  = Q8A16MmaSchedule<32, 48, 64, 16, 16, 2, 3>;
using MmaR32T64  = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using MmaR32T96  = Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>;
using MmaR32T128 = Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>;
using MmaR48T64  = Q8A16MmaSchedule<48, 64, 64, 48, 16, 2, 3>;
using MmaR64T32  = Q8A16MmaSchedule<64, 32, 64, 32, 16, 2, 3>;
using MmaR64T48  = Q8A16MmaSchedule<64, 48, 64, 32, 16, 2, 3>;
using MmaR64T96  = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
// The wide route of the linear family is the one schedule measured to want cg on the predicated
// path; every other schedule, here and in the six other Ops that instantiate this same tile, keeps
// the inherited ca.
using MmaR64T128 = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Q8MmaFragmentPipeline::PingPong,
                                    Cache::cg, Cache::cg, Cache::cg>;
using MmaR96T96  = Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>;
using MmaR128T64 = Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>;
using MmaR128T80 = Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>;
// K128 single-activation-stage schedules shared with fused consumers.
using MmaR64x16T48K128A1  = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using MmaR64x32T64K128A1  = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using MmaR64x32T96K128A1  = Q8A16MmaSchedule<64, 96, 128, 32, 24, 1, 2>;
using MmaR64x16T128K128A2 = Q8A16MmaSchedule<64, 128, 128, 16, 16, 2, 1>;
using MmaR64x16T256K64A2  = Q8A16MmaSchedule<64, 256, 64, 16, 32, 2, 1>;
using MmaR64x32T64K64A2   = Q8A16MmaSchedule<64, 64, 64, 32, 16, 2, 3>;
using MmaR64x32T128K64A2  = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2>;
using MmaR128x32T64K64A2  = Q8A16MmaSchedule<128, 64, 64, 32, 32, 2, 2>;
using MmaR128x64T128K64A1 = Q8A16MmaSchedule<128, 128, 64, 64, 32, 1, 2>;
using MmaR128x32T128K64A2 = Q8A16MmaSchedule<128, 128, 64, 32, 32, 2, 1>;

using SimtR8T4         = Q8A16SimtSchedule<8, 4, 1, 32, 2, Cache::cg, 2>;
using SimtR8T5         = Q8A16SimtSchedule<8, 5, 1, 32, 2, Cache::cg, 2>;
using SimtR8T6         = Q8A16SimtSchedule<8, 6, 1, 32, 2, Cache::cg, 2>;
using SimtR8T8         = Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 2>;
using SimtR1T4W8       = Q8A16SimtSchedule<1, 4, 8, 32, 2, Cache::cg, 1>;
using SimtR1T8W8       = Q8A16SimtSchedule<1, 8, 8, 32, 2, Cache::cg, 1>;
using GemvR4W1K16384   = Q8A16GemvSchedule<4, 1, 2, 16384>;
using SimtR4T4W2G16S2  = Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 2>;
using SimtR4T5W2G16S2  = Q8A16SimtSchedule<4, 5, 2, 16, 2, Cache::cg, 2>;
using SimtR4T8W2G16S2  = Q8A16SimtSchedule<4, 8, 2, 16, 2, Cache::cg, 2>;
using SlicedR16T16W8S2 = Q8A16SlicedKMmaSchedule<16, 8, 2, 2, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::RuntimeActive>;
using SlicedR16T16W4S2 = Q8A16SlicedKMmaSchedule<16, 4, 2, 3, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::RuntimeActive>;
using SlicedR16T32W4S2 = Q8A16SlicedKMmaSchedule<32, 4, 2, 2, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::RuntimeActive>;
using SlicedR16T8W16S2 = Q8A16SlicedKMmaSchedule<8, 16, 2, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                 Cache::cg, Q8ActivationStage::RuntimeActive>;
using SlicedR32T16W4S2 =
    Q8A16SlicedKMmaSchedule<16, 4, 2, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                            Q8ActivationStage::RuntimeActive, 0, 16, false, 2>;
using SlicedR32T16W8S2 =
    Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                            Q8ActivationStage::RuntimeActive, 0, 16, false, 2>;
} // namespace q8_instances
} // namespace infernix::ops::detail
