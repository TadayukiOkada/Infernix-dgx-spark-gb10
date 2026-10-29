#pragma once

#include "ops/common/dflash_rope_table.h"
#include "ops/common/sincos_reduced.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace infernix::ops {

// D128 full-head 1-D RoPE frequencies for theta=1e7.
static __device__ __constant__ double kDflashRopeInvFrequency[64] = {
    INFERNIX_DFLASH_ROPE_1E7_INV_FREQUENCY};

__device__ __forceinline__ void dflash_rope_sincos(const std::int32_t* positions, int token,
                                                   int pair, float* sine, float* cosine) {
    const double angle = static_cast<double>(positions[token]) * kDflashRopeInvFrequency[pair];
    sincos_turns(angle, sine, cosine);
}

} // namespace infernix::ops
