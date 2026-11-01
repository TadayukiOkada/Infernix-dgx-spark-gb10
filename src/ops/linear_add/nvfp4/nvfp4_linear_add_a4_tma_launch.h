#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream);
// The 2-term (A4 residual profile) lane: each column's activation planes multiplied by the same
// weight tile into one accumulator, then added to `residual`. Reads a `Tiled128` scale plane.
void launch_nvfp4_a4_tma_linear_add_residual(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                             cudaStream_t stream);
} // namespace infernix::ops::detail
