#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream);
// The same 1-term residual-add GEMM on a 128-token tile (scales `Tiled128`) instead of the
// 256-token one, for the widths where the 256-token grid leaves a short last wave on GB10's
// 48 SMs -- see nvfp4_a4_prefer_narrow_token_tile.
void launch_nvfp4_a4_tma_linear_add_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                           cudaStream_t stream);
// The 2-term (A4 residual profile) lane: each column's activation planes multiplied by the same
// weight tile into one accumulator, then added to `residual`. Reads a `Tiled128` scale plane.
void launch_nvfp4_a4_tma_linear_add_residual(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                             cudaStream_t stream);
} // namespace infernix::ops::detail
