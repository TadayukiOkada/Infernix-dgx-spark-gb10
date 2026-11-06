#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream);
// The same 1-term SwiGLU GEMM on a 128-token tile (scales `Tiled128`) instead of the 256-token
// one, for the widths where the 256-token grid leaves a short last wave on GB10's 48 SMs --
// see nvfp4_a4_prefer_narrow_token_tile.
void launch_nvfp4_linear_swiglu_a4_tma_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                              cudaStream_t stream);
// The A4 residual planes (kNvfp4A4ResidualTerms) over one 128-token tile, scales Tiled128.
void launch_nvfp4_linear_swiglu_a4_residual_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                                cudaStream_t stream);
} // namespace infernix::ops::detail
