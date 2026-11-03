#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
void launch_nvfp4_a4_tma_linear(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                cudaStream_t stream);

// The 1-term TMA GEMM on a 128-token tile rather than the 256-token prefill tile, for the prefill
// widths whose 256-token grid leaves a short last wave across the 48 SMs or pads its last token tile
// (nvfp4_a4_prefer_narrow_token_tile picks them). Measured 2026-11-03 over 161 shape/width points:
// 2 to 18% ahead where the rule fires, 1 to 40% behind where it does not -- halving the token tile
// doubles the weight traffic each token tile re-reads, and that cost grows with the weight matrix.
// Registered for the K whose shapes were measured; K = 5120 was not.
void launch_nvfp4_a4_tma_linear_narrow(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream);

void launch_nvfp4_a4_residual_tma_linear(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                         cudaStream_t stream);

// The 2-term residual pair with a 64-row weight panel rather than the 128-row panel the grid
// heuristic in launch_nvfp4_a4_residual_tma_linear picks. Halving the panel each CTA owns doubles
// the grid, which pays on the shapes whose 128-row grid leaves a large fraction of the 48 SMs idle
// in its last wave -- the decode and MTP-verification band, up to 128 tokens. It measures 10..30%
// behind at prefill widths, so launch_nvfp4_a4_pair_r64 bounds it there. 3 stages at 1 block/SM is
// the only stage/occupancy pair that both fits the 99 KiB shared-memory limit with two activation
// planes per stage and won its A/B; the other measured pairs (4/1, 2/2) are candidates in
// bench/ops/nvfp4_route_sweep.cu, not product routes.
void launch_nvfp4_a4_residual_tma_linear_r64(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                             cudaStream_t stream);
} // namespace infernix::ops::detail
