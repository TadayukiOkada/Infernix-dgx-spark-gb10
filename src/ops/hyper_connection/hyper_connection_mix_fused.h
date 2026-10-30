#pragma once

// The fused route of hyper_connection_mix (docs/maintainer/qwen3_8-flash-next-design.md §8.3
// K1a/K1b, §19.3.3 Phase 1b) for Q8_G32_FP16 RowSplit and BF16 contiguous weights. Private to the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// Columns the fused route serves in one call, per weight codec. GB10 measurement (2026-10-30,
// infernix_hyper_connection_bench, interleaved against the composed route at every width): the fused
// SIMT mix wins while its per-column dot-product loop amortizes under the MMA linear - for Q8
// weights to 3 columns (T=3 39.7 us against the composed route's 41.9; T=4 43.8 composed against
// 45.2 fused), for BF16 weights to 4 (T=4 64.1 fused against 67.1 composed; T=5 68.7 against 71.0),
// because the BF16 linear reads 2x the bytes. The previous 16 left the speculative widths (MTP's 4,
// DFlash2's 8/9 and 16) on the losing side of that crossover.
inline constexpr std::int32_t kHcMixFusedMaxColumnsQ8   = 3;
inline constexpr std::int32_t kHcMixFusedMaxColumnsBf16 = 4;

// Upper bound of the fused band across the codecs: sizes the kernels' shared arrays, the fused
// workspace and the launch attributes; the dispatch itself uses the per-codec limits above.
inline constexpr std::int32_t kHcMixFusedMaxColumns = 4;

// Whether the fused route serves this problem: both weights Q8_G32_FP16 RowSplit or both BF16
// contiguous, the registered geometry (4 streams of 2560, rank a multiple of 8 up to 512) and the
// codec's columns above.
[[nodiscard]] bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden,
                                          std::int32_t streams, std::int32_t rank, std::int32_t columns) noexcept;

// FP32 split-K partials [S][down rows][T] and per-stream inverse RMS [S][T].
[[nodiscard]] std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams,
                                                       std::int32_t columns) noexcept;

// K1a then K1b. `partials` holds hc_mix_fused_workspace_bytes; inject (FP32 [S, T]) may be null.
void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream);

} // namespace infernix::ops::detail
