#pragma once

// Q8 W8A8 fast-path prefill (q8_g32_fp16; LinearPolicy::AllowA8 or AllowA4 only — the A16,
// A16ColumnInvariant and reproducible routes never take it): BF16 activations quantized per
// 32-element K group to int8 codes with an FP32 scale (ops/ggml/ggml_a8.h's codec), products on the
// int8 Tensor Cores (m16n8k32 s8, ops/linear/q8/q8_a8_mma.cuh) with the exact int32 group dots
// scaled by the weight's FP16 group scale and the activation's group scale into FP32 accumulators.
// It serves the registered problems from kQ8A8MinTokens columns; every other call keeps the exact
// A16 routes. LOSSY (activation quantization), opt-in by the artifact's activation policy.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The route's admission widths (prefill), as the f16-accumulate route's.
inline constexpr std::int32_t kQ8A8MinTokens = 128;

// Encoded columns are batched through the workspace at this width, so the encoded batch stays L2
// resident while the row tiles sweep it.
inline constexpr std::int32_t kQ8A8BatchColumns = 2048;

// Whether [n, k] registers the route (a K that slices into whole 32-element groups and complete
// 64-element MMA stages, with rows the tile or its predicate can cover).
[[nodiscard]] bool q8_a8_problem(std::int32_t n, std::int32_t k) noexcept;

// Whether a call of `tokens` columns on [n, k] takes the route.
[[nodiscard]] bool q8_a8_admits(std::int32_t n, std::int32_t k, std::int32_t tokens) noexcept;

// The encoded activations a call of up to max_tokens columns needs (0 when the route cannot run).
[[nodiscard]] std::size_t q8_a8_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                         std::int32_t max_tokens);

// out [n, T] BF16 on the route (a registered problem, T >= kQ8A8MinTokens), the encoded
// activations in `workspace` (scoped).
void q8_linear_a8(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream);

} // namespace infernix::ops::detail
