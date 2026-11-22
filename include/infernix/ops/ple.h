#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Per-layer n-gram embedding injection (PLE) into hyper-connection residual streams
 * (docs/maintainer/qwen3_8-flash-next-design.md §12). For each column t, with S streams of H
 * features (residual R [S*H, T], stream-major):
 *
 *   e        = bf16(fp8_row * scale) for each of the column's n-gram rows       -> ple_embed
 *   key      = W_key e  [S*H],  value = W_value e  [H]                          (caller GEMMs)
 *   g_s      = <norm(key_s; w_key), norm(R_s; w_query)> / sqrt(H)
 *   g_s      = sign(g_s) * sqrt(max(|g_s|, 1e-6))
 *   gated_s  = sigmoid(g_s) * value                                             -> ple_gate
 *   nrm      = norm(gated; w_conv)                                              -> ple_gate
 *   R       += gated + SiLU(sum_j w_conv1d[j] * nrm[t - (K-1-j) * dilation])     -> ple_conv_inject
 *
 * norm() is the per-stream RMSNorm with unit-offset weight. The convolution is depthwise over the
 * S*H channels and causal; its state holds the last (K-1)*dilation columns of nrm of each
 * sequence. Every Op evaluates in FP32 from represented inputs and rounds each output once;
 * the oracle is the formula in FP64.
 */

/// rows: U8 [W, heads, T] n-gram codes; scale: [1]. out: BF16 [heads*head_width, T],
/// out[h*head_width + d, t] = the decoded cell times the global scale. With a BF16 scale, W is
/// head_width FP8 E4M3FN codes and the cell is e4m3(rows[h, t][d]) * scale. With an FP32 scale, W
/// is head_width/2 packed E2M1 codes followed by head_width/16 E4M3FN block scales (one per 16
/// codes) and the cell is e2m1(code[d]) * e4m3(scale[d / 16]) * scale.
void ple_embed(const Tensor& rows, const Tensor& scale, Tensor& out, cudaStream_t stream);

/// NVFP4 rows (NVIDIA FP4-Local's table): rows U8 [W/2 + W/16, heads, T], each row its W/2 code
/// bytes (E2M1, element 2i in the low nibble of byte i) then its W/16 E4M3FN block scales;
/// multiplier: FP32 [1], positive and finite. out: BF16 [heads*W, T],
/// out[h*W + j, t] = bf16(e2m1(c_j) * e4m3fn(s_{j/16}) * multiplier), the exact product rounded once
/// to nearest even (W a multiple of 16).
void ple_embed_nvfp4(const Tensor& rows, const Tensor& multiplier, Tensor& out, cudaStream_t stream);

/// Hash table and device-mapped resident NVFP4 volume for fused GPU-side PLE hashing and
/// decoding (GPU-direct UMA / unified-memory resident table).
struct PleNgramHashTable {
    const void* resident_blocks   = nullptr; // device pointer to resident [blocks, block_bytes]
    const Tensor* multipliers     = nullptr; // I64 [ngram_size] (bitwise U64)
    const Tensor* sizes           = nullptr; // I64 [heads] (bitwise U64 primes)
    const Tensor* offsets         = nullptr; // I64 [heads] (bitwise U64 row offsets)
    const Tensor* multiplier      = nullptr; // FP32 [1] global NVFP4 table multiplier
    std::int32_t ngram_size       = 0;
    std::int32_t heads_per_ngram  = 0;
    std::int32_t eos_token_id     = 0;
    std::int32_t vocab_size       = 0;
    std::int32_t head_width       = 0;
    std::int32_t row_bytes        = 0;
    std::int32_t rows_per_block   = 0;
    std::int32_t block_bytes      = 0;
    std::uint64_t total_rows      = 0;
};

/// Fused GPU hash + resident NVFP4 lookup + BF16 embed. windows: I32 [ngram_size, T] where
/// windows[s, t] is the context token s steps before column t (s = 0 is the current column's
/// token; earlier positions before sequence start hold eos_token_id). out: BF16 [heads*W, T].
/// Produces bit-identical output to host NgramHash::row_ids + NgramVolume::read_rows +
/// ple_embed_nvfp4.
void ple_hash_embed_nvfp4(const Tensor& windows, const PleNgramHashTable& table, Tensor& out,
                          cudaStream_t stream);

/// key: BF16 [S*H, T]; value: BF16 [H, T]; residual: BF16 [S*H, T]; key_norm, query_norm,
/// conv_norm: [S*H], all BF16 or all FP32 (NVIDIA FP4-Local's). gated and normalized: BF16 [S*H, T].
void ple_gate(const Tensor& key, const Tensor& value, const Tensor& residual,
              const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
              std::int32_t streams, float eps, Tensor& gated, Tensor& normalized,
              cudaStream_t stream);

/// The T columns form `sequences` runs of T / sequences consecutive columns. Sequence i reads its
/// convolution history from state slot source_slots[i] and writes the updated history to slot
/// destination_slots[i] (both I32 [sequences]; equal slots update in place). An empty
/// destination_slots leaves every state unchanged (speculative verification, committed later by
/// ple_conv_commit). When tree_rows is supplied (I32 [kSpeculativeTreeRowWords, sequences]), each
/// sequence whose tree != 0 walks parent[col] per tap lag instead of col - lag. weight: BF16 [C, K]
/// (tap j of channel c at j*C + c); states: BF16 [C, span, slots] with span = (K-1) * dilation,
/// oldest column first. residual is updated in place. When next_norm_weight and next_norm_out
/// are supplied, also writes the per-stream unit-offset RMSNorm of the updated residual (matching
/// hyper_connection_norm over `streams`).
void ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                     std::int32_t dilation, Tensor& states, const Tensor& source_slots,
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream,
                     const Tensor* next_norm_weight = nullptr, Tensor* next_norm_out = nullptr,
                     float next_norm_eps = 0.0F, std::int32_t streams = 4,
                     const Tensor* tree_rows = nullptr);

/// ple_gate immediately followed by ple_conv_inject, fused into one kernel when the geometry is a
/// decode column per sequence (one column each, so width == 1) with a state update, an aligned
/// 8-element-vectorizable gate and convolution, and a history span within the folded limit: the
/// gated and normalized values then stay in registers between the two stages and never reach
/// global memory. The fused result is bitwise-identical to the composed pair; any other geometry
/// (or an invalid one, which throws) takes exactly the composed ple_gate + ple_conv_inject, for
/// which gated and normalized are the scratch (and must satisfy ple_gate's contract). Optional
/// next_norm_weight / next_norm_out / next_norm_eps fuse the following block's per-stream HC
/// RMSNorm of the updated residual.
void ple_gate_conv_inject(const Tensor& key, const Tensor& value, Tensor& residual,
                          const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
                          std::int32_t streams, float eps, const Tensor& weight, std::int32_t dilation,
                          Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                          Tensor& gated, Tensor& normalized, cudaStream_t stream,
                          const Tensor* next_norm_weight = nullptr, Tensor* next_norm_out = nullptr,
                          float next_norm_eps = 0.0F, const Tensor* tree_rows = nullptr);

/// Commits the first commit_columns[b] (I32 [B], each in [0, W]) of row b's verified columns to
/// slot slots[b] (I32 [B]): the history becomes the trailing span columns of the old history
/// followed by normalized[:, 0:n, b] (normalized: BF16 [C, W, B], the conv inputs ple_gate
/// produced for those columns). Zero leaves the row's history unchanged. The result equals the
/// in-place ple_conv_inject state update over those n columns.
void ple_conv_commit(const Tensor& normalized, const Tensor& commit_columns, Tensor& states,
                     const Tensor& slots, cudaStream_t stream);

} // namespace infernix::ops
