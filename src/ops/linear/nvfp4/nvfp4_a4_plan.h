#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

struct Nvfp4A4Workspace {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    // Extent of the scale plane. The tiled layout writes over whole tiles, so it reaches past the
    // real token count; carrying the size lets the quantizer check that rather than trust that the
    // caller allocated through allocate_nvfp4_a4_workspace.
    std::size_t scale_bytes = 0;
    // The A4 residual profile stores `terms` row-major planes, plane p at codes + p * code_plane and
    // scales + p * scale_plane; A4 has one.
    int terms               = 1;
    std::size_t code_plane  = 0;
    std::size_t scale_plane = 0;
    float* row_scales       = nullptr;
};

// LinearPolicy::AllowA4Residual's term count. Two terms passed the decode gate with DFlash2 and
// n-gram copies on the official 27B (TV excess 4.2e-5 against a 2.4e-3 budget, INT8 KV), and run
// 1,787 / 1,217 / 887 tok/s there at one, two and four requests against three terms' 1,696 /
// 1,074 / 783 (RTX 5090, 2026-10-14): a third term costs time the gate does not need.
inline constexpr int kNvfp4A4ResidualTerms = 2;

inline std::size_t nvfp4_a4_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 A4 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("nvfp4 A4 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Nvfp4A4Workspace allocate_nvfp4_a4_workspace(Arena& arena, std::int32_t tokens,
                                             std::int32_t input_rows, int terms = 1,
                                             bool row_scaled = false) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("nvfp4 A4 workspace: invalid K");
    }
    if (terms != 1 && terms != kNvfp4A4ResidualTerms) {
        throw std::invalid_argument("nvfp4 A4 workspace: invalid activation planes");
    }
    const std::size_t code_bytes =
        nvfp4_a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 2);
    if (terms > 1) {
        // Residual scale planes are allocated over whole tiles like the one-term plane, so either
        // the row-major (MMA) or the 128-token tiled (TMA) layout fits. Each plane's extent is a
        // multiple of 16 bytes for every K that is a multiple of 256.
        const std::size_t scale_plane = nvfp4_a4_checked_bytes(
            nvfp4_a4_padded_tokens(tokens), static_cast<std::size_t>(input_rows) / 16);
        const DeviceSpan codes  = arena.alloc_bytes(code_bytes * terms, 256);
        const DeviceSpan scales = arena.alloc_bytes(scale_plane * terms, 256);
        float* row_scales       = nullptr;
        if (row_scaled) {
            const DeviceSpan rs =
                arena.alloc_bytes(nvfp4_a4_checked_bytes(tokens, sizeof(float)), 256);
            row_scales = static_cast<float*>(rs.data);
        }
        return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
                scale_plane * terms, terms, code_bytes, scale_plane, row_scales};
    }
    // The tiled layout addresses whole tiles, so the scale plane is allocated for the padded token
    // count: at most 255 tokens of scales, under 0.3 MiB on the widest registered K, and paid
    // whichever layout the quantizer then writes.
    const std::size_t scale_bytes = nvfp4_a4_checked_bytes(
        nvfp4_a4_padded_tokens(tokens), static_cast<std::size_t>(input_rows) / 16);
    const DeviceSpan codes  = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales = arena.alloc_bytes(scale_bytes, 256);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
            scale_bytes};
}

inline std::size_t nvfp4_a4_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                     int terms = 1, bool row_scaled = false) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_a4_workspace(layout, tokens, input_rows, terms, row_scaled);
    return layout.peak_bytes(1);
}

// Which 1-term A4 prefill token tile to run, on the 48 SMs of GB10 (both tiles hold one CTA/SM).
// A 256-token tile packs twice the work of a 128-token tile into one CTA, so it wins while the grid
// fills whole waves; when ceil(rows/128) * ceil(T/256) leaves a short last wave, or the last token
// tile is only partly real, the 128-token tile fills the machine better. Wave fill alone is not
// enough: halving the token tile doubles how many times each weight panel is streamed, so the win
// stops once that extra traffic exceeds what the 24 MiB L2 absorbs. Measured on GB10 over 161
// shape/width points (bench/ops/nvfp4_route_sweep.cu, cold 48 MiB L2, 15 repeats): the two
// conditions below pick the faster tile at 19 of 20 held-out points, average 6.5% ahead across the
// widths they select (worst +1.0%), and the widths they decline are up to 40% behind. Every route
// that picks a 1-term A4 TMA token tile -- the standalone shape tables, linear_add and
// linear_swiglu -- reads this one predicate, because the quantizer must write the scale layout the
// chosen tile reads.
constexpr bool nvfp4_a4_prefer_narrow_token_tile(std::int32_t rows, std::int32_t k,
                                                 std::int32_t tokens) {
    constexpr std::int32_t kWaveCtas = 48;
    const std::int32_t row_tiles    = rows / 128;
    const std::int32_t wide_tiles   = (tokens + 255) / 256;
    const std::int32_t narrow_tiles = (tokens + 127) / 128;
    // Whole waves times the token work one CTA owns, in units of one 128-token tile.
    const std::int64_t wide_cost =
        ((static_cast<std::int64_t>(row_tiles) * wide_tiles + kWaveCtas - 1) / kWaveCtas) * 2;
    const std::int64_t narrow_cost =
        (static_cast<std::int64_t>(row_tiles) * narrow_tiles + kWaveCtas - 1) / kWaveCtas;
    if (narrow_cost * 8 > 7 * wide_cost) return false;
    // Weight bytes = codes (k/2) + scales (k/16) per row. The narrow grid re-reads them
    // (narrow_tiles - wide_tiles) panel-times more; the measured boundary sits between 51 MiB,
    // where the tile is still 5% ahead, and 66 MiB, where it is 8% behind.
    const std::int64_t weight_bytes = static_cast<std::int64_t>(rows) * k * 9 / 16;
    return weight_bytes * (narrow_tiles - wide_tiles) <= (52LL << 20);
}

// reciprocal_quotient selects the reciprocal-multiply quotient formulation. It is measured
// faster only on latency-bound small grids; callers opt in explicitly per measured route and
// the default keeps historical division behavior. A workspace with more than one term is quantized
// as the A4 residual profile (row-major only): plane 0 holds the A4 encoding, and each later plane
// the NVFP4 encoding of what the earlier ones left.
void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream,
                              bool reciprocal_quotient = false);

void launch_nvfp4_rht16_bf16(const Tensor& x, __nv_bfloat16* output, cudaStream_t stream);


} // namespace infernix::ops::detail
