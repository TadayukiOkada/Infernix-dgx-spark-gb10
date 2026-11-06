#pragma once

// The wide-route grouped GEMM of offloaded_sparse_moe (wide_expert.h). It is the warp-specialized
// TMA pipeline of the dense NVFP4 W4A4 route (ops/linear/nvfp4/nvfp4_a4_tma.cuh) made grouped and
// persistent:
//   - A operand, activations: the same TMA path. A token tile's A4 codes arrive through a
//     64-byte-swizzled tensor map and are read with the same ldmatrix fragments, its scales through
//     a second map; both planes hold one row per routed entry, grouped by expert.
//   - B operand, weights: each record's `nvfp4_expert_rg16_v1` units are bulk-copied unchanged
//     (one cp.async.bulk per row group and K tile, from wherever the job's record lives) and turned
//     into mxf4nvf4 fragments in registers. No tensor-core layout leaks into the record, and no
//     weight is repacked in memory.
//   - Work: a device-built list of (job, 64-column tile) pairs times the matrix's 128-row blocks,
//     walked by a persistent grid, so the pipeline runs on across tiles and experts.
//   - Epilogues receive FP32 accumulators and apply the canonical boundaries of §16.2.

#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include "core/device.h"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::offloaded_moe::wide {

// The activation planes of one call: A4(x) for the gate/up GEMM and A4(h) for the down GEMM, and with
// two-term A4 (W4A16 fast path) the later terms' planes.
struct alignas(128) Descriptors {
    CUtensorMap x_codes;
    CUtensorMap x_scales;
    CUtensorMap h_codes;
    CUtensorMap h_scales;
    CUtensorMap x2_codes;
    CUtensorMap x2_scales;
    CUtensorMap h2_codes;
    CUtensorMap h2_scales;
    CUtensorMap x3_codes;
    CUtensorMap x3_scales;
    CUtensorMap h3_codes;
    CUtensorMap h3_scales;
};

enum class Matrix { GateUp, Down };

// A matrix of an expert record of geometry G (w4a4_expert.h): its 16-row groups, its K blocks and its
// offset in the record. Flash-Next: gate/up 80 x 160, down 160 x 40; Qwen3.5-122B-A10B (the A4 residual
// routes only): gate/up 128 x 192, down 192 x 64.
template <Matrix M, class G = FlashNextExperts>
struct MatrixShape {
    static constexpr int kRowGroups      = M == Matrix::GateUp ? G::kGateUpRowGroups : G::kDownRowGroups;
    static constexpr int kBlocks         = M == Matrix::GateUp ? G::kGateUpBlocks : G::kDownBlocks;
    static constexpr std::size_t kOffset = M == Matrix::GateUp ? 0 : G::kGateUpBytes;
};

// The tile shape follows Nvfp4A4TmaMmaSchedule (128 rows, 128-wide K tiles of two 64-wide MMA
// steps, one producer warp beside the consumers) with a 64-column token tile: an expert sees
// about 20 columns per 1K-token chunk and 80 per 4K chunk, where the dense schedule's 128- or
// 256-token tiles (fixed by its tiled scale layout) would mostly compute padding. Eight consumer
// warps tile it 2 (columns) x 4 (rows); a warp's 32 rows are one 16-intermediate block of h.
struct Schedule {
    static constexpr int kTerms            = 1;            // A4 activation terms per element
    static constexpr int kBlockTokens      = kTileColumns; // 64
    static constexpr int kBlockRows        = 128;
    static constexpr int kRowGroups        = kBlockRows / 16; // 8
    static constexpr int kBlockK           = 128;
    static constexpr int kUnitsPerKTile    = kBlockK / 16; // 8 record units per row group
    static constexpr int kStages           = 3;
    static constexpr int kMinBlocksPerSm   = 2;
    static constexpr int kWarpsTokens      = 2;
    static constexpr int kWarpsRows        = 4;
    static constexpr int kConsumerWarps    = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads  = kConsumerWarps * 32;
    static constexpr int kProducerThreads  = 32;
    static constexpr int kThreads          = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens       = kBlockTokens / kWarpsTokens; // 32
    static constexpr int kWarpRows         = kBlockRows / kWarpsRows;     // 32
    static constexpr int kMmaTokens        = kWarpTokens / 16;            // 2
    static constexpr int kMmaRows          = kWarpRows / 8;               // 4: lo/hi halves of 2 row groups
    static constexpr int kK64PerStage      = kBlockK / 64;                // 2
    static constexpr int kCodeRowBytes     = kBlockK / 2;                 // 64
    static constexpr int kScaleWordsPerRow = 4;                           // a 16-byte scale box: 2 K tiles
    static constexpr int kUnitRunBytes     = kUnitsPerKTile * static_cast<int>(kUnitBytes); // 1,152
    static constexpr std::uint32_t kStageBytes =
        kBlockTokens * kCodeRowBytes + kBlockTokens * kScaleWordsPerRow * 4 + kRowGroups * kUnitRunBytes;
    static constexpr std::uint32_t kHalfStageBytes =
        kWarpTokens * kCodeRowBytes + kWarpTokens * kScaleWordsPerRow * 4 + kRowGroups * kUnitRunBytes;
    // Consumer warps compute only the 16-column MMA tiles of the 64-column token tile that hold a
    // valid column (the epilogues never write a column past the tile's count, so a skipped tile's
    // zeroed accumulators are unobservable and every output bit is unchanged). An expert sees about
    // 20 columns per 1K-token chunk and 80 per 4K chunk, so most tiles' upper MMA tiles are padding.
    static constexpr bool kSkipIdleColumns = true;
    static_assert(kWarpRows == 32, "a warp's rows are one A4 block of h");
    static_assert(kUnitRunBytes % 16 == 0, "bulk copies move multiples of 16 bytes");
};

// A4 residual (W4A16 fast paths) at expert geometry G: each stage holds every term's activation tile
// beside one weight tile, and two stages (39 KB at two terms, 48 KB at three) keep two CTAs per SM.
// The tile is Flash-Next's measured one (recipe C and FP4-Local: +33-35 % at 32K-64K prefill). At
// Qwen3.5-122B-A10B's geometry it is also the right one:
//   - its K loops are longer (gate/up 24 K tiles, down 8, against 20 and 5), so each work tile's
//     pipeline fill and epilogue are a smaller share; its 16 and 24 row blocks of 128 keep a call's
//     work items in several persistent waves even at 1K-token chunks;
//   - its experts see about 32 columns per 1K chunk and 128 per 4K (256 experts, top 8), so wider
//     token tiles would compute mostly padding, and a 64-column warp tile (twice the weight fragment
//     reuse) needs about 140 registers a thread, past the 112 that two CTAs of 288 threads allow;
//   - a third stage would cost the second CTA (58 KB a CTA at two terms).
// Its consumer warps inherit Schedule's kSkipIdleColumns: at 1K-token chunks about half of the
// tiles' columns are padding, which the outputs never read, so the skip removes that tensor work and
// its fragment loads without changing any output bit.
template <int Terms, class G = FlashNextExperts>
struct ScheduleResidual : Schedule {
    static_assert(Terms >= 2 && Terms <= kMaxResidualTerms);
    static constexpr int kTerms  = Terms;
    static constexpr int kStages = 2;
    static constexpr std::uint32_t kStageBytes =
        kTerms * (kBlockTokens * kCodeRowBytes + kBlockTokens * kScaleWordsPerRow * 4) + kRowGroups * kUnitRunBytes;
    static constexpr std::uint32_t kHalfStageBytes =
        kTerms * (kWarpTokens * kCodeRowBytes + kWarpTokens * kScaleWordsPerRow * 4) + kRowGroups * kUnitRunBytes;
};

// Schedule Base with a 64-token warp tile (Basalt ebf6928), for calls of kWarpTile64MinColumns columns per
// expert or more on average (wide_expert.h). Four consumer warps each take every column of the 64-column token tile against
// their 32 rows, so each B fragment (rg16_fragments' decode of a row group's staged units) feeds four
// 16-column MMA tiles instead of two: half the B decodes and shared loads per MMA. The stages, the K
// order of every output's MMA chain and so every output bit are Base's; the 160-thread CTA keeps two
// per SM within its shared memory, with registers to spare for the doubled accumulators.
template <class Base>
struct WarpTile64 : Base {
    static constexpr int kWarpsTokens     = 1;
    static constexpr int kWarpsRows       = 4;
    static constexpr int kConsumerWarps   = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kThreads         = kConsumerThreads + Base::kProducerThreads;
    static constexpr int kWarpTokens      = Base::kBlockTokens / kWarpsTokens; // 64
    static constexpr int kMmaTokens       = kWarpTokens / 16;                  // 4
    static_assert(Base::kBlockRows / kWarpsRows == Base::kWarpRows, "the warp's rows stay one A4 block of h");
};

template <class S>
using AccumulatorsOf = float[S::kMmaTokens][S::kMmaRows][4];

template <class S>
struct Storage {
    alignas(1024) std::uint8_t a_codes[S::kStages][S::kTerms][S::kBlockTokens * S::kCodeRowBytes];
    alignas(128) std::uint32_t a_scales[S::kStages][S::kTerms][S::kBlockTokens * S::kScaleWordsPerRow];
    alignas(128) std::uint8_t b_units[S::kStages][S::kRowGroups * S::kUnitRunBytes];
    alignas(8) std::uint64_t full[S::kStages];
    alignas(8) std::uint64_t empty[S::kStages];
};

// One work tile as the consumer threads see it.
struct TileContext {
    int expert;    // the job's expert
    int first;     // packed row (plane row, dispatch.entries index) of the tile's first column
    int columns;   // valid columns in the tile, 1..64
    int row_block; // 128-row block of the matrix
    int warp_m;    // column half of the tile (32 columns)
    int warp_n;    // 32-row quarter of the block
    int group;     // lane / 4
    int tig;       // lane % 4
    int col_base;  // column of the warp's first live 16-column MMA tile (the two column warps split
                   // the tile's live tiles evenly, so this is not always warp_m * kWarpTokens)
    int live_tiles; // the warp's live MMA tiles at col_base (the epilogues write exactly these)
};

// Accumulator acc[m][r][v] of a consumer thread holds column
//   col_base + m * 16 + group + (v >= 2 ? 8 : 0)
// of the tile and matrix row
//   row_block * 128 + tile_row(warp_n, r, tig) + (v & 1).
// Rows 2i and 2i + 1 are therefore gate_i and up_i of the gate/up matrix in the same thread.
__device__ __forceinline__ int tile_row(int warp_n, int r, int tig) {
    return (2 * warp_n + (r >> 1)) * 16 + 8 * (r & 1) + 2 * tig;
}

using Accumulators = float[Schedule::kMmaTokens][Schedule::kMmaRows][4];

__device__ __forceinline__ void bulk_load(void* destination, const void* source, std::uint32_t bytes,
                                          std::uint64_t* barrier) {
    // The CTA-local destination form: with a shared::cluster destination ptxas guards every copy with
    // a call to a driver routine for remote CTAs, and the driver then reserves ~14 KB of stack per
    // resident thread (3.3 GiB on 170 SMs) at the kernel's first launch.
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :
                 : "r"(smem_addr(destination)), "l"(source), "r"(bytes), "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ std::uint32_t lds32(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

// The B fragments of one row group for one 64-wide K step from its four staged record units
// (unit u at units + 144 u covers K block u of the step). MMA K slots 0-15, 16-31, 32-47, 48-63
// take blocks 0, 2, 1, 3 (the plane_block order of the activations), so the four lanes of a
// group read four distinct 16-byte bank ranges. Within a block, a thread's eight slots pair
// record nibbles k and k + 4 of one half block, which is exactly a masked OR of two code words:
// rows 0-7 of the group are the low nibbles, rows 8-15 the high ones.
__device__ __forceinline__ void rg16_fragments(const std::uint8_t* units, int group, int tig, unsigned (&lo)[2],
                                               unsigned (&hi)[2], unsigned& scale_lo, unsigned& scale_hi) {
    const int quad          = 64 * (tig & 1) + 4 * group;
    const std::uint8_t* b0  = units + 2 * (tig >> 1) * kUnitBytes + quad;
    const std::uint8_t* b1  = units + (1 + 2 * (tig >> 1)) * kUnitBytes + quad;
    const std::uint32_t w0 = lds32(b0), w1 = lds32(b0 + 32), v0 = lds32(b1), v1 = lds32(b1 + 32);
    lo[0] = (w0 & 0x0F0F0F0FU) | ((w1 & 0x0F0F0F0FU) << 4);
    hi[0] = ((w0 >> 4) & 0x0F0F0F0FU) | (w1 & 0xF0F0F0F0U);
    lo[1] = (v0 & 0x0F0F0F0FU) | ((v1 & 0x0F0F0F0FU) << 4);
    hi[1] = ((v0 >> 4) & 0x0F0F0F0FU) | (v1 & 0xF0F0F0F0U);
    // Row r's scale of unit u is byte 128 + r; this lane supplies column (row) `group` and
    // `group` + 8, in slot-block order 0, 2, 1, 3.
    const int word          = 128 + 4 * (group >> 2);
    const unsigned select   = static_cast<unsigned>(group & 3) | (static_cast<unsigned>(4 + (group & 3)) << 4);
    const std::uint32_t s0 = lds32(units + word), s1 = lds32(units + kUnitBytes + word);
    const std::uint32_t s2 = lds32(units + 2 * kUnitBytes + word), s3 = lds32(units + 3 * kUnitBytes + word);
    const std::uint32_t t0 = lds32(units + word + 8), t1 = lds32(units + kUnitBytes + word + 8);
    const std::uint32_t t2 = lds32(units + 2 * kUnitBytes + word + 8), t3 = lds32(units + 3 * kUnitBytes + word + 8);
    scale_lo = __byte_perm(__byte_perm(s0, s2, select), __byte_perm(s1, s3, select), 0x5410);
    scale_hi = __byte_perm(__byte_perm(t0, t2, select), __byte_perm(t1, t3, select), 0x5410);
}

// The work-unit mapping of grouped_kernel. A job's column tiles are contiguous in the pass's
// (job, column) list, so its unit span is [first, first + job_tiles) * kRowBlocks and the mapping
// enumerates it (row_block, t_local) with t_local innermost: adjacent CTAs of a multi-column-tile
// job then read the same 128-row weight block at the same time (one fetch feeds every column tile,
// instead of every tile streaming the block again). The job's tile span is found by two binary
// searches of the job-sorted list; probe = tile_begin + w / kRowBlocks is a tile of the job owning
// w because every job contributes whole kRowBlocks-sized unit spans.
//
// ColTileInner is false for the small weight blocks (kKTiles * kRowGroups * kUnitRunBytes below
// 64 KiB), where the old (tile-outer, row-block-inner) mapping stays: Flash-Next's down matrix
// (45 KiB of weight per unit, 5 K tiles) gained nothing from the new order and measured 0.6-1.1 %
// slower on it (two A/B passes, against ~1 % run-to-run noise), while its gate/up matrix (90 KiB a
// unit, 10 K tiles) measured 4-5 % faster on it (pp8192 prefill, 2026-11-05).
template <bool ColTileInner>
__device__ __forceinline__ void map_work_unit(const Call& call, int tile_begin, int tile_end, int w, int kRowBlocks,
                                              int& tile, int& row_block) {
    if constexpr (!ColTileInner) {
        tile      = tile_begin + w / kRowBlocks;
        row_block = w % kRowBlocks;
        return;
    }
    const int probe = tile_begin + w / kRowBlocks;
    const int job   = __ldg(&call.tiles[2 * probe]);
    int lo = tile_begin, hi = tile_end; // the job's first tile
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (__ldg(&call.tiles[2 * mid]) < job) { lo = mid + 1; } else { hi = mid; }
    }
    const int first = lo;
    lo = first;
    hi = tile_end; // one past the job's last tile
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (__ldg(&call.tiles[2 * mid]) == job) { lo = mid + 1; } else { hi = mid; }
    }
    const int unit = w - (first - tile_begin) * kRowBlocks; // [0, (lo - first) * kRowBlocks)
    row_block      = unit / (lo - first);
    tile           = first + unit % (lo - first);
}

template <Matrix M, class Epilogue, class S = Schedule, class G = FlashNextExperts>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void grouped_kernel(
#ifdef _WIN32
    // As in nvfp4_a4_tma_kernel: MSVC cannot pass the over-aligned descriptors by value.
    const Descriptors* descriptors_pointer,
#else
    const __grid_constant__ Descriptors descriptors,
#endif
    const Call call, const std::int32_t pass, const Epilogue epilogue) {
#ifdef _WIN32
    const Descriptors& descriptors = *descriptors_pointer;
#endif
    using Shape                = MatrixShape<M, G>;
    constexpr int kRowBlocks   = Shape::kRowGroups / S::kRowGroups;
    constexpr int kKTiles      = Shape::kBlocks / S::kUnitsPerKTile;
    static_assert(Shape::kRowGroups % S::kRowGroups == 0 && Shape::kBlocks % S::kUnitsPerKTile == 0);
    const CUtensorMap* codes_map[3]  = {M == Matrix::GateUp ? &descriptors.x_codes : &descriptors.h_codes,
                                        M == Matrix::GateUp ? &descriptors.x2_codes : &descriptors.h2_codes,
                                        M == Matrix::GateUp ? &descriptors.x3_codes : &descriptors.h3_codes};
    const CUtensorMap* scales_map[3] = {M == Matrix::GateUp ? &descriptors.x_scales : &descriptors.h_scales,
                                        M == Matrix::GateUp ? &descriptors.x2_scales : &descriptors.h2_scales,
                                        M == Matrix::GateUp ? &descriptors.x3_scales : &descriptors.h3_scales};

    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<Storage<S>*>(shared_bytes);
    const int tile_begin = call.pass_tiles[pass];
    const int tile_end   = call.pass_tiles[pass + 1];
    const int work       = (tile_end - tile_begin) * kRowBlocks;
    if (static_cast<int>(blockIdx.x) >= work) { return; }
    // The 16-byte scale box of the TMA map covers the (2m, 2m + 1) K-tile pair, so an odd k_tile's
    // load is the even one's duplicate. The odd tile instead reads the box from the even tile's
    // stage, whose a_scales array stays intact across the odd tile's consumer phase: a later scale
    // write needs that stage's empty barrier, which the consumer only releases after the odd tile
    // (k_tile + 2's refill touches the stage's codes and units, disjoint arrays, and with an even
    // kKTiles and an odd stage count (>= 3) it is itself an odd-parity k_tile that loads no scales).
    // kKTiles even also aligns the global stage counter's parity with the k_tile's, so a pair never
    // straddles two work tiles.
    constexpr bool kScalesDedupe = kKTiles % 2 == 0 && S::kStages >= 3 && S::kStages % 2 == 1;
    constexpr std::uint32_t kScaleStageBytes =
        static_cast<std::uint32_t>(S::kTerms) * S::kBlockTokens * S::kScaleWordsPerRow * 4U;
    // The column-tile-inner work mapping where its weight block is big enough to pay for itself
    // (map_work_unit's comment; the 64 KiB threshold splits Flash-Next's measured 90 KiB gate/up
    // win from its 45 KiB down loss).
    constexpr bool kColTileInner = static_cast<std::uint64_t>(kKTiles) * S::kRowGroups * S::kUnitRunBytes >=
                                   (1ULL << 16);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S::kStages; ++stage) {
            cta_mbarrier_init(&shared.full[stage], 1);
            cta_mbarrier_init(&shared.empty[stage], S::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < S::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
#pragma unroll
            for (int term = 0; term < S::kTerms; ++term) {
                acquire_staged_tensor_map(codes_map[term]);
                acquire_staged_tensor_map(scales_map[term]);
            }
#endif
            int it = 0;
#pragma unroll 1
            for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
                int tile = 0, row_block = 0;
                map_work_unit<kColTileInner>(call, tile_begin, tile_end, w, kRowBlocks, tile, row_block);
                const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
                const int row = call.dispatch.offsets[call.dispatch.jobs[job]] + column;
                const std::uint8_t* matrix = call.job_records[job] + Shape::kOffset +
                                             static_cast<std::size_t>(row_block) * S::kRowGroups * Shape::kBlocks *
                                                 kUnitBytes;
#pragma unroll 1
                for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
                    const int stage = it % S::kStages;
                    const bool skip_scales = kScalesDedupe && (k_tile & 1) != 0;
                    cta_mbarrier_wait(&shared.empty[stage], 1U ^ ((static_cast<unsigned>(it) / S::kStages) & 1U));
                    cta_mbarrier_arrive_expect_tx(&shared.full[stage],
                                                  S::kStageBytes - (skip_scales ? kScaleStageBytes : 0U));
#pragma unroll
                    for (int term = 0; term < S::kTerms; ++term) {
                        detail::nvfp4_tma_load_2d(shared.a_codes[stage][term], codes_map[term],
                                                  k_tile * S::kCodeRowBytes, row, &shared.full[stage]);
                        // A 16-byte scale box covers two K tiles; with the deduplication the odd
                        // tile reads the even one's staged copy (its stage holds no scale box).
                        if (!skip_scales) {
                            detail::nvfp4_tma_load_2d(shared.a_scales[stage][term], scales_map[term], (k_tile / 2) * 16,
                                                      row, &shared.full[stage]);
                        }
                    }
#pragma unroll
                    for (int g = 0; g < S::kRowGroups; ++g) {
                        bulk_load(shared.b_units[stage] + g * S::kUnitRunBytes,
                                  matrix + (static_cast<std::size_t>(g) * Shape::kBlocks +
                                            static_cast<std::size_t>(k_tile) * S::kUnitsPerKTile) *
                                               kUnitBytes,
                                  S::kUnitRunBytes, &shared.full[stage]);
                    }
                }
            }
        }
        return;
    }

    const int consumer = static_cast<int>(threadIdx.x) - S::kProducerThreads;
    const int lane = consumer & 31, warp = consumer >> 5;
    TileContext ctx{};
    ctx.warp_m = warp / S::kWarpsRows;
    ctx.warp_n = warp % S::kWarpsRows;
    ctx.group  = lane >> 2;
    ctx.tig    = lane & 3;
    // The A fragment and scale lanes of nvfp4_a4_tma_kernel.
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);

    int it = 0;
#pragma unroll 1
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        int tile = 0;
        map_work_unit<kColTileInner>(call, tile_begin, tile_end, w, kRowBlocks, tile, ctx.row_block);
        const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
        ctx.expert    = call.dispatch.jobs[job];
        const int begin = call.dispatch.offsets[ctx.expert];
        ctx.first       = begin + column;
        ctx.columns     = min(S::kBlockTokens, call.dispatch.offsets[ctx.expert + 1] - ctx.first);
        // The warp's 16-column MMA tiles that hold a valid column (warp-uniform), starting at the
        // warp's column base. A skipped tile's accumulators stay zero and its columns are never
        // written (the epilogues stop at ctx.columns). With two column warps the live tiles are
        // split evenly between them: an expert of at most 32 columns used to leave the upper warp
        // idle while the lower one ran both tiles' K loops and epilogues; now each warp takes one
        // tile, and the upper warp's MMA chain per column is exactly the chain the lower one used
        // to run (same K order, same A rows, same B fragments), so every output bit is unchanged.
        int col_base = ctx.warp_m * S::kWarpTokens;
        int live_mma = S::kMmaTokens;
        if constexpr (S::kSkipIdleColumns) {
            if constexpr (S::kWarpsTokens == 2) {
                const int total = (ctx.columns + 15) / 16; // the tile's live MMA tiles: 1..4
                const int split = (total + 1) / 2;         // the lower column warp's share: 1..2
                col_base = ctx.warp_m == 0 ? 0 : split * 16;
                live_mma = ctx.warp_m == 0 ? min(total, split) : max(total - split, 0);
            } else {
                live_mma = min(max(ctx.columns - col_base + 15, 0) / 16, S::kMmaTokens);
            }
        }
        ctx.col_base = col_base;
        ctx.live_tiles = live_mma;

        AccumulatorsOf<S> acc = {};
#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
            const int stage = it % S::kStages;
            // With the deduplication an odd k_tile reads its scales from the even tile's stage (the
            // same 16-byte box, staged one stage back); the word index below is unchanged.
            const int scales_stage =
                kScalesDedupe && (k_tile & 1) != 0 ? (it + S::kStages - 1) % S::kStages : stage;
            cta_mbarrier_wait(&shared.full[stage], (static_cast<unsigned>(it) / S::kStages) & 1U);
            unsigned b_fragments[S::kMmaRows][2];
            unsigned b_scales[S::kMmaRows];
            const auto load_b_step = [&](int local_k64) {
#pragma unroll
                for (int g = 0; g < S::kMmaRows / 2; ++g) {
                    const std::uint8_t* units = shared.b_units[stage] + (2 * ctx.warp_n + g) * S::kUnitRunBytes +
                                                local_k64 * 4 * static_cast<int>(kUnitBytes);
                    rg16_fragments(units, ctx.group, ctx.tig, b_fragments[2 * g], b_fragments[2 * g + 1],
                                   b_scales[2 * g], b_scales[2 * g + 1]);
                }
            };
            if constexpr (S::kTerms == 1) {
                // The A fragments of one 64-wide K step, double-buffered across the stage's two
                // steps (nvfp4_a4_tma_kernel's pattern): the next step's ldmatrix and scale reads
                // are issued before the current step's OMMA block, so their shared-memory latency
                // overlaps the tensor pipe instead of standing in front of every mma_m's four
                // multiplies. Only the one-term schedule takes this: with two or three terms the
                // next step's fragments of every term would stay live together and the two-CTA
                // 112-register cap spills (the term-loop comment below).
                unsigned a_fragments[2][S::kMmaTokens][4];
                unsigned a_step_scales[2][S::kMmaTokens];
                const auto load_a_step = [&](int slot, int local_k64) {
#pragma unroll
                    for (int mma_m = 0; mma_m < S::kMmaTokens; ++mma_m) {
                        if (mma_m >= live_mma) { continue; }
                        const int row           = col_base + mma_m * 16 + a_row_offset;
                        const int logical_byte  = local_k64 * 32 + a_column_byte;
                        const int physical_byte = ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                        ldmatrix_x4(a_fragments[slot][mma_m][0], a_fragments[slot][mma_m][1],
                                    a_fragments[slot][mma_m][2], a_fragments[slot][mma_m][3],
                                    smem_addr(shared.a_codes[stage][0] + row * S::kCodeRowBytes + physical_byte));
                        const int scale_row = col_base + mma_m * 16 + sfa_row;
                        a_step_scales[slot][mma_m] =
                            shared.a_scales[scales_stage][0][scale_row * S::kScaleWordsPerRow + (k_tile & 1) * 2 +
                                                           local_k64];
                    }
                };
                if (live_mma != 0) { load_a_step(0, 0); }
#pragma unroll
                for (int local_k64 = 0; local_k64 < S::kK64PerStage; ++local_k64) {
                    if (live_mma == 0) { break; } // an idle warp only releases the stage
                    load_b_step(local_k64);
                    if (local_k64 + 1 < S::kK64PerStage) { load_a_step((local_k64 + 1) & 1, local_k64 + 1); }
#pragma unroll
                    for (int mma_m = 0; mma_m < S::kMmaTokens; ++mma_m) {
                        if (mma_m >= live_mma) { continue; }
#pragma unroll
                        for (int r = 0; r < S::kMmaRows; ++r) {
                            mma_nvfp4_e4m3(acc[mma_m][r][0], acc[mma_m][r][1], acc[mma_m][r][2], acc[mma_m][r][3],
                                           a_fragments[local_k64 & 1][mma_m][0], a_fragments[local_k64 & 1][mma_m][1],
                                           a_fragments[local_k64 & 1][mma_m][2], a_fragments[local_k64 & 1][mma_m][3],
                                           b_fragments[r][0], b_fragments[r][1], a_step_scales[local_k64 & 1][mma_m],
                                           b_scales[r]);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int local_k64 = 0; local_k64 < S::kK64PerStage; ++local_k64) {
                    if (live_mma == 0) { break; } // an idle warp only releases the stage
                    load_b_step(local_k64);
                    // The three-term residual schedule at the two-CTA 112-register cap spills when
                    // all three terms' fragments stay live together, so its A fragments stay
                    // inline in the term loop.
#pragma unroll (S::kTerms == 3 ? 1 : S::kTerms)
                    for (int term = 0; term < S::kTerms; ++term) {
#pragma unroll
                        for (int mma_m = 0; mma_m < S::kMmaTokens; ++mma_m) {
                            if (mma_m >= live_mma) { continue; }
                            unsigned a_fragments[4];
                            const int row           = col_base + mma_m * 16 + a_row_offset;
                            const int logical_byte  = local_k64 * 32 + a_column_byte;
                            const int physical_byte = ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                            ldmatrix_x4(a_fragments[0], a_fragments[1], a_fragments[2], a_fragments[3],
                                        smem_addr(shared.a_codes[stage][term] + row * S::kCodeRowBytes + physical_byte));
                            const int scale_row = col_base + mma_m * 16 + sfa_row;
                            const unsigned a_scale =
                                shared.a_scales[scales_stage][term][scale_row * S::kScaleWordsPerRow +
                                                                    (k_tile & 1) * 2 + local_k64];
#pragma unroll
                            for (int r = 0; r < S::kMmaRows; ++r) {
                                mma_nvfp4_e4m3(acc[mma_m][r][0], acc[mma_m][r][1], acc[mma_m][r][2], acc[mma_m][r][3],
                                               a_fragments[0], a_fragments[1], a_fragments[2], a_fragments[3],
                                               b_fragments[r][0], b_fragments[r][1], a_scale, b_scales[r]);
                            }
                        }
                    }
                }
            }
            fence_reads_before_async_refill(); // the shared reads before the TMA and bulk refills
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
        }
        // Registers and global memory only, so the producer already fills the next tile's stages.
        if (S::kTerms > 1 || ctx.col_base < ctx.columns) { epilogue(acc, ctx); }
    }
}

// ------------------------------------------------------------------------------------- epilogues

// SwiGLU of the gate/up tile and A4 of h with the down input scale, into the entry's h row. The
// four lanes of a group hold the 16 intermediates of one h block for a column (lane tig: block
// elements tig, 4 + tig, 8 + tig, 12 + tig), so the block maximum and the code bytes are formed
// with two shuffles each, and lane tig owns code bytes tig and 4 + tig of the plane layout.
// Every epilogue takes a warp's T 16-column MMA tiles from the warp's column base (T = 2: Schedule's;
// 4: WarpTile64's).
struct GateUpEpilogue {
    const ExpertScales* scales;
    std::uint8_t* h_plane;

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][Schedule::kMmaRows][4], const TileContext& ctx) const {
        using S               = Schedule;
        const ExpertScales sc = scales[ctx.expert];
        const int block       = ctx.row_block * 4 + ctx.warp_n;
        const int position    = plane_block(block);
#pragma unroll
        for (int m = 0; m < T; ++m) {
            if (m >= ctx.live_tiles) { break; } // the epilogue writes exactly the warp's live tiles
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column    = ctx.col_base + m * 16 + ctx.group + 8 * half;
                const unsigned mask = __ballot_sync(0xFFFFFFFFU, column < ctx.columns);
                if (column >= ctx.columns) { continue; }
                std::uint16_t h[S::kMmaRows];
                float amax = 0.0F;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], sc.alpha_gate));
                    const std::uint16_t up = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], sc.alpha_up));
                    h[r]           = canon::swiglu_bf16(gate, up);
                    const float a  = canon::f32_from_bits(canon::f32_bits(canon::bf16_to_f32(h[r])) & 0x7FFFFFFFU);
                    amax           = a > amax ? a : amax;
                }
#pragma unroll
                for (int offset = 1; offset <= 2; offset <<= 1) {
                    const float other = __shfl_xor_sync(mask, amax, offset);
                    amax              = other > amax ? other : amax;
                }
                const std::uint8_t scale = canon::a4_scale_word(amax, sc.input_down);
                unsigned code[S::kMmaRows] = {};
                if (canon::e4m3_scaled(scale) != 0) {
                    const float d = canon::mul_rn(canon::e4m3_value(scale), sc.input_down);
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        code[r] = canon::e2m1_rn_satfinite(canon::div_rn(canon::bf16_to_f32(h[r]), d));
                    }
                }
                unsigned low  = (code[0] | (code[1] << 4)) << (8 * ctx.tig);
                unsigned high = (code[2] | (code[3] << 4)) << (8 * ctx.tig);
#pragma unroll
                for (int offset = 1; offset <= 2; offset <<= 1) {
                    low |= __shfl_xor_sync(mask, low, offset);
                    high |= __shfl_xor_sync(mask, high, offset);
                }
                if (ctx.tig == 0) {
                    std::uint8_t* row = h_plane + static_cast<std::size_t>(ctx.first + column) * kHRowStride;
                    *reinterpret_cast<uint2*>(row + 8 * position) = make_uint2(low, high);
                    row[kHCodeBytes + position] = scale;
                }
            }
        }
    }
};

// y = bf16_rn(acc * alpha_down) into the entry's output column.
struct DownEpilogue {
    const ExpertScales* scales;
    const std::int32_t* entries;
    std::uint16_t* outputs;

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][Schedule::kMmaRows][4], const TileContext& ctx) const {
        using S           = Schedule;
        const float alpha = scales[ctx.expert].alpha_down;
#pragma unroll
        for (int m = 0; m < T; ++m) {
            if (m >= ctx.live_tiles) { break; } // the epilogue writes exactly the warp's live tiles
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.col_base + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * kHidden +
                                     ctx.row_block * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], alpha));
                    const std::uint32_t y1 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], alpha));
                    *reinterpret_cast<std::uint32_t*>(out + tile_row(ctx.warp_n, r, ctx.tig)) = y0 | (y1 << 16);
                }
            }
        }
    }
};

// Two-term A4 (W4A16 fast path) at expert geometry G: the accumulators are in units of the row scale,
// so the gate/up outputs are bf16_rn(acc * row_scale(token) * alpha), h = swiglu_bf16(gate, up) is
// written as the entry's BF16 h row (the A16 h_rows layout), and its terms are formed by a separate
// pass. An entry's token is entry_column (a shared-expert job's entries follow the routed ones).
template <class G = FlashNextExperts>
struct GateUpEpilogueResidual {
    const ExpertScales* scales;
    const std::int32_t* entries;
    const float* x_row_scale; // per token
    std::uint8_t* h_rows;
    int top_k;
    int shared_first; // MoeDispatch::shared_first (0x7FFFFFFF: routed entries only)

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][Schedule::kMmaRows][4], const TileContext& ctx) const {
        using S               = Schedule;
        const ExpertScales sc = scales[ctx.expert];
#pragma unroll
        for (int m = 0; m < T; ++m) {
            if (m >= ctx.live_tiles) { break; } // the epilogue writes exactly the warp's live tiles
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.col_base + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int entry = ctx.first + column;
                const float rs  = x_row_scale[entry_column(entries[entry], top_k, shared_first)];
                auto* h = reinterpret_cast<std::uint16_t*>(h_rows + static_cast<std::size_t>(entry) *
                                                                        A16Planes<G>::kHRowBytes);
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate =
                        canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(acc[m][r][2 * half], rs), sc.alpha_gate));
                    const std::uint16_t up =
                        canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(acc[m][r][2 * half + 1], rs), sc.alpha_up));
                    h[(ctx.row_block * S::kBlockRows + tile_row(ctx.warp_n, r, ctx.tig)) / 2] = canon::swiglu_bf16(gate, up);
                }
            }
        }
    }
};

// y = bf16_rn(acc * row_scale(entry's h) * alpha_down) into the entry's output column (G::kHidden rows).
template <class G = FlashNextExperts>
struct DownEpilogueResidual {
    const ExpertScales* scales;
    const std::int32_t* entries;
    const float* h_row_scale; // per entry
    std::uint16_t* outputs;

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][Schedule::kMmaRows][4], const TileContext& ctx) const {
        using S           = Schedule;
        const float alpha = scales[ctx.expert].alpha_down;
#pragma unroll
        for (int m = 0; m < T; ++m) {
            if (m >= ctx.live_tiles) { break; } // the epilogue writes exactly the warp's live tiles
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.col_base + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const float rs     = h_row_scale[ctx.first + column];
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * G::kHidden +
                                     ctx.row_block * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(acc[m][r][2 * half], rs), alpha));
                    const std::uint32_t y1 =
                        canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(acc[m][r][2 * half + 1], rs), alpha));
                    *reinterpret_cast<std::uint32_t*>(out + tile_row(ctx.warp_n, r, ctx.tig)) = y0 | (y1 << 16);
                }
            }
        }
    }
};

// ------------------------------------------------------------------------------------ launching

[[nodiscard]] int persistent_ctas();

template <Matrix M, class S = Schedule, class G = FlashNextExperts, class Epilogue>
void launch(const Call& call, std::int32_t pass, const Epilogue& epilogue, cudaStream_t stream) {
    constexpr int bytes = static_cast<int>(sizeof(Storage<S>));
    // Static: MSVC 2022 rejects the lambda's use of an automatic constexpr local (C2326).
    static constexpr auto kernel = grouped_kernel<M, Epilogue, S, G>;
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void)configured;
    const int dynamic        = detail::nvfp4_prepare_shared<bytes, kernel, true>();
    constexpr int kRowBlocks = MatrixShape<M, G>::kRowGroups / S::kRowGroups;
    const int ctas           = call.mean_job_columns <= 1 ? std::min(persistent_ctas(), 2 * kRowBlocks) : persistent_ctas();
#ifdef _WIN32
    kernel<<<ctas, S::kThreads, dynamic, stream>>>(static_cast<const Descriptors*>(call.descriptors),
                                                   call, pass, epilogue);
#else
    kernel<<<ctas, S::kThreads, dynamic, stream>>>(*static_cast<const Descriptors*>(call.descriptors),
                                                   call, pass, epilogue);
#endif
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::offloaded_moe::wide
