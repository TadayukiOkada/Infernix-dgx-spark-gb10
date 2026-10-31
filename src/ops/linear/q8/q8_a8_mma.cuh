#pragma once

// Q8G32 W8A8 prefill route (LinearPolicy::AllowA8, LOSSY, never on the A16 or column-invariant
// routes): the BF16 activations are quantized per 32-element K slice to signed byte codes with an
// FP32 slice scale (ops/ggml/ggml_a8.h's codec, the same one the GGUF A8 routes run), and the
// product runs on the int8 Tensor Cores as m16n8k32 s8 MMA with the exact int32 slice dot scaled
// into FP32 per group.
//
// ---- Arithmetic of one output (row r, column t)
//
// With P_s the exact int32 dot of the row's codes of slice s with the column's codes (every weight
// code in [-128, 127] and every activation code in [-127, 127], so |P_s| < 2^19), w_s the row's
// binary16 group scale exactly in binary32 and d_s the column's slice scale,
//
//   acc = +0; for s = 0 .. K/32 - 1, in order:
//     acc = fma(fl32(P_s * w_s), d_s, acc)
//
// (ggml_a8_tile.cuh's kPlain term: fl32(P * w) comes from the int32 MMA without a conversion
// instruction, ggml_a8_mma.cuh's magic accumulator). The row's output is acc rounded once to the
// output type. A slice whose activation maximum is below 2^-120 has d = 0 and codes 0; an
// activation slice holding Inf or NaN has d = NaN, so its column's outputs are NaN; a non-finite
// weight scale makes its rows NaN.
//
// ---- Tiles
//
// One CTA computes 128 rows x 128 columns with eight warps of 64 rows x 32 columns (four m16 by
// four n8 MMA tiles), K in stages of 64 elements (two 32-element slices, hence two IMMA.16832.S8
// steps per stage) in a three-deep cp.async ring. The quantizer emits the activation codes in
// B-fragment order under the permuted K labelling of ggml_a8_tile.cuh (the k32 dot does not
// depend on the K labels when A and B agree): lane (g, t) of a pair of n8 tiles holds chunk t
// (code words 2t and 2t + 1) of columns g and 8 + g of the pair's sixteen columns as one int4, so
// the ring copies each 512-byte fragment block as 16-byte cp.async blocks and the MMA step reads
// two tiles' B fragments with one conflict-free LDS.128 (the entry index is XOR-swizzled so any
// phase grouping of the LDS hits distinct 16-byte bank groups). The weight codes stage in row
// order (two 16-byte copies per row and slice) and the A fragments read as the row pair's two
// int2 chunks.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_shared.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

inline constexpr int kQ8A8BlockRows    = 128;
inline constexpr int kQ8A8BlockTokens  = 128;
inline constexpr int kQ8A8BlockK       = 64;
inline constexpr int kQ8A8Stages       = 3;
inline constexpr int kQ8A8Threads      = 256;
inline constexpr int kQ8A8MinBlocks    = 1;
inline constexpr int kQ8A8RowWarps     = 2;
inline constexpr int kQ8A8ColumnWarps  = kQ8A8Threads / 32 / kQ8A8RowWarps;
inline constexpr int kQ8A8RowsPerWarp  = kQ8A8BlockRows / kQ8A8RowWarps;
inline constexpr int kQ8A8RowTiles     = kQ8A8RowsPerWarp / 16; // m16 tiles per warp
inline constexpr int kQ8A8TokenTiles   = kQ8A8BlockTokens / kQ8A8ColumnWarps / 8; // n8 tiles per warp
inline constexpr int kQ8A8StageSlices  = kQ8A8BlockK / 32;
static_assert(kQ8A8BlockRows % (kQ8A8RowWarps * 16) == 0);
static_assert(kQ8A8BlockTokens == kQ8A8ColumnWarps * kQ8A8TokenTiles * 8);

// The B-fragment block of sixteen columns and one slice: 32 int4 entries, entry l (g = l >> 2,
// t = l & 3) = {word 2t, word 2t + 1} of column g followed by the same pair of column 8 + g
// (word w of a column holds its elements 4w..4w+3). Entry l physically sits at p(l) = l ^
// (((l >> 3) & 3) << 2), which spreads any eight consecutive lanes over distinct 16-byte groups.
__device__ __forceinline__ int q8_a8_bentry(int l) { return l ^ (((l >> 3) & 3) << 2); }

// The quantizer's encoded batch: column t, slice s. Four threads (u = 0..3) encode one item;
// thread u codes the slice's elements 8u..8u+7 and writes them as the two code words 2u, 2u + 1
// into entry (r & 7) * 4 + u (r = t mod 16) of the (column block t/16, slice) fragment block at
// word slot 2 * (r / 8). The slice scale d goes to xscales[t * slices + s] (natural order; the
// tile reads four columns' scales as one float4). `items` counts the items (columns * slices).
__global__ void __launch_bounds__(kQ8A8Threads) q8_a8_quantize_x_kernel(
    const __nv_bfloat16* __restrict__ x, int k, int slices, std::int32_t* __restrict__ xcodes,
    float* __restrict__ xscales, int items) {
    namespace a8 = ::infernix::ggml::a8;
    const int i    = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int item = i >> 2, u = i & 3;
    if (item >= items) { return; }
    const int t              = item / slices, s = item - t * slices;
    const uint2 val0         = load_vec<uint2>(x + static_cast<std::size_t>(t) * k + 32 * s + 4 * u);
    const uint2 val1         = load_vec<uint2>(x + static_cast<std::size_t>(t) * k + 32 * s + 16 + 4 * u);
    const unsigned words[4]  = {val0.x, val0.y, val1.x, val1.y};
    std::uint16_t e[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        e[2 * m]     = static_cast<std::uint16_t>(words[m] & 0xFFFFU);
        e[2 * m + 1] = static_cast<std::uint16_t>(words[m] >> 16);
    }
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) { key = max(key, static_cast<unsigned>(e[m] & 0x7FFFU)); }
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 1));
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 2));
    const a8::SliceScale sc  = a8::slice_scale(key);
    const auto pack = [](int a, int b, int c, int d) {
        return static_cast<std::int32_t>(
            (static_cast<unsigned>(a) & 0xFFU) | ((static_cast<unsigned>(b) & 0xFFU) << 8) |
            ((static_cast<unsigned>(c) & 0xFFU) << 16) | (static_cast<unsigned>(d) << 24));
    };
    int q[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) { q[m] = a8::code(e[m], sc.inv); }
    const int r       = t & 15;
    const int entry   = q8_a8_bentry((r & 7) * 4 + u);
    const int block   = t >> 4;
    std::int32_t* dst = xcodes + ((static_cast<std::size_t>(block) * slices + s) * 32 + entry) * 4 +
                       ((r >> 3) << 1);
    store_vec(dst, make_int2(pack(q[0], q[1], q[2], q[3]), pack(q[4], q[5], q[6], q[7])));
    if (u == 0) { xscales[item] = sc.d; }
}

// One K stage of a CTA. a[j][row][0..7] is the row's slice-j code words (chunk c = words 2c,
// 2c + 1); the A fragment of lane (g, t) reads the two rows' chunk-t int2s. xb[j][pair][l] is
// the swizzled B-fragment entry l of the pair's sixteen columns (quantizer order). xs holds the
// columns' slice-scale pairs, ws the rows' raw binary16 group scales (expanded to float at the
// MMA's use, so the stage needs no separate pass and only one barrier).
struct Q8A8Stage {
    alignas(16) std::int32_t a[kQ8A8StageSlices][kQ8A8BlockRows][8];
    alignas(16) int4 xb[kQ8A8StageSlices][8][32];
    alignas(16) float2 xs[kQ8A8BlockTokens];
    alignas(16) std::uint16_t ws[kQ8A8BlockRows][kQ8A8StageSlices];
};
static_assert(sizeof(Q8A8Stage) == 17920);
inline constexpr int kQ8A8SharedBytes = kQ8A8Stages * static_cast<int>(sizeof(Q8A8Stage));

// CTA (x, y): rows 128x.. against the batch columns column_begin + 128y.. (the batch holds
// `batch` encoded columns, of which `live` are call columns). Full skips every row and column
// predicate: whole row tiles and whole column tiles of live columns.
template <bool Full, class Output>
__global__ void __launch_bounds__(kQ8A8Threads, kQ8A8MinBlocks) q8_a8_mma_kernel(
    const std::int32_t* __restrict__ xcodes, const float* __restrict__ xscales,
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ wscales, Output output,
    int rows, int k, int slices, int kg, int batch, int live, int column_begin) {
    namespace a8 = ::infernix::ggml::a8;
    constexpr int BM       = kQ8A8BlockRows;
    constexpr int BN       = kQ8A8BlockTokens;
    constexpr int MT       = kQ8A8RowTiles;
    constexpr int NT       = kQ8A8TokenTiles;
    constexpr int SLICES   = kQ8A8StageSlices;
    constexpr int STAGES   = kQ8A8Stages;
    constexpr int kThreads = kQ8A8Threads;
    constexpr int ROWS_PER_WARP = kQ8A8RowsPerWarp;
    constexpr int COLS_PER_WARP = BN / kQ8A8ColumnWarps;

    auto* scratch = q8_shared_storage<kQ8A8SharedBytes>();
    auto* stages  = reinterpret_cast<Q8A8Stage*>(scratch);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / kQ8A8ColumnWarps;
    const int wn   = warp % kQ8A8ColumnWarps;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    // A grouped raster (four row tiles walk the column tiles fastest, the measured best on GB10):
    // a wave re-reads each staged weight tile from L2 across the columns and each encoded column
    // slab across the rows.
    const int tiles_m = static_cast<int>(gridDim.x);
    const int tiles_n = static_cast<int>(gridDim.y);
    int tile_m        = static_cast<int>(blockIdx.x);
    int tile_n        = static_cast<int>(blockIdx.y);
    if (tiles_n > 1 && tiles_m >= 8) {
        constexpr int kGroupRows = 4;
        const int linear         = tile_n * tiles_m + tile_m;
        const int group_span     = kGroupRows * tiles_n;
        const int group          = linear / group_span;
        const int first_m        = group * kGroupRows;
        const int active_m       = min(kGroupRows, tiles_m - first_m);
        const int within         = linear - group * group_span;
        tile_m                   = first_m + within % active_m;
        tile_n                   = within / active_m;
    }
    const int m0  = tile_m * BM;
    const int n0  = column_begin + tile_n * BN;
    const int nkt = k / kQ8A8BlockK;
    // The CTA's B slab is the batch's column blocks tile_n*8.. (16 columns each).
    const int base_pair = tile_n * 8;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc[mi][nt][q] = 0.0F; }
        }
    }

    // The weight rows' slice groups and the columns' fragment blocks copy as contiguous 16-byte
    // cp.async; the column scales copy 8 bytes per column, the weight's raw group scales 4 bytes
    // per row. One 16-byte unit per thread and slice (the stage's copies are a multiple of the
    // thread count), so the loop body is a single copy and the address math amortizes over slices.
    auto issue = [&](int kt) {
        const int buf = kt % STAGES;
        const int g0  = kt * SLICES;
        Q8A8Stage& sm = stages[buf];
#pragma unroll
        for (int j = 0; j < SLICES; ++j) {
            { // the rows' 32-byte code rows: two 16-byte units, one per thread, XOR-swizzled for ldmatrix
                const int row   = tid >> 1;
                const int chunk = tid & 1;
                const int swz   = chunk ^ ((row >> 2) & 1);
                const int grow  = m0 + row;
                auto* dst       = &sm.a[j][row][swz * 4];
                if constexpr (Full) {
                    cp_async<16, Cache::cg>(
                        dst, codes + (static_cast<std::size_t>(grow) * kg + g0 + j) * 32 + chunk * 16);
                } else {
                    const std::size_t gi =
                        static_cast<std::size_t>(grow < rows ? grow : 0) * kg + g0 + j;
                    cp_async_zfill<16, Cache::cg>(dst, codes + gi * 32 + chunk * 16,
                                                  grow < rows ? 16 : 0);
                }
            }
            { // the CTA's B slab: eight 512-byte fragment blocks, 32 units each — one unit per thread
                const int pair  = tid >> 5;
                const int chunk = tid & 31;
                const int gp    = base_pair + pair;
                const int col   = gp * 16;
                auto* dst       = reinterpret_cast<std::int32_t*>(&sm.xb[j][pair][0]) + chunk * 4;
                if constexpr (Full) {
                    cp_async<16, Cache::cg>(
                        dst, xcodes + ((static_cast<std::size_t>(gp) * slices + g0 + j) * 32 + chunk) * 4);
                } else {
                    const std::size_t at =
                        (static_cast<std::size_t>(col < batch ? gp : 0) * slices + g0 + j) * 32 + chunk;
                    cp_async_zfill<16, Cache::cg>(dst, xcodes + at * 4, col < batch ? 16 : 0);
                }
            }
        }
        for (int item = tid; item < BN; item += kThreads) {
            const int col = base_pair * 16 + item;
            auto* dst     = &sm.xs[item];
            if constexpr (Full) {
                cp_async<8>(dst, xscales + static_cast<std::size_t>(col) * slices + g0);
            } else {
                const std::size_t at =
                    static_cast<std::size_t>(col < batch ? col : 0) * slices + g0;
                cp_async_zfill<8>(dst, xscales + at, col < batch ? 8 : 0);
            }
        }
        if (tid < BM) {
            const int grow = m0 + tid;
            auto* dst      = &sm.ws[tid][0];
            if constexpr (Full) {
                cp_async<4>(dst, wscales + static_cast<std::size_t>(grow) * kg + g0);
            } else {
                const std::size_t at = static_cast<std::size_t>(grow < rows ? grow : 0) * kg + g0;
                cp_async_zfill<4>(dst, wscales + at, grow < rows ? 4 : 0);
            }
        }
    };

    auto mma_stage = [&](int kt) {
        const Q8A8Stage& sm = stages[kt % STAGES];
#pragma unroll
        for (int j = 0; j < SLICES; ++j) {
            int4 bp[NT / 2];
            float2 xd[NT];
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                bp[np] = sm.xb[j][wn * (NT / 2) + np][q8_a8_bentry(lane)];
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float4 raw = load_vec<float4>(&sm.xs[wn * COLS_PER_WARP + 8 * nt + 2 * t]);
                xd[nt]           = j == 0 ? make_float2(raw.x, raw.z) : make_float2(raw.y, raw.w);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int rowA  = wm * ROWS_PER_WARP + mi * 16 + g;
                const int lrow  = wm * ROWS_PER_WARP + mi * 16 + (lane & 15);
                const int lswz  = (lane >> 4) ^ ((lrow >> 2) & 1);
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(&sm.a[j][lrow][lswz * 4]));
                const float w0  = __half2float(__ushort_as_half(sm.ws[rowA][j]));
                const float w1  = __half2float(__ushort_as_half(sm.ws[rowA + 8][j]));
                const float mw0 = __fmul_rn(a8::kMinusMagic, w0);
                const float mw1 = __fmul_rn(a8::kMinusMagic, w1);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int4 b   = bp[nt >> 1];
                    const int bx   = (nt & 1) ? b.z : b.x;
                    const int by   = (nt & 1) ? b.w : b.y;
                    const float c0 = xd[nt].x;
                    const float c1 = xd[nt].y;
                    int p[4];
                    a8::mma_k32_magic(p, static_cast<int>(a0), static_cast<int>(a1),
                                      static_cast<int>(a2), static_cast<int>(a3), bx, by);
                    acc[mi][nt][0] = __fmaf_rn(a8::scaled_term(p[0], w0, mw0), c0, acc[mi][nt][0]);
                    acc[mi][nt][1] = __fmaf_rn(a8::scaled_term(p[1], w0, mw0), c1, acc[mi][nt][1]);
                    acc[mi][nt][2] = __fmaf_rn(a8::scaled_term(p[2], w1, mw1), c0, acc[mi][nt][2]);
                    acc[mi][nt][3] = __fmaf_rn(a8::scaled_term(p[3], w1, mw1), c1, acc[mi][nt][3]);
                }
            }
        }
    };

    // STAGES-1 stages of prefetch: stage kt's arrival is awaited while its MMAs run, with the
    // next STAGES-2 already in flight (the ring holds STAGES buffers).
    for (int s = 0; s < STAGES - 1 && s < nkt; ++s) {
        issue(s);
        cp_commit();
    }
    for (int kt = 0; kt < nkt; ++kt) {
        // Groups beyond the one this stage consumes may stay in flight only up to the ring's
        // spare buffers; on the last stages the newest groups are the ones being consumed, so
        // progressively fewer (down to none) may remain outstanding.
        const int slack = nkt - 1 - kt;
        if (slack >= STAGES - 2) { cp_wait<STAGES - 2>(); }
        else if (slack == 1)     { cp_wait<1>(); }
        else                     { cp_wait<0>(); }
        __syncthreads();
        if (kt + STAGES - 1 < nkt) {
            issue(kt + STAGES - 1);
            cp_commit();
        }
        mma_stage(kt);
    }

    const auto output_tile = linear_output_tile<BM>(output, m0);
    const int live_columns = live;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row   = m0 + wm * ROWS_PER_WARP + mi * 16 + g + 8 * (q >> 1);
                const int token = n0 + wn * COLS_PER_WARP + nt * 8 + 2 * t + (q & 1);
                if (Full || (row < rows && token < live_columns)) {
                    output_tile.store(row, token, acc[mi][nt][q]);
                }
            }
        }
    }
}

} // namespace infernix::ops::detail
