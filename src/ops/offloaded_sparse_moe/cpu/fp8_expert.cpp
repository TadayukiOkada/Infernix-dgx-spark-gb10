#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"

#include "ops/common/canonical_math.h"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/offloaded_sparse_moe/cpu/neon_mad.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include <sys/auxv.h>

#define INFERNIX_MOE_NEON 1

namespace infernix::ops::offloaded_moe {
namespace {

using fp8b128::kTile;

constexpr std::size_t kGateBytes   = static_cast<std::size_t>(kIntermediate) * kHidden;
constexpr std::size_t kScaleOffset = 3 * kGateBytes;
constexpr int kGateTiles           = (kIntermediate / kTile) * (kHidden / kTile); // per matrix

// Exact R of rows [row_begin, row_end) of a [rows, k] block-FP8 matrix (codes row-major; its tile
// scales [rows / 128][k / 128]) against `ncols` encoded columns: r[(row - row_begin) * ncols + c],
// with e_min[row - row_begin] each row's smallest tile exponent. The reference form.
void row_sums_scalar(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, exact::I128* r, int* e_min) {
    const int tiles = k / kTile;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w      = codes + static_cast<std::size_t>(row) * k;
        const std::uint16_t* tile  = scales + static_cast<std::size_t>(row / kTile) * tiles;
        const int e                = fp8b128::min_exponent(tile, tiles);
        e_min[row - row_begin]     = e;
        for (int c = 0; c < ncols; ++c) {
            exact::I128 acc;
            for (int t = 0; t < tiles; ++t) {
                std::int64_t p = 0;
                for (int j = t * kTile; j < (t + 1) * kTile; ++j) {
                    const CpuLimbSlice& s = columns[c][j / 32];
                    const std::int64_t x  = static_cast<std::int64_t>(s.hi[j % 32]) * 2048 + s.lo[j % 32];
                    p += fp8b128::fp8_int(w[j]) * x;
                }
                fp8b128::add_tile(acc, p, tile[t], e);
            }
            r[static_cast<std::size_t>(row - row_begin) * ncols + c] = acc;
        }
    }
}

#if defined(INFERNIX_MOE_NEON)

// A weight's integer w = e4m3fn(c) * 2^9 (fp8_int) is split by its exponent field e into
//
//   w = A + 2^8 * B,  A = (e <= 8 ? +-(lead(e) + m) * 2^max(e - 1, 0) : 0)  in [-1920, 1920],
//                     B = (e >= 9 ? +-(8 + m) * 2^(e - 9)              : 0)  in [-896, 896],
//
// (lead(e) = 8 for a normal code, 0 for a subnormal one), so exactly one of A and B is nonzero and
// both are a signed 4-bit mantissa times a power of two at most 128. x86 additionally splits each
// column limb slice into even and odd elements, because vpmaddubsw adds neighbouring bytes'
// products. On AArch64 the decoded A and B are already int16 lanes and vmlal_s16 multiplies them
// against the column's int16 limbs, so that split does not exist here: SplitLimbSlice, PairSlice
// and the ~80 KiB of per-thread split storage they need on every call are all gone.

// Per exponent field e: the implicit leading 8, and A's and B's power of two (0: not that part).
constexpr std::uint8_t kLead[16] = {0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
constexpr std::uint8_t kPowA[16] = {1, 1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0};
constexpr std::uint8_t kPowB[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64};

// The A and B parts of 16 weights as int16: [0] weights 0-7, [1] weights 8-15.
inline void decode16(const std::uint8_t* codes, int16x8_t (&a)[2], int16x8_t (&b)[2]) {
    const uint8x16_t c    = vld1q_u8(codes);
    const uint8x16_t e    = vandq_u8(vshrq_n_u8(c, 3), vdupq_n_u8(15));
    const uint8x16_t base = vorrq_u8(vqtbl1q_u8(vld1q_u8(kLead), e), vandq_u8(c, vdupq_n_u8(7)));
    // The signed mantissa: negate where the code's sign bit is set (a zero code's is zero anyway).
    const uint8x16_t neg  = vceqq_u8(vandq_u8(c, vdupq_n_u8(0x80)), vdupq_n_u8(0x80));
    const int8x16_t  mant = vreinterpretq_s8_u8(vbslq_u8(neg, vsubq_u8(vdupq_n_u8(0), base), base));
    const uint8x16_t pa   = vqtbl1q_u8(vld1q_u8(kPowA), e);
    const uint8x16_t pb   = vqtbl1q_u8(vld1q_u8(kPowB), e);
    const int16x8_t  ml   = vmovl_s8(vget_low_s8(mant));
    const int16x8_t  mh   = vmovl_s8(vget_high_s8(mant));
    a[0]                  = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pa))), ml);
    a[1]                  = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pa))), mh);
    b[0]                  = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pb))), ml);
    b[1]                  = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pb))), mh);
}

// acc += sum over the 16 weights of `p` against `limb` (p[0] covers limbs 0-7, p[1] 8-15).
inline void mla16(int32x4_t& acc, const int16x8_t (&p)[2], const std::int16_t* limb) {
    const int16x8_t l0 = vld1q_s16(limb);
    const int16x8_t l1 = vld1q_s16(limb + 8);
    acc                = vmlal_s16(acc, vget_low_s16(p[0]), vget_low_s16(l0));
    acc                = vmlal_high_s16(acc, p[0], l0);
    acc                = vmlal_s16(acc, vget_low_s16(p[1]), vget_low_s16(l1));
    acc                = vmlal_high_s16(acc, p[1], l1);
}

// Rows [row_begin, row_end) against NC columns. Per tile and column four int32 lane streams
// (A * lo, A * hi, B * lo, B * hi; each lane sums 32 products of at most 1920 * 2047 per tile,
// below 2^26) give the tile's exact P = 2^19 S_Bhi + 2^11 S_Ahi + 2^8 S_Blo + S_Alo (int64),
// which enters R as fp8_exact.h's. Four columns keep 16 accumulators live, which fits the 32
// vector registers alongside the decode temporaries.
template <int NC>
INFERNIX_I8MM_TARGET void row_sums_neon_n(const std::uint8_t* codes, const std::uint16_t* scales, int k,
                                          int row_begin, int row_end, const CpuLimbSlice* const* columns, int c0,
                                          int ncols, exact::I128* r, int* e_min) {
    const int tiles = k / kTile;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w     = codes + static_cast<std::size_t>(row) * k;
        const std::uint16_t* tile = scales + static_cast<std::size_t>(row / kTile) * tiles;
        const int e               = fp8b128::min_exponent(tile, tiles);
        e_min[row - row_begin]    = e;
        exact::I128 out[NC];
        for (int t = 0; t < tiles; ++t) {
            int32x4_t v[NC][4]; // A*lo, A*hi, B*lo, B*hi
            for (int c = 0; c < NC; ++c) {
                for (int i = 0; i < 4; ++i) { v[c][i] = vdupq_n_s32(0); }
            }
            for (int s = 4 * t; s < 4 * t + 4; ++s) {
                int16x8_t a[2][2], b[2][2]; // [half of the slice][weights 0-7 or 8-15]
                decode16(w + 32 * s, a[0], b[0]);
                decode16(w + 32 * s + 16, a[1], b[1]);
                for (int c = 0; c < NC; ++c) {
                    const CpuLimbSlice& l = columns[c0 + c][s];
                    mla16(v[c][0], a[0], l.lo);
                    mla16(v[c][0], a[1], l.lo + 16);
                    mla16(v[c][1], a[0], l.hi);
                    mla16(v[c][1], a[1], l.hi + 16);
                    mla16(v[c][2], b[0], l.lo);
                    mla16(v[c][2], b[1], l.lo + 16);
                    mla16(v[c][3], b[0], l.hi);
                    mla16(v[c][3], b[1], l.hi + 16);
                }
            }
            for (int c = 0; c < NC; ++c) {
                const std::int64_t p = (static_cast<std::int64_t>(neon::hsum(v[c][3])) << 19) +
                                       (static_cast<std::int64_t>(neon::hsum(v[c][1])) << 11) +
                                       (static_cast<std::int64_t>(neon::hsum(v[c][2])) << 8) +
                                       static_cast<std::int64_t>(neon::hsum(v[c][0]));
                fp8b128::add_tile(out[c], p, tile[t], e);
            }
        }
        for (int c = 0; c < NC; ++c) { r[static_cast<std::size_t>(row - row_begin) * ncols + c0 + c] = out[c]; }
    }
}

void row_sums_neon(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,
                   const CpuLimbSlice* const* columns, int ncols, exact::I128* r, int* e_min) {
    for (int c0 = 0; c0 < ncols; c0 += 4) {
        switch (std::min(4, ncols - c0)) {
        case 1: row_sums_neon_n<1>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        case 2: row_sums_neon_n<2>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        case 3: row_sums_neon_n<3>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        default: row_sums_neon_n<4>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        }
    }
}

#endif // INFERNIX_MOE_NEON

// The columns of one call as the row sums read them. AArch64 reads CpuLimbSlice directly, so there
// is nothing to pre-split: the class just selects the route once per call instead of per row.
class Columns {
public:
    Columns(CpuIsa isa, const CpuLimbSlice* const* columns, int ncols, int slices)
        : isa_(isa), columns_(columns) {
        (void)ncols;
        (void)slices;
        if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe fp8: unsupported CPU ISA"); }
    }

    void row_sums(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end, int ncols,
                  exact::I128* r, int* e_min) const {
        switch (isa_) {
#if defined(INFERNIX_MOE_NEON)
        case CpuIsa::kNeon: row_sums_neon(codes, scales, k, row_begin, row_end, columns_, ncols, r, e_min); return;
#endif
        default: row_sums_scalar(codes, scales, k, row_begin, row_end, columns_, ncols, r, e_min); return;
        }
    }

private:
    CpuIsa isa_;
    const CpuLimbSlice* const* columns_;
};

} // namespace

void fp8_gate_up(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* x, const int* x_emax, int ncols,
                 int i_begin, int i_end, std::uint16_t* const* h) {
    constexpr int kUnit = 16;
    exact::I128 g[kUnit * kMaxColumns], u[kUnit * kMaxColumns];
    int ge[kUnit], ue[kUnit];
    const auto* scales = reinterpret_cast<const std::uint16_t*>(record + kScaleOffset);
    const Columns columns(isa, x, ncols, kHidden / 32);
    for (int i0 = i_begin; i0 < i_end; i0 += kUnit) {
        const int i1 = std::min(i_end, i0 + kUnit);
        columns.row_sums(record, scales, kHidden, i0, i1, ncols, g, ge);
        columns.row_sums(record + kGateBytes, scales + kGateTiles, kHidden, i0, i1, ncols, u, ue);
        for (int i = i0; i < i1; ++i) {
            for (int c = 0; c < ncols; ++c) {
                const std::size_t at = static_cast<std::size_t>(i - i0) * ncols + c;
                h[c][i] = canon::swiglu_bf16(fp8b128::output_bf16(g[at], ge[i - i0], x_emax[c]),
                                             fp8b128::output_bf16(u[at], ue[i - i0], x_emax[c]));
            }
        }
    }
}

void fp8_down(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* h, const int* h_emax, int ncols,
              int row_begin, int row_end, std::uint16_t* const* y) {
    constexpr int kRows = 32;
    exact::I128 r[kRows * kMaxColumns];
    int e[kRows];
    const auto* scales = reinterpret_cast<const std::uint16_t*>(record + kScaleOffset) + 2 * kGateTiles;
    const Columns columns(isa, h, ncols, kIntermediate / 32);
    for (int r0 = row_begin; r0 < row_end; r0 += kRows) {
        const int r1 = std::min(row_end, r0 + kRows);
        columns.row_sums(record + 2 * kGateBytes, scales, kIntermediate, r0, r1, ncols, r, e);
        for (int row = r0; row < r1; ++row) {
            for (int c = 0; c < ncols; ++c) {
                y[c][row] = fp8b128::output_bf16(r[static_cast<std::size_t>(row - r0) * ncols + c], e[row - r0], h_emax[c]);
            }
        }
    }
}

} // namespace infernix::ops::offloaded_moe
