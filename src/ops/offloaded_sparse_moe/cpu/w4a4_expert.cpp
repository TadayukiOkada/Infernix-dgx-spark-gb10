#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include "ops/offloaded_sparse_moe/cpu/neon_mad.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <sys/auxv.h>

#define INFERNIX_MOE_NEON 1

namespace infernix::ops::offloaded_moe {
namespace {

using canon::A4Block;

// Code nibble of row r (0..15) at k (0..15) inside one 144-byte unit (design §6.2).
inline unsigned unit_code(const std::uint8_t* unit, int r, int k) {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

void row_sums_scalar(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A4Block* const* acts, int ncols, std::int64_t* out) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* unit =
                    matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
                const A4Block& a = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    std::int32_t p = 0;
                    for (int k = 0; k < 16; ++k) { p += canon::e2m1_x2(unit_code(unit, r, k)) * a.c2[k]; }
                    s[r] += static_cast<std::int64_t>(p * a.scale_scaled) * canon::e4m3_scaled(unit[128 + r]);
                }
            }
            for (int r = 0; r < 16; ++r) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = s[r]; }
        }
    }
}

#if defined(INFERNIX_MOE_NEON)

// Software prefetch `distance` bytes ahead along the unit stream (design §10.3). The best distance
// depends on the host's memory latency and core speed, so calibration chooses it (§14.2).
inline void prefetch_unit(const std::uint8_t* unit, int distance) {
    if (distance > 0) {
        const char* p = reinterpret_cast<const char*>(unit) + distance;
        __builtin_prefetch(p);
        __builtin_prefetch(p + 64);
        __builtin_prefetch(p + 128);
    }
}

// u8 table: 2 * e2m1(code) + 12, so a weight is a positive byte and the code bias is removed from
// each dot product again (exactly x86's vpdpbusd scheme, so the integer P is identical).
constexpr std::array<std::int8_t, 16> kBiasedCodes = {12, 13, 14, 15, 16, 18, 20, 24,
                                                      12, 11, 10, 9,  8,  6,  4,  0};
constexpr int kBias = 12;

inline std::int32_t quad_word(const A4Block& a, int q) {
    std::int32_t w;
    std::memcpy(&w, &a.c2[4 * q], 4);
    return w;
}

// The 16 per-row scales of a unit (16 e4m3 bytes at unit + 128) as four int32x4 for rows 0-3, 4-7,
// 8-11 and 12-15: m << (e - 1), or m when e == 0.
inline void unit_scales(const std::uint8_t* s, int32x4_t (&out)[4]) {
    // Two 8-byte loads, one per group of eight rows: widening 8 -> 16 -> 32 bits lands rows 0-3,
    // 4-7, 8-11 and 12-15 in the four output vectors with no shuffles. Loading four bytes per group
    // instead lets the compiler merge the loads into overlapping 8-byte ones that read past the
    // record, so the row-group's own 16 scale bytes are the ones loaded.
    const uint16x8_t lo = vmovl_u8(vld1_u8(s));
    const uint16x8_t hi = vmovl_u8(vld1_u8(s + 8));
    const uint32x4_t g[4] = {vmovl_u16(vget_low_u16(lo)), vmovl_u16(vget_high_u16(lo)),
                             vmovl_u16(vget_low_u16(hi)), vmovl_u16(vget_high_u16(hi))};
    for (int i = 0; i < 4; ++i) {
        const uint32x4_t m = vandq_u32(g[i], vdupq_n_u32(7));
        const uint32x4_t e = vandq_u32(vshrq_n_u32(g[i], 3), vdupq_n_u32(15));
        // vshlq with the huge count (e - 1 when e == 0) yields zero, which vbsl replaces by m.
        const uint32x4_t norm =
            vshlq_u32(vorrq_u32(m, vdupq_n_u32(8)), vreinterpretq_s32_u32(vsubq_u32(e, vdupq_n_u32(1))));
        out[i] = vreinterpretq_s32_u32(vbslq_u32(vceqq_u32(e, vdupq_n_u32(0)), m, norm));
    }
}

// One column's exact row sums for rows [rg, rg + 16) of a row-group-major matrix (design §6.2).
//
// Unit byte 32 * (k / 4) + 4 * (r % 8) + k % 4 holds row r's code for k-values 4 * (k / 4) .. +3,
// low nibble for r < 8 and high nibble for r >= 8. A 16-byte window at 32 * q + 16 * w therefore
// holds, in 32-bit lane l, the four k-codes of row 4w + l (low nibbles) and of row 4w + l + 8 (high
// nibbles). With the activation's k-quad broadcast over all four lanes, each dot product
// accumulates four rows at once, one row per lane, so the lanes are the rows: no cross-lane
// reduction is needed, and the lane holding row r reads scale byte r.
//
// Bit-identical to the scalar route and to x86's: the same biased-code integers, the same bias
// removal, and the same int64 accumulation of (P - bias * c2_sum) * scale_scaled * row_scale, one
// term per block.
// GCC cannot inline an `usdot` callee into a caller without FEAT_I8MM, and a template body carries
// one target for every instantiation, so the two ISA levels are generated separately: the x86 route
// does the same with INFERNIX_AVX2_ROW_SUMS. `mad` and the kernel share the target attribute, which
// is what turns each dot product into one instruction instead of a call.
#define INFERNIX_NEON_ROW_SUMS(NAME, MAD, TARGET)                                                     \
    TARGET void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,                \
                     const A4Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {  \
        for (int rg = rg_begin; rg < rg_end; ++rg) {                                                  \
            for (int column = 0; column < ncols; ++column) {                                          \
                const A4Block* act = acts[column];                                                    \
                const int8x16_t lut   = vld1q_s8(kBiasedCodes.data());                                \
                const uint8x16_t low4 = vdupq_n_u8(0x0F);                                             \
                /* Rows of window w with nibble n are 4w + 8n + l: two int64 accumulators per row. */ \
                int64x2_t sum[8];                                                                     \
                for (int i = 0; i < 8; ++i) { sum[i] = vdupq_n_s64(0); }                              \
                const std::size_t row_group = static_cast<std::size_t>(rg) * blocks;                   \
                for (int b = 0; b < blocks; ++b) {                                                    \
                    const std::uint8_t* unit = matrix + (row_group + b) * kUnitBytes;                  \
                    prefetch_unit(unit, prefetch_bytes);                                              \
                    /* Each block has its own activation block, so bias and column scale are per block. */ \
                    const A4Block& ab = act[b];                                                       \
                    int8x16_t aq[4];                                                                  \
                    for (int q = 0; q < 4; ++q) {                                                      \
                        aq[q] = vreinterpretq_s8_s32(vdupq_n_s32(quad_word(ab, q)));                    \
                    }                                                                                  \
                    int32x4_t p[4]; /* [2 * w + n]: this block's dot products, one row per lane */      \
                    for (int i = 0; i < 4; ++i) { p[i] = vdupq_n_s32(0); }                             \
                    for (int q = 0; q < 4; ++q) {                                                      \
                        for (int w = 0; w < 2; ++w) {                                                  \
                            const uint8x16_t raw = vld1q_u8(unit + 32 * q + 16 * w);                    \
                            const uint8x16_t wl =                                                      \
                                vreinterpretq_u8_s8(vqtbl1q_s8(lut, vandq_u8(raw, low4)));              \
                            const uint8x16_t wh = vreinterpretq_u8_s8(                                 \
                                vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(raw, 4), low4)));                  \
                            p[2 * w]     = MAD::mad(p[2 * w], wl, aq[q]);                               \
                            p[2 * w + 1] = MAD::mad(p[2 * w + 1], wh, aq[q]);                           \
                        }                                                                              \
                    }                                                                                  \
                    int32x4_t sw[4];                                                                   \
                    unit_scales(unit + 128, sw);                                                       \
                    const int32x4_t bias = vdupq_n_s32(kBias * ab.c2_sum);                             \
                    const int32x4_t col  = vdupq_n_s32(ab.scale_scaled);                               \
                    for (int w = 0; w < 2; ++w) {                                                      \
                        for (int n = 0; n < 2; ++n) {                                                  \
                            const int32x4_t pa = vmulq_s32(vsubq_s32(p[2 * w + n], bias), col);         \
                            const int32x4_t sc = sw[w + 2 * n];                                        \
                            sum[4 * w + 2 * n] = vaddq_s64(                                            \
                                sum[4 * w + 2 * n], vmull_s32(vget_low_s32(pa), vget_low_s32(sc)));     \
                            sum[4 * w + 2 * n + 1] = vaddq_s64(                                        \
                                sum[4 * w + 2 * n + 1], vmull_high_s32(pa, sc));                        \
                        }                                                                              \
                    }                                                                                  \
                }                                                                                      \
                alignas(16) std::int64_t v[4];                                                          \
                for (int w = 0; w < 2; ++w) {                                                          \
                    for (int n = 0; n < 2; ++n) {                                                      \
                        vst1q_s64(v, sum[4 * w + 2 * n]);                                              \
                        const std::size_t base =                                                       \
                            (static_cast<std::size_t>(rg - rg_begin) * 16 + 4 * w + 8 * n) * ncols;     \
                        out[base + 0 * ncols + column] = v[0];                                         \
                        out[base + 1 * ncols + column] = v[1];                                         \
                        vst1q_s64(v, sum[4 * w + 2 * n + 1]);                                          \
                        out[base + 2 * ncols + column] = v[0];                                         \
                        out[base + 3 * ncols + column] = v[1];                                         \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
    }

INFERNIX_NEON_ROW_SUMS(row_sums_neon_i8mm, neon::I8mmMad, INFERNIX_I8MM_TARGET)
INFERNIX_NEON_ROW_SUMS(row_sums_neon_asimd, neon::AsimdMad, /* baseline */)

#undef INFERNIX_NEON_ROW_SUMS

struct CpuFeatures {
    bool neon   = false; // FEAT_ASIMD, the baseline AArch64 vector unit
    bool dotprod = false; // FEAT_DotProd (HWCAP_ASIMDDP)
    bool i8mm   = false; // FEAT_I8MM (HWCAP2_I8MM)
};

// The kernel reports optional features through AT_HWCAP/AT_HWCAP2 (the AArch64 analogue of
// CPUID/xgetbv).
CpuFeatures detect_features() {
    CpuFeatures f;
    f.neon = true; // AArch64 always has Advanced SIMD
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
#    ifndef HWCAP_ASIMDDP
#        define HWCAP_ASIMDDP (1UL << 20)
#    endif
#    ifndef HWCAP2_I8MM
#        define HWCAP2_I8MM (1UL << 13)
#    endif
    f.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
    f.i8mm    = (hwcap2 & HWCAP2_I8MM) != 0;
    return f;
}

const CpuFeatures& features() {
    static const CpuFeatures f = detect_features();
    return f;
}

#endif // INFERNIX_MOE_NEON

} // namespace

const char* cpu_isa_name(CpuIsa isa) {
    switch (isa) {
    case CpuIsa::kScalar: return "scalar";
    case CpuIsa::kNeon: return "neon";
    }
    return "unknown";
}

bool cpu_isa_supported(CpuIsa isa) {
#if defined(INFERNIX_MOE_NEON)
    switch (isa) {
    case CpuIsa::kScalar: return true;
    case CpuIsa::kNeon: return features().neon;
    }
    return false;
#else
    return isa == CpuIsa::kScalar;
#endif
}

CpuIsa best_cpu_isa() {
    for (CpuIsa isa : {CpuIsa::kNeon}) {
        if (cpu_isa_supported(isa)) { return isa; }
    }
    return CpuIsa::kScalar;
}

void rg16_row_sums(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                   const canon::A4Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe: unsupported CPU ISA"); }
    switch (isa) {
    case CpuIsa::kScalar: row_sums_scalar(matrix, blocks, rg_begin, rg_end, acts, ncols, out); return;
#if defined(INFERNIX_MOE_NEON)
    case CpuIsa::kNeon:
        if (features().i8mm) {
            row_sums_neon_i8mm(matrix, blocks, rg_begin, rg_end, acts, ncols, out, prefetch_bytes);
        } else {
            row_sums_neon_asimd(matrix, blocks, rg_begin, rg_end, acts, ncols, out, prefetch_bytes);
        }
        return;
#else
    default: break;
#endif
    }
    throw std::invalid_argument("offloaded_moe: unsupported CPU ISA");
}

void quantize_a4(const std::uint16_t* v, int n, float input_scale, canon::A4Block* out) {
    for (int b = 0; b < n / 16; ++b) { out[b] = canon::quantize_a4_block(v + 16 * b, input_scale); }
}

void gate_up_units(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* x_gate, const canon::A4Block* const* x_up,
                   int ncols, int unit_begin, int unit_end, canon::A4Block* const* h_blocks,
                   int prefetch_bytes) {
    // No allocation on the miss path: one unit's sums fit on the stack.
    std::int64_t s_gate[32 * kMaxColumns];
    std::int64_t s_up[32 * kMaxColumns];
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_gate, ncols, s_gate, prefetch_bytes);
        const std::int64_t* up_sums = s_gate;
        if (x_up != x_gate) {
            rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_up, ncols, s_up, prefetch_bytes);
            up_sums = s_up;
        }
        for (int c = 0; c < ncols; ++c) {
            std::uint16_t h[16];
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = canon::a4_row_output(s_gate[static_cast<std::size_t>(2 * i) * ncols + c], scales.alpha_gate);
                const std::uint16_t p = canon::a4_row_output(up_sums[static_cast<std::size_t>(2 * i + 1) * ncols + c], scales.alpha_up);
                h[i] = canon::swiglu_bf16(g, p);
            }
            h_blocks[c][u] = canon::quantize_a4_block(h, scales.input_down);
        }
    }
}

void down_rows(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
               const canon::A4Block* const* h_blocks, int ncols, int rg_begin, int rg_end,
               std::uint16_t* const* y, int prefetch_bytes) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time: no allocation on the miss path
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums(isa, record + kGateUpBytes, kDownBlocks, rg, rg + 1, h_blocks, ncols, s, prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a4_row_output(s[r * ncols + c], scales.alpha_down);
            }
        }
    }
}

void expert_forward(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                    const std::uint16_t* const* x, std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    std::vector<canon::A4Block> xg(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    std::vector<canon::A4Block> xu;
    std::vector<canon::A4Block> hb(static_cast<std::size_t>(ncols) * kHBlocks);
    const canon::A4Block* g_ptr[kMaxColumns];
    const canon::A4Block* u_ptr[kMaxColumns];
    canon::A4Block* h_ptr[kMaxColumns];
    const canon::A4Block* h_cptr[kMaxColumns];
    const bool shared = scales.input_gate == scales.input_up;
    if (!shared) { xu.resize(xg.size()); }
    for (int c = 0; c < ncols; ++c) {
        quantize_a4(x[c], kHidden, scales.input_gate, &xg[static_cast<std::size_t>(c) * kGateUpBlocks]);
        g_ptr[c] = &xg[static_cast<std::size_t>(c) * kGateUpBlocks];
        if (!shared) {
            quantize_a4(x[c], kHidden, scales.input_up, &xu[static_cast<std::size_t>(c) * kGateUpBlocks]);
            u_ptr[c] = &xu[static_cast<std::size_t>(c) * kGateUpBlocks];
        }
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * kHBlocks];
        h_cptr[c] = h_ptr[c];
    }
    gate_up_units(isa, record, scales, g_ptr, shared ? g_ptr : u_ptr, ncols, 0, kHBlocks, h_ptr, prefetch_bytes);
    down_rows(isa, record, scales, h_cptr, ncols, 0, kDownRowGroups, y, prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
