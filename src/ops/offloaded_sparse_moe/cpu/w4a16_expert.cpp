#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"

#include "ops/offloaded_sparse_moe/cpu/neon_mad.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <sys/auxv.h>

#define INFERNIX_MOE_NEON 1

namespace infernix::ops::offloaded_moe {
namespace {

using canon::A16Block;

// Code nibble of row r (0..15) at k (0..15) inside one 144-byte unit (design §6.2).
inline unsigned unit_code(const std::uint8_t* unit, int r, int k) {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

void row_sums_scalar(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A16Block* const* acts, int ncols, std::int64_t* out) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
                const A16Block& a        = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    std::int32_t p = 0; // |P| < 2^31 (design §16.2.1)
                    for (int k = 0; k < 16; ++k) { p += canon::e2m1_x2(unit_code(unit, r, k)) * canon::a16_x(a, k); }
                    s[r] += static_cast<std::int64_t>(p) * canon::e4m3_scaled(unit[128 + r]);
                }
            }
            for (int r = 0; r < 16; ++r) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = s[r]; }
        }
    }
}

#if defined(INFERNIX_MOE_NEON)

// Software prefetch `distance` bytes ahead along the unit stream (design §10.3, §14.2).
inline void prefetch_unit(const std::uint8_t* unit, int distance) {
    if (distance > 0) {
        const char* p = reinterpret_cast<const char*>(unit) + distance;
        __builtin_prefetch(p);
        __builtin_prefetch(p + 64);
        __builtin_prefetch(p + 128);
    }
}

// Code -> doubled E2M1 value as a signed byte, and biased by +12 as an unsigned byte.
constexpr std::array<std::int8_t, 16> kSignedCodes = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr std::array<std::int8_t, 16> kBiasedCodes = {12, 13, 14, 15, 16, 18, 20, 24,
                                                      12, 11, 10, 9,  8,  6,  4,  0};
constexpr int kBias = 12;

template <class T>
inline std::int32_t quad_word(const T (&bytes)[16], int q) {
    std::int32_t w;
    std::memcpy(&w, &bytes[4 * q], 4);
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

// One row group's exact row sums for every column of a row-group-major matrix (design §6.2).
//
// Unit byte 32 * (k / 4) + 4 * (r % 8) + k % 4 holds row r's code for k-values 4 * (k / 4) .. +3,
// low nibble for r < 8 and high nibble for r >= 8. A 16-byte window at 32 * q + 16 * w therefore
// holds, in 32-bit lane l, the four k-codes of row 4w + l (low nibbles) and of row 4w + l + 8 (high
// nibbles), so one dot product accumulates four rows at once, one row per lane, and the lanes are
// the rows.
//
// P = lo + 256 mid + 65536 hi with X split into unsigned lo/mid bytes and a signed hi byte, exactly
// as the scalar route and x86's AVX-512 path compute it: lo and mid are unsigned activations times
// signed codes, hi is a signed activation times biased codes with kBias * hi_sum removed. The int64
// accumulation of P * row_scale is then identical.
//
// GCC cannot inline an `usdot` callee into a caller without FEAT_I8MM, and a template body carries
// one target for every instantiation, so the two ISA levels are generated separately: the x86 route
// does the same with INFERNIX_AVX2_ROW_SUMS. `mad` and the kernel share the target attribute, which
// is what turns each dot product into one instruction instead of a call.
#define INFERNIX_NEON_ROW_SUMS(NAME, MAD, TARGET)                                                   \
    TARGET void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,              \
                     const A16Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) { \
        const int8x16_t lut_s   = vld1q_s8(kSignedCodes.data());                                    \
        const uint8x16_t lut_b  = vreinterpretq_u8_s8(vld1q_s8(kBiasedCodes.data()));               \
        const uint8x16_t low4   = vdupq_n_u8(0x0F);                                                 \
        for (int rg = rg_begin; rg < rg_end; ++rg) {                                                \
            for (int column = 0; column < ncols; ++column) {                                        \
                const A16Block* act = acts[column];                                                 \
                /* sum[4w + 2n + i]: rows 4w + 8n + 2i and +1 as int64 across all blocks. */        \
                int64x2_t sum[8];                                                                   \
                for (int i = 0; i < 8; ++i) { sum[i] = vdupq_n_s64(0); }                            \
                const std::size_t row_group = static_cast<std::size_t>(rg) * blocks;                \
                for (int b = 0; b < blocks; ++b) {                                                  \
                    const std::uint8_t* unit = matrix + (row_group + b) * kUnitBytes;                \
                    prefetch_unit(unit, prefetch_bytes);                                            \
                    const A16Block& ab = act[b];                                                    \
                    uint8x16_t alo[4], amd[4];                                                      \
                    int8x16_t ahi[4];                                                               \
                    for (int q = 0; q < 4; ++q) {                                                   \
                        alo[q] = vreinterpretq_u8_s32(vdupq_n_s32(quad_word(ab.lo, q)));             \
                        amd[q] = vreinterpretq_u8_s32(vdupq_n_s32(quad_word(ab.mid, q)));            \
                        ahi[q] = vreinterpretq_s8_s32(vdupq_n_s32(quad_word(ab.hi, q)));             \
                    }                                                                               \
                    int32x4_t plo[4], pmd[4], phi[4];                                               \
                    for (int i = 0; i < 4; ++i) {                                                   \
                        plo[i] = pmd[i] = phi[i] = vdupq_n_s32(0);                                  \
                    }                                                                               \
                    for (int q = 0; q < 4; ++q) {                                                   \
                        for (int w = 0; w < 2; ++w) {                                               \
                            const uint8x16_t raw = vld1q_u8(unit + 32 * q + 16 * w);                 \
                            const int8x16_t cs = vqtbl1q_s8(lut_s, vandq_u8(raw, low4));             \
                            const int8x16_t ch =                                                     \
                                vqtbl1q_s8(lut_s, vandq_u8(vshrq_n_u8(raw, 4), low4));               \
                            const uint8x16_t cb =                                                    \
                                vreinterpretq_u8_s8(vqtbl1q_s8(vreinterpretq_s8_u8(lut_b),            \
                                                               vandq_u8(raw, low4)));               \
                            const uint8x16_t cb2 =                                                   \
                                vreinterpretq_u8_s8(vqtbl1q_s8(vreinterpretq_s8_u8(lut_b),            \
                                                               vandq_u8(vshrq_n_u8(raw, 4), low4))); \
                            plo[2 * w]     = MAD::mad(plo[2 * w], alo[q], cs);                       \
                            pmd[2 * w]     = MAD::mad(pmd[2 * w], amd[q], cs);                       \
                            phi[2 * w]     = MAD::mad(phi[2 * w], cb, ahi[q]);                       \
                            plo[2 * w + 1] = MAD::mad(plo[2 * w + 1], alo[q], ch);                   \
                            pmd[2 * w + 1] = MAD::mad(pmd[2 * w + 1], amd[q], ch);                   \
                            phi[2 * w + 1] = MAD::mad(phi[2 * w + 1], cb2, ahi[q]);                  \
                        }                                                                           \
                    }                                                                               \
                    int32x4_t sw[4];                                                                \
                    unit_scales(unit + 128, sw);                                                    \
                    const int32x4_t bias = vdupq_n_s32(kBias * ab.hi_sum);                           \
                    for (int w = 0; w < 2; ++w) {                                                   \
                        for (int n = 0; n < 2; ++n) {                                               \
                            const int i = 2 * w + n;                                                \
                            const int32x4_t hi = vsubq_s32(phi[i], bias);                            \
                            const int32x4_t p  = vaddq_s32(                                          \
                                vaddq_s32(plo[i], vshlq_n_s32(pmd[i], 8)), vshlq_n_s32(hi, 16));     \
                            const int32x4_t sc = sw[w + 2 * n];                                     \
                            sum[4 * w + 2 * n] = vaddq_s64(                                          \
                                sum[4 * w + 2 * n], vmull_s32(vget_low_s32(p), vget_low_s32(sc)));    \
                            sum[4 * w + 2 * n + 1] = vaddq_s64(                                      \
                                sum[4 * w + 2 * n + 1], vmull_high_s32(p, sc));                       \
                        }                                                                           \
                    }                                                                               \
                }                                                                                   \
                alignas(16) std::int64_t v[4];                                                       \
                for (int w = 0; w < 2; ++w) {                                                       \
                    for (int n = 0; n < 2; ++n) {                                                   \
                        vst1q_s64(v, sum[4 * w + 2 * n]);                                           \
                        const std::size_t base =                                                    \
                            (static_cast<std::size_t>(rg - rg_begin) * 16 + 4 * w + 8 * n) * ncols;  \
                        out[base + 0 * ncols + column] = v[0];                                      \
                        out[base + 1 * ncols + column] = v[1];                                      \
                        vst1q_s64(v, sum[4 * w + 2 * n + 1]);                                       \
                        out[base + 2 * ncols + column] = v[0];                                      \
                        out[base + 3 * ncols + column] = v[1];                                      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }

INFERNIX_NEON_ROW_SUMS(row_sums_neon_i8mm, neon::I8mmMad, INFERNIX_I8MM_TARGET)
INFERNIX_NEON_ROW_SUMS(row_sums_neon_asimd, neon::AsimdMad, /* baseline */)

#undef INFERNIX_NEON_ROW_SUMS

struct CpuFeatures {
    bool neon    = false; // FEAT_ASIMD, the baseline AArch64 vector unit
    bool dotprod = false; // FEAT_DotProd (HWCAP_ASIMDDP)
    bool i8mm    = false; // FEAT_I8MM (HWCAP2_I8MM)
};

// The kernel reports optional features through AT_HWCAP/AT_HWCAP2 (the AArch64 analogue of
// CPUID/xgetbv). macOS exposes the same facts through sysctl.
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

int encode_a16(const std::uint16_t* v, int n, canon::A16Block* out) {
    if (n <= 0 || n % 16 != 0) { throw std::invalid_argument("offloaded_moe: A16 columns hold whole blocks"); }
    const int emax = canon::a16_column_exponent(v, n);
    encode_a16_blocks(v, emax, 0, n / 16, out);
    return emax;
}

void encode_a16_blocks(const std::uint16_t* v, int emax, int block_begin, int block_end, canon::A16Block* out) {
    for (int b = block_begin; b < block_end; ++b) { out[b] = canon::a16_block(v + 16 * b, emax); }
}

void rg16_row_sums_a16(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                       const canon::A16Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {
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

void gate_up_units_a16(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record,
                       const ExpertScales& scales, const canon::A16Block* const* x, const int* x_exp, int ncols,
                       int unit_begin, int unit_end, std::uint16_t* const* h, int prefetch_bytes) {
    std::int64_t s[32 * kMaxColumns]; // one unit's sums: no allocation on the miss path
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums_a16(isa, record, shape.gate_up_blocks(), 2 * u, 2 * u + 2, x, ncols, s, prefetch_bytes);
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = canon::a16_row_output(s[static_cast<std::size_t>(2 * i) * ncols + c], x_exp[c], scales.alpha_gate);
                const std::uint16_t p = canon::a16_row_output(s[static_cast<std::size_t>(2 * i + 1) * ncols + c], x_exp[c], scales.alpha_up);
                h[c][16 * u + i] = canon::swiglu_bf16(g, p);
            }
        }
    }
}

void down_rows_a16(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record,
                   const ExpertScales& scales, const canon::A16Block* const* h, const int* h_exp, int ncols,
                   int rg_begin, int rg_end, std::uint16_t* const* y, int prefetch_bytes) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums_a16(isa, record + shape.gate_up_bytes(), shape.down_blocks(), rg, rg + 1, h, ncols, s,
                          prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a16_row_output(s[r * ncols + c], h_exp[c], scales.alpha_down);
            }
        }
    }
}

void expert_forward_a16(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record,
                        const ExpertScales& scales, int ncols, const std::uint16_t* const* x,
                        std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!registered_shape(shape)) { throw std::invalid_argument("offloaded_moe: unregistered expert geometry"); }
    const int x_blocks = shape.gate_up_blocks(), h_blocks = shape.h_blocks();
    std::vector<canon::A16Block> xb(static_cast<std::size_t>(ncols) * x_blocks);
    std::vector<canon::A16Block> hb(static_cast<std::size_t>(ncols) * h_blocks);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * shape.intermediate);
    const canon::A16Block* x_ptr[kMaxColumns];
    const canon::A16Block* h_ptr[kMaxColumns];
    std::uint16_t* hv_ptr[kMaxColumns];
    int x_exp[kMaxColumns], h_exp[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        x_exp[c]  = encode_a16(x[c], shape.hidden, &xb[static_cast<std::size_t>(c) * x_blocks]);
        x_ptr[c]  = &xb[static_cast<std::size_t>(c) * x_blocks];
        hv_ptr[c] = &hv[static_cast<std::size_t>(c) * shape.intermediate];
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * h_blocks];
    }
    gate_up_units_a16(isa, shape, record, scales, x_ptr, x_exp, ncols, 0, h_blocks, hv_ptr, prefetch_bytes);
    for (int c = 0; c < ncols; ++c) {
        h_exp[c] = encode_a16(hv_ptr[c], shape.intermediate, &hb[static_cast<std::size_t>(c) * h_blocks]);
    }
    down_rows_a16(isa, shape, record, scales, h_ptr, h_exp, ncols, 0, shape.down_row_groups(), y,
                  prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
