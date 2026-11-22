// PLE Ops for NVIDIA Qwen3.8-Flash-Next-FP4-Local (docs/maintainer/qwen3_8-flash-next-design.md §12.5):
//
// - ple_embed_nvfp4 decodes NVFP4 n-gram rows (W/2 E2M1 code bytes, then W/16 E4M3FN scales) times the
//   table's FP32 multiplier. The oracle forms the exact product in FP64 (at most 30 significant bits)
//   and rounds it once to BF16, to nearest even; the Op must equal it bit for bit, on every code and
//   scale word, with multipliers whose products land on BF16 ties and beside them.
// - ple_gate with FP32 norm weights (FP4-Local's trained norms) against the FP64 closed formula,
//   under the bound it meets with BF16 norms: |out - ref| <= 2^-8 |ref| + 2^-12 rms(ref).

#include "ops/op_family.h"
#include "infernix/ops/ple.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

double e2m1(unsigned code) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8U) ? -kValues[code & 7U] : kValues[code & 7U];
}

double e4m3fn(std::uint8_t word) {
    const int exponent = (word >> 3) & 15, mantissa = word & 7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa, -9) : std::ldexp(8 + mantissa, exponent - 10);
    return (word & 0x80U) ? -magnitude : magnitude;
}

// Round a binary64 value to BF16 once, to nearest even (the values here are normal or zero).
std::uint16_t bf16_rne(double value) {
    if (value == 0.0) { return std::signbit(value) ? 0x8000U : 0U; }
    const double magnitude = std::fabs(value);
    int exponent           = 0;
    (void)std::frexp(magnitude, &exponent); // magnitude in [2^(exponent-1), 2^exponent)
    const double ulp     = std::ldexp(1.0, exponent - 8);
    double quotient      = magnitude / ulp; // in [128, 256), exact
    double integral      = std::floor(quotient);
    const double half    = quotient - integral;
    if (half > 0.5 || (half == 0.5 && std::fmod(integral, 2.0) != 0.0)) { integral += 1.0; }
    const float rounded = static_cast<float>(integral * ulp); // exact: 9 significant bits at most
    std::uint32_t bits;
    std::memcpy(&bits, &rounded, 4);
    return static_cast<std::uint16_t>((bits >> 16) | (value < 0 ? 0x8000U : 0U));
}

void embed_exact() {
    const auto one = [](const int kColumns) {
    constexpr int kWidth = 160, kHeads = 16, kRowBytes = kWidth / 2 + kWidth / 16;
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(kRowBytes) * kHeads * kColumns);
    std::uint32_t state = 2026U;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::size_t within = i % kRowBytes;
        auto byte                = static_cast<std::uint8_t>(next(state) >> 24);
        if (within >= kWidth / 2) { byte &= 0x7FU; } // block scales are nonnegative
        if (within >= kWidth / 2 && byte == 0x7FU) { byte = 0x7EU; }
        rows[i] = byte;
    }
    // Every code and every scale word appears; multipliers: FP4-Local's, a power of two, and 1 + 2^-7
    // (every product then has a 1 just past BF16's last bit, so many land on exact ties) and 1 + 2^-23.
    for (const float multiplier : {3.324236240587197e-05F, 0x1p-14F, 1.0F + 0x1p-7F, 1.0F + 0x1p-23F}) {
        Device rd(rows.size()), md(4), od(static_cast<std::size_t>(kHeads) * kWidth * kColumns * 2);
        t::cuda_check(cudaMemcpy(rd.p, rows.data(), rows.size(), cudaMemcpyHostToDevice), "rows");
        t::cuda_check(cudaMemcpy(md.p, &multiplier, 4, cudaMemcpyHostToDevice), "multiplier");
        const Tensor r(rd.p, DType::U8, {kRowBytes, kHeads, kColumns});
        const Tensor m(md.p, DType::FP32, {1});
        Tensor out(od.p, DType::BF16, {kHeads * kWidth, kColumns});
        infernix::ops::ple_embed_nvfp4(r, m, out, nullptr);
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * kColumns);
        t::cuda_check(cudaMemcpy(got.data(), od.p, got.size() * 2, cudaMemcpyDeviceToHost), "read");
        std::size_t bad = 0;
        for (int c = 0; c < kColumns; ++c) {
            for (int h = 0; h < kHeads; ++h) {
                const std::uint8_t* row = rows.data() + (static_cast<std::size_t>(c) * kHeads + h) * kRowBytes;
                for (int j = 0; j < kWidth; ++j) {
                    const unsigned code = (row[j / 2] >> (4 * (j % 2))) & 15U;
                    const double exact  = e2m1(code) * e4m3fn(row[kWidth / 2 + j / 16]) * static_cast<double>(multiplier);
                    bad += got[(static_cast<std::size_t>(c) * kHeads + h) * kWidth + j] != bf16_rne(exact);
                }
            }
        }
        check(bad == 0, "ple_embed_nvfp4 T=" + std::to_string(kColumns) +
                            " equals the exact product rounded once (multiplier " +
                            std::to_string(multiplier) + "): " + std::to_string(bad) + " differ");
    }
    };
    // Decode (one column, whose blocks split over warp-sized CTAs), a short chunk below one CTA per
    // SM, and a chunk wide enough to keep one CTA per column.
    for (const int columns : {1, 37, 64}) { one(columns); }
}

// The PLE gate's closed formula in FP64 (include/infernix/ops/ple.h).
void gate_oracle(const std::vector<std::uint16_t>& key, const std::vector<std::uint16_t>& value,
                 const std::vector<std::uint16_t>& residual, const std::vector<double>& wk, const std::vector<double>& wq,
                 const std::vector<double>& wc, int S, int H, int T, double eps, std::vector<double>& gated,
                 std::vector<double>& normalized) {
    const int W = S * H;
    gated.assign(static_cast<std::size_t>(W) * T, 0.0);
    normalized.assign(gated.size(), 0.0);
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < S; ++s) {
            const std::size_t base = static_cast<std::size_t>(c) * W + static_cast<std::size_t>(s) * H;
            double kk = 0, qq = 0;
            for (int d = 0; d < H; ++d) {
                const double k = t::bf16_to_f32(key[base + d]), q = t::bf16_to_f32(residual[base + d]);
                kk += k * k;
                qq += q * q;
            }
            const double ki = 1.0 / std::sqrt(kk / H + eps), qi = 1.0 / std::sqrt(qq / H + eps);
            double dot = 0;
            for (int d = 0; d < H; ++d) {
                const std::size_t ch = static_cast<std::size_t>(s) * H + d;
                dot += t::bf16_to_f32(key[base + d]) * ki * (1.0 + wk[ch]) * t::bf16_to_f32(residual[base + d]) * qi *
                       (1.0 + wq[ch]);
            }
            double g = dot / std::sqrt(static_cast<double>(H));
            g        = std::copysign(std::sqrt(std::max(std::fabs(g), 1e-6)), g);
            const double gate = 1.0 / (1.0 + std::exp(-g));
            double vv         = 0;
            for (int d = 0; d < H; ++d) {
                const double v = gate * t::bf16_to_f32(value[static_cast<std::size_t>(c) * H + d]);
                gated[base + d] = v;
                vv += v * v;
            }
            const double vi = 1.0 / std::sqrt(vv / H + eps);
            for (int d = 0; d < H; ++d) {
                normalized[base + d] = gated[base + d] * vi * (1.0 + wc[static_cast<std::size_t>(s) * H + d]);
            }
        }
    }
}

void gate_fp32_norms() {
    constexpr int S = 4, H = 2560, W = S * H, T = 5;
    constexpr float kEps = 1e-6F;
    std::uint32_t state = 77U;
    const auto uniform  = [&] {
        return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    };
    std::vector<std::uint16_t> key(static_cast<std::size_t>(W) * T), residual(key.size()), value(static_cast<std::size_t>(H) * T);
    for (auto& v : key) { v = t::f32_to_bf16(uniform()); }
    for (auto& v : residual) { v = t::f32_to_bf16(3.0F * uniform()); }
    for (auto& v : value) { v = t::f32_to_bf16(uniform()); }
    for (const bool fp32 : {false, true}) {
        std::vector<double> norms[3];
        std::vector<std::uint16_t> norm16[3];
        std::vector<float> norm32[3];
        Device nd[3] = {Device(W * 4), Device(W * 4), Device(W * 4)};
        for (int n = 0; n < 3; ++n) {
            norms[n].resize(W);
            norm16[n].resize(W);
            norm32[n].resize(W);
            for (int i = 0; i < W; ++i) {
                const float v = 0.25F * uniform() + 0x1p-20F * uniform(); // finer than BF16 for the FP32 case
                norm32[n][i]  = v;
                norm16[n][i]  = t::f32_to_bf16(v);
                norms[n][i]   = fp32 ? v : t::bf16_to_f32(norm16[n][i]);
            }
            t::cuda_check(cudaMemcpy(nd[n].p, fp32 ? static_cast<const void*>(norm32[n].data()) : norm16[n].data(),
                                     static_cast<std::size_t>(W) * (fp32 ? 4 : 2), cudaMemcpyHostToDevice),
                          "norm");
        }
        Device kd(key.size() * 2), vd(value.size() * 2), rd(residual.size() * 2), gd(key.size() * 2), od(key.size() * 2);
        t::cuda_check(cudaMemcpy(kd.p, key.data(), key.size() * 2, cudaMemcpyHostToDevice), "key");
        t::cuda_check(cudaMemcpy(vd.p, value.data(), value.size() * 2, cudaMemcpyHostToDevice), "value");
        t::cuda_check(cudaMemcpy(rd.p, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice), "residual");
        const DType norm_type = fp32 ? DType::FP32 : DType::BF16;
        Tensor gated(gd.p, DType::BF16, {W, T}), normalized(od.p, DType::BF16, {W, T});
        infernix::ops::ple_gate(Tensor(kd.p, DType::BF16, {W, T}), Tensor(vd.p, DType::BF16, {H, T}),
                                Tensor(rd.p, DType::BF16, {W, T}), Tensor(nd[0].p, norm_type, {W}),
                                Tensor(nd[1].p, norm_type, {W}), Tensor(nd[2].p, norm_type, {W}), S, kEps, gated,
                                normalized, nullptr);
        std::vector<std::uint16_t> got_gated(key.size()), got_normalized(key.size());
        t::cuda_check(cudaMemcpy(got_gated.data(), gd.p, key.size() * 2, cudaMemcpyDeviceToHost), "gated");
        t::cuda_check(cudaMemcpy(got_normalized.data(), od.p, key.size() * 2, cudaMemcpyDeviceToHost), "normalized");
        std::vector<double> ref_gated, ref_normalized;
        gate_oracle(key, value, residual, norms[0], norms[1], norms[2], S, H, T, kEps, ref_gated, ref_normalized);
        for (const auto& [got, ref, name] : {std::tuple{&got_gated, &ref_gated, "gated"},
                                             std::tuple{&got_normalized, &ref_normalized, "normalized"}}) {
            double ss = 0;
            for (const double v : *ref) { ss += v * v; }
            const double rms = std::sqrt(ss / static_cast<double>(ref->size()));
            std::size_t bad  = 0;
            for (std::size_t i = 0; i < ref->size(); ++i) {
                const double e = std::fabs(t::bf16_to_f32((*got)[i]) - (*ref)[i]);
                bad += e > std::ldexp(std::fabs((*ref)[i]), -8) + std::ldexp(rms, -12);
            }
            check(bad == 0, std::string("ple_gate ") + (fp32 ? "FP32" : "BF16") + " norms: " + name + " within its bound (" +
                                std::to_string(bad) + " outside)");
        }
    }
}

void hash_embed_exact() {
    constexpr int kNgramSize     = 5;
    constexpr int kHeadsPerNgram = 4;
    constexpr int kHeads         = (kNgramSize - 1) * kHeadsPerNgram;
    constexpr int kWidth         = 160;
    constexpr int kRowBytes      = kWidth / 2 + kWidth / 16;
    constexpr int kRowsPerBlock  = 4096 / kRowBytes;
    constexpr int kBlockBytes    = 4096;
    constexpr int kEos           = 248044;

    const auto splitmix64 = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    };
    const auto is_prime = [](std::uint64_t v) {
        if (v < 2) { return false; }
        if (v % 2 == 0) { return v == 2; }
        for (std::uint64_t d = 3; d * d <= v; d += 2) {
            if (v % d == 0) { return false; }
        }
        return true;
    };

    std::vector<std::uint64_t> mults(kNgramSize), sizes(kHeads), offsets(kHeads);
    for (int i = 0; i < kNgramSize; ++i) {
        mults[i] = 2 * (splitmix64(1234ULL + 0x9E3779B97F4A7C15ULL * (i + 1)) % 1000000000ULL) + 1;
    }
    std::uint64_t prime = 16, total_rows = 0;
    for (int i = 0; i < kHeads; ++i) {
        do { ++prime; } while (!is_prime(prime));
        sizes[i]   = prime;
        offsets[i] = total_rows;
        total_rows += prime;
    }
    const std::uint64_t blocks = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
    std::vector<std::uint8_t> table_blocks(static_cast<std::size_t>(blocks) * kBlockBytes, 0);
    std::uint32_t state = 9001U;
    for (std::uint64_t r = 0; r < total_rows; ++r) {
        std::uint8_t* dst = table_blocks.data() + (r / kRowsPerBlock) * kBlockBytes + (r % kRowsPerBlock) * kRowBytes;
        for (int b = 0; b < kRowBytes; ++b) {
            auto byte = static_cast<std::uint8_t>(next(state) >> 24);
            if (b >= kWidth / 2) { byte &= 0x7FU; }
            if (b >= kWidth / 2 && byte == 0x7FU) { byte = 0x7EU; }
            dst[b] = byte;
        }
    }

    Device d_blocks(table_blocks.size()), d_mults(mults.size() * 8), d_sizes(sizes.size() * 8),
        d_offsets(offsets.size() * 8), d_mult(4);
    t::cuda_check(cudaMemcpy(d_blocks.p, table_blocks.data(), table_blocks.size(), cudaMemcpyHostToDevice), "blocks");
    t::cuda_check(cudaMemcpy(d_mults.p, mults.data(), mults.size() * 8, cudaMemcpyHostToDevice), "mults");
    t::cuda_check(cudaMemcpy(d_sizes.p, sizes.data(), sizes.size() * 8, cudaMemcpyHostToDevice), "sizes");
    t::cuda_check(cudaMemcpy(d_offsets.p, offsets.data(), offsets.size() * 8, cudaMemcpyHostToDevice), "offsets");

    const Tensor t_mults(d_mults.p, DType::I64, {kNgramSize});
    const Tensor t_sizes(d_sizes.p, DType::I64, {kHeads});
    const Tensor t_offsets(d_offsets.p, DType::I64, {kHeads});
    const Tensor t_mult(d_mult.p, DType::FP32, {1});

    for (const int columns : {1, 8, 37}) {
        std::vector<std::int32_t> windows(static_cast<std::size_t>(kNgramSize) * columns);
        for (int c = 0; c < columns; ++c) {
            for (int s = 0; s < kNgramSize; ++s) {
                std::int32_t tok = static_cast<std::int32_t>((next(state) >> 12) % 248000U);
                if (c % 5 == 0 && s == 0) { tok = kEos; }     // EOS at s=0 must NOT close context
                if (c % 5 == 2 && s == 2) { tok = kEos; }     // EOS at s>0 closes s>=2
                if (c == 0 && s > 0) { tok = kEos; }          // BOS padding
                windows[static_cast<std::size_t>(c) * kNgramSize + s] = tok;
            }
        }
        Device d_win(windows.size() * 4), d_out(static_cast<std::size_t>(kHeads) * kWidth * columns * 2);
        t::cuda_check(cudaMemcpy(d_win.p, windows.data(), windows.size() * 4, cudaMemcpyHostToDevice), "windows");
        const Tensor t_win(d_win.p, DType::I32, {kNgramSize, columns});
        Tensor t_out(d_out.p, DType::BF16, {kHeads * kWidth, columns});

        for (const float multiplier : {3.324236240587197e-05F, 1.0F + 0x1p-7F}) {
            t::cuda_check(cudaMemcpy(d_mult.p, &multiplier, 4, cudaMemcpyHostToDevice), "multiplier");
            infernix::ops::PleNgramHashTable tbl{};
            tbl.resident_blocks = d_blocks.p;
            tbl.multipliers     = &t_mults;
            tbl.sizes           = &t_sizes;
            tbl.offsets         = &t_offsets;
            tbl.multiplier      = &t_mult;
            tbl.ngram_size      = kNgramSize;
            tbl.heads_per_ngram = kHeadsPerNgram;
            tbl.eos_token_id    = kEos;
            tbl.head_width      = kWidth;
            tbl.row_bytes       = kRowBytes;
            tbl.rows_per_block  = kRowsPerBlock;
            tbl.block_bytes     = kBlockBytes;
            tbl.total_rows      = total_rows;
            infernix::ops::ple_hash_embed_nvfp4(t_win, tbl, t_out, nullptr);

            std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * columns);
            t::cuda_check(cudaMemcpy(got.data(), d_out.p, got.size() * 2, cudaMemcpyDeviceToHost), "read");

            std::size_t bad = 0;
            for (int c = 0; c < columns; ++c) {
                const std::int32_t* w = windows.data() + static_cast<std::size_t>(c) * kNgramSize;
                bool closed           = false;
                std::uint64_t ctx[kNgramSize]{};
                for (int s = 0; s < kNgramSize; ++s) {
                    if (s > 0 && w[s] == kEos) { closed = true; }
                    ctx[s] = static_cast<std::uint64_t>(closed ? kEos : w[s]);
                }
                std::uint64_t mixed = ctx[0] * mults[0];
                for (int n = 2; n <= kNgramSize; ++n) {
                    mixed ^= ctx[n - 1] * mults[n - 1];
                    const int base = (n - 2) * kHeadsPerNgram;
                    for (int h = base; h < base + kHeadsPerNgram; ++h) {
                        const std::uint64_t row_id = mixed % sizes[h] + offsets[h];
                        const std::uint8_t* row =
                            table_blocks.data() + (row_id / kRowsPerBlock) * kBlockBytes +
                            (row_id % kRowsPerBlock) * kRowBytes;
                        for (int j = 0; j < kWidth; ++j) {
                            const unsigned code = (row[j / 2] >> (4 * (j % 2))) & 15U;
                            const double exact =
                                e2m1(code) * e4m3fn(row[kWidth / 2 + j / 16]) * static_cast<double>(multiplier);
                            bad += got[(static_cast<std::size_t>(c) * kHeads + h) * kWidth + j] != bf16_rne(exact);
                        }
                    }
                }
            }
            check(bad == 0, "ple_hash_embed_nvfp4 T=" + std::to_string(columns) + " mult=" +
                                std::to_string(multiplier) + ": " + std::to_string(bad) + " differ");
        }
    }
}

void gather_embed_nvfp4_exact() {
    constexpr int kWidth     = 160;
    constexpr int kHeads     = 16;
    constexpr int kRowBytes  = kWidth / 2 + kWidth / 16;
    constexpr int kCacheSlots = 128;

    struct MappedHost {
        void* p = nullptr;
        explicit MappedHost(std::size_t bytes) {
            t::cuda_check(cudaHostAlloc(&p, bytes, cudaHostAllocMapped), "cudaHostAlloc");
        }
        ~MappedHost() {
            if (p != nullptr) { cudaFreeHost(p); }
        }
        MappedHost(const MappedHost&)            = delete;
        MappedHost& operator=(const MappedHost&) = delete;
    };

    std::uint32_t state = 4242U;
    const auto fill_nvfp4_bytes = [&](std::uint8_t* dst, std::size_t n_rows) {
        for (std::size_t r = 0; r < n_rows; ++r) {
            std::uint8_t* row = dst + r * kRowBytes;
            for (int b = 0; b < kRowBytes; ++b) {
                auto byte = static_cast<std::uint8_t>(next(state) >> 24);
                if (b >= kWidth / 2) { byte &= 0x7FU; }
                if (b >= kWidth / 2 && byte == 0x7FU) { byte = 0x7EU; }
                row[b] = byte;
            }
        }
    };

    const auto check_output = [&](const std::vector<std::uint16_t>& got, const std::uint8_t* cache_p,
                                  const std::uint8_t* fallback_p, const std::uint32_t* slots_p, int columns,
                                  float multiplier, const std::string& label) {
        std::size_t bad = 0;
        for (int c = 0; c < columns; ++c) {
            for (int h = 0; h < kHeads; ++h) {
                const std::uint32_t slot_raw = slots_p[static_cast<std::size_t>(c) * kHeads + h];
                const std::uint8_t* row =
                    (slot_raw & infernix::ops::kPleCacheStagingBit) != 0U
                        ? (fallback_p + (static_cast<std::size_t>(c) * kHeads + h) * kRowBytes)
                        : (cache_p + static_cast<std::size_t>(slot_raw) * kRowBytes);
                for (int j = 0; j < kWidth; ++j) {
                    const unsigned code = (row[j / 2] >> (4 * (j % 2))) & 15U;
                    const double exact =
                        e2m1(code) * e4m3fn(row[kWidth / 2 + j / 16]) * static_cast<double>(multiplier);
                    bad += got[(static_cast<std::size_t>(c) * kHeads + h) * kWidth + j] != bf16_rne(exact);
                }
            }
        }
        check(bad == 0, label + ": " + std::to_string(bad) + " differ");
    };

    MappedHost cache(static_cast<std::size_t>(kCacheSlots) * kRowBytes);
    auto* cache_p = static_cast<std::uint8_t*>(cache.p);
    fill_nvfp4_bytes(cache_p, kCacheSlots);

    for (const int columns : {1, 8, 16, 64, 256}) {
        MappedHost fallback(static_cast<std::size_t>(columns) * kHeads * kRowBytes);
        MappedHost slots(static_cast<std::size_t>(columns) * kHeads * sizeof(std::uint32_t));
        auto* fallback_p = static_cast<std::uint8_t*>(fallback.p);
        auto* slots_p    = static_cast<std::uint32_t*>(slots.p);
        fill_nvfp4_bytes(fallback_p, static_cast<std::size_t>(columns) * kHeads);
        for (int i = 0; i < columns * kHeads; ++i) {
            slots_p[i] = (i % 13 == 0) ? infernix::ops::kPleCacheStagingBit : static_cast<std::uint32_t>(next(state) % kCacheSlots);
        }
        std::atomic_thread_fence(std::memory_order_release);

        Device md(sizeof(float)), od(static_cast<std::size_t>(kHeads) * kWidth * columns * sizeof(std::uint16_t));
        const Tensor t_scale(md.p, DType::FP32, {1});
        Tensor t_out(od.p, DType::BF16, {kHeads * kWidth, columns});

        for (const float multiplier : {3.324236240587197e-05F, 1.0F + 0x1p-7F}) {
            t::cuda_check(cudaMemcpy(md.p, &multiplier, sizeof(float), cudaMemcpyHostToDevice), "multiplier");
            infernix::ops::PleCacheGather gather{};
            gather.cache_base    = cache_p;
            gather.fallback_rows = fallback_p;
            gather.slots         = slots_p;
            gather.scale         = &t_scale;
            gather.cache_slots   = kCacheSlots;
            gather.heads         = kHeads;
            gather.head_width    = kWidth;
            gather.row_bytes     = kRowBytes;
            infernix::ops::ple_gather_embed_nvfp4(gather, t_out, nullptr);
            t::cuda_check(cudaDeviceSynchronize(), "gather embed nvfp4");
            std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * columns);
            t::cuda_check(cudaMemcpy(got.data(), od.p, got.size() * 2, cudaMemcpyDeviceToHost), "read");
            check_output(got, cache_p, fallback_p, slots_p, columns, multiplier,
                         "ple_gather_embed_nvfp4 ungated T=" + std::to_string(columns) + " mult=" +
                             std::to_string(multiplier));
        }
    }

    // Gated launch + CUDA Graph capture/replay with host-side slot/cache mutation between replays (.cv coherency).
    constexpr int kGraphCols    = 16;
    constexpr float kMultiplier = 1.0F + 0x1p-7F;
    MappedHost fallback(static_cast<std::size_t>(kGraphCols) * kHeads * kRowBytes);
    MappedHost slots(static_cast<std::size_t>(kGraphCols) * kHeads * sizeof(std::uint32_t));
    MappedHost ready_host(sizeof(std::uint32_t));
    auto* fallback_p = static_cast<std::uint8_t*>(fallback.p);
    auto* slots_p    = static_cast<std::uint32_t*>(slots.p);
    auto* ready_p    = static_cast<volatile std::uint32_t*>(ready_host.p);
    *ready_p         = 0U;

    Device md(sizeof(float)), d_expected(sizeof(std::uint32_t)), d_wait_stats(2 * sizeof(std::uint64_t)),
        od(static_cast<std::size_t>(kHeads) * kWidth * kGraphCols * sizeof(std::uint16_t));
    t::cuda_check(cudaMemcpy(md.p, &kMultiplier, sizeof(float), cudaMemcpyHostToDevice), "multiplier");
    t::cuda_check(cudaMemset(d_wait_stats.p, 0, 2 * sizeof(std::uint64_t)), "wait stats");
    const Tensor t_scale(md.p, DType::FP32, {1});
    Tensor t_out(od.p, DType::BF16, {kHeads * kWidth, kGraphCols});

    cudaStream_t stream = nullptr;
    t::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");

    infernix::ops::PleCacheGather gather{};
    gather.cache_base    = cache_p;
    gather.fallback_rows = fallback_p;
    gather.slots         = slots_p;
    gather.scale         = &t_scale;
    gather.cache_slots   = kCacheSlots;
    gather.heads         = kHeads;
    gather.head_width    = kWidth;
    gather.row_bytes     = kRowBytes;
    gather.ready         = static_cast<const std::uint32_t*>(ready_host.p);
    gather.expected      = static_cast<const std::uint32_t*>(d_expected.p);
    gather.wait_stats    = static_cast<std::uint64_t*>(d_wait_stats.p);

    cudaGraph_t graph         = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    infernix::ops::ple_gather_embed_nvfp4(gather, t_out, stream);
    t::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    t::cuda_check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "instantiate");

    const auto run_replay = [&](std::uint32_t word, bool delayed_host) {
        t::cuda_check(cudaMemcpyAsync(d_expected.p, &word, sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
                      "expected");
        const auto stage_data = [&] {
            fill_nvfp4_bytes(cache_p, kCacheSlots);
            fill_nvfp4_bytes(fallback_p, static_cast<std::size_t>(kGraphCols) * kHeads);
            for (int i = 0; i < kGraphCols * kHeads; ++i) {
                slots_p[i] =
                    (i % 7 == 0) ? infernix::ops::kPleCacheStagingBit : static_cast<std::uint32_t>(next(state) % kCacheSlots);
            }
            std::atomic_thread_fence(std::memory_order_release);
            *ready_p = word;
        };
        std::thread host_writer;
        if (delayed_host) {
            t::cuda_check(cudaGraphLaunch(graph_exec, stream), "graph launch gated");
            host_writer = std::thread([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                stage_data();
            });
        } else {
            stage_data();
            t::cuda_check(cudaGraphLaunch(graph_exec, stream), "graph launch ungated");
        }
        t::cuda_check(cudaStreamSynchronize(stream), "stream sync");
        if (host_writer.joinable()) { host_writer.join(); }

        std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * kGraphCols);
        t::cuda_check(cudaMemcpy(got.data(), od.p, got.size() * 2, cudaMemcpyDeviceToHost), "read");
        check_output(got, cache_p, fallback_p, slots_p, kGraphCols, kMultiplier,
                     "ple_gather_embed_nvfp4 graph replay word=" + std::to_string(word));
    };

    run_replay(1U, false);
    run_replay(2U, true);
    std::uint64_t wait_stats[2]{};
    t::cuda_check(cudaMemcpy(wait_stats, d_wait_stats.p, sizeof(wait_stats), cudaMemcpyDeviceToHost), "read wait_stats");
    check(wait_stats[0] > 0 && wait_stats[1] == 1ULL,
          "ple_gather_embed_nvfp4 gate wait stats expected >0 ns and 1 wait, got ns=" +
              std::to_string(wait_stats[0]) + " waits=" + std::to_string(wait_stats[1]));

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    // 256-iteration CUDA Graph host-mutation coherency stress test on slot 0 and slots[].
    for (const int stress_cols : {1, 64}) {
        MappedHost s_fallback(static_cast<std::size_t>(stress_cols) * kHeads * kRowBytes);
        MappedHost s_slots(static_cast<std::size_t>(stress_cols) * kHeads * sizeof(std::uint32_t));
        MappedHost s_ready_host(sizeof(std::uint32_t));
        auto* s_fallback_p = static_cast<std::uint8_t*>(s_fallback.p);
        auto* s_slots_p    = static_cast<std::uint32_t*>(s_slots.p);
        auto* s_ready_p    = static_cast<volatile std::uint32_t*>(s_ready_host.p);
        *s_ready_p         = 0U;

        Device d_s_expected(sizeof(std::uint32_t)),
            d_s_out(static_cast<std::size_t>(kHeads) * kWidth * stress_cols * sizeof(std::uint16_t));
        Tensor t_s_out(d_s_out.p, DType::BF16, {kHeads * kWidth, stress_cols});

        infernix::ops::PleCacheGather s_gather{};
        s_gather.cache_base    = cache_p;
        s_gather.fallback_rows = s_fallback_p;
        s_gather.slots         = s_slots_p;
        s_gather.scale         = &t_scale;
        s_gather.cache_slots   = kCacheSlots;
        s_gather.heads         = kHeads;
        s_gather.head_width    = kWidth;
        s_gather.row_bytes     = kRowBytes;
        s_gather.ready         = static_cast<const std::uint32_t*>(s_ready_host.p);
        s_gather.expected      = static_cast<const std::uint32_t*>(d_s_expected.p);

        cudaGraph_t s_graph         = nullptr;
        cudaGraphExec_t s_graph_exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "stress begin capture");
        infernix::ops::ple_gather_embed_nvfp4(s_gather, t_s_out, stream);
        t::cuda_check(cudaStreamEndCapture(stream, &s_graph), "stress end capture");
        t::cuda_check(cudaGraphInstantiate(&s_graph_exec, s_graph, nullptr, nullptr, 0), "stress instantiate");

        std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * stress_cols);
        for (std::uint32_t iter = 1; iter <= 256U; ++iter) {
            t::cuda_check(cudaMemcpyAsync(d_s_expected.p, &iter, sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
                          "stress expected");
            t::cuda_check(cudaGraphLaunch(s_graph_exec, stream), "stress graph launch");
            std::thread host_writer([&] {
                if (iter % 16U == 0U) { std::this_thread::sleep_for(std::chrono::microseconds(150)); }
                // Mutate slot 0 of mapped cache and fallback row 0 on every iteration.
                fill_nvfp4_bytes(cache_p, 1);
                fill_nvfp4_bytes(s_fallback_p, static_cast<std::size_t>(stress_cols) * kHeads);
                s_slots_p[0] = 0U;
                for (int i = 1; i < stress_cols * kHeads; ++i) {
                    s_slots_p[i] =
                        (i % 7 == 0) ? infernix::ops::kPleCacheStagingBit : ((iter + static_cast<std::uint32_t>(i)) % 3U == 0U ? 0U : static_cast<std::uint32_t>(next(state) % kCacheSlots));
                }
                std::atomic_thread_fence(std::memory_order_release);
                *s_ready_p = iter;
            });
            t::cuda_check(cudaStreamSynchronize(stream), "stress stream sync");
            host_writer.join();
            t::cuda_check(cudaMemcpy(got.data(), d_s_out.p, got.size() * 2, cudaMemcpyDeviceToHost), "stress read");
            check_output(got, cache_p, s_fallback_p, s_slots_p, stress_cols, kMultiplier,
                         "ple_gather_embed_nvfp4 coherency stress T=" + std::to_string(stress_cols) +
                             " iter=" + std::to_string(iter));
        }

        cudaGraphExecDestroy(s_graph_exec);
        cudaGraphDestroy(s_graph);
    }

    cudaStreamDestroy(stream);
}

} // namespace

INFERNIX_OP_TEST(ple_nvfp4) {
    try {
        if (!t::linear::cuda_available()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        embed_exact();
        hash_embed_exact();
        gather_embed_nvfp4_exact();
        gate_fp32_norms();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "PLE NVFP4 checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
