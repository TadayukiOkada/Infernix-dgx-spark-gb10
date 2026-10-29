// PLE n-gram embedding decode (design §12.2, §12.3). ple_embed dispatches on the scale dtype:
//   - BF16 scale  -> FP8 rows: each head-row is head_width E4M3FN codes, out = bf16(e4m3 * scale).
//   - FP32 scale  -> NVFP4 rows: each head-row packs head_width/2 E2M1 codes (low nibble first)
//                     then head_width/16 E4M3FN block scales (one per 16 codes), out =
//                     bf16(e2m1(code[d]) * e4m3(scale[d / 16]) * scale).
// Both paths are checked bit for bit against an independent host oracle that reconstructs the
// represented values from the raw row bytes, so a layout/indexing change in the kernel fails here.
#include "infernix/ops/ple.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { cuda_check(cudaMalloc(&p, sizeof(T) * count), "cudaMalloc"); }
    Buffer(const std::vector<T>& v) : Buffer(v.size()) { upload(v); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v) {
        cuda_check(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    std::vector<T> download() const {
        std::vector<T> v(n);
        cuda_check(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "cudaMemcpy");
        return v;
    }
};

std::uint16_t bf16_rn(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, 4);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

// Independent reconstructions of the represented values (not a copy of the kernel's expressions).
float e2m1_code(unsigned code) {
    static const float mag[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float m = mag[code & 7U];
    return (code & 8U) != 0 ? -m : m;
}

float e4m3_byte(unsigned word) {
    const unsigned s = (word >> 7) & 1U, e = (word >> 3) & 0xFU, m = word & 7U;
    // E4M3FN: denormals (e == 0) are m * 2^-9; normals are (1 + m/8) * 2^(e - 7).
    const float value = (e == 0) ? static_cast<float>(m) * (1.0F / 512.0F)
                                 : ldexpf(1.0F + static_cast<float>(m) / 8.0F, static_cast<int>(e) - 7);
    return s != 0 ? -value : value;
}

// ----------------------------------------------------------------------------- NVFP4 path

void test_nvfp4(std::int32_t heads, std::int32_t head_width, std::int32_t columns) {
    const std::int32_t row_bytes = head_width / 2 + head_width / 16;
    const std::int32_t embed_dim = head_width * heads;
    std::mt19937 rng(static_cast<std::uint32_t>(heads * 31 + head_width * 13 + columns * 7));
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(0, 0x7E); // finite non-negative E4M3FN words
    std::uniform_real_distribution<float> global_dist(0.05F, 2.0F);
    const float global_scale = global_dist(rng);

    // Physical layout the kernel reads: column t, then head h, then row_bytes packed bytes.
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(columns) * heads * row_bytes);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(embed_dim) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t h = 0; h < heads; ++h) {
            std::uint8_t* row = rows.data() + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
            for (std::int32_t b = 0; b < head_width / 2; ++b) { row[b] = static_cast<std::uint8_t>(byte_dist(rng)); }
            for (std::int32_t b = 0; b < head_width / 16; ++b) {
                row[head_width / 2 + b] = static_cast<std::uint8_t>(scale_dist(rng));
            }
            for (std::int32_t d = 0; d < head_width; ++d) {
                const float cell = e2m1_code((row[d >> 1] >> (4 * (d & 1))) & 0xFU) * e4m3_byte(row[head_width / 2 + d / 16]);
                expected[static_cast<std::size_t>(t) * embed_dim + h * head_width + d] = bf16_rn(cell * global_scale);
            }
        }
    }
    Buffer<std::uint8_t> drows(rows);
    Buffer<float> dscale(std::vector<float>{global_scale});
    Buffer<std::uint16_t> dout(expected.size());
    Tensor tr(drows.p, DType::U8, {row_bytes, heads, columns});
    Tensor ts(dscale.p, DType::FP32, {1});
    Tensor to(dout.p, DType::BF16, {embed_dim, columns});
    infernix::ops::ple_embed(tr, ts, to, nullptr);
    cuda_check(cudaDeviceSynchronize(), "nvfp4 embed");
    const std::string tag = "nvfp4 heads=" + std::to_string(heads) + " width=" + std::to_string(head_width) +
                            " cols=" + std::to_string(columns);
    check(dout.download() == expected, tag + ": decoded rows differ from the host oracle");
}

// --------------------------------------------------------------------------------- FP8 path

void test_fp8(std::int32_t heads, std::int32_t head_width, std::int32_t columns) {
    const std::int32_t row_bytes = head_width;
    const std::int32_t elements = row_bytes * heads;
    std::mt19937 rng(static_cast<std::uint32_t>(heads * 17 + head_width * 5 + columns * 3));
    std::uniform_int_distribution<int> byte_dist(0, 0x7E); // finite non-negative codes
    const float global_f = 1.75F;
    const std::uint16_t scale_bf16 = bf16_rn(global_f);
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(columns) * heads * row_bytes);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(elements) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t h = 0; h < heads; ++h) {
            std::uint8_t* row = rows.data() + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
            for (std::int32_t d = 0; d < head_width; ++d) {
                row[d] = static_cast<std::uint8_t>(byte_dist(rng));
                __nv_bfloat16 sb;
                std::memcpy(&sb, &scale_bf16, 2);
                const float v = e4m3_byte(row[d]) * __bfloat162float(sb);
                expected[static_cast<std::size_t>(t) * elements + h * head_width + d] = bf16_rn(v);
            }
        }
    }
    Buffer<std::uint8_t> drows(rows);
    Buffer<std::uint16_t> dscale(std::vector<std::uint16_t>{scale_bf16}), dout(expected.size());
    Tensor tr(drows.p, DType::U8, {row_bytes, heads, columns});
    Tensor ts(dscale.p, DType::BF16, {1});
    Tensor to(dout.p, DType::BF16, {elements, columns});
    infernix::ops::ple_embed(tr, ts, to, nullptr);
    cuda_check(cudaDeviceSynchronize(), "fp8 embed");
    const std::string tag = "fp8 heads=" + std::to_string(heads) + " width=" + std::to_string(head_width) +
                            " cols=" + std::to_string(columns);
    check(dout.download() == expected, tag + ": decoded rows differ from the host oracle");
}

// Hand-computed spot checks: a positive and a negative E2M1 magnitude at scale 1.0 and global 2.0.
void test_nvfp4_spot(std::int32_t head_width) {
    const std::int32_t heads = 1, row_bytes = head_width / 2 + head_width / 16, embed_dim = head_width;
    const float global_scale = 2.0F;
    std::vector<std::uint8_t> row(row_bytes, 0);
    // Low nibble of byte 0 is E2M1 code 0x7 (+6); its block scale byte is E4M3FN 0x38 (1.0).
    row[0] = 0x07;
    row[head_width / 2] = 0x38;
    // Low nibble of byte 1 is E2M1 code 0xF (-6); shares the same block scale (d/16 == 0).
    row[1] = 0x0F;
    std::vector<std::uint16_t> expected(embed_dim, 0);
    expected[0] = bf16_rn(6.0F * 1.0F * global_scale);   // d0: +6 * 1 * 2 = +12
    expected[1] = 0;                                     // d1: high nibble of byte 0 is 0
    expected[2] = bf16_rn(-6.0F * 1.0F * global_scale);  // d2: -6 * 1 * 2 = -12
    Buffer<std::uint8_t> drows(row);
    Buffer<float> dscale(std::vector<float>{global_scale});
    Buffer<std::uint16_t> dout(embed_dim);
    Tensor tr(drows.p, DType::U8, {row_bytes, heads, 1});
    Tensor ts(dscale.p, DType::FP32, {1});
    Tensor to(dout.p, DType::BF16, {embed_dim, 1});
    infernix::ops::ple_embed(tr, ts, to, nullptr);
    cuda_check(cudaDeviceSynchronize(), "nvfp4 spot");
    check(dout.download() == expected, "nvfp4 spot: hand-computed cells differ");
}

} // namespace

INFERNIX_OP_TEST(ple_embed) {
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const auto [heads, width, cols] : {std::tuple{16, 160, 1}, std::tuple{16, 160, 5},
                                                std::tuple{8, 32, 3}, std::tuple{16, 160, 64}}) {
            test_nvfp4(heads, width, cols);
            test_fp8(heads, width, cols);
        }
        test_nvfp4_spot(160);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all PLE embed decode checks passed\n");
    return 0;
}
