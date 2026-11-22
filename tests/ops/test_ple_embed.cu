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

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
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

void test_gather_embed_fp8(int width, std::initializer_list<int> cols_list, bool run_stress) {
    constexpr int kHeads      = 16;
    constexpr int kCacheSlots = 96;

    struct MappedHost {
        void* p = nullptr;
        explicit MappedHost(std::size_t bytes) {
            cuda_check(cudaHostAlloc(&p, bytes, cudaHostAllocMapped), "cudaHostAlloc");
        }
        ~MappedHost() {
            if (p != nullptr) { cudaFreeHost(p); }
        }
        MappedHost(const MappedHost&)            = delete;
        MappedHost& operator=(const MappedHost&) = delete;
    };

    std::mt19937 rng(7331U + static_cast<std::uint32_t>(width));
    std::uniform_int_distribution<int> byte_dist(0, 0x7E);
    std::uniform_int_distribution<int> slot_dist(0, kCacheSlots - 1);
    const std::uint16_t scale_bf16 = bf16_rn(1.75F);
    __nv_bfloat16 sb;
    std::memcpy(&sb, &scale_bf16, 2);
    const float scale_f = __bfloat162float(sb);
    Buffer<std::uint16_t> dscale(std::vector<std::uint16_t>{scale_bf16});
    Tensor ts(dscale.p, DType::BF16, {1});

    MappedHost cache(static_cast<std::size_t>(kCacheSlots) * width);
    auto* cache_p = static_cast<std::uint8_t*>(cache.p);
    for (int i = 0; i < kCacheSlots * width; ++i) { cache_p[i] = static_cast<std::uint8_t>(byte_dist(rng)); }

    const auto compute_oracle = [&](const std::uint8_t* fallback_p, const std::uint32_t* slots_p, int columns) {
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(kHeads) * width * columns);
        for (int t = 0; t < columns; ++t) {
            for (int h = 0; h < kHeads; ++h) {
                const std::uint32_t slot_raw = slots_p[static_cast<std::size_t>(t) * kHeads + h];
                const std::uint8_t* row =
                    (slot_raw & infernix::ops::kPleCacheStagingBit) != 0U
                        ? (fallback_p + (static_cast<std::size_t>(t) * kHeads + h) * width)
                        : (cache_p + static_cast<std::size_t>(slot_raw) * width);
                for (int d = 0; d < width; ++d) {
                    expected[static_cast<std::size_t>(t) * (kHeads * width) + h * width + d] =
                        bf16_rn(e4m3_byte(row[d]) * scale_f);
                }
            }
        }
        return expected;
    };

    for (const int columns : cols_list) {
        MappedHost fallback(static_cast<std::size_t>(columns) * kHeads * width);
        MappedHost slots(static_cast<std::size_t>(columns) * kHeads * sizeof(std::uint32_t));
        auto* fallback_p = static_cast<std::uint8_t*>(fallback.p);
        auto* slots_p    = static_cast<std::uint32_t*>(slots.p);
        for (int i = 0; i < columns * kHeads * width; ++i) {
            fallback_p[i] = static_cast<std::uint8_t>(byte_dist(rng));
        }
        for (int i = 0; i < columns * kHeads; ++i) {
            slots_p[i] = (i % 11 == 0) ? infernix::ops::kPleCacheStagingBit : static_cast<std::uint32_t>(slot_dist(rng));
        }
        std::atomic_thread_fence(std::memory_order_release);

        Buffer<std::uint16_t> dout(static_cast<std::size_t>(kHeads) * width * columns);
        Tensor to(dout.p, DType::BF16, {kHeads * width, columns});
        infernix::ops::PleCacheGather gather{};
        gather.cache_base    = cache_p;
        gather.fallback_rows = fallback_p;
        gather.slots         = slots_p;
        gather.scale         = &ts;
        gather.cache_slots   = kCacheSlots;
        gather.heads         = kHeads;
        gather.head_width    = width;
        gather.row_bytes     = width;
        infernix::ops::ple_gather_embed_fp8(gather, to, nullptr);
        cuda_check(cudaDeviceSynchronize(), "fp8 gather embed");
        check(dout.download() == compute_oracle(fallback_p, slots_p, columns),
              "fp8 gather ungated width=" + std::to_string(width) + " cols=" + std::to_string(columns) +
                  ": differs from host oracle");
    }

    // Gated CUDA Graph capture/replay with host cache mutation between replays.
    constexpr int kGraphCols = 16;
    MappedHost fallback(static_cast<std::size_t>(kGraphCols) * kHeads * width);
    MappedHost slots(static_cast<std::size_t>(kGraphCols) * kHeads * sizeof(std::uint32_t));
    MappedHost ready_host(sizeof(std::uint32_t));
    auto* fallback_p = static_cast<std::uint8_t*>(fallback.p);
    auto* slots_p    = static_cast<std::uint32_t*>(slots.p);
    auto* ready_p    = static_cast<volatile std::uint32_t*>(ready_host.p);
    *ready_p         = 0U;

    Buffer<std::uint32_t> d_expected(1);
    Buffer<std::uint64_t> d_wait_stats(std::vector<std::uint64_t>{0ULL, 0ULL});
    Buffer<std::uint16_t> dout(static_cast<std::size_t>(kHeads) * width * kGraphCols);
    Tensor to(dout.p, DType::BF16, {kHeads * width, kGraphCols});

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");

    infernix::ops::PleCacheGather gather{};
    gather.cache_base    = cache_p;
    gather.fallback_rows = fallback_p;
    gather.slots         = slots_p;
    gather.scale         = &ts;
    gather.cache_slots   = kCacheSlots;
    gather.heads         = kHeads;
    gather.head_width    = width;
    gather.row_bytes     = width;
    gather.ready         = static_cast<const std::uint32_t*>(ready_host.p);
    gather.expected      = d_expected.p;
    gather.wait_stats    = d_wait_stats.p;

    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    infernix::ops::ple_gather_embed_fp8(gather, to, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "instantiate");

    for (const std::uint32_t word : {1U, 2U}) {
        cuda_check(cudaMemcpyAsync(d_expected.p, &word, sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
                   "expected");
        const auto stage = [&] {
            for (int i = 0; i < kCacheSlots * width; ++i) { cache_p[i] = static_cast<std::uint8_t>(byte_dist(rng)); }
            for (int i = 0; i < kGraphCols * kHeads * width; ++i) {
                fallback_p[i] = static_cast<std::uint8_t>(byte_dist(rng));
            }
            for (int i = 0; i < kGraphCols * kHeads; ++i) {
                slots_p[i] =
                    (i % 9 == 0) ? infernix::ops::kPleCacheStagingBit : static_cast<std::uint32_t>(slot_dist(rng));
            }
            std::atomic_thread_fence(std::memory_order_release);
            *ready_p = word;
        };
        std::thread writer;
        if (word == 2U) {
            cuda_check(cudaGraphLaunch(graph_exec, stream), "graph launch gated");
            writer = std::thread([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                stage();
            });
        } else {
            stage();
            cuda_check(cudaGraphLaunch(graph_exec, stream), "graph launch ungated");
        }
        cuda_check(cudaStreamSynchronize(stream), "stream sync");
        if (writer.joinable()) { writer.join(); }
        check(dout.download() == compute_oracle(fallback_p, slots_p, kGraphCols),
              "fp8 gather graph replay width=" + std::to_string(width) + " word=" + std::to_string(word) +
                  ": differs from oracle");
    }
    const auto stats = d_wait_stats.download();
    check(stats[0] > 0 && stats[1] == 1ULL, "fp8 gather gate wait_stats mismatch");

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);

    if (run_stress) {
        for (const int stress_cols : {1, 64}) {
            MappedHost s_fallback(static_cast<std::size_t>(stress_cols) * kHeads * width);
            MappedHost s_slots(static_cast<std::size_t>(stress_cols) * kHeads * sizeof(std::uint32_t));
            MappedHost s_ready_host(sizeof(std::uint32_t));
            auto* s_fallback_p = static_cast<std::uint8_t*>(s_fallback.p);
            auto* s_slots_p    = static_cast<std::uint32_t*>(s_slots.p);
            auto* s_ready_p    = static_cast<volatile std::uint32_t*>(s_ready_host.p);
            *s_ready_p         = 0U;

            Buffer<std::uint32_t> d_s_expected(1);
            Buffer<std::uint16_t> d_s_out(static_cast<std::size_t>(kHeads) * width * stress_cols);
            Tensor t_s_out(d_s_out.p, DType::BF16, {kHeads * width, stress_cols});

            infernix::ops::PleCacheGather s_gather{};
            s_gather.cache_base    = cache_p;
            s_gather.fallback_rows = s_fallback_p;
            s_gather.slots         = s_slots_p;
            s_gather.scale         = &ts;
            s_gather.cache_slots   = kCacheSlots;
            s_gather.heads         = kHeads;
            s_gather.head_width    = width;
            s_gather.row_bytes     = width;
            s_gather.ready         = static_cast<const std::uint32_t*>(s_ready_host.p);
            s_gather.expected      = d_s_expected.p;

            cudaGraph_t s_graph          = nullptr;
            cudaGraphExec_t s_graph_exec = nullptr;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "stress begin capture");
            infernix::ops::ple_gather_embed_fp8(s_gather, t_s_out, stream);
            cuda_check(cudaStreamEndCapture(stream, &s_graph), "stress end capture");
            cuda_check(cudaGraphInstantiate(&s_graph_exec, s_graph, nullptr, nullptr, 0), "stress instantiate");

            for (std::uint32_t iter = 1; iter <= 256U; ++iter) {
                cuda_check(cudaMemcpyAsync(d_s_expected.p, &iter, sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream),
                           "stress expected");
                cuda_check(cudaGraphLaunch(s_graph_exec, stream), "stress launch");
                std::thread writer([&] {
                    if (iter % 16U == 0U) { std::this_thread::sleep_for(std::chrono::microseconds(150)); }
                    for (int d = 0; d < width; ++d) { cache_p[d] = static_cast<std::uint8_t>(byte_dist(rng)); }
                    for (int i = 0; i < stress_cols * kHeads * width; ++i) {
                        s_fallback_p[i] = static_cast<std::uint8_t>(byte_dist(rng));
                    }
                    s_slots_p[0] = 0U;
                    for (int i = 1; i < stress_cols * kHeads; ++i) {
                        s_slots_p[i] =
                            (i % 9 == 0) ? infernix::ops::kPleCacheStagingBit
                                         : (((iter + static_cast<std::uint32_t>(i)) % 3U == 0U)
                                                ? 0U
                                                : static_cast<std::uint32_t>(slot_dist(rng)));
                    }
                    std::atomic_thread_fence(std::memory_order_release);
                    *s_ready_p = iter;
                });
                cuda_check(cudaStreamSynchronize(stream), "stress stream sync");
                writer.join();
                check(d_s_out.download() == compute_oracle(s_fallback_p, s_slots_p, stress_cols),
                      "fp8 gather coherency stress width=" + std::to_string(width) +
                          " cols=" + std::to_string(stress_cols) + " iter=" + std::to_string(iter) +
                          ": differs from oracle");
            }

            cudaGraphExecDestroy(s_graph_exec);
            cudaGraphDestroy(s_graph);
        }
    }

    cudaStreamDestroy(stream);
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
        test_gather_embed_fp8(160, {1, 4, 32, 64, 256}, true);
        test_gather_embed_fp8(24, {1, 4, 32}, false);
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
