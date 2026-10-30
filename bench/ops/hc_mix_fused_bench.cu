// Public-Op benchmark for hyper_connection_mix fused route (K1a + K1b).
// Production dispatch is owned exclusively by hyper_connection_mix().

#include "core/weight.h"
#include "infernix/ops/hyper_connection.h"
#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"
#include "direct_bf16_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

infernix::bench::ColdFlushBytes kFlushBytes{};
constexpr int kHidden             = 2560;
constexpr int kStreams            = 4;
constexpr int kRank               = 320; // production rank (hc_lowrank in conversion.json)
constexpr int kDownRows           = kRank + kStreams; // 324 when combine=true
constexpr float kEps              = 1e-6F;

enum class Codec : std::uint8_t { Q8, Bf16 };

struct Options {
    Codec codec          = Codec::Q8;
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 12, 16};
    int warmup   = 8;
    int repeat   = 40;
    bool check   = false;
    bool graph   = false;
    std::string csv_out;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--codec" && i + 1 < argc) {
            ++i;
            opt.codec = std::string(argv[i]) == "bf16" ? Codec::Bf16 : Codec::Q8;
        } else if (arg == "--tokens" && i + 1 < argc) {
            ++i;
            opt.tokens.clear();
            std::string spec = argv[i];
            std::string token;
            for (char c : spec) {
                if (c == ',') {
                    opt.tokens.push_back(std::stoi(token));
                    token.clear();
                } else {
                    token += c;
                }
            }
            if (!token.empty()) opt.tokens.push_back(std::stoi(token));
        } else if (arg == "--warmup" && i + 1 < argc) {
            opt.warmup = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            opt.repeat = std::stoi(argv[++i]);
        } else if (arg == "--check") {
            opt.check = true;
        } else if (arg == "--graph") {
            opt.graph = true;
        } else if (arg == "-o" && i + 1 < argc) {
            opt.csv_out = argv[++i];
        }
    }
    return opt;
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s [--codec q8|bf16] [--tokens 1,2,4,...] [--warmup N] [--repeat N]\n"
                 "                  [--check] [--graph] [-o out.csv]\n",
                 prog);
}

struct HcWeights {
    DeviceBuffer down_storage;
    Weight down{};
    DeviceBuffer up_storage;
    Weight up{};
    DeviceBuffer norm;
    std::uint64_t weight_bytes = 0;
};

HcWeights make_weights(Codec codec, std::uint64_t seed) {
    HcWeights w;
    const int width = kStreams * kHidden; // 10240
    // norm_weight: BF16 [width], unit-offset (values near 0 for realistic RMSNorm gain)
    w.norm = bench::make_bf16(width, seed + 1, -0.01F, 0.01F);

    if (codec == Codec::Q8) {
        // RowSplit Q8: codes [rows, padded_k] where padded_k = align_up(k, 128).
        //   down: k = width = 10240, align_up(10240,128) = 10240 (divides evenly)
        //   up:   k = rank  =   320, align_up(320,128)   =   384
        constexpr int up_padded_k = (kRank + 127) / 128 * 128; // 384
        auto dq = bench::make_row_split_weight(QType::Q8_G32_FP16, kDownRows, width, width, seed + 2);
        w.down_storage = std::move(dq.storage);
        w.down         = dq.weight;
        auto uq = bench::make_row_split_weight(QType::Q8_G32_FP16, width, kRank, up_padded_k, seed + 3);
        w.up_storage = std::move(uq.storage);
        w.up         = uq.weight;
        if (!ops::detail::hc_mix_fused_supported(w.down, w.up, kHidden, kStreams, kRank,
                                                 ops::detail::kHcMixFusedMaxColumnsQ8)) {
            throw std::runtime_error("constructed Q8 weights rejected by hc_mix_fused_supported");
        }
        w.weight_bytes = dq.model_weight_bytes() + uq.model_weight_bytes();
    } else {
        auto dw = bench::make_direct_bf16_weight(kDownRows, width, static_cast<std::uint32_t>(seed + 2));
        w.down_storage = std::move(dw.storage);
        w.down         = dw.weight;
        auto uw = bench::make_direct_bf16_weight(width, kRank, static_cast<std::uint32_t>(seed + 3));
        w.up_storage = std::move(uw.storage);
        w.up         = uw.weight;
        w.weight_bytes = dw.model_weight_bytes() + uw.model_weight_bytes();
    }
    return w;
}

// Run the fused route directly via detail::hc_mix_fused().
void run_fused(const HcWeights& w, const DeviceBuffer& residual_buf, const DeviceBuffer& x_buf,
               const DeviceBuffer& inject_buf, void* partials, int T, cudaStream_t stream) {
    Tensor residual(residual_buf.p, DType::BF16, {kStreams * kHidden, T});
    Tensor norm_w(w.norm.p, DType::BF16, {kStreams * kHidden, 1});
    Tensor x(x_buf.p, DType::BF16, {kHidden, T});
    Tensor inject(inject_buf.p, DType::FP32, {kStreams, T});
    ops::detail::hc_mix_fused(residual, norm_w, w.down, w.up, kStreams, kRank, kEps, x, &inject,
                              partials, stream);
}

// Run the composed route via the public hyper_connection_mix() API.
void run_composed(const HcWeights& w, const DeviceBuffer& residual_buf, const DeviceBuffer& x_buf,
                  const DeviceBuffer& inject_buf, WorkspaceArena& workspace, int T,
                  cudaStream_t stream) {
    Tensor residual(residual_buf.p, DType::BF16, {kStreams * kHidden, T});
    Tensor norm_w(w.norm.p, DType::BF16, {kStreams * kHidden, 1});
    Tensor x(x_buf.p, DType::BF16, {kHidden, T});
    Tensor inject(inject_buf.p, DType::FP32, {kStreams, T});
    ops::hyper_connection_mix(residual, norm_w, w.down, w.up, ops::LinearPolicy::AllowA8, kStreams,
                              kRank, kEps, x, &inject, workspace, stream);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlushBytes);

        const HcWeights weights = make_weights(opt.codec, 0x51U);
        const char* codec_name  = opt.codec == Codec::Q8 ? "q8_g32_fp16" : "bf16";
        // The fused band is per-codec (the columns where it beats the composed route on this GPU).
        const int fused_max = opt.codec == Codec::Q8 ? ops::detail::kHcMixFusedMaxColumnsQ8
                                                     : ops::detail::kHcMixFusedMaxColumnsBf16;

        // Fused workspace: partials buffer sized for max T in the grid.
        const int max_T = *std::max_element(opt.tokens.begin(), opt.tokens.end());
        const std::size_t partials_bytes =
            ops::detail::hc_mix_fused_workspace_bytes(kDownRows, kStreams, max_T);
        DeviceBuffer partials(partials_bytes);
        partials.fill();

        // Composed workspace for correctness check / comparison.
        const std::size_t composed_ws_bytes = ops::hyper_connection_mix_workspace_capacity_bytes(
            weights.down, weights.up, ops::LinearPolicy::AllowA8, kStreams, kRank, max_T);
        DeviceArena composed_arena(std::max(composed_ws_bytes, static_cast<std::size_t>(256)));

        // Residual activation: BF16 [S*H, max_T] with realistic scale (~N(0,1) per stream).
        const std::uint64_t residual_elements =
            static_cast<std::uint64_t>(kStreams * kHidden) * max_T;
        DeviceBuffer residual_buf(residual_elements * sizeof(__nv_bfloat16));
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(residual_buf.p),
                                               residual_elements, 0xABCD1234ULL, -1.0F, 1.0F));
        // Output buffers
        DeviceBuffer x_buf(static_cast<std::uint64_t>(kHidden) * max_T * sizeof(__nv_bfloat16));
        DeviceBuffer inject_buf(static_cast<std::uint64_t>(kStreams) * max_T * sizeof(float));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::fprintf(stderr,
                     "hc_mix_fused_bench: codec=%s H=%d S=%d rank=%d down_rows=%d weight_bytes=%.2fMB\n",
                     codec_name, kHidden, kStreams, kRank, kDownRows,
                     weights.weight_bytes / (1024.0 * 1024.0));

        // Header
        std::printf("codec\ttokens\troute\tmedian_us\tmin_us\tp95_us\tgb_s\tweight_gb\n");

        struct Row {
            int tokens;
            const char* route;
            bench::ColdTiming timing;
            double gb_s;
        };
        std::vector<Row> results;

        for (int T : opt.tokens) {
            if (T > fused_max && opt.check) {
                // fused only serves its codec's band; skip the fused arm beyond that
                continue;
            }

            // --- Fused route timing ---
            if (T <= fused_max) {
                auto launch_fused = [&](cudaStream_t s) {
                    run_fused(weights, residual_buf, x_buf, inject_buf, partials.p, T, s);
                };

                bench::ColdTiming cold;
                if (opt.graph) {
                    bench::TimedGraph graph;
                    graph.capture(stream, [&](cudaStream_t s) { launch_fused(s); });
                    cold = bench::measure_cold_graph(graph, flush, stream, opt.warmup, opt.repeat);
                } else {
                    cold = bench::measure_cold_launch(launch_fused, flush, stream, opt.warmup,
                                                      opt.repeat);
                }

                // Also measure warm (no L2 flush) for reference.
                bench::ColdTiming warm = bench::measure_launch(launch_fused, stream, opt.warmup,
                                                               opt.repeat);

                const double elapsed_s = cold.median_us * 1e-6;
                const double total_bytes =
                    static_cast<double>(weights.weight_bytes) +
                    static_cast<double>(kStreams * kHidden * T * 2) + // residual read
                    static_cast<double>(kHidden * T * 2) +            // x write
                    static_cast<double>(kStreams * T * 4);            // inject write
                const double gb_s = total_bytes / elapsed_s / 1e9;

                std::printf("%s\t%d\tfused_cold\t%.2f\t%.2f\t%.2f\t%.1f\t%.3f\n", codec_name, T,
                            cold.median_us, cold.min_us, cold.p95_us, gb_s,
                            weights.weight_bytes / 1e9);
                std::printf("%s\t%d\tfused_warm\t%.2f\t%.2f\t%.2f\t%.1f\t%.3f\n", codec_name, T,
                            warm.median_us, warm.min_us, warm.p95_us,
                            total_bytes / (warm.median_us * 1e-6) / 1e9,
                            weights.weight_bytes / 1e9);
                results.push_back({T, "fused_cold", cold, gb_s});
                results.push_back({T, "fused_warm", warm, total_bytes / (warm.median_us * 1e-6) / 1e9});
            }

            // --- Composed route timing (for comparison at all T) ---
            {
                auto launch_composed = [&](cudaStream_t s) {
                    run_composed(weights, residual_buf, x_buf, inject_buf, composed_arena, T, s);
                };
                bench::ColdTiming cold_c =
                    bench::measure_cold_launch(launch_composed, flush, stream, opt.warmup,
                                               opt.repeat);
                const double elapsed_s   = cold_c.median_us * 1e-6;
                const double total_bytes = static_cast<double>(weights.weight_bytes) +
                                           static_cast<double>(kStreams * kHidden * T * 2) +
                                           static_cast<double>(kHidden * T * 2);
                const double gb_s        = total_bytes / elapsed_s / 1e9;
                std::printf("%s\t%d\tcomposed_cold\t%.2f\t%.2f\t%.2f\t%.1f\t%.3f\n", codec_name, T,
                            cold_c.median_us, cold_c.min_us, cold_c.p95_us, gb_s,
                            weights.weight_bytes / 1e9);
                results.push_back({T, "composed_cold", cold_c, gb_s});
            }

            // --- Correctness check: fused vs composed ---
            if (opt.check && T <= fused_max) {
                CUDA_CHECK(cudaMemsetAsync(x_buf.p, 0, x_buf.bytes, stream));
                CUDA_CHECK(cudaMemsetAsync(inject_buf.p, 0, inject_buf.bytes, stream));
                run_fused(weights, residual_buf, x_buf, inject_buf, partials.p, T, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                DeviceBuffer x_ref(x_buf.bytes);
                DeviceBuffer inj_ref(inject_buf.bytes);
                CUDA_CHECK(cudaMemcpyAsync(x_ref.p, x_buf.p, x_buf.bytes, cudaMemcpyDeviceToDevice,
                                           stream));
                CUDA_CHECK(cudaMemcpyAsync(inj_ref.p, inject_buf.p, inject_buf.bytes,
                                           cudaMemcpyDeviceToDevice, stream));

                // Run composed into the same buffers
                CUDA_CHECK(cudaMemsetAsync(x_buf.p, 0, x_buf.bytes, stream));
                CUDA_CHECK(cudaMemsetAsync(inject_buf.p, 0, inject_buf.bytes, stream));
                run_composed(weights, residual_buf, x_buf, inject_buf, composed_arena, T, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Compare on host (only the first T columns are valid output)
                const int x_count    = kHidden * T;
                const int inj_count  = kStreams * T;
                std::vector<__nv_bfloat16> x_host(static_cast<std::size_t>(x_count));
                std::vector<__nv_bfloat16> x_comp(static_cast<std::size_t>(x_count));
                std::vector<float> inj_host(static_cast<std::size_t>(inj_count));
                std::vector<float> inj_comp(static_cast<std::size_t>(inj_count));
                const std::size_t x_bytes_to_copy   = static_cast<std::size_t>(x_count) * sizeof(__nv_bfloat16);
                const std::size_t inj_bytes_to_copy = static_cast<std::size_t>(inj_count) * sizeof(float);
                CUDA_CHECK(cudaMemcpy(x_host.data(), x_ref.p, x_bytes_to_copy, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(x_comp.data(), x_buf.p, x_bytes_to_copy, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(inj_host.data(), inj_ref.p, inj_bytes_to_copy,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(inj_comp.data(), inject_buf.p, inj_bytes_to_copy,
                                      cudaMemcpyDeviceToHost));

                float max_x_diff = 0, max_inj_diff = 0;
                for (int i = 0; i < x_count; ++i) {
                    float a = __bfloat162float(x_host[i]);
                    float b = __bfloat162float(x_comp[i]);
                    max_x_diff = std::max(max_x_diff, std::fabs(a - b));
                }
                for (int i = 0; i < inj_count; ++i) {
                    max_inj_diff = std::max(max_inj_diff, std::fabs(inj_host[i] - inj_comp[i]));
                }
                std::fprintf(stderr, "  [check T=%d] max|x_fused-x_composed|=%.6f  "
                                     "max|inject_diff|=%.6f\n",
                             T, max_x_diff, max_inj_diff);
                // The fused path keeps intermediates in FP32 while composed rounds to BF16 between
                // steps, so we expect differences of order ~2^-8 (bf16 mantissa). A threshold of
                // 2e-3 is generous enough to catch real bugs while tolerating rounding divergence.
                if (max_x_diff > 2e-3F || max_inj_diff > 2e-3F) {
                    std::fprintf(stderr, "  FAIL: difference exceeds tolerance\n");
                    return 3;
                }
            }
        }

        // CSV output
        if (!opt.csv_out.empty()) {
            std::FILE* out = std::fopen(opt.csv_out.c_str(), "w");
            if (!out) throw std::runtime_error("cannot open csv output");
            std::fprintf(out, "codec,tokens,route,median_us,min_us,p95_us,gb_s,weight_gb\n");
            for (const Row& r : results) {
                std::fprintf(out, "%s,%d,%s,%.2f,%.2f,%.2f,%.1f,%.3f\n", codec_name, r.tokens,
                             r.route, r.timing.median_us, r.timing.min_us, r.timing.p95_us, r.gb_s,
                             weights.weight_bytes / 1e9);
            }
            std::fclose(out);
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_hc_mix_fused_bench: %s\n", error.what());
        usage(argv[0]);
        return 2;
    }
}
