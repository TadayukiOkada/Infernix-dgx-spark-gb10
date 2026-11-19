// Route screening for the narrow-T band of one Q8 Linear shape. No production dispatch change.
//
// Motivation: the production tables were fitted on RTX 5090. A T-continuity sweep on GB10
// (bench/ops/linear_bench --sweep) shows sharp cliffs where one more token costs 30-45% more,
// which can only happen when the selector jumps to a route that is worse, not better, for that
// width. This bench measures every available route at the same T so the true crossover can be
// re-derived on GB10 instead of inherited.
//
// Every route here accumulates each output over K in the same order as production
// (kChunk/FMA/reduction order is untouched), so the outputs are bit-identical; only speed differs.
// The harness verifies that against the production route before timing anything.

#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "quantized_weight.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;
using infernix::ops::Cache; // outer-namespace enum, not pulled in by the detail using-directive

namespace {

struct Route {
    const char* name;
    Q8Launch launch;
};

// Generic route builders: instantiate an arbitrary schedule and adapt it to the Q8Launch signature
// so the sweep can time any point of the schedule grid, not just the pre-registered globals. A
// launch that rejects the operands (static-K / exact-token guards) is caught so one bad combo
// cannot abort a multi-shape sweep: it leaves stale output and fails the exactness gate rather
// than being mistaken for a winner.
template <class Schedule>
void mma_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
    try {
        launch_q8_a16_mma<Schedule>(q8_linear_operands(x, w),
                                    LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                    LinearIdentityEpilogue{}, s);
    } catch (...) {
    }
}
template <class Schedule>
void sliced_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
    try {
        launch_q8_a16_sliced_k_mma<Schedule>(
            q8_linear_operands(x, w),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
            LinearIdentityEpilogue{}, s);
    } catch (...) {
    }
}
template <class Schedule>
void grouped_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
    try {
        launch_q8_a16_grouped_sliced_k_mma<Schedule>(
            q8_linear_operands(x, w),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
            LinearIdentityEpilogue{}, s);
    } catch (...) {
    }
}

// The register-streamed route is the exception to the K-generic grid above: its schedule carries a
// static K, so only the instantiations built for the screened K can match it. A mismatched K throws
// inside the launcher and is swallowed here, which leaves the zeroed output to fail the exactness
// gate rather than being timed as an empty launch and mistaken for a winner.
template <class Schedule>
void stream_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
    try {
        launch_q8_a16_stream<Schedule>(
            q8_linear_operands(x, w),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
            LinearIdentityEpilogue{}, s);
    } catch (...) {
    }
}

template <int K, int RowsPerWarp, int KSplit, int Warps, int MaxCols, int ColsPerPass,
          Q8StreamX XStage, int MinBlocksPerSm>
using StreamSched = Q8A16StreamSchedule<K, RowsPerWarp, KSplit, Warps, MaxCols, ColsPerPass, XStage,
                                        MinBlocksPerSm>;

// The routes the production tables can reach for these shapes, plus the nearest neighbours on
// either side of the boundary under test.
const Route kRoutes[] = {
    {"production", [](const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
         ops::linear(x, w, out, s);
     }},
    {"simt_r8_t4", launch_q8_a16_simt_r8_t4},
    {"simt_r8_t5", launch_q8_a16_simt_r8_t5},
    {"simt_r8_t6", launch_q8_a16_simt_r8_t6},
    {"simt_r8_t8", launch_q8_a16_simt_r8_t8},
    {"simt_r1_t4_w8", launch_q8_a16_simt_r1_t4_w8},
    {"simt_r1_t8_w8", launch_q8_a16_simt_r1_t8_w8},
    {"mma_r32_t16", launch_q8_a16_mma_r32_t16},
    {"mma_r32_t24", launch_q8_a16_mma_r32_t24},
    {"mma_r32_t32", launch_q8_a16_mma_r32_t32},
    {"mma_r32_t48", launch_q8_a16_mma_r32_t48},
    {"mma_r32_t64", launch_q8_a16_mma_r32_t64},
    {"mma_r32_t96", launch_q8_a16_mma_r32_t96},
    {"mma_r32_t128", launch_q8_a16_mma_r32_t128},
    {"mma_r48_t64", launch_q8_a16_mma_r48_t64},
    {"mma_r64_t32", launch_q8_a16_mma_r64_t32},
    {"mma_r64_t48", launch_q8_a16_mma_r64_t48},
    {"mma_r64_t96", launch_q8_a16_mma_r64_t96},
    {"mma_r64_t128", launch_q8_a16_mma_r64_t128},
    {"mma_r96_t96", launch_q8_a16_mma_r96_t96},
    {"mma_r128_t64", launch_q8_a16_mma_r128_t64},
    {"mma_r128_t80", launch_q8_a16_mma_r128_t80},
    {"sliced_r16_t16_w8_s2", launch_q8_a16_sliced_r16_t16_w8_s2},
    {"sliced_r16_t16_w4_s2", launch_q8_a16_sliced_r16_t16_w4_s2},
    {"sliced_r16_t32_w4_s2", launch_q8_a16_sliced_r16_t32_w4_s2},
    {"sliced_r32_t16_w4_s2", launch_q8_a16_sliced_r32_t16_w4_s2},
    {"sliced_r32_t16_w8_s2", launch_q8_a16_sliced_r32_t16_w8_s2},
    {"simt_r4_t4_w2_g16_s2", launch_q8_a16_simt_r4_t4_w2_g16_s2},
    {"simt_r4_t5_w2_g16_s2", launch_q8_a16_simt_r4_t5_w2_g16_s2},
    {"simt_r4_t8_w2_g16_s2", launch_q8_a16_simt_r4_t8_w2_g16_s2},
    {"mma_r64x16_t48_k128", launch_q8_a16_mma_r64x16_t48_k128_a1},
    {"mma_r64x32_t64_k128", launch_q8_a16_mma_r64x32_t64_k128_a1},
    {"mma_r64x32_t96_k128_a1", launch_q8_a16_mma_r64x32_t96_k128_a1},
    {"mma_r64x16_t128_k128_a2", launch_q8_a16_mma_r64x16_t128_k128_a2},
    {"mma_r64x16_t256_k64_a2", launch_q8_a16_mma_r64x16_t256_k64_a2},
    {"mma_r64x32_t64_k64_a2", launch_q8_a16_mma_r64x32_t64_k64_a2},
    {"mma_r64x32_t128_k64_a2", launch_q8_a16_mma_r64x32_t128_k64_a2},
    {"mma_r128x32_t64_k64_a2", launch_q8_a16_mma_r128x32_t64_k64_a2},
    {"mma_r128x64_t128_k64_a1", launch_q8_a16_mma_r128x64_t128_k64_a1},
    {"mma_r128x32_t128_k64_a2", launch_q8_a16_mma_r128x32_t128_k64_a2},
    // Register-streamed variants for the two Qwen3.8-Flash-Next hyper-connection shapes, whose
    // tables name a Stream alias (K=320 up, K=10240 down). The SIMT route's column count is a
    // compile-time kCols, so a padded width does dead FMAs; the streamed route bounds each pass by
    // the runtime live-column count, which is the difference being screened here. Instantiations
    // whose static K does not match the screened shape throw and fail the exactness gate.
    {"stream_k320_m8c8", stream_route<StreamSched<320, 4, 1, 8, 8, 8, Q8StreamX::Global, 2>>},
    {"stream_k320_m8c4", stream_route<StreamSched<320, 4, 1, 8, 8, 4, Q8StreamX::Global, 2>>},
    {"stream_k320_m8c2", stream_route<StreamSched<320, 4, 1, 8, 8, 2, Q8StreamX::Global, 2>>},
    {"stream_k320_m8c1", stream_route<StreamSched<320, 4, 1, 8, 8, 1, Q8StreamX::Global, 2>>},
    {"stream_k320_m16c8", stream_route<StreamSched<320, 4, 1, 8, 16, 8, Q8StreamX::Global, 2>>},
    {"stream_k320_m4c4", stream_route<StreamSched<320, 4, 1, 8, 4, 4, Q8StreamX::Global, 2>>},
    {"stream_k320_m2c2", stream_route<StreamSched<320, 4, 1, 8, 2, 2, Q8StreamX::Global, 2>>},
    {"stream_k320_m1c1", stream_route<StreamSched<320, 4, 1, 8, 1, 1, Q8StreamX::Global, 2>>},
    {"stream_k320_r8m8c8", stream_route<StreamSched<320, 8, 1, 8, 8, 8, Q8StreamX::Global, 2>>},
    {"stream_k320_m8c8s1", stream_route<StreamSched<320, 4, 1, 8, 8, 8, Q8StreamX::Global, 1>>},
    {"stream_k320_shm8c8", stream_route<StreamSched<320, 4, 1, 8, 8, 8, Q8StreamX::Shared, 2>>},
    {"stream_k10240_m8c8", stream_route<StreamSched<10240, 2, 8, 8, 8, 8, Q8StreamX::Global, 2>>},
    {"stream_k10240_m4c4", stream_route<StreamSched<10240, 2, 8, 8, 4, 4, Q8StreamX::Global, 2>>},
    {"stream_k10240_m64c8", stream_route<StreamSched<10240, 2, 8, 8, 64, 8, Q8StreamX::Global, 2>>},
    {"stream_k10240_m8c2", stream_route<StreamSched<10240, 2, 8, 8, 8, 2, Q8StreamX::Global, 2>>},
    {"stream_k10240_m8c1", stream_route<StreamSched<10240, 2, 8, 8, 8, 1, Q8StreamX::Global, 2>>},
    {"stream_k640_m8c8", stream_route<StreamSched<640, 4, 1, 8, 8, 8, Q8StreamX::Global, 2>>},
    {"stream_k2560_m8c8", stream_route<StreamSched<2560, 4, 1, 8, 8, 8, Q8StreamX::Global, 2>>},
    // Full schedule-grid permutations (MMA / sliced-K / grouped), each constraint-checked by the
    // generating script so they cannot violate a schedule static_assert.
#include "q8_permutation_routes.inc"
};

struct Point {
    int tokens;
    double median_us;
    double min_us;
    double p95_us;
    double gbs;
    std::string route;
    bool bit_exact;    // reproduces production's bits at this same width
    bool t_invariant;  // reproduces the per-column results of a T=1 call (the linear contract)
    bool numeric_ok;   // numerically agrees with production within BF16 tolerance
};

bool check_numeric_bf16(const std::vector<std::uint16_t>& got,
                        const std::vector<std::uint16_t>& ref) {
    if (got.size() != ref.size() || got.empty()) return false;
    bool any_nonzero = false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        __nv_bfloat16 gb, rb;
        std::memcpy(&gb, &got[i], 2);
        std::memcpy(&rb, &ref[i], 2);
        const float g = __bfloat162float(gb);
        const float r = __bfloat162float(rb);
        if (!std::isfinite(g) || !std::isfinite(r)) return false;
        if (g != 0.0f) any_nonzero = true;
        const float tol = std::max(0.0625f, 0.03125f * std::fabs(r));
        if (std::fabs(g - r) > tol) return false;
    }
    return any_nonzero;
}

std::vector<std::uint16_t> download(const Tensor& t) {
    std::vector<std::uint16_t> host(t.ne[0] * t.ne[1]);
    CUDA_CHECK(cudaMemcpy(host.data(), t.data, host.size() * 2, cudaMemcpyDeviceToHost));
    return host;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int n = 2560, k = 6144, repeat = 30, warmup = 5, flush_mib = 32;
        std::vector<int> tokens;
        std::string csv_out;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--n" && i + 1 < argc) {
                n = std::stoi(argv[++i]);
            } else if (arg == "--k" && i + 1 < argc) {
                k = std::stoi(argv[++i]);
            } else if (arg == "--tokens" && i + 1 < argc) {
                std::string spec = argv[++i], token;
                for (char c : spec) {
                    if (c == ',') {
                        if (!token.empty()) tokens.push_back(std::stoi(token));
                        token.clear();
                    } else {
                        token += c;
                    }
                }
                if (!token.empty()) tokens.push_back(std::stoi(token));
            } else if (arg == "--repeat" && i + 1 < argc) {
                repeat = std::stoi(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            } else if (arg == "--flush-mib" && i + 1 < argc) {
                flush_mib = std::stoi(argv[++i]);
            } else if (arg == "-o" && i + 1 < argc) {
                csv_out = argv[++i];
            } else if (arg == "-h" || arg == "--help") {
                std::printf("usage: %s [--n N] [--k K] [--tokens 1,2,...] [--repeat N] "
                            "[--warmup N] [--flush-mib N] [-o out.csv]\n", argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        if (tokens.empty()) tokens = {1, 2, 4, 5, 8, 9, 12, 16, 17, 32, 33, 64, 65};

        const int max_t = *std::max_element(tokens.begin(), tokens.end());
        const std::size_t flush_bytes = static_cast<std::size_t>(flush_mib) << 20;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(flush_bytes);

        auto qw = bench::make_row_split_weight(QType::Q8_G32_FP16, n, k,
                                               (k + 127) / 128 * 128, 0x5eed1234ULL);
        const std::size_t out_count = static_cast<std::size_t>(n) * max_t;
        DeviceBuffer x_storage(static_cast<std::size_t>(k) * max_t * 2);
        DeviceBuffer out_storage(out_count * 2);
        DeviceBuffer ref_storage(out_count * 2);
        CUDA_CHECK(bench::fixture::fill_values(
            static_cast<__nv_bfloat16*>(x_storage.p), static_cast<std::size_t>(k) * max_t,
            0xa5a5ULL, -0.5F, 0.5F));
        CUDA_CHECK(cudaDeviceSynchronize());

        const Tensor x = Tensor(x_storage.p, DType::BF16, {k, max_t});
        const Tensor out = Tensor(out_storage.p, DType::BF16, {n, max_t});
        // Reference: the production route at each T, used for the bit-exactness gate.
        const Tensor ref = Tensor(ref_storage.p, DType::BF16, {n, max_t});

        std::printf("shape N=%d K=%d routes=%zu tokens=%zu\n", n, k,
                    sizeof(kRoutes) / sizeof(kRoutes[0]), tokens.size());
        std::printf("%6s %-24s %11s %11s %11s %8s %7s %s\n", "T", "route", "median_us",
                    "min_us", "p95_us", "GB/s", "%peak", "exact");

        std::vector<Point> points;
        const auto cold = [&](const Route& route, int t) {
            Tensor xs = x;
            xs.ne[1] = t;
            Tensor os = out;
            os.ne[1] = t;
            for (int i = 0; i < warmup; ++i) {
                bench::flush_l2(flush, stream);
                route.launch(xs, qw.weight, os, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<double> samples;
            for (int i = 0; i < repeat; ++i) {
                bench::flush_l2(flush, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                cudaEvent_t a, b;
                CUDA_CHECK(cudaEventCreate(&a));
                CUDA_CHECK(cudaEventCreate(&b));
                CUDA_CHECK(cudaEventRecord(a, stream));
                route.launch(xs, qw.weight, os, stream);
                CUDA_CHECK(cudaEventRecord(b, stream));
                CUDA_CHECK(cudaEventSynchronize(b));
                float ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                samples.push_back(ms * 1000.0F);
                CUDA_CHECK(cudaEventDestroy(a));
                CUDA_CHECK(cudaEventDestroy(b));
            }
            std::sort(samples.begin(), samples.end());
            return samples;
        };

        for (int t : tokens) {
            Tensor xs = x;
            xs.ne[1] = t;
            Tensor rs = ref;
            rs.ne[1] = t;
            ops::linear(xs, qw.weight, rs, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto ref_host = download(rs);

            // The Linear contract (tests/ops/linear/test_q8_a16.cpp): a T-column call must reproduce,
            // column for column, what T calls of width 1 produce. That reference is independent of the
            // route under test, so a faster route that changes the bits has to be caught here rather
            // than in CI. The production route is what defines width-1 bits.
            //
            // Each width-1 output is staged into its own aligned buffer before it is copied into the
            // reference: with an odd N the column stride is not a multiple of 16, and the linear entry
            // point requires 16-byte-aligned operands.
            std::vector<std::uint16_t> alone(static_cast<std::size_t>(n) * t);
            DeviceBuffer one_out(static_cast<std::size_t>(n) * 2);
            for (int column = 0; column < t; ++column) {
                Tensor one = Tensor(static_cast<unsigned char*>(x.data) +
                                        static_cast<std::size_t>(column) * k * 2,
                                    DType::BF16, {k, 1});
                Tensor out1 = Tensor(one_out.p, DType::BF16, {n, 1});
                ops::linear(one, qw.weight, out1, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(alone.data() + static_cast<std::size_t>(column) * n,
                                      out1.data, static_cast<std::size_t>(n) * 2,
                                      cudaMemcpyDeviceToHost));
            }

            // Bytes actually read for this call: one pass over the weight plus activation+output.
            const double bytes = static_cast<double>(qw.model_weight_bytes()) +
                                 static_cast<double>(k) * t * 2 + static_cast<double>(n) * t * 2;
            for (const Route& route : kRoutes) {
                Tensor os = out;
                os.ne[1] = t;
                CUDA_CHECK(cudaMemsetAsync(os.data, 0, static_cast<std::size_t>(n) * t * 2, stream));
                route.launch(xs, qw.weight, os, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto got = download(os);
                const bool exact = std::memcmp(got.data(), ref_host.data(), got.size() * 2) == 0;
                const bool invariant = std::memcmp(got.data(), alone.data(), got.size() * 2) == 0;
                const bool num_ok = exact || check_numeric_bf16(got, ref_host);
                if (!num_ok && std::strcmp(route.name, "production") != 0) { continue; }

                const auto samples = cold(route, t);
                const double median = samples[samples.size() / 2];
                const double gbs = bytes / (median * 1.0e-6) / 1.0e9;
                Point p{t, median, samples.front(), samples[(samples.size() * 95) / 100], gbs,
                        route.name, exact, invariant, num_ok};
                points.push_back(p);
                std::printf("%6d %-24s %11.2f %11.2f %11.2f %8.1f %6.1f%% %s %s\n", t, route.name,
                            p.median_us, p.min_us, p.p95_us, p.gbs,
                            100.0 * p.gbs / bench::kSustainedReadGBs, exact ? "yes" : "num",
                            invariant ? "exact" : "T-DEPENDENT");
            }
            const Point* prod = nullptr;
            const Point* best_inv = nullptr;
            const Point* best_num = nullptr;
            for (const Point& p : points) {
                if (p.tokens != t) continue;
                if (p.route == "production") prod = &p;
                if (p.bit_exact && p.t_invariant &&
                    (best_inv == nullptr || p.median_us < best_inv->median_us)) {
                    best_inv = &p;
                }
                if (p.numeric_ok &&
                    (best_num == nullptr || p.median_us < best_num->median_us)) {
                    best_num = &p;
                }
            }
            if (prod != nullptr && best_inv != nullptr && prod->median_us > best_inv->median_us * 1.01) {
                std::printf("  => T=%d: production %.2f us, best exact '%s' %.2f us  (%.1f%% faster)\n",
                            t, prod->median_us, best_inv->route.c_str(), best_inv->median_us,
                            100.0 * (prod->median_us - best_inv->median_us) / prod->median_us);
            }
            if (prod != nullptr && best_num != nullptr && best_num != best_inv &&
                prod->median_us > best_num->median_us * 1.01) {
                std::printf("  => T=%d: production %.2f us, best numeric '%s' %.2f us  (%.1f%% faster, exact=%s)\n",
                            t, prod->median_us, best_num->route.c_str(), best_num->median_us,
                            100.0 * (prod->median_us - best_num->median_us) / prod->median_us,
                            best_num->bit_exact ? "yes" : "no");
            }
            std::printf("\n");
        }

        if (!csv_out.empty()) {
            std::FILE* f = std::fopen(csv_out.c_str(), "w");
            if (f == nullptr) throw std::runtime_error("cannot open " + csv_out);
            std::fprintf(f, "N,K,T,route,median_us,min_us,p95_us,gbs,peak_pct,bit_exact,t_invariant\n");
            for (const Point& p : points) {
                std::fprintf(f, "%d,%d,%d,%s,%.3f,%.3f,%.3f,%.2f,%.3f,%d,%d\n", n, k, p.tokens,
                             p.route.c_str(), p.median_us, p.min_us, p.p95_us, p.gbs,
                             100.0 * p.gbs / bench::kSustainedReadGBs, p.bit_exact ? 1 : 0,
                             p.t_invariant ? 1 : 0);
            }
            std::fclose(f);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q8_route_sweep: %s\n", error.what());
        return 1;
    }
}
