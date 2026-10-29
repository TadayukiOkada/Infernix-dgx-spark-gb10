// Candidate + production benchmark for the two registered public Q8 LinearPair geometries
// (k=2048 dflash2 K/V row views, k=5120 companion). Emits the shared bench CSV schema so the
// awk before/after + winner scripts used across the GB10 retune work unchanged.

#include "core/weight.h"
#include "infernix/ops/linear_pair.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/linear_pair/q8/q8_pair_kernels.h"
#include "ops/linear_pair/q8/q8_pair_plan.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;

namespace {

infernix::bench::ColdFlushBytes kFlushBytes{};
constexpr std::int32_t kParentRows = 6144;
constexpr std::int32_t kFirstRow   = 4096;
constexpr std::int32_t kSecondRow  = 5120;
constexpr std::int32_t kRows       = 1024;

struct Options {
    std::vector<std::int32_t> t_sweep;
    std::int32_t k            = 2048;
    int warmup                = 5;
    int repeat                = 30;
    std::string csv_out;
    bool graph               = false;
    bool candidates          = false;
    bool production_only     = false;
};

struct Result {
    std::string path;
    std::int32_t t;
    bench::ColdTiming timing;
    std::size_t workspace_bytes;
    std::size_t graph_nodes;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--k") {
            options.k = std::stoi(std::string(next("--k value")));
            if (options.k != 2048 && options.k != 5120) {
                throw std::invalid_argument("--k must be 2048 or 5120");
            }
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--execution") {
            const auto mode = next("--execution value");
            if (mode != "eager" && mode != "graph")
                throw std::invalid_argument("execution must be eager or graph");
            options.graph = mode == "graph";
        } else if (arg == "--candidates") {
            options.candidates = true;
        } else if (arg == "--production-only") {
            options.production_only = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--k 2048|5120] [--t-sweep 1,2,...] [--warmup N] [--repeat N] "
                        "[--csv-out PATH] [--execution eager|graph] [--candidates] "
                        "[--production-only]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.t_sweep.empty()) {
        if (options.k == 5120) {
            options.t_sweep = {1, 8, 16, 32, 48, 64, 80, 96, 128, 256, 512, 768, 960, 1024, 2048};
        } else {
            options.t_sweep = {1,  2,  8,  16,  24,  32,  33,  48,  64,  80,  88,  96,  104,
                               112, 128, 160, 192, 193, 224, 256, 384, 480, 481, 640, 641, 672,
                               673, 680, 681, 784, 785, 896, 897, 960, 961, 976, 977, 1280, 1281,
                               1316, 1317, 1344, 1345, 1346, 1440, 1441, 1466, 1467, 1680, 1681,
                               1708, 1709, 1920, 1921, 1922, 1923, 2016, 2017, 2018, 2019, 2208,
                               2209, 2270, 2271, 3072, 4096};
        }
    }
    return options;
}

void append(std::vector<Result>& results, const Options& options, const char* path, int t,
            bench::ColdTiming timing, std::size_t workspace_bytes, std::size_t graph_nodes) {
    std::printf("entry=linear_pair k=%d path=%s T=%d execution=%s graph_nodes=%zu "
                "workspace=%zu median=%.3f us min=%.3f us p95=%.3f us\n",
                options.k, path, t, options.graph ? "graph" : "eager", graph_nodes, workspace_bytes,
                timing.median_us, timing.min_us, timing.p95_us);
    results.push_back({path, t, timing, workspace_bytes, graph_nodes});
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) return;
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open timing CSV");
    out << "problem,entry,path,T,execution,workspace_bytes,graph_nodes,median_us,min_us,p95_us,"
           "warmup,repeat,flush_bytes\n";
    for (const auto& r : results) {
        out << "k" << options.k << ",linear_pair," << r.path << ',' << r.t << ','
            << (options.graph ? "graph" : "eager") << ',' << r.workspace_bytes << ',';
        if (r.graph_nodes) out << r.graph_nodes;
        out << ',' << r.timing.median_us << ',' << r.timing.min_us << ',' << r.timing.p95_us << ','
            << options.warmup << ',' << options.repeat << ',' << kFlushBytes << '\n';
    }
}

using SD = ops::detail::Q8PairScheduleId;

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        const cudaStream_t stream = context.stream;
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d k=%d cache=cold\n", context.props.name,
                    context.props.major, context.props.minor, CUDART_VERSION, options.k);
        const auto [minimum, maximum] =
            std::minmax_element(options.t_sweep.begin(), options.t_sweep.end());
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(options.k) * (*maximum), 101U);
        DeviceBuffer first_output(static_cast<std::size_t>(kRows) * (*maximum) * 2);
        DeviceBuffer second_output(static_cast<std::size_t>(kRows) * (*maximum) * 2);
        bench::PackedQuantizedWeight parent = bench::make_row_split_weight(
            QType::Q8_G32_FP16, kParentRows, options.k, options.k, 501U);
        const Weight first_weight  = bench::row_view(parent.weight, kFirstRow, kRows);
        const Weight second_weight = bench::row_view(parent.weight, kSecondRow, kRows);

        std::vector<Result> results;
        for (const auto t : options.t_sweep) {
            Tensor x(input.p, DType::BF16, {options.k, t});
            Tensor first(first_output.p, DType::BF16, {kRows, t});
            Tensor second(second_output.p, DType::BF16, {kRows, t});
            const auto body = [&](cudaStream_t launch_stream) {
                ops::linear_pair(x, first_weight, second_weight, first, second, launch_stream);
            };
            bench::TimedGraph graph;
            if (options.graph) graph.capture(stream, body);
            const auto prod_timing =
                options.graph ? bench::measure_cold_graph(graph, flush, stream, options.warmup,
                                                          options.repeat)
                              : bench::measure_cold_launch(body, flush, stream, options.warmup,
                                                           options.repeat);
            append(results, options, "production", t, prod_timing, 0, graph.nodes());
            if (!options.candidates || options.production_only) { continue; }

            const auto run = [&](const char* path, auto&& launch) {
                const auto timing =
                    bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
                append(results, options, path, t, timing, 0, 0);
            };

            namespace K = ops::detail;
            if (options.k == 2048) {
                if (t == 1) {
                    run("decode_r4", [&](cudaStream_t cs) {
                        K::q8_pair_decode_r4_launch(x, first_weight, second_weight, first, second,
                                                     cs);
                    });
                    run("decode_r8", [&](cudaStream_t cs) {
                        K::q8_pair_decode_r8_launch(x, first_weight, second_weight, first, second,
                                                     cs);
                    });
                    run("decode_r16", [&](cudaStream_t cs) {
                        K::q8_pair_decode_r16_launch(x, first_weight, second_weight, first, second,
                                                      cs);
                    });
                }
                if (t >= 2 && t <= 32) {
                    run("splitk_exact_t", [&](cudaStream_t cs) {
                        K::q8_pair_splitk_exact_t_launch(x, first_weight, second_weight, first,
                                                         second, cs);
                    });
                }
#define RUN_MEDIUM(NAME, HI)                                                                       \
    if (t >= 33 && t <= (HI)) {                                                                    \
        run(#NAME, [&](cudaStream_t cs) {                                                          \
            K::q8_pair_splitk_medium_launch(SD::NAME, x, first_weight, second_weight, first,      \
                                            second, cs);                                           \
        });                                                                                        \
    }
                RUN_MEDIUM(DualSplitKMediumC48, 48)
                RUN_MEDIUM(DualSplitKMediumC64, 64)
                RUN_MEDIUM(DualSplitKMediumC80, 80)
                RUN_MEDIUM(DualSplitKMediumC88, 88)
                RUN_MEDIUM(DualSplitKMediumC96, 96)
                RUN_MEDIUM(DualSplitKMediumC104, 104)
                RUN_MEDIUM(DualSplitKMediumC112, 112)
                RUN_MEDIUM(DualSplitKMediumC128, 128)
                RUN_MEDIUM(DualSplitKMediumC160, 160)
                RUN_MEDIUM(DualSplitKMediumC192, 192)
                RUN_MEDIUM(DualSplitKMediumC224, 224)
                RUN_MEDIUM(DualSplitKMediumC256, 256)
#undef RUN_MEDIUM
#define RUN_CONCAT(NAME)                                                                           \
    run(#NAME, [&](cudaStream_t cs) {                                                              \
        K::q8_pair_concat_mma_launch(SD::NAME, x, first_weight, second_weight, first, second,     \
                                     cs);                                                          \
    })
                RUN_CONCAT(ConcatMmaR32C64);
                RUN_CONCAT(ConcatMmaR32C80);
                RUN_CONCAT(ConcatMmaR32C96);
                RUN_CONCAT(ConcatMmaR32C112);
                RUN_CONCAT(ConcatMmaR32C128);
                RUN_CONCAT(ConcatMmaR48C64);
                RUN_CONCAT(ConcatMmaR48C96);
                RUN_CONCAT(ConcatMmaR48C112);
                RUN_CONCAT(ConcatMmaR48C128);
                RUN_CONCAT(ConcatMmaR64C64);
                RUN_CONCAT(ConcatMmaR64C80);
                RUN_CONCAT(ConcatMmaR64C96);
                RUN_CONCAT(ConcatMmaR64C128);
                RUN_CONCAT(ConcatMmaR96C64);
                RUN_CONCAT(ConcatMmaR96C80);
                RUN_CONCAT(ConcatMmaR96C96);
                RUN_CONCAT(ConcatMmaR96C112);
                RUN_CONCAT(ConcatMmaR128C64);
                RUN_CONCAT(ConcatMmaR128C80);
#undef RUN_CONCAT
            } else {
                // k=5120: only the SIMT and dual-MMA families are shape-valid.
                const bool c64_full  = (t % 64) == 0;
                const bool c128_full = (t % 128) == 0;
                run("simt_r8_c4", [&](cudaStream_t cs) {
                    K::q8_pair_simt_r8_c4_launch(x, first_weight, second_weight, first, second, cs);
                });
                run("dual_mma_r32_c64", [&](cudaStream_t cs) {
                    K::q8_pair_gemm_mma_r32_c64_launch(c64_full, x, first_weight, second_weight,
                                                       first, second, cs);
                });
                run("dual_mma_r32_c128", [&](cudaStream_t cs) {
                    K::q8_pair_gemm_mma_r32_c128_launch(c128_full, x, first_weight, second_weight,
                                                        first, second, cs);
                });
            }
        }
        write_csv(options, results);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_linear_pair_bench: %s\n", error.what());
        return 1;
    }
}
