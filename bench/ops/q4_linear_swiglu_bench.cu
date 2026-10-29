// Cold-cache public Op benchmark + candidate harness for the registered Q4 LinearSwiGLU profile.
//
// The default rows measure the complete production-dispatched `ops::linear_swiglu` call. With
// `--candidates` the benchmark also times each internal Q4 kernel variant and the
// `linear`+`silu_mul` materialization, so a schedule decision can be made against the production
// row measured in the same process. Candidate rows are cold/eager launches (never graphs) and
// report no workspace, matching the note in the GB10 tuning roadmap.

#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/silu_mul.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;

namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
infernix::bench::ColdFlushBytes kFlushBytes{};

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 24, 32, 48};
    int warmup           = 5;
    int repeat           = 30;
    bool profile         = false;
    bool graph           = false;
    bool candidates      = false;
    bool production_only = false;
    std::string csv_out;
    ops::LinearRoutes routes = ops::LinearRoutes::Exact;
};

struct Result {
    std::int32_t tokens               = 0;
    const char* execution             = "eager";
    std::size_t graph_nodes           = 0;
    double median_us                  = 0.0;
    double min_us                     = 0.0;
    double p95_us                     = 0.0;
    double weight_bytes               = 0.0;
    double logical_bytes              = 0.0;
    double projection_flops           = 0.0;
    double logical_gbs                = 0.0;
    double useful_tflops              = 0.0;
    std::size_t workspace_sweep_bytes = 0;
    std::size_t workspace_exact_bytes = 0;
    // "production" for the public call, the internal kernel name for a candidate row.
    const char* path = "production";
};

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
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
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--t-sweep") {
            options.tokens = parse_tokens(next("--t-sweep value"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (argument == "--execution") {
            const std::string_view mode = next("--execution value");
            if (mode != "eager" && mode != "graph") {
                throw std::invalid_argument("--execution must be eager or graph");
            }
            options.graph = mode == "graph";
        } else if (argument == "--csv-out") {
            options.csv_out = std::string(next("--csv-out value"));
        } else if (argument == "--candidates") {
            options.candidates = true;
        } else if (argument == "--production-only") {
            options.production_only = true;
        } else if (argument == "--routes") {
            const std::string_view routes = next("--routes value");
            if (routes != "exact" && routes != "fast") {
                throw std::invalid_argument("--routes must be exact or fast");
            }
            options.routes = routes == "fast" ? ops::LinearRoutes::Q4W4A8 : ops::LinearRoutes::Exact;
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s [--t-sweep 1,2,...] [--warmup N] [--repeat N] [--execution "
                "eager|graph]\n"
                "          [--routes exact|fast] [--csv-out PATH] [--profile] "
                "[--candidates] [--production-only]\n"
                "  --execution graph captures ONE complete public ops::linear_swiglu call per\n"
                "  graph; the reported graph_nodes is the captured count, not an assumption.\n"
                "  --routes fast passes LinearRoutes::Q4W4A8 (the q4-w4a8 fast path, which serves\n"
                "  T >= 256, int8 activation quantization included); exact (default) passes none.\n"
                "  --profile runs a single eager or single graph launch for external profilers.\n"
                "  --candidates also times every internal Q4 kernel variant eagerly per token;\n"
                "  those rows are cold/eager and report workspace_exact_bytes=0.\n"
                "  --production-only keeps the public call only (no candidate rows).\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.tokens.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.profile && !options.csv_out.empty()) {
        throw std::invalid_argument("--csv-out is incompatible with --profile");
    }
    return options;
}

// Fills the derived per-row metrics once a timing is known.
void fill(Result& result, const bench::ColdTiming& timing, std::size_t graph_nodes, bool graph,
          std::size_t workspace_sweep_bytes, std::size_t workspace_exact_bytes,
          double weight_bytes) {
    const double seconds = timing.median_us * 1.0e-6;
    const double flops   = 2.0 * static_cast<double>(kGateUpRows) * kHidden * result.tokens;
    const double bytes =
        weight_bytes + 2.0 * static_cast<double>(kHidden + kOutputRows) * result.tokens;
    result.execution             = graph ? "graph" : "eager";
    result.graph_nodes           = graph_nodes;
    result.median_us             = timing.median_us;
    result.min_us                = timing.min_us;
    result.p95_us                = timing.p95_us;
    result.weight_bytes          = weight_bytes;
    result.logical_bytes         = bytes;
    result.projection_flops      = flops;
    result.logical_gbs           = bytes / seconds / 1.0e9;
    result.useful_tflops         = flops / seconds / 1.0e12;
    result.workspace_sweep_bytes = workspace_sweep_bytes;
    result.workspace_exact_bytes = workspace_exact_bytes;
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { throw std::runtime_error("cannot open csv output: " + path); }
    std::fprintf(file, "t,execution,graph_nodes,median_us,min_us,p95_us,weight_bytes,logical_bytes,"
                       "projection_flops,effective_gbs,dram_spec_pct,useful_tflops,"
                       "workspace_sweep_bytes,workspace_exact_bytes,path\n");
    for (const Result& result : results) {
        const double seconds = result.median_us * 1.0e-6;
        std::fprintf(file, "%d,%s,%zu,%.3f,%.3f,%.3f,%.0f,%.0f,%.0f,%.4f,%.4f,%.4f,%zu,%zu,%s\n",
                     result.tokens, result.execution, result.graph_nodes, result.median_us,
                     result.min_us, result.p95_us, result.weight_bytes, result.logical_bytes,
                     result.projection_flops, result.logical_bytes / seconds / 1.0e9,
                     result.logical_bytes / seconds / 1.0e9 / 1792.0 * 100.0,
                     result.projection_flops / seconds / 1.0e12, result.workspace_sweep_bytes,
                     result.workspace_exact_bytes, result.path);
    }
    std::fclose(file);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        const cudaStream_t stream = context.stream;
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d cache=cold\n", context.props.name,
                    context.props.major, context.props.minor, CUDART_VERSION);
        const auto [min_t, max_t] =
            std::minmax_element(options.tokens.begin(), options.tokens.end());
        const std::int32_t min_tokens = *min_t;
        const std::int32_t max_tokens = *max_t;
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 101U);
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_tokens * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed =
            bench::make_row_split_weight(QType::Q4_G64_FP16, kGateUpRows, kHidden, kHidden, 501U);
        const std::size_t workspace_capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::Q4_G64_FP16, kGateUpRows, kHidden, ops::LinearPolicy::A16Only, min_tokens,
            max_tokens, options.routes);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));
        // Dedicated gate/up scratch so the `materialized` candidate mirrors production's
        // linear+silu_mul body without churning the arena on every repeat. Only allocated when a
        // candidate row can run.
        const bool run_candidates =
            options.candidates && !options.production_only && !options.profile;
        DeviceBuffer gate_up_scratch(
            run_candidates
                ? static_cast<std::size_t>(kGateUpRows) * max_tokens * sizeof(std::uint16_t)
                : 0);

        const auto launch = [&](std::int32_t tokens, cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
            ops::linear_swiglu(x, packed.weight, out, ops::LinearPolicy::A16Only, workspace,
                               launch_stream, options.routes);
        };
        const char* routes_name = options.routes == ops::LinearRoutes::Exact ? "exact" : "fast";

        if (options.profile) {
            bench::TimedGraph graph;
            if (options.graph) {
                graph.capture(stream, [&](cudaStream_t capture_stream) {
                    launch(options.tokens.front(), capture_stream);
                });
                graph.launch(stream);
            } else {
                launch(options.tokens.front(), stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu Q4 T=%d execution=%s routes=%s graph_nodes=%zu "
                        "workspace_sweep=%zu\n",
                        options.tokens.front(), options.graph ? "graph" : "eager", routes_name,
                        graph.nodes(), workspace_capacity);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::printf("# execution=%s routes=%s warmup=%d repeat=%d flush_bytes=%zu "
                    "workspace_sweep_bytes=%zu candidates=%s\n",
                    options.graph ? "graph" : "eager", routes_name, options.warmup, options.repeat,
                    static_cast<std::size_t>(kFlushBytes), workspace_capacity,
                    run_candidates ? "yes" : "no");
        std::vector<Result> results;
        results.reserve(options.tokens.size() * (run_candidates ? 6U : 1U));
        for (const std::int32_t tokens : options.tokens) {
            bench::TimedGraph graph;
            if (options.graph) {
                graph.capture(stream,
                              [&](cudaStream_t capture_stream) { launch(tokens, capture_stream); });
            }
            const bench::ColdTiming timing =
                options.graph
                    ? bench::measure_cold_graph(graph, flush, stream, options.warmup,
                                                options.repeat)
                    : bench::measure_cold_launch(
                          [&](cudaStream_t launch_stream) { launch(tokens, launch_stream); }, flush,
                          stream, options.warmup, options.repeat);
            const std::size_t workspace_exact = ops::linear_swiglu_workspace_capacity_bytes(
                QType::Q4_G64_FP16, kGateUpRows, kHidden, ops::LinearPolicy::A16Only, tokens,
                tokens, options.routes);
            Result production;
            production.tokens = tokens;
            fill(production, timing, graph.nodes(), options.graph, workspace_capacity,
                 workspace_exact, static_cast<double>(packed.model_weight_bytes()));
            std::printf("T=%-4d path=production median=%9.3f us min=%9.3f p95=%9.3f %7.1f GB/s "
                        "%7.2f TFLOP/s workspace_sweep=%zu workspace_exact=%zu nodes=%zu\n",
                        tokens, timing.median_us, timing.min_us, timing.p95_us,
                        production.logical_gbs, production.useful_tflops, workspace_capacity,
                        workspace_exact, graph.nodes());
            results.push_back(production);
            if (!run_candidates) { continue; }

            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
            const auto run = [&](const char* path, auto&& body) {
                const bench::ColdTiming candidate =
                    bench::measure_cold_launch(body, flush, stream, options.warmup, options.repeat);
                Result row;
                row.tokens = tokens;
                row.path   = path;
                fill(row, candidate, 0, false, workspace_capacity, 0,
                     static_cast<double>(packed.model_weight_bytes()));
                std::printf("T=%-4d path=%-20s median=%9.3f us min=%9.3f p95=%9.3f %7.1f GB/s "
                            "%7.2f TFLOP/s\n",
                            tokens, path, candidate.median_us, candidate.min_us, candidate.p95_us,
                            row.logical_gbs, row.useful_tflops);
                results.push_back(row);
            };

            if (tokens == 1) {
                run("gemv_pair", [&](cudaStream_t cs) {
                    ops::detail::q4_linear_swiglu_gemv_pair_launch(x, packed.weight, out, cs);
                });
            }
            if (tokens >= 2 && tokens <= 32) {
                run("small_t_tiled", [&](cudaStream_t cs) {
                    ops::detail::q4_linear_swiglu_small_t_tiled_launch(x, packed.weight, out, cs);
                });
            }
            run("mma_r32_c128", [&](cudaStream_t cs) {
                ops::detail::q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, packed.weight,
                                                                                 out, cs);
            });
            run("mma_r32_c128_tail", [&](cudaStream_t cs) {
                ops::detail::q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(
                    x, packed.weight, out, cs);
            });
            run("materialized", [&](cudaStream_t cs) {
                Tensor gate_up(gate_up_scratch.p, DType::BF16, {kGateUpRows, tokens});
                ops::linear(x, packed.weight, gate_up, cs);
                ops::silu_mul(gate_up.slice(0, 0, kOutputRows),
                              gate_up.slice(0, kOutputRows, kOutputRows), out, cs);
            });
        }
        if (!options.csv_out.empty()) { write_csv(options.csv_out, results); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_q4_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
