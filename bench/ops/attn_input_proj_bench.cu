// Public-Op benchmark for every registered Attention input-projection contract.
// Production dispatch is owned exclusively by attn_input_proj().

#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/linear.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "core/device.h"
#include "core/temporary_switches.h"
#include "direct_bf16_weight.cuh"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace infernix;

namespace {

infernix::bench::ColdFlushBytes kFlushBytes{};


enum class Format : std::uint8_t { Q4Q5, Q8Qgkv, Q8Qkv, Q8DFlash2Qkv, Bf16, Nvfp4, Fp8, All };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };
enum class Fp8Problem : std::uint8_t { TwentySevenB, OneTwentyTwoB };

struct Options {
    Format format                  = Format::All;
    ops::LinearPolicy nvfp4_policy = ops::LinearPolicy::AllowA4;
    ops::LinearPolicy fp8_policy   = ops::LinearPolicy::AllowA8;
    ops::LinearPolicy q4q5_policy  = ops::LinearPolicy::A16Only;
    Fp8Problem fp8_problem         = Fp8Problem::TwentySevenB;
    CacheMode cache                = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile     = false;
    bool graph       = false;
    bool candidates  = false;
    std::string cand_filter;  // --candidates: keep only schedules with this exact name
    std::string csv_out;
};

struct Result {
    const char* format;
    const char* policy;
    std::int32_t tokens;
    CacheState cache;
    std::size_t workspace_bytes;
    std::uint64_t logical_bytes;
    double useful_flops;
    bench::ColdTiming timing;
    std::size_t graph_nodes              = 0;
    std::size_t workspace_capacity_bytes = 0;
    std::size_t workspace_peak_bytes     = 0;
};

struct Measurement {
    bench::ColdTiming timing;
    std::size_t graph_nodes = 0;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: infernix_attn_input_proj_bench "
                 "[--format q4q5|q8-qgkv|q8-qkv|q8-dflash2-qkv|bf16|nvfp4|fp8|all] "
                 "[--nvfp4-policy a16|a4|invariant] [--fp8-policy a16|a8|a8r|invariant] "
                 "[--fp8-problem 27b|122b] "
                 "[--q4q5-policy a16|invariant] "
                 "[--tokens T,...] [--cache cold|warm|both] [--execution eager|graph] "
                 "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, 1, std::numeric_limits<std::int32_t>::max(), flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--format") {
            const std::string_view value(next("--format requires a value"));
            if (value == "q4q5")
                options.format = Format::Q4Q5;
            else if (value == "q8-qgkv")
                options.format = Format::Q8Qgkv;
            else if (value == "q8-qkv")
                options.format = Format::Q8Qkv;
            else if (value == "q8-dflash2-qkv")
                options.format = Format::Q8DFlash2Qkv;
            else if (value == "bf16")
                options.format = Format::Bf16;
            else if (value == "nvfp4")
                options.format = Format::Nvfp4;
            else if (value == "fp8")
                options.format = Format::Fp8;
            else if (value == "all")
                options.format = Format::All;
            else
                usage("--format expects q4q5, q8-qgkv, q8-qkv, q8-dflash2-qkv, bf16, nvfp4, "
                      "fp8, or all");
        } else if (argument == "--nvfp4-policy") {
            const std::string_view value(next("--nvfp4-policy requires a value"));
            if (value == "a16")
                options.nvfp4_policy = ops::LinearPolicy::A16Only;
            else if (value == "a4")
                options.nvfp4_policy = ops::LinearPolicy::AllowA4;
            else if (value == "invariant")
                options.nvfp4_policy = ops::LinearPolicy::A16ColumnInvariant;
            else
                usage("--nvfp4-policy expects a16, a4 or invariant");
        } else if (argument == "--q4q5-policy") {
            const std::string_view value(next("--q4q5-policy requires a value"));
            if (value == "a16")
                options.q4q5_policy = ops::LinearPolicy::A16Only;
            else if (value == "invariant")
                options.q4q5_policy = ops::LinearPolicy::A16ColumnInvariant;
            else
                usage("--q4q5-policy expects a16 or invariant");
        } else if (argument == "--fp8-policy") {
            const std::string_view value(next("--fp8-policy requires a value"));
            if (value == "a16")
                options.fp8_policy = ops::LinearPolicy::A16Only;
            else if (value == "a8r")
                options.fp8_policy = ops::LinearPolicy::AllowA8Residual;
            else if (value == "a8")
                options.fp8_policy = ops::LinearPolicy::AllowA8;
            else if (value == "invariant")
                options.fp8_policy = ops::LinearPolicy::A16ColumnInvariant;
            else
                usage("--fp8-policy expects a16, a8, a8r or invariant");
        } else if (argument == "--fp8-problem") {
            const std::string_view value(next("--fp8-problem requires a value"));
            if (value == "27b")
                options.fp8_problem = Fp8Problem::TwentySevenB;
            else if (value == "122b")
                options.fp8_problem = Fp8Problem::OneTwentyTwoB;
            else
                usage("--fp8-problem expects 27b or 122b");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), "--tokens");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value != "eager" && value != "graph") usage("--execution expects eager or graph");
            options.graph = value == "graph";
        } else if (argument == "--candidates") {
            options.candidates = true;
        } else if (argument == "--cand-filter") {
            options.cand_filter = std::string(next("--cand-filter requires a value"));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.profile && (options.format == Format::All || options.tokens.size() != 1 ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one format, one T, and one cache state");
    }
    return options;
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "a16";
    case ops::LinearPolicy::AllowA8:
        return "a8";
    case ops::LinearPolicy::AllowA4:
        return "a4";
    case ops::LinearPolicy::AllowA4Residual:
        return "a4r";
    case ops::LinearPolicy::AllowA8Residual:
        return "a8r";
    case ops::LinearPolicy::A16ColumnInvariant:
        return "invariant";
    }
    throw std::invalid_argument("unknown linear policy");
}

template <class Launch>
Measurement measure_public(Launch&& launch, CacheState cache, bench::L2FlushBuffer& flush,
                           cudaStream_t stream, int warmup, int repeat, bool graph) {
    if (graph) {
        bench::TimedGraph captured;
        captured.capture(stream, launch);
        return {cache == CacheState::Cold
                    ? bench::measure_cold_graph(captured, flush, stream, warmup, repeat)
                    : bench::measure_graph(captured, stream, warmup, repeat),
                captured.nodes()};
    }
    return {cache == CacheState::Cold
                ? bench::measure_cold_launch(std::forward<Launch>(launch), flush, stream, warmup,
                                             repeat)
                : bench::measure_launch(std::forward<Launch>(launch), stream, warmup, repeat),
            0};
}

template <class Launch>
void profile_public(Launch&& launch, const char* format, const char* policy, CacheState cache,
                    bench::L2FlushBuffer& flush, cudaStream_t stream, int warmup, bool graph) {
    bench::TimedGraph captured;
    if (graph) captured.capture(stream, launch);
    const auto invoke = [&] {
        if (graph)
            captured.launch(stream);
        else
            launch(stream);
    };
    for (int index = 0; index < warmup; ++index) { invoke(); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=attn_input_proj format=%s policy=%s dispatch=public execution=%s "
                "graph_nodes=%zu cache=%s\n",
                format, policy, graph ? "graph" : "eager", captured.nodes(), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    invoke();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = static_cast<double>(result.logical_bytes) / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=attn_input_proj format=%-8s policy=%-3s cache=%-4s execution=%-5s "
                "graph_nodes=%zu T=%4d "
                "workspace=%9zu capacity=%9zu peak=%9zu median=%9.3f us min=%9.3f us p95=%9.3f us "
                "logical=%8.1f GB/s math=%8.2f TFLOP/s\n",
                result.format, result.policy, cache_name(result.cache),
                result.graph_nodes ? "graph" : "eager", result.graph_nodes, result.tokens,
                result.workspace_bytes, result.workspace_capacity_bytes,
                result.workspace_peak_bytes, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, tflops);
}

void append_result(std::vector<Result>& results, const char* format, const char* policy,
                   std::int32_t tokens, CacheState cache, std::size_t workspace_bytes,
                   std::uint64_t logical_bytes, double useful_flops, Measurement measurement,
                   std::size_t capacity = 0, std::size_t peak = 0) {
    Result result{format,
                  policy,
                  tokens,
                  cache,
                  workspace_bytes,
                  logical_bytes,
                  useful_flops,
                  measurement.timing,
                  measurement.graph_nodes,
                  capacity,
                  peak};
    report(result);
    results.push_back(result);
}

std::uint64_t tensor_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(tokens) * 2ULL;
}

void run_q4q5(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
              std::vector<Result>& results) {
    constexpr std::int32_t hidden      = 5120;
    constexpr std::int32_t q_rows      = 6144;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = q_rows + kv_rows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk =
        bench::make_row_split_weight(QType::Q4_G64_FP16, parent_rows, hidden, hidden, 501U);
    bench::PackedQuantizedWeight gv =
        bench::make_row_split_weight(QType::Q5_G64_FP16, parent_rows, hidden, hidden, 503U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 199U);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tg(gate.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto launch = [&](cudaStream_t launch_stream) {
            ops::attn_input_proj(x, qk.weight, gv.weight, tq, tg, tk, tv, options.q4q5_policy,
                                 launch_stream);
        };
        const char* policy = policy_name(options.q4q5_policy);
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, "q4q5", policy, profile_cache, flush, stream, options.warmup,
                           options.graph);
            continue;
        }
        const std::uint64_t logical = qk.model_weight_bytes() + gv.model_weight_bytes() +
                                      tensor_bytes(hidden, tokens) +
                                      tensor_bytes(2 * q_rows + 2 * kv_rows, tokens);
        const double flops = 4.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            append_result(results, "q4q5", policy, tokens, cache, 0, logical, flops,
                          measure_public(launch, cache, flush, stream, options.warmup,
                                         options.repeat, options.graph));
        }
    }
}

template <class WeightFixture>
void run_four_output(const Options& options, const char* format, QType qtype,
                     ops::LinearPolicy policy, bool implicit_a16_entry, std::int32_t hidden,
                     std::int32_t q_rows, std::int32_t kv_rows, std::int32_t parent_rows,
                     WeightFixture& fixture, bench::L2FlushBuffer& flush, cudaStream_t stream,
                     std::vector<Result>& results) {
    const std::int32_t min_tokens = *std::min_element(options.tokens.begin(), options.tokens.end());
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    const std::size_t workspace_bytes = ops::attn_input_proj_workspace_capacity_bytes(
        qtype, parent_rows, hidden, policy, min_tokens, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 101U);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tg(gate.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto exact_workspace = ops::attn_input_proj_workspace_capacity_bytes(
            qtype, parent_rows, hidden, policy, tokens, tokens);
        const auto launch = [&](cudaStream_t launch_stream) {
            if (implicit_a16_entry) {
                ops::attn_input_proj(x, fixture.weight, tq, tg, tk, tv, launch_stream);
            } else {
                ops::attn_input_proj(x, fixture.weight, tq, tg, tk, tv, policy, workspace,
                                     launch_stream);
            }
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, format, policy_name(policy), profile_cache, flush, stream,
                           options.warmup, options.graph);
            continue;
        }
        const std::uint64_t logical = fixture.model_weight_bytes() + tensor_bytes(hidden, tokens) +
                                      tensor_bytes(2 * q_rows + 2 * kv_rows, tokens);
        const double flops = 2.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            workspace.reset_peak();
            const auto measurement = measure_public(launch, cache, flush, stream, options.warmup,
                                                    options.repeat, options.graph);
            if (workspace.used() != 0 || workspace.peak_used() > exact_workspace)
                throw std::runtime_error("public projection exceeds its workspace query");
            append_result(results, format, policy_name(policy), tokens, cache, exact_workspace,
                          logical, flops, measurement, workspace_bytes, workspace.peak_used());
        }
    }
}

void run_q8_qkv(const Options& options, const char* label, std::int32_t hidden,
                bench::L2FlushBuffer& flush, cudaStream_t stream, std::vector<Result>& results) {
    constexpr std::int32_t q_rows      = 4096;
    constexpr std::int32_t kv_rows     = 1024;
    constexpr std::int32_t parent_rows = 6144;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight weight =
        bench::make_row_split_weight(QType::Q8_G32_FP16, parent_rows, hidden, hidden, 505U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 103U);
    DeviceBuffer q(static_cast<std::size_t>(q_rows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kv_rows) * max_tokens * 2);
    for (const std::int32_t tokens : options.tokens) {
        Tensor x(input.p, DType::BF16, {hidden, tokens});
        Tensor tq(q.p, DType::BF16, {q_rows, tokens});
        Tensor tk(k.p, DType::BF16, {kv_rows, tokens});
        Tensor tv(v.p, DType::BF16, {kv_rows, tokens});
        const auto launch = [&](cudaStream_t launch_stream) {
            ops::attn_input_proj(x, weight.weight, tq, tk, tv, launch_stream);
        };
        const CacheState profile_cache =
            options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
        if (options.profile) {
            profile_public(launch, label, "a16", profile_cache, flush, stream, options.warmup,
                           options.graph);
            continue;
        }
        const std::uint64_t logical = weight.model_weight_bytes() + tensor_bytes(hidden, tokens) +
                                      tensor_bytes(q_rows + 2 * kv_rows, tokens);
        const double flops = 2.0 * parent_rows * hidden * static_cast<double>(tokens);
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm))
                continue;
            append_result(results, label, "a16", tokens, cache, 0, logical, flops,
                          measure_public(launch, cache, flush, stream, options.warmup,
                                         options.repeat, options.graph));
        }
    }
}

void run_nvfp4_candidates(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
                          std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kOutputRows = 2 * kQueryRows + 2 * kKvRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_nvfp4_weight(kOutputRows, kHidden);
    const std::size_t maximum_workspace =
        ops::detail::nvfp4_a4_workspace_capacity_bytes(max_tokens, kHidden);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 105U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Nvfp4AttnInputA4ScheduleId, 7> kSchedules{{
        ops::detail::Nvfp4AttnInputA4ScheduleId::M32N64,
        ops::detail::Nvfp4AttnInputA4ScheduleId::M32N128,
        ops::detail::Nvfp4AttnInputA4ScheduleId::M64N128,
        ops::detail::Nvfp4AttnInputA4ScheduleId::M128N128Pipelined,
        ops::detail::Nvfp4AttnInputA4ScheduleId::M128N128Resident,
        ops::detail::Nvfp4AttnInputA4ScheduleId::Tma128,
        ops::detail::Nvfp4AttnInputA4ScheduleId::Tma256,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Nvfp4AttnInputA4Problem problem{kHidden, kQueryRows, kKvRows, kOutputRows,
                                                           kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::nvfp4_attn_input_a4_candidate_is_legal(schedule, problem)) {
                continue;
            }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::nvfp4_attn_input_a4_execute_candidate(schedule, x, parent.weight, tq,
                                                                    tg, tk, tv, workspace,
                                                                    launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(kOutputRows, tokens);
            const double flops = 2.0 * static_cast<double>(kOutputRows) * kHidden * tokens;
            append_result(results, "nvfp4",
                          ops::detail::nvfp4_attn_input_a4_schedule_name(schedule), tokens,
                          CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

void run_fp8(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
             std::vector<Result>& results) {
    if (options.fp8_problem == Fp8Problem::OneTwentyTwoB) {
        // Qwen3.5-122B-A10B (Ornith-1.5-122B-NVFP4): fp8_e4m3fn_row_f32 parent [17408,3072],
        // query 8192, key 512, gate 8192, value 512.
        auto weight = bench::make_fp8_weight(17408, 3072, 0x51U, 0.05F, true);
        run_four_output(options, "fp8-f32", QType::FP8_E4M3FN_ROW_F32, options.fp8_policy, false,
                        3072, 8192, 512, 17408, weight, flush, stream, results);
        return;
    }
    auto weight = bench::make_fp8_weight(14336, 5120);
    run_four_output(options, "fp8", QType::FP8_E4M3FN_ROW_BF16, options.fp8_policy, false, 5120,
                    6144, 1024, 14336, weight, flush, stream, results);
}

void run_fp8_candidates(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
                        std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kOutputRows = 2 * kQueryRows + 2 * kKvRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_fp8_weight(kOutputRows, kHidden);
    const std::size_t candidate_partials =
        ops::detail::fp8_attn_input_candidate_partial_capacity_bytes();
    const std::size_t maximum_workspace =
        ops::detail::fp8_a8_workspace_capacity_bytes(max_tokens, kHidden, candidate_partials);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 107U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Fp8AttnInputScheduleId, 6> kSchedules{{
        ops::detail::Fp8AttnInputScheduleId::A8T32R32K128,
        ops::detail::Fp8AttnInputScheduleId::A8T32R128K128,
        ops::detail::Fp8AttnInputScheduleId::A8Tma64x128,
        ops::detail::Fp8AttnInputScheduleId::A8Tma64x256,
        ops::detail::Fp8AttnInputScheduleId::A8Tma96x256,
        ops::detail::Fp8AttnInputScheduleId::A8Bulk,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Fp8AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kOutputRows,
                                                       kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::fp8_attn_input_candidate_is_legal(schedule, problem)) {
                continue;
            }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::fp8_attn_input_execute_candidate(schedule, x, parent.weight, tq, tg, tk,
                                                              tv, workspace, launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(kOutputRows, tokens);
            const double flops = 2.0 * static_cast<double>(kOutputRows) * kHidden * tokens;
            append_result(results, "fp8", ops::detail::fp8_attn_input_schedule_name(schedule), tokens,
                          CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

// Candidate sweep for the 122B [17408,3072] fp8_e4m3fn_row_f32 A8 bands. Same shape semantics as
// run_fp8_candidates: every legal schedule is driven directly and the argmin per T is the ground
// truth for route retuning.
void run_fp8_122b_candidates(const Options& options, bench::L2FlushBuffer& flush,
                             cudaStream_t stream, std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 3072;
    constexpr std::int32_t kQueryRows  = 8192;
    constexpr std::int32_t kKvRows     = 512;
    constexpr std::int32_t kOutputRows = 2 * kQueryRows + 2 * kKvRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_fp8_weight(kOutputRows, kHidden, 0x51U, 0.05F, true);
    const std::size_t candidate_partials =
        ops::detail::fp8_attn_input_n17408_candidate_partial_capacity_bytes();
    const std::size_t maximum_workspace =
        ops::detail::fp8_a8_workspace_capacity_bytes(max_tokens, kHidden, candidate_partials);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 107U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Fp8AttnInputN17408ScheduleId, 12> kSchedules{{
        ops::detail::Fp8AttnInputN17408ScheduleId::A8T32R32K128,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8T32R128K128,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma64x128,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma64x128G8,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma64x256,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma64x256G8,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma96x256,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma96x256G8,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Tma128x128G8,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8Bulk,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8BulkG8,
        ops::detail::Fp8AttnInputN17408ScheduleId::A8BulkG8Wave48,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Fp8AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kOutputRows,
                                                       kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::fp8_attn_input_n17408_candidate_is_legal(schedule, problem)) {
                continue;
            }
            const char* schedule_name =
                ops::detail::fp8_attn_input_n17408_schedule_name(schedule);
            if (!options.cand_filter.empty() && options.cand_filter != schedule_name) {
                continue;
            }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::fp8_attn_input_n17408_execute_candidate(schedule, x, parent.weight, tq,
                                                                     tg, tk, tv, workspace,
                                                                     launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(kOutputRows, tokens);
            const double flops = 2.0 * static_cast<double>(kOutputRows) * kHidden * tokens;
            append_result(results, "fp8-f32", schedule_name, tokens,
                          CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

// Q8 candidate sweep: for each shape (target four-output, companion three-output, dflash2
// three-output) drive every legal schedule directly so the bench sees per-schedule timings.
void run_q8_target_candidates(const Options& options, bench::L2FlushBuffer& flush,
                              cudaStream_t stream, std::vector<Result>& results) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQueryRows   = 4096;
    constexpr std::int32_t kKvRows      = 512;
    constexpr std::int32_t kParentRows  = 9216;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent =
        bench::make_row_split_weight(QType::Q8_G32_FP16, kParentRows, kHidden, kHidden, 507U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 109U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Q8AttnInputScheduleId, 6> kSchedules{{
        ops::detail::Q8AttnInputScheduleId::DecodeR8Direct,
        ops::detail::Q8AttnInputScheduleId::SplitKMmaDirect,
        ops::detail::Q8AttnInputScheduleId::SimtR8C4,
        ops::detail::Q8AttnInputScheduleId::MmaR32C128,
        ops::detail::Q8AttnInputScheduleId::MmaR32C64,
        ops::detail::Q8AttnInputScheduleId::MmaR64C128,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Q8AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kParentRows,
                                                      kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::q8_attn_input_candidate_is_legal(schedule, problem)) { continue; }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::q8_attn_input_execute_candidate(schedule, x, parent.weight, tq, tg, tk,
                                                             tv, launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(kParentRows, tokens);
            const double flops = 2.0 * static_cast<double>(kParentRows) * kHidden * tokens;
            append_result(results, "q8-qgkv", ops::detail::q8_attn_input_schedule_name(schedule),
                          tokens, CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

void run_q8_qkv_candidates(const Options& options, const char* label, std::int32_t hidden,
                           bench::L2FlushBuffer& flush, cudaStream_t stream,
                           std::vector<Result>& results) {
    constexpr std::int32_t kQueryRows  = 4096;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kParentRows = 6144;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent =
        bench::make_row_split_weight(QType::Q8_G32_FP16, kParentRows, hidden, hidden, 505U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(hidden) * max_tokens, 111U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Q8AttnInputScheduleId, 10> kSchedules{{
        ops::detail::Q8AttnInputScheduleId::DecodeR8Direct,
        ops::detail::Q8AttnInputScheduleId::SplitKMmaDirect,
        ops::detail::Q8AttnInputScheduleId::MmaR32C64,
        ops::detail::Q8AttnInputScheduleId::MmaR32C128,
        ops::detail::Q8AttnInputScheduleId::MmaR64C64,
        ops::detail::Q8AttnInputScheduleId::MmaR64C96,
        ops::detail::Q8AttnInputScheduleId::MmaR64C128,
        ops::detail::Q8AttnInputScheduleId::MmaR128C64,
        ops::detail::Q8AttnInputScheduleId::MmaR128C80,
        ops::detail::Q8AttnInputScheduleId::SimtR8C4,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Q8AttnInputProblem problem{hidden, kQueryRows, kKvRows, kParentRows,
                                                      hidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::q8_attn_input_candidate_is_legal(schedule, problem)) { continue; }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {hidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::q8_attn_input_execute_candidate(schedule, x, parent.weight, tq, tk, tv,
                                                             launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(hidden, tokens) +
                                          tensor_bytes(kQueryRows + 2 * kKvRows, tokens);
            const double flops = 2.0 * static_cast<double>(kParentRows) * hidden * tokens;
            append_result(results, label, ops::detail::q8_attn_input_schedule_name(schedule), tokens,
                          CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

void run_q8_dflash2_candidates(const Options& options, bench::L2FlushBuffer& flush,
                               cudaStream_t stream, std::vector<Result>& results) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQueryRows   = 4096;
    constexpr std::int32_t kKvRows      = 1024;
    constexpr std::int32_t kParentRows  = 6144;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent =
        bench::make_row_split_weight(QType::Q8_G32_FP16, kParentRows, kHidden, kHidden, 509U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 113U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Q8AttnInputScheduleId, 6> kSchedules{{
        ops::detail::Q8AttnInputScheduleId::DFlash2SmallT,
        ops::detail::Q8AttnInputScheduleId::DFlash2MmaR16C64K128,
        ops::detail::Q8AttnInputScheduleId::DFlash2MmaR32C32K128,
        ops::detail::Q8AttnInputScheduleId::DFlash2MmaR32C64K128,
        ops::detail::Q8AttnInputScheduleId::DFlash2MmaR32C64,
        ops::detail::Q8AttnInputScheduleId::DFlash2MmaR64C128,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Q8AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kParentRows,
                                                      kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::q8_attn_input_candidate_is_legal(schedule, problem)) { continue; }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::q8_attn_input_execute_candidate(schedule, x, parent.weight, tq, tk, tv,
                                                             launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = parent.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(kQueryRows + 2 * kKvRows, tokens);
            const double flops = 2.0 * static_cast<double>(kParentRows) * kHidden * tokens;
            append_result(results, "q8-dflash2-qkv",
                          ops::detail::q8_attn_input_schedule_name(schedule), tokens,
                          CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

// Q4/Q5 candidate sweep: drive every legal schedule directly on the four-output target shape so
// the bench sees per-schedule timings for crossover retuning. Uses the same two-weight layout as
// run_q4q5 (Q4_G64_FP16 query_key + Q5_G64_FP16 gate_value).
void run_q4q5_candidates(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
                         std::vector<Result>& results) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQueryRows   = 6144;
    constexpr std::int32_t kKvRows      = 1024;
    constexpr std::int32_t kParentRows  = kQueryRows + kKvRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk =
        bench::make_row_split_weight(QType::Q4_G64_FP16, kParentRows, kHidden, kHidden, 501U);
    bench::PackedQuantizedWeight gv =
        bench::make_row_split_weight(QType::Q5_G64_FP16, kParentRows, kHidden, kHidden, 503U);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 199U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Q4Q5AttnInputScheduleId, 6> kSchedules{{
        ops::detail::Q4Q5AttnInputScheduleId::ParentSplitFixed,
        ops::detail::Q4Q5AttnInputScheduleId::MixedR32C32S2,
        ops::detail::Q4Q5AttnInputScheduleId::MixedR32C64S3,
        ops::detail::Q4Q5AttnInputScheduleId::PairR32C64S3,
        ops::detail::Q4Q5AttnInputScheduleId::MixedR64C128S2,
        ops::detail::Q4Q5AttnInputScheduleId::PairR32C64S4,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Q4Q5AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kHidden,
                                                        tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::q4_q5_attn_input_candidate_is_legal(schedule, problem)) { continue; }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::q4_q5_attn_input_execute_candidate(schedule, x, qk.weight, gv.weight,
                                                                tq, tg, tk, tv, launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = qk.model_weight_bytes() + gv.model_weight_bytes() +
                                          tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(2 * kQueryRows + 2 * kKvRows, tokens);
            const double flops = 2.0 * static_cast<double>(kParentRows) * kHidden * tokens;
            append_result(results, "q4q5", ops::detail::q4_q5_attn_input_schedule_name(schedule),
                          tokens, CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

// BF16 candidate sweep: drive every legal schedule directly on the four-output target shape so the
// bench sees per-schedule timings for crossover retuning.
void run_bf16_candidates(const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream,
                         std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kParentRows = 14336;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::DirectBf16Weight weight = bench::make_direct_bf16_weight(kParentRows, kHidden);
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, 101U);
    DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * max_tokens * 2);
    DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
    DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_tokens * 2);

    constexpr std::array<ops::detail::Bf16AttnInputScheduleId, 7> kSchedules{{
        ops::detail::Bf16AttnInputScheduleId::Decode,
        ops::detail::Bf16AttnInputScheduleId::SimtSmall,
        ops::detail::Bf16AttnInputScheduleId::SlicedR32T16W4,
        ops::detail::Bf16AttnInputScheduleId::MmaR32T32K128S3,
        ops::detail::Bf16AttnInputScheduleId::TmaR64T64K64S3,
        ops::detail::Bf16AttnInputScheduleId::MmaR64T32K64S3,
        ops::detail::Bf16AttnInputScheduleId::TmaR64T128K64S2,
    }};

    for (const std::int32_t tokens : options.tokens) {
        const ops::detail::Bf16AttnInputProblem problem{kHidden, kQueryRows, kKvRows, kParentRows,
                                                        kHidden, tokens};
        for (const auto schedule : kSchedules) {
            if (!ops::detail::bf16_attn_input_candidate_is_legal(schedule, problem)) { continue; }
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor tq(q.p, DType::BF16, {kQueryRows, tokens});
                Tensor tg(gate.p, DType::BF16, {kQueryRows, tokens});
                Tensor tk(k.p, DType::BF16, {kKvRows, tokens});
                Tensor tv(v.p, DType::BF16, {kKvRows, tokens});
                ops::detail::bf16_attn_input_execute_candidate(schedule, x, weight.weight, tq, tg,
                                                               tk, tv, launch_stream);
            };
            const Measurement measurement =
                measure_public(launch, CacheState::Cold, flush, stream, options.warmup,
                               options.repeat, false);
            const std::uint64_t logical = weight.model_weight_bytes() + tensor_bytes(kHidden, tokens) +
                                          tensor_bytes(2 * kQueryRows + 2 * kKvRows, tokens);
            const double flops = 2.0 * static_cast<double>(kParentRows) * kHidden * tokens;
            append_result(results, "bf16", ops::detail::bf16_attn_input_schedule_name(schedule),
                          tokens, CacheState::Cold, 0, logical, flops, measurement);
        }
    }
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) return;
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to open CSV output");
    output << "entry,format,policy,cache,T,workspace_bytes,logical_bytes,useful_flops,"
              "median_us,min_us,p95_us,execution,graph_nodes,workspace_capacity_bytes,workspace_"
              "peak_bytes\n";
    for (const Result& result : results) {
        output << "attn_input_proj," << result.format << ',' << result.policy << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.useful_flops << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << (result.graph_nodes ? "graph" : "eager") << ',';
        output << result.graph_nodes << ',' << result.workspace_capacity_bytes << ','
               << result.workspace_peak_bytes << '\n';
    }
}

bool selected(Format configured, Format candidate) {
    return configured == Format::All || configured == candidate;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        int device            = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d %s\n", props.name, props.major, props.minor,
                    CUDART_VERSION, temporary_switches::describe(temporary_switches::Scope::Qwen3_5).c_str());
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlushBytes);
        std::vector<Result> results;

        if (selected(options.format, Format::Q4Q5)) {
            if (options.candidates) {
                run_q4q5_candidates(options, flush, stream, results);
            } else {
                run_q4q5(options, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Q8Qgkv)) {
            if (options.candidates) {
                run_q8_target_candidates(options, flush, stream, results);
            } else {
                auto weight =
                    bench::make_row_split_weight(QType::Q8_G32_FP16, 9216, 2048, 2048, 507U);
                run_four_output(options, "q8-qgkv", QType::Q8_G32_FP16, ops::LinearPolicy::A16Only,
                                true, 2048, 4096, 512, 9216, weight, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Q8Qkv)) {
            if (options.candidates) {
                run_q8_qkv_candidates(options, "q8-qkv", 2048, flush, stream, results);
            } else {
                run_q8_qkv(options, "q8-qkv", 2048, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Q8DFlash2Qkv)) {
            if (options.candidates) {
                run_q8_dflash2_candidates(options, flush, stream, results);
            } else {
                run_q8_qkv(options, "q8-dflash2-qkv", 5120, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Bf16)) {
            if (options.candidates) {
                run_bf16_candidates(options, flush, stream, results);
            } else {
                auto weight = bench::make_direct_bf16_weight(14336, 5120);
                run_four_output(options, "bf16", QType::BF16, ops::LinearPolicy::A16Only, false, 5120,
                                6144, 1024, 14336, weight, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Nvfp4)) {
            if (options.candidates) {
                run_nvfp4_candidates(options, flush, stream, results);
            } else {
                auto weight = bench::make_nvfp4_weight(14336, 5120);
                run_four_output(options, "nvfp4", QType::NVFP4, options.nvfp4_policy, false, 5120,
                                6144, 1024, 14336, weight, flush, stream, results);
            }
        }
        if (selected(options.format, Format::Fp8)) {
            if (options.candidates) {
                if (options.fp8_problem == Fp8Problem::OneTwentyTwoB) {
                    run_fp8_122b_candidates(options, flush, stream, results);
                } else {
                    run_fp8_candidates(options, flush, stream, results);
                }
            } else {
                run_fp8(options, flush, stream, results);
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_attn_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
