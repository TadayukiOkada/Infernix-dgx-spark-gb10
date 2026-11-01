#include "core/weight.h"
#include "infernix/ops/linear_swiglu.h"

#include "core/device.h"
#include "core/temporary_switches.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"

// Per-variant harness headers: every internal A16/A4 schedule that the production swiglu plan
// selects behind its token bands, exposed so each can be timed independently on this device.
#include "core/pdl.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

namespace d = ops::detail;
constexpr std::int32_t kIntermediate = kOutputRows / 2;

// Direct instantiations of every internal NVFP4 schedule the production swiglu plans select behind
// their token bands, so each can be timed independently on this device (GB10). These mirror the
// exact template arguments used in src/ops/linear_swiglu/nvfp4/{a4,a4_tma,small_t}.cu.
template <class Schedule>
void bench_a4_gemm(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                   cudaStream_t stream) {
    using S = d::Nvfp4ScheduleInstance<Schedule, kHidden>;
    d::launch_nvfp4_a4_mma<S>(
        d::nvfp4_a4_operands(weight, workspace, tokens, d::Nvfp4ScaleLayout::RowMajor),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        d::SwiGluTokenMajorMmaEpilogue{}, stream, d::SwiGluTokenMajorMmaRows<S>{},
        pdl::Dependency::Programmatic);
}

void bench_a4_tma(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                  cudaStream_t stream) {
    d::launch_nvfp4_linear_swiglu_a4_tma(
        d::nvfp4_a4_operands(weight, workspace, tokens, d::Nvfp4ScaleLayout::Tiled256),
        static_cast<__nv_bfloat16*>(out.data), stream);
}

// The 2-term (FusedA4Residual) lanes with an explicit schedule, so the TMA tile that replaces the
// RTX 5090 non-TMA `ResidualM64N128` band is chosen against its neighbours in one process.
template <class Schedule>
void bench_a4r_gemm(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                    cudaStream_t stream) {
    using S = d::Nvfp4ScheduleInstance<Schedule, kHidden>;
    d::launch_nvfp4_a4_mma<S>(
        d::nvfp4_a4_operands(weight, workspace, tokens, d::Nvfp4ScaleLayout::RowMajor),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        d::SwiGluTokenMajorMmaEpilogue{}, stream, d::SwiGluTokenMajorMmaRows<S>{},
        pdl::Dependency::Programmatic);
}

template <int BlockTokens, int Stages, int MinBlocksPerSm, int BlockRows>
void bench_a4r_tma(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                   cudaStream_t stream) {
    using S = d::Nvfp4ScheduleInstance<
        d::Nvfp4A4TmaMmaSchedule<BlockTokens, Stages, MinBlocksPerSm,
                                 d::kNvfp4A4TmaWeightCodeL2_128B, d::kNvfp4A4ResidualTerms,
                                 BlockRows>,
        kHidden>;
    d::launch_nvfp4_a4_tma_mma<S>(
        d::nvfp4_a4_operands(weight, workspace, tokens, S::kScaleLayout),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        d::SwiGluTokenMajorMmaEpilogue{}, stream, d::SwiGluTokenMajorMmaRows<S>{});
}

template <int Tokens, int Warps, int Stages>
void bench_a16_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_sliced_k_mma<
        d::Nvfp4ScheduleInstance<d::Nvfp4SlicedInstance<Tokens, Warps, Stages>, kHidden>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        d::Nvfp4SwiGluEpilogue{}, stream, d::Nvfp4SwiGluRows<8>{});
}

template <class Schedule>
void bench_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_mma<d::Nvfp4ScheduleInstance<Schedule, kHidden>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        d::Nvfp4SwiGluEpilogue{}, stream, d::Nvfp4SwiGluMmaRows{});
}

struct Options {
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<std::int32_t> t_sweep{1, 4, 8, 16};
    int warmup   = 5;
    int repeat   = 30;
    bool profile  = false;
    bool variants = false;
    std::string csv_out;
};

struct Result {
    std::int32_t tokens;
    bench::ColdTiming timing;
    double effective_gbs;
    double useful_tflops;
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
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy  = ops::LinearPolicy::A16Only;
                options.t_sweep = {1, 4, 8, 16};
            } else if (value == "a4") {
                options.policy  = ops::LinearPolicy::AllowA4;
                options.t_sweep = {5, 48, 49, 56, 64, 65, 96, 97, 128, 256, 1024};
            } else if (value == "invariant") {
                options.policy  = ops::LinearPolicy::A16ColumnInvariant;
                options.t_sweep = {1, 2, 4, 8, 16};
            } else if (value == "a4r") {
                options.policy  = ops::LinearPolicy::AllowA4Residual;
                options.t_sweep = {5, 8, 16, 32, 64};
            } else {
                throw std::invalid_argument("--policy must be a16, a4, a4r or invariant");
            }
        } else if (argument == "--variants") {
            options.variants = true;
        } else if (argument == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat")));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--policy a16|a4|a4r|invariant] [--variants] [--t-sweep 1,4,...] [--warmup N] [--repeat N] "
                        "[--profile] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    return options;
}

const char* policy_name(ops::LinearPolicy policy) {
    if (policy == ops::LinearPolicy::AllowA4Residual) { return "A4R"; }
    if (policy == ops::LinearPolicy::A16ColumnInvariant) { return "INV"; }
    return policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16";
}

void write_csv(const Options& options, const std::vector<Result>& results,
               std::uint64_t weight_bytes) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV: " + options.csv_out); }
    out << "op,weight_type,policy,N,K,T,weight_bytes,median_us,min_us,p95_us,effective_gbs,"
           "useful_tflops,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        out << "linear_swiglu,NVFP4," << policy_name(options.policy) << ',' << kGateUpRows << ','
            << kHidden << ',' << result.tokens << ',' << weight_bytes << ','
            << result.timing.median_us << ',' << result.timing.min_us << ',' << result.timing.p95_us
            << ',' << result.effective_gbs << ',' << result.useful_tflops << ',' << options.warmup
            << ',' << options.repeat << ',' << kFlushBytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t =
            *std::min_element(options.t_sweep.begin(), options.t_sweep.end());
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t, 101U);
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed  = bench::make_nvfp4_weight(kGateUpRows, kHidden);
        const std::size_t one_term = d::nvfp4_a4_workspace_capacity_bytes(max_t, kHidden);
        const std::size_t two_term =
            d::nvfp4_a4_workspace_capacity_bytes(max_t, kHidden, d::kNvfp4A4ResidualTerms);
        const std::size_t workspace_capacity = options.variants
            ? std::max(ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden,
                                                                   ops::LinearPolicy::AllowA4, min_t,
                                                                   max_t),
                       one_term + two_term)
            : ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden,
                                                          options.policy, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto make_launch = [&](std::int32_t tokens) {
            return [&, tokens](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
                ops::linear_swiglu(x, packed.weight, out, options.policy, workspace, launch_stream);
            };
        };

        if (options.variants) {
            // Per-schedule cold sweep: drive every internal A16/A4 schedule the production swiglu
            // plan hides behind its token bands, each over its full route (quantize included for
            // A4), so the GB10 winner can be read per token count rather than inferred from the
            // policy gate.
            using LaunchFn = std::function<void(cudaStream_t)>;
            struct Cand {
                const char* name;
                LaunchFn launch;
            };
            std::FILE* cfile = nullptr;
            if (!options.csv_out.empty()) {
                std::filesystem::create_directories(std::filesystem::path(options.csv_out).parent_path());
                cfile = std::fopen(options.csv_out.c_str(), "w");
                std::fprintf(cfile, "path,kind,K,T,median_us,min_us,p95_us\n");
            }
            std::printf("# variants N=%d K=%d warmup=%d repeat=%d\n", kGateUpRows, kHidden,
                        options.warmup, options.repeat);
            for (const std::int32_t tokens : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
                workspace.reset();
                const d::Nvfp4A4Workspace scratch =
                    d::allocate_nvfp4_a4_workspace(workspace, tokens, kHidden);
                const auto a4 = [&](const char* name, void (*gemm)(const Weight&, Tensor&,
                                                                   d::Nvfp4A4Workspace, int,
                                                                   cudaStream_t)) {
                    return Cand{name, [&](cudaStream_t s) {
                                   d::launch_nvfp4_a4_quantize(x, packed.weight, scratch,
                                                               d::Nvfp4ScaleLayout::RowMajor, s);
                                   gemm(packed.weight, out, scratch, tokens, s);
                               }};
                };
                const auto a16 = [&](const char* name, void (*fn)(const Tensor&, const Weight&,
                                                                  Tensor&, cudaStream_t)) {
                    return Cand{name, [&](cudaStream_t s) { fn(x, packed.weight, out, s); }};
                };
                std::vector<Cand> cands;
                cands.push_back(a16("a16_sliced_t8_w8_s2", [](const Tensor& x_, const Weight& w,
                                                              Tensor& o, cudaStream_t s) {
                    bench_a16_sliced<8, 8, 2>(x_, w, o, s);
                }));
                cands.push_back(a16("a16_sliced_t16_w8_s2", [](const Tensor& x_, const Weight& w,
                                                               Tensor& o, cudaStream_t s) {
                    bench_a16_sliced<16, 8, 2>(x_, w, o, s);
                }));
                cands.push_back(a16("a16_sliced_t32_w4_s2", [](const Tensor& x_, const Weight& w,
                                                               Tensor& o, cudaStream_t s) {
                    bench_a16_sliced<32, 4, 2>(x_, w, o, s);
                }));
                cands.push_back(a16("a16_mma_32x64", [](const Tensor& x_, const Weight& w, Tensor& o,
                                                        cudaStream_t s) {
                    bench_a16_mma<d::Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x_, w, o, s);
                }));
                cands.push_back(a4("a4_m32n64_s3", [](const Weight& w, Tensor& o,
                                                      d::Nvfp4A4Workspace ws, int t,
                                                      cudaStream_t s) {
                    bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>>(w, o, ws, t, s);
                }));
                cands.push_back(a4("a4_m64n64", [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws,
                                                   int t, cudaStream_t s) {
                    bench_a4_gemm<d::Nvfp4A4MmaSchedule<64, 64, 256, 4, 2, 2, 2>>(w, o, ws, t, s);
                }));
                cands.push_back(a4("a4_m96n128", [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws,
                                                    int t, cudaStream_t s) {
                    bench_a4_gemm<d::Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1>>(w, o, ws, t, s);
                }));
                cands.push_back(a4("a4_m128n128", [](const Weight& w, Tensor& o,
                                                     d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                    bench_a4_gemm<d::Nvfp4A4MmaSchedule<128, 128, 256, 4, 4, 2, 1>>(w, o, ws, t, s);
                }));
                cands.push_back(Cand{"a4_tma", [&](cudaStream_t s) {
                    d::launch_nvfp4_a4_quantize(x, packed.weight, scratch, d::Nvfp4ScaleLayout::Tiled256,
                                                s);
                    bench_a4_tma(packed.weight, out, scratch, tokens, s);
                }});
                // T4: the 2-term FusedA4Residual lanes the production plan hides behind its token
                // bands (quantize included), so the TMA / non-TMA crossover is read in one process.
                const d::Nvfp4A4Workspace residual_scratch = d::allocate_nvfp4_a4_workspace(
                    workspace, tokens, kHidden, d::kNvfp4A4ResidualTerms);
                using A4RLaunch = void (*)(const Weight&, Tensor&, d::Nvfp4A4Workspace, int,
                                           cudaStream_t);
                const auto a4r = [&](const char* name, A4RLaunch gemm,
                                     d::Nvfp4ScaleLayout layout) {
                    return Cand{name, [=, &x, &packed, &out](cudaStream_t s) {
                                   d::launch_nvfp4_a4_quantize(x, packed.weight, residual_scratch,
                                                               layout, s);
                                   gemm(packed.weight, out, residual_scratch, tokens, s);
                               }};
                };
                cands.push_back(a4r(
                    "a4r_m16n64",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_gemm<d::Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 3, 2, ops::Cache::cg,
                                                             ops::Cache::cg,
                                                             d::kNvfp4A4ResidualTerms>>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::RowMajor));
                cands.push_back(a4r(
                    "a4r_m32n64",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_gemm<d::Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2, ops::Cache::cg,
                                                             ops::Cache::cg,
                                                             d::kNvfp4A4ResidualTerms>>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::RowMajor));
                cands.push_back(a4r(
                    "a4r_m64n128",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_gemm<d::Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1, ops::Cache::cg,
                                                             ops::Cache::cg,
                                                             d::kNvfp4A4ResidualTerms>>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::RowMajor));
                cands.push_back(a4r(
                    "a4r_m96n128",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_gemm<d::Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1, ops::Cache::cg,
                                                             ops::Cache::cg,
                                                             d::kNvfp4A4ResidualTerms>>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::RowMajor));
                cands.push_back(a4r(
                    "a4r_tma128_r128_s3",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_tma<128, 3, 1, 128>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::Tiled128));
                cands.push_back(a4r(
                    "a4r_tma128_r64_s3",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_tma<128, 3, 1, 64>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::Tiled128));
                cands.push_back(a4r(
                    "a4r_tma256_r64_s2",
                    [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4r_tma<256, 2, 1, 64>(w, o, ws, t, s);
                    },
                    d::Nvfp4ScaleLayout::Tiled256));
                for (const Cand& cand : cands) {
                    const bench::ColdTiming timing =
                        bench::measure_cold_launch(cand.launch, flush, stream, options.warmup,
                                                   options.repeat);
                    std::printf("K=%-5d T=%-4d %-24s median=%9.3f us min=%9.3f p95=%9.3f\n", kHidden,
                                tokens, cand.name, timing.median_us, timing.min_us, timing.p95_us);
                    if (cfile) {
                        std::fprintf(cfile, "%s,a4,%d,%d,%.3f,%.3f,%.3f\n", cand.name, kHidden, tokens,
                                     timing.median_us, timing.min_us, timing.p95_us);
                        std::fflush(cfile);
                    }
                }
            }
            if (cfile) { std::fclose(cfile); }
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        if (options.profile) {
            const std::int32_t tokens = options.t_sweep.front();
            const auto launch         = make_launch(tokens);
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                bench::flush_l2(flush, stream);
                launch(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu weight_type=NVFP4 policy=%s T=%d\n",
                        policy_name(options.policy), tokens);
            std::fflush(stdout);
            CUDA_CHECK(cudaProfilerStart());
            launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        results.reserve(options.t_sweep.size());
        std::printf("# %s\n", temporary_switches::describe(temporary_switches::Scope::Qwen3_5).c_str());
        std::printf("%-14s %3s %8s %8s %6s %11s %11s %11s %10s %10s\n", "op", "pol", "N", "K", "T",
                    "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s");
        for (const std::int32_t tokens : options.t_sweep) {
            const auto launch = make_launch(tokens);
            const bench::ColdTiming timing =
                bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
            const double seconds      = timing.median_us * 1.0e-6;
            const double useful_flops = 2.0 * static_cast<double>(kGateUpRows) * kHidden * tokens;
            const double model_bytes  = static_cast<double>(packed.model_weight_bytes()) +
                                       2.0 * static_cast<double>(kHidden + kOutputRows) * tokens;
            const double useful_tflops = useful_flops / seconds / 1.0e12;
            const double effective_gbs = model_bytes / seconds / 1.0e9;
            std::printf("%-14s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f\n",
                        "linear_swiglu", policy_name(options.policy), kGateUpRows, kHidden, tokens,
                        timing.median_us, timing.min_us, timing.p95_us, effective_gbs,
                        useful_tflops);
            results.push_back({tokens, timing, effective_gbs, useful_tflops});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_nvfp4_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
