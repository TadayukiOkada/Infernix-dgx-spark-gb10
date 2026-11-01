#include "core/weight.h"
#include "infernix/ops/linear_add.h"

#include "core/device.h"
#include "core/temporary_switches.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"

// Per-variant harness headers: every internal A16/A4 schedule that the production plan selects
// behind its token bands, exposed so each can be timed independently on this device.
#include "core/pdl.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"

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

constexpr std::int32_t kRows      = 5120;
infernix::bench::ColdFlushBytes kFlushBytes{};

namespace d = ops::detail;

// Direct instantiations of every internal NVFP4 schedule that the production A16/A4 plans select
// behind their token bands, so each can be timed independently on this device (GB10). These mirror
// the exact template arguments used in src/ops/linear_add/nvfp4/{a16,a4}.cu.
template <class Schedule, int K>
void bench_a4_gemm(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                   cudaStream_t stream) {
    d::launch_nvfp4_a4_mma<d::Nvfp4ScheduleInstance<Schedule, K>>(
        d::nvfp4_a4_operands(weight, workspace, tokens, d::Nvfp4ScaleLayout::RowMajor),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream, {},
        pdl::Dependency::Programmatic);
}

template <d::Nvfp4GeometryId Geometry>
void bench_a4_tma(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace, int tokens,
                  cudaStream_t stream) {
    d::launch_nvfp4_a4_tma_linear_add(
        d::nvfp4_a4_operands(weight, workspace, tokens, d::Nvfp4ScaleLayout::Tiled256),
        static_cast<__nv_bfloat16*>(out.data), stream);
}

template <int Tokens, int Warps, int Stages, int K>
void bench_a16_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_sliced_k_mma<
        d::Nvfp4ScheduleInstance<d::Nvfp4SlicedInstance<Tokens, Warps, Stages>, K>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

// The K-baked tile with the RowTiles lever exposed, so RowTiles and the runtime-K form are
// measured as separate changes rather than bundled into one candidate.
template <int Tokens, int Warps, int Stages, int RowTiles, int K>
void bench_a16_sliced_baked(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    d::launch_nvfp4_a16_sliced_k_mma<
        d::Nvfp4ScheduleInstance<d::Nvfp4SlicedInstance<Tokens, Warps, Stages, RowTiles>, K>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

// The runtime-K (bare-schedule) form of the sliced-K tile, with the RowTiles lever exposed: baking
// K lets ptxas unroll the K-group loop, which the standalone tables found slower at these widths.
template <int Tokens, int Warps, int Stages, int RowTiles>
void bench_a16_sliced_rt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_sliced_k_mma<d::Nvfp4SlicedInstance<Tokens, Warps, Stages, RowTiles>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

template <class Schedule, int K>
void bench_a16_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_gemv<d::Nvfp4ScheduleInstance<Schedule, K>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

// A TMA lane with an explicit schedule, so the BlockTokens/BlockRows/Stages shape can be swept
// against the one the production launcher hardcodes.
template <int BlockTokens, int Stages, int MinBlocksPerSm, int BlockRows, int K>
void bench_a4_tma_schedule(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace,
                           int tokens, cudaStream_t stream) {
    using S = d::Nvfp4ScheduleInstance<
        d::Nvfp4A4TmaMmaSchedule<BlockTokens, Stages, MinBlocksPerSm,
                                 d::kNvfp4A4TmaWeightCodeL2_128B, 1, BlockRows>,
        K>;
    d::launch_nvfp4_a4_tma_mma<S>(
        d::nvfp4_a4_operands(weight, workspace, tokens, S::kScaleLayout),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

template <int BlockTokens>
inline constexpr d::Nvfp4ScaleLayout kBenchTmaScaleLayout =
    BlockTokens == 128 ? d::Nvfp4ScaleLayout::Tiled128 : d::Nvfp4ScaleLayout::Tiled256;

// The 2-term (FusedA4Residual) TMA lane with an explicit schedule: the route that replaces the RTX
// 5090 RowMajor ResidualM64N128 band above T = 64.
template <int BlockTokens, int Stages, int MinBlocksPerSm, int BlockRows, int K>
void bench_a4_tma_residual(const Weight& weight, Tensor& out, d::Nvfp4A4Workspace workspace,
                           int tokens, cudaStream_t stream) {
    using S = d::Nvfp4ScheduleInstance<
        d::Nvfp4A4TmaMmaSchedule<BlockTokens, Stages, MinBlocksPerSm,
                                 d::kNvfp4A4TmaWeightCodeL2_128B, d::kNvfp4A4ResidualTerms,
                                 BlockRows>,
        K>;
    d::launch_nvfp4_a4_tma_mma<S>(
        d::nvfp4_a4_operands(weight, workspace, tokens, S::kScaleLayout),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

template <class Schedule, int K>
void bench_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    d::launch_nvfp4_a16_mma<d::Nvfp4ScheduleInstance<Schedule, K>>(
        d::nvfp4_a16_operands(x, weight),
        d::LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        d::LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(out.data), weight.n}}, stream);
}

struct Options {
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<std::int32_t> t_sweep{1, 4, 8, 16, 17, 128, 512, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile   = false;
    bool variants  = false;
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
        if (argument == "--n") {
            options.n = std::stoi(std::string(next("--n")));
        } else if (argument == "--k") {
            options.k = std::stoi(std::string(next("--k")));
        } else if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy = ops::LinearPolicy::A16Only;
            } else if (value == "a4") {
                options.policy = ops::LinearPolicy::AllowA4;
            } else if (value == "invariant") {
                options.policy = ops::LinearPolicy::A16ColumnInvariant;
            } else if (value == "a4r") {
                options.policy = ops::LinearPolicy::AllowA4Residual;
            } else {
                throw std::invalid_argument("--policy must be a16, a4, a4r or invariant");
            }
        } else if (argument == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat")));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--variants") {
            options.variants = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s --n 5120 --k 6144|17408 [--policy a16|a4|a4r|invariant] [--variants] "
                "[--t-sweep 1,4,...] [--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n"
                "  --variants: time every internal A16/A4 schedule variant independently per token.\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.n != kRows || (options.k != 6144 && options.k != 17408)) {
        throw std::invalid_argument("NVFP4 LinearAdd supports N=5120 and K=6144|17408");
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
        out << "linear_add,NVFP4," << policy_name(options.policy) << ',' << options.n << ','
            << options.k << ',' << result.tokens << ',' << weight_bytes << ','
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
        DeviceBuffer input    = bench::make_bf16(static_cast<std::size_t>(options.k) * max_t, 101U);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(options.n) * max_t, 103U);
        bench::SavedBuffer residual_initial(residual);
        const auto restore                   = [&](cudaStream_t s) { residual_initial.restore(s); };
        bench::PackedQuantizedWeight packed  = bench::make_nvfp4_weight(options.n, options.k);
        const std::size_t one_term = d::nvfp4_a4_workspace_capacity_bytes(max_t, options.k);
        const std::size_t two_term =
            d::nvfp4_a4_workspace_capacity_bytes(max_t, options.k, d::kNvfp4A4ResidualTerms);
        const std::size_t workspace_capacity = options.variants
            ? std::max(ops::linear_add_workspace_capacity_bytes(QType::NVFP4, options.n, options.k,
                                                                ops::LinearPolicy::AllowA4, min_t,
                                                                max_t),
                       one_term + two_term)
            : ops::linear_add_workspace_capacity_bytes(QType::NVFP4, options.n, options.k,
                                                       options.policy, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto make_launch = [&](std::int32_t tokens) {
            return [&, tokens](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {options.k, tokens});
                Tensor out(residual.p, DType::BF16, {options.n, tokens});
                ops::linear_add(x, packed.weight, out, options.policy, workspace, launch_stream);
            };
        };

        if (options.variants) {
            // Per-schedule cold sweep: drive every internal A16/A4 schedule the production plan
            // hides behind its token bands, each over its full route, so the GB10 winner can be
            // read per token count rather than inferred from the policy gate.
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
            const bool k6144 = options.k == 6144;
            std::printf("# variants N=%d K=%d warmup=%d repeat=%d\n", options.n, options.k,
                        options.warmup, options.repeat);
            for (const std::int32_t tokens : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {options.k, tokens});
                Tensor out(residual.p, DType::BF16, {options.n, tokens});
                workspace.reset();
                const d::Nvfp4A4Workspace scratch =
                    d::allocate_nvfp4_a4_workspace(workspace, tokens, options.k);
                const auto a4 = [&](const char* name, void (*gemm)(const Weight&, Tensor&,
                                                                   d::Nvfp4A4Workspace, int,
                                                                   cudaStream_t)) {
                    // Capture the function pointer by value: the stored std::function outlives this
                    // helper's frame, so a by-reference capture would dangle.
                    return Cand{name, [=, &x, &packed, &out, &scratch](cudaStream_t s) {
                                   d::launch_nvfp4_a4_quantize(x, packed.weight, scratch,
                                                               d::Nvfp4ScaleLayout::RowMajor, s);
                                   gemm(packed.weight, out, scratch, tokens, s);
                               }};
                };
                const auto a16 = [&](const char* name, void (*fn)(const Tensor&, const Weight&,
                                                                  Tensor&, cudaStream_t)) {
                    return Cand{name, [=, &x, &packed, &out](cudaStream_t s) {
                                   fn(x, packed.weight, out, s);
                               }};
                };
                std::vector<Cand> cands;
                if (k6144) {
                    cands.push_back(a16("a16_sliced_t8_w8_s2", [](const Tensor& x_, const Weight& w,
                                                                  Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<8, 8, 2, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t16_w8_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<16, 8, 2, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t32_w8_s1", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<32, 8, 1, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t32_w4_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<32, 4, 2, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t32_w2_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<32, 2, 2, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t64_w2_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<64, 2, 2, 6144>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_mma_64x64", [](const Tensor& x_, const Weight& w,
                                                            Tensor& o, cudaStream_t s) {
                        bench_a16_mma<d::Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, 6144>(x_, w,
                                                                                              o, s);
                    }));
                    cands.push_back(a16("a16_mma_64x128", [](const Tensor& x_, const Weight& w,
                                                             Tensor& o, cudaStream_t s) {
                        bench_a16_mma<d::Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 6144>(x_, w,
                                                                                              o, s);
                    }));
                    cands.push_back(a4("a4_m32n64", [](const Weight& w, Tensor& o,
                                                       d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>, 6144>(w, o, ws,
                                                                                            t, s);
                    }));
                    cands.push_back(a4("a4_m32n128", [](const Weight& w, Tensor& o,
                                                        d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>, 6144>(w, o, ws,
                                                                                             t, s);
                    }));
                    cands.push_back(a4("a4_m64n128", [](const Weight& w, Tensor& o,
                                                        d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>, 6144>(w, o, ws,
                                                                                             t, s);
                    }));
                    cands.push_back(a4("a4_m128n128_resident",
                                       [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                                          cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 1>, 6144>(w, o,
                                                                                              ws, t, s);
                    }));
                    cands.push_back(a4("a4_m128n128_pipelined",
                                       [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                                          cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>, 6144>(w, o,
                                                                                              ws, t, s);
                    }));
                    cands.push_back(Cand{"a4_tma", [&](cudaStream_t s) {
                        d::launch_nvfp4_a4_quantize(x, packed.weight, scratch,
                                                    d::Nvfp4ScaleLayout::Tiled256, s);
                        bench_a4_tma<d::Nvfp4GeometryId::N5120K6144>(packed.weight, out, scratch,
                                                                     tokens, s);
                    }});
                } else {
                    cands.push_back(a16("a16_sliced_t8_w4_s2", [](const Tensor& x_, const Weight& w,
                                                                  Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<8, 4, 2, 17408>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t16_w4_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<16, 4, 2, 17408>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t32_w4_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<32, 4, 2, 17408>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t32_w4_s1", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<32, 4, 1, 17408>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_sliced_t64_w2_s2", [](const Tensor& x_, const Weight& w,
                                                                   Tensor& o, cudaStream_t s) {
                        bench_a16_sliced<64, 2, 2, 17408>(x_, w, o, s);
                    }));
                    cands.push_back(a16("a16_mma_32x64", [](const Tensor& x_, const Weight& w,
                                                            Tensor& o, cudaStream_t s) {
                        bench_a16_mma<d::Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, 17408>(x_, w,
                                                                                               o, s);
                    }));
                    cands.push_back(a16("a16_mma_64x128", [](const Tensor& x_, const Weight& w,
                                                             Tensor& o, cudaStream_t s) {
                        bench_a16_mma<d::Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 17408>(x_, w,
                                                                                               o, s);
                    }));
                    cands.push_back(a4("a4_m32n32k512", [](const Weight& w, Tensor& o,
                                                           d::Nvfp4A4Workspace ws, int t,
                                                           cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 32, 512, 2, 2, 3, 2>, 17408>(w, o, ws,
                                                                                             t, s);
                    }));
                    cands.push_back(a4("a4_m32n64", [](const Weight& w, Tensor& o,
                                                       d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>, 17408>(w, o, ws,
                                                                                             t, s);
                    }));
                    cands.push_back(a4("a4_m32n128", [](const Weight& w, Tensor& o,
                                                        d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>, 17408>(w, o, ws,
                                                                                              t, s);
                    }));
                    cands.push_back(a4("a4_m64n128", [](const Weight& w, Tensor& o,
                                                        d::Nvfp4A4Workspace ws, int t, cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>, 17408>(w, o, ws,
                                                                                              t, s);
                    }));
                    cands.push_back(a4("a4_m128n128_resident",
                                       [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                                          cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 1>, 17408>(w, o,
                                                                                               ws, t, s);
                    }));
                    cands.push_back(a4("a4_m128n128_pipelined",
                                       [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                                          cudaStream_t s) {
                        bench_a4_gemm<d::Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>, 17408>(w, o,
                                                                                               ws, t, s);
                    }));
                    cands.push_back(Cand{"a4_tma", [&](cudaStream_t s) {
                        d::launch_nvfp4_a4_quantize(x, packed.weight, scratch,
                                                    d::Nvfp4ScaleLayout::Tiled256, s);
                        bench_a4_tma<d::Nvfp4GeometryId::N5120K17408>(packed.weight, out, scratch,
                                                                      tokens, s);
                    }});
                }
                // T7 (2026-10-31) candidates: the levers the task proposes for the fused add route,
                // measured in the same process against the production schedules above.
                const auto add_t7 = [&](auto ktag) {
                    constexpr int K = decltype(ktag)::value;
                    const auto gemv = [&](const char* name, void (*fn)(const Tensor&, const Weight&,
                                                                       Tensor&, cudaStream_t)) {
                        cands.push_back(Cand{name, [=, &x, &packed, &out](cudaStream_t s) {
                                               fn(x, packed.weight, out, s);
                                           }});
                    };
                    if (tokens == 1) {
                        gemv("gemv_w2_staged_prod", [](const Tensor& x_, const Weight& w, Tensor& o,
                                                       cudaStream_t s) {
                            bench_a16_gemv<d::Nvfp4A16GemvSchedule<
                                               8, 2, 16, 4, d::Nvfp4ScaleAccess::StagedRaw,
                                               d::Nvfp4CodeCache::Default, 2>,
                                           K>(x_, w, o, s);
                        });
                        gemv("gemv_w1_direct", [](const Tensor& x_, const Weight& w, Tensor& o,
                                                  cudaStream_t s) {
                            bench_a16_gemv<d::Nvfp4A16GemvSchedule<
                                               8, 1, 16, 4, d::Nvfp4ScaleAccess::Direct,
                                               d::Nvfp4CodeCache::Default, 2>,
                                           K>(x_, w, o, s);
                        });
                        gemv("gemv_w1_staged", [](const Tensor& x_, const Weight& w, Tensor& o,
                                                  cudaStream_t s) {
                            bench_a16_gemv<d::Nvfp4A16GemvSchedule<
                                               8, 1, 16, 4, d::Nvfp4ScaleAccess::StagedRaw,
                                               d::Nvfp4CodeCache::Default, 2>,
                                           K>(x_, w, o, s);
                        });
                        gemv("gemv_w2_direct", [](const Tensor& x_, const Weight& w, Tensor& o,
                                                  cudaStream_t s) {
                            bench_a16_gemv<d::Nvfp4A16GemvSchedule<
                                               8, 2, 16, 4, d::Nvfp4ScaleAccess::Direct,
                                               d::Nvfp4CodeCache::Default, 2>,
                                           K>(x_, w, o, s);
                        });
                    }
                    // The 2..5-token SIMT kernel this candidate used to measure is gone: the
                    // runtime-K RowTiles-2 sliced-K tiles of every one of those bands beat it on
                    // GB10, so nvfp4_linear_add_small_t.cu was removed with its route.
                    const auto sliced_rt = [&](const char* name, void (*fn)(const Tensor&,
                                                                            const Weight&, Tensor&,
                                                                            cudaStream_t)) {
                        cands.push_back(Cand{name, [=, &x, &packed, &out](cudaStream_t s) {
                                               fn(x, packed.weight, out, s);
                                           }});
                    };
#define INFER_NIX_BENCH_SLICED_RT(NAME, T, W, ST, RT)                                            \
    sliced_rt(NAME, [](const Tensor& x_, const Weight& w, Tensor& o, cudaStream_t s) {            \
        bench_a16_sliced_rt<T, W, ST, RT>(x_, w, o, s);                                           \
    });
#define INFER_NIX_BENCH_SLICED_BAKED(NAME, T, W, ST, RT)                                         \
    sliced_rt(NAME, [](const Tensor& x_, const Weight& w, Tensor& o, cudaStream_t s) {            \
        bench_a16_sliced_baked<T, W, ST, RT, K>(x_, w, o, s);                                     \
    });
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t8_w8_s2_r2", 8, 8, 2, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t8_w4_s2_r2", 8, 4, 2, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t16_w8_s2_r2", 16, 8, 2, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t16_w4_s2_r2", 16, 4, 2, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t32_w4_s1_r2", 32, 4, 1, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t32_w4_s2_r2", 32, 4, 2, 2)
                    INFER_NIX_BENCH_SLICED_RT("slicedrt_t64_w2_s2_r2", 64, 2, 2, 2)
                    INFER_NIX_BENCH_SLICED_BAKED("slicedbkt8_w8_s2_r2", 8, 8, 2, 2)
                    INFER_NIX_BENCH_SLICED_BAKED("slicedbkt8_w4_s2_r2", 8, 4, 2, 2)
                    INFER_NIX_BENCH_SLICED_BAKED("slicedbkt16_w8_s2_r2", 16, 8, 2, 2)
                    INFER_NIX_BENCH_SLICED_BAKED("slicedbkt16_w4_s2_r2", 16, 4, 2, 2)
                    INFER_NIX_BENCH_SLICED_BAKED("slicedbkt32_w4_s1_r2", 32, 4, 1, 2)
#undef INFER_NIX_BENCH_SLICED_BAKED
#undef INFER_NIX_BENCH_SLICED_RT
                    const auto tma = [&](const char* name,
                                         void (*gemm)(const Weight&, Tensor&, d::Nvfp4A4Workspace,
                                                      int, cudaStream_t),
                                         d::Nvfp4ScaleLayout layout) {
                        cands.push_back(Cand{name, [=, &x, &packed, &out, &scratch](cudaStream_t s) {
                                               d::launch_nvfp4_a4_quantize(x, packed.weight, scratch,
                                                                           layout, s);
                                               gemm(packed.weight, out, scratch, tokens, s);
                                           }});
                    };
                    tma("a4_tma128_b3",
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_tma_schedule<128, 3, 1, 128, K>(w, o, ws, t, s);
                        },
                        kBenchTmaScaleLayout<128>);
                    tma("a4_tma256_b3_r64",
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_tma_schedule<256, 3, 1, 64, K>(w, o, ws, t, s);
                        },
                        kBenchTmaScaleLayout<256>);
                    // T4: the 2-term FusedA4Residual lanes (quantize included) that the production
                    // plan hides behind its token bands, so the TMA lane that replaces the RTX 5090
                    // RowMajor ResidualM64N128 band above T = 64 is picked against its neighbours.
                    const d::Nvfp4A4Workspace rscratch = d::allocate_nvfp4_a4_workspace(
                        workspace, tokens, K, d::kNvfp4A4ResidualTerms);
                    const auto a4r = [&](const char* name, d::Nvfp4ScaleLayout layout,
                                         void (*gemm)(const Weight&, Tensor&, d::Nvfp4A4Workspace,
                                                      int, cudaStream_t)) {
                        cands.push_back(Cand{name, [=, &x, &packed, &out](cudaStream_t s) {
                                               d::launch_nvfp4_a4_quantize(x, packed.weight,
                                                                           rscratch, layout, s);
                                               gemm(packed.weight, out, rscratch, tokens, s);
                                           }});
                    };
                    a4r("a4r_m32n64", d::Nvfp4ScaleLayout::RowMajor,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_gemm<d::Nvfp4A4MmaSchedule<
                                              32, 64, 256, 2, 4, 3, 2, ops::Cache::cg, ops::Cache::cg,
                                              d::kNvfp4A4ResidualTerms>,
                                          K>(w, o, ws, t, s);
                        });
                    a4r("a4r_m64n64", d::Nvfp4ScaleLayout::RowMajor,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_gemm<d::Nvfp4A4MmaSchedule<
                                              64, 64, 256, 4, 2, 2, 2, ops::Cache::cg, ops::Cache::cg,
                                              d::kNvfp4A4ResidualTerms>,
                                          K>(w, o, ws, t, s);
                        });
                    a4r("a4r_m64n128", d::Nvfp4ScaleLayout::RowMajor,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_gemm<d::Nvfp4A4MmaSchedule<
                                              64, 128, 256, 4, 2, 2, 1, ops::Cache::cg, ops::Cache::cg,
                                              d::kNvfp4A4ResidualTerms>,
                                          K>(w, o, ws, t, s);
                        });
                    a4r("a4r_tma128_r128_s3", kBenchTmaScaleLayout<128>,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_tma_residual<128, 3, 1, 128, K>(w, o, ws, t, s);
                        });
                    a4r("a4r_tma128_r64_s3", kBenchTmaScaleLayout<128>,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_tma_residual<128, 3, 1, 64, K>(w, o, ws, t, s);
                        });
                    a4r("a4r_tma256_r64_s2", kBenchTmaScaleLayout<256>,
                        [](const Weight& w, Tensor& o, d::Nvfp4A4Workspace ws, int t,
                           cudaStream_t s) {
                            bench_a4_tma_residual<256, 2, 1, 64, K>(w, o, ws, t, s);
                        });
                };
                if (k6144)
                    add_t7(std::integral_constant<int, 6144>{});
                else
                    add_t7(std::integral_constant<int, 17408>{});
                for (const Cand& cand : cands) {
                    const bench::ColdTiming timing = bench::measure_cold_launch_prepared(
                        restore, cand.launch, flush, stream, options.warmup, options.repeat);
                    std::printf("K=%-5d T=%-4d %-28s median=%9.3f us min=%9.3f p95=%9.3f\n",
                                options.k, tokens, cand.name, timing.median_us, timing.min_us,
                                timing.p95_us);
                    if (cfile) {
                        std::fprintf(cfile, "%s,%s,%d,%d,%.3f,%.3f,%.3f\n", cand.name,
                                     k6144 ? "6144" : "17408", options.k, tokens, timing.median_us,
                                     timing.min_us, timing.p95_us);
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
                restore(stream);
                bench::flush_l2(flush, stream);
                launch(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            restore(stream);
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_add weight_type=NVFP4 policy=%s N=%d K=%d T=%d\n",
                        policy_name(options.policy), options.n, options.k, tokens);
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
        std::printf("%-11s %3s %8s %8s %6s %11s %11s %11s %10s %10s\n", "op", "pol", "N", "K", "T",
                    "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s");
        for (const std::int32_t tokens : options.t_sweep) {
            const auto launch              = make_launch(tokens);
            const bench::ColdTiming timing = bench::measure_cold_launch_prepared(
                restore, launch, flush, stream, options.warmup, options.repeat);
            const double seconds      = timing.median_us * 1.0e-6;
            const double useful_flops = 2.0 * static_cast<double>(options.n) * options.k * tokens;
            const double model_bytes  = static_cast<double>(packed.model_weight_bytes()) +
                                       2.0 * static_cast<double>(options.k) * tokens +
                                       4.0 * static_cast<double>(options.n) * tokens;
            const double useful_tflops = useful_flops / seconds / 1.0e12;
            const double effective_gbs = model_bytes / seconds / 1.0e9;
            std::printf("%-11s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f\n", "linear_add",
                        policy_name(options.policy), options.n, options.k, tokens, timing.median_us,
                        timing.min_us, timing.p95_us, effective_gbs, useful_tflops);
            results.push_back({tokens, timing, effective_gbs, useful_tflops});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_nvfp4_linear_add_bench: %s\n", error.what());
        return 1;
    }
}
