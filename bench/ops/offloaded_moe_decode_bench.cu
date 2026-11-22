// Device-resident moe_experts benchmark (not the unrelated sparse_moe operator).
// Routing/setup and CPU bit-exact validation are outside CUDA-event timing.
// Defaults cover decode, repeated-column verification and the eight-column path.
#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_moe_fixtures.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;

namespace {
template <class T> DeviceBuffer copy(const std::vector<T>& values) {
    DeviceBuffer out(values.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(out.p, values.data(), out.bytes, cudaMemcpyHostToDevice));
    return out;
}

double bf16_to_f64(std::uint16_t bits) {
    return static_cast<double>(canon::bf16_to_f32(bits));
}

std::vector<canon::A4Block> quantize_a4(const std::uint16_t* v, int n, float scale) {
    std::vector<canon::A4Block> out(static_cast<std::size_t>(n / 16));
    for (int b = 0; b < n / 16; ++b) {
        out[static_cast<std::size_t>(b)] = canon::quantize_a4_block(v + 16 * b, scale);
    }
    return out;
}

std::int64_t rg16_exact_row_sum(const std::uint8_t* matrix, int blocks, int row,
                                const canon::A4Block* a) {
    const std::uint8_t* group =
        matrix + static_cast<std::size_t>(row / 16) * blocks * moe::kUnitBytes;
    const int r     = row % 16;
    const int i     = r % 8;
    const int shift = r < 8 ? 0 : 4;
    std::int64_t s  = 0;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = group + static_cast<std::size_t>(b) * moe::kUnitBytes;
        int p                    = 0;
        for (int k = 0; k < 16; ++k) {
            const unsigned byte = unit[32 * (k / 4) + 4 * i + (k % 4)];
            p += canon::e2m1_x2((byte >> shift) & 15U) * a[b].c2[k];
        }
        s += static_cast<std::int64_t>(p) * canon::e4m3_scaled(unit[128 + r]) * a[b].scale_scaled;
    }
    return s;
}

std::vector<double> fp64_expert_column(const fixtures::Expert& expert, const std::uint16_t* x_bf16) {
    const std::uint8_t* gate_up = expert.record.data();
    const std::uint8_t* down    = expert.record.data() + moe::kGateUpBytes;
    const auto xg               = quantize_a4(x_bf16, moe::kHidden, expert.scales.input_gate);
    const auto xu               = quantize_a4(x_bf16, moe::kHidden, expert.scales.input_up);
    std::vector<std::uint16_t> h(moe::kIntermediate);
    for (int i = 0; i < moe::kIntermediate; ++i) {
        const double g =
            static_cast<double>(rg16_exact_row_sum(gate_up, moe::kGateUpBlocks, 2 * i, xg.data())) *
            0x1p-20 * static_cast<double>(expert.scales.alpha_gate);
        const double u =
            static_cast<double>(rg16_exact_row_sum(gate_up, moe::kGateUpBlocks, 2 * i + 1, xu.data())) *
            0x1p-20 * static_cast<double>(expert.scales.alpha_up);
        const double gb = bf16_to_f64(canon::f32_to_bf16_rn(static_cast<float>(g)));
        const double ub = bf16_to_f64(canon::f32_to_bf16_rn(static_cast<float>(u)));
        h[static_cast<std::size_t>(i)] =
            canon::f32_to_bf16_rn(static_cast<float>(gb / (1.0 + std::exp(-gb)) * ub));
    }
    const auto hq = quantize_a4(h.data(), moe::kIntermediate, expert.scales.input_down);
    std::vector<double> out(moe::kHidden);
    for (int r = 0; r < moe::kHidden; ++r) {
        out[static_cast<std::size_t>(r)] =
            static_cast<double>(rg16_exact_row_sum(down, moe::kDownBlocks, r, hq.data())) *
            0x1p-20 * static_cast<double>(expert.scales.alpha_down);
    }
    return out;
}

void run(int columns, bool split, bool omma, int repeat, DeviceContext& context, bench::L2FlushBuffer& flush) {
    constexpr int top_k = 10;
    const int entries = columns * top_k;
    std::mt19937 rng(20261015);
    const auto x = fixtures::random_activations(rng, columns);
    std::vector<std::uint8_t> records;
    std::vector<moe::ExpertScales> scales;
    std::vector<std::uint16_t> expected(omma ? 0 : static_cast<std::size_t>(entries) * moe::kHidden);
    std::vector<double> expected_fp64(omma ? static_cast<std::size_t>(entries) * moe::kHidden : 0);
    for (int e = 0; e < top_k; ++e) {
        const auto expert = fixtures::random_expert(rng, split);
        records.insert(records.end(), expert.record.begin(), expert.record.end());
        scales.push_back(expert.scales);
        if (omma) {
            for (int c = 0; c < columns; ++c) {
                const auto ref = fp64_expert_column(expert, x.data() + static_cast<std::size_t>(c) * moe::kHidden);
                std::copy(ref.begin(), ref.end(),
                          expected_fp64.begin() + static_cast<std::size_t>(c * top_k + e) * moe::kHidden);
            }
        } else {
            std::vector<const std::uint16_t*> xp(columns);
            std::vector<std::uint16_t*> yp(columns);
            for (int c = 0; c < columns; ++c) {
                xp[c] = x.data() + c * moe::kHidden;
                yp[c] = expected.data() + (c * top_k + e) * moe::kHidden;
            }
            moe::expert_forward(moe::best_cpu_isa(), expert.record.data(), expert.scales,
                                columns, xp.data(), yp.data());
        }
    }
    auto d_records = copy(records);
    auto d_scales = copy(scales);
    auto d_x = copy(x);
    std::vector<std::int32_t> frames(top_k), ids(entries);
    std::iota(frames.begin(), frames.end(), 0);
    for (int i = 0; i < entries; ++i) { ids[i] = i % top_k; }
    auto d_frames = copy(frames);
    auto d_ids = copy(ids);
    DeviceBuffer weights(entries * sizeof(float)), gate(columns * sizeof(float));
    ops::MoeRouting routing{Tensor(d_ids.p, DType::I32, {top_k, columns}),
                            Tensor(weights.p, DType::FP32, {top_k, columns}),
                            Tensor(gate.p, DType::FP32, {columns})};
    DeviceBuffer dispatch_memory(ops::moe_dispatch_bytes(top_k, entries));
    auto dispatch = ops::carve_moe_dispatch(dispatch_memory.p, top_k, entries);
    ops::moe_dispatch(routing, top_k, dispatch, nullptr, context.stream);
    // All frames are resident. host_records satisfies the source contract but is never read.
    ops::MoeExpertSource source{.frame_base = static_cast<const std::uint8_t*>(d_records.p),
                                .frames = static_cast<const std::int32_t*>(d_frames.p),
                                .host_records = static_cast<const std::uint8_t*>(d_records.p),
                                .record_stride = moe::kRecordBytes,
                                .scales = static_cast<const moe::ExpertScales*>(d_scales.p),
                                .omma_narrow = omma};
    DeviceBuffer workspace(
        ops::moe_experts_workspace_bytes(moe::kFlashNextShape, top_k, entries, columns, source.activation, source.wide_a16));
    DeviceBuffer output(static_cast<std::size_t>(entries) * moe::kHidden * sizeof(std::uint16_t));
    Tensor tx(d_x.p, DType::BF16, {moe::kHidden, columns});
    Tensor ty(output.p, DType::BF16, {moe::kHidden, entries});
    auto launch = [&](cudaStream_t stream) {
        ops::moe_experts(tx, dispatch, source, top_k, top_k, workspace.p, ty, stream);
    };
    auto verify = [&] {
        std::vector<std::uint16_t> actual(static_cast<std::size_t>(entries) * moe::kHidden);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.p, output.bytes,
                                    cudaMemcpyDeviceToHost, context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        if (omma) {
            for (int e = 0; e < entries; ++e) {
                double diff2 = 0.0, ref2 = 0.0;
                const std::size_t base = static_cast<std::size_t>(e) * moe::kHidden;
                for (int i = 0; i < moe::kHidden; ++i) {
                    const double y = bf16_to_f64(actual[base + i]);
                    const double r = expected_fp64[base + i];
                    if (!std::isfinite(y)) {
                        throw std::runtime_error("OMMA non-finite output at entry " + std::to_string(e));
                    }
                    diff2 += (y - r) * (y - r);
                    ref2 += r * r;
                }
                const double rel = ref2 > 0.0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
                if (rel > 0x1p-4) {
                    throw std::runtime_error("OMMA FP64 relative L2 " + std::to_string(rel) +
                                             " exceeds 2^-4 at entry " + std::to_string(e));
                }
            }
        } else {
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (actual[i] != expected[i]) {
                    throw std::runtime_error("CPU bit mismatch at output " + std::to_string(i));
                }
            }
        }
    };
    launch(context.stream); // initialize kernel attributes before graph capture
    verify();
    bench::TimedGraph graph;
    graph.capture(context.stream, launch);
    auto replay = [&](cudaStream_t stream) { graph.launch(stream); };
    for (bool cold : {false, true}) {
        const auto timing = cold
            ? bench::measure_cold_launch(replay, flush, context.stream, 10, repeat)
            : bench::measure_graph(graph, context.stream, 10, repeat);
        verify();
        std::printf("T=%d top_k=%d split=%d omma=%d cache=%s graph_nodes=%zu exact=%s "
                    "median_us=%.3f min_us=%.3f p95_us=%.3f\n",
                    columns, top_k, split, omma, cold ? "cold" : "warm", graph.nodes(),
                    omma ? "fp64_rel<=2^-4" : "yes", timing.median_us, timing.min_us, timing.p95_us);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        int repeat = 100, tokens = 0;
        bool omma = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--omma") {
                omma = true;
                continue;
            }
            if ((arg != "--repeat" && arg != "--tokens") || i + 1 >= argc) {
                throw std::invalid_argument("usage: --repeat N --tokens 1|2|4|8 [--omma]");
            }
            const int value = std::stoi(argv[++i]);
            if (arg == "--repeat") { repeat = value; }
            else { tokens = value; }
        }
        if (repeat <= 0 || (tokens != 0 && tokens != 1 && tokens != 2 && tokens != 4 && tokens != 8)) {
            throw std::invalid_argument("repeat must be positive; tokens must be 1, 2, 4 or 8");
        }
        DeviceContext context;
        const std::uint64_t flush_bytes = infernix::bench::default_cold_flush_bytes();
        bench::L2FlushBuffer flush(flush_bytes);
        std::printf("# gpu=%s sm=%d shared_per_sm=%zu timed=moe_experts graph resident "
                    "routing=untimed cold_flush_mib=%zu omma=%d seed=20261015 repeat=%d\n",
                    context.props.name, context.props.multiProcessorCount,
                    context.props.sharedMemPerMultiprocessor,
                    static_cast<std::size_t>(flush_bytes >> 20), omma, repeat);
        for (int columns : {1, 2, 4, 8}) {
            if (tokens != 0 && tokens != columns) { continue; }
            for (bool split : {false, true}) { run(columns, split, omma, repeat, context, flush); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "offloaded_moe_decode_bench: %s\n", error.what());
        return 1;
    }
}
