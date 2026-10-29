// Device-resident moe_experts benchmark (not the unrelated sparse_moe operator).
// Routing/setup and CPU bit-exact validation are outside CUDA-event timing.
// Defaults cover decode, repeated-column verification and the eight-column path.
#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/offloaded_moe_fixtures.h"

#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace moe = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;

namespace {
template <class T> DeviceBuffer copy(const std::vector<T>& values) {
    DeviceBuffer out(values.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(out.p, values.data(), out.bytes, cudaMemcpyHostToDevice));
    return out;
}

void run(int columns, bool split, int repeat, DeviceContext& context, bench::L2FlushBuffer& flush) {
    constexpr int top_k = 10;
    const int entries = columns * top_k;
    std::mt19937 rng(20261015);
    const auto x = fixtures::random_activations(rng, columns);
    std::vector<std::uint8_t> records;
    std::vector<moe::ExpertScales> scales;
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(entries) * moe::kHidden);
    for (int e = 0; e < top_k; ++e) {
        const auto expert = fixtures::random_expert(rng, split);
        records.insert(records.end(), expert.record.begin(), expert.record.end());
        scales.push_back(expert.scales);
        std::vector<const std::uint16_t*> xp(columns);
        std::vector<std::uint16_t*> yp(columns);
        for (int c = 0; c < columns; ++c) {
            xp[c] = x.data() + c * moe::kHidden;
            yp[c] = expected.data() + (c * top_k + e) * moe::kHidden;
        }
        moe::expert_forward(moe::best_cpu_isa(), expert.record.data(), expert.scales,
                            columns, xp.data(), yp.data());
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
                                .scales = static_cast<const moe::ExpertScales*>(d_scales.p)};
    DeviceBuffer workspace(
        ops::moe_experts_workspace_bytes(moe::kFlashNextShape, top_k, entries, columns, source.activation, source.wide_a16));
    DeviceBuffer output(expected.size() * sizeof(std::uint16_t));
    Tensor tx(d_x.p, DType::BF16, {moe::kHidden, columns});
    Tensor ty(output.p, DType::BF16, {moe::kHidden, entries});
    auto launch = [&](cudaStream_t stream) {
        ops::moe_experts(tx, dispatch, source, top_k, top_k, workspace.p, ty, stream);
    };
    auto verify = [&] {
        std::vector<std::uint16_t> actual(expected.size());
        CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.p, output.bytes,
                                    cudaMemcpyDeviceToHost, context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                throw std::runtime_error("CPU bit mismatch at output " + std::to_string(i));
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
        std::printf("T=%d top_k=%d split=%d cache=%s graph_nodes=%zu exact=yes "
                    "median_us=%.3f min_us=%.3f p95_us=%.3f\n",
                    columns, top_k, split, cold ? "cold" : "warm", graph.nodes(),
                    timing.median_us, timing.min_us, timing.p95_us);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        int repeat = 100, tokens = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg != "--repeat" && arg != "--tokens") || i + 1 >= argc) {
                throw std::invalid_argument("usage: --repeat N --tokens 1|4|8 (default: all)");
            }
            const int value = std::stoi(argv[++i]);
            if (arg == "--repeat") { repeat = value; }
            else { tokens = value; }
        }
        if (repeat <= 0 || (tokens != 0 && tokens != 1 && tokens != 4 && tokens != 8)) {
            throw std::invalid_argument("repeat must be positive; tokens must be 1, 4 or 8");
        }
        DeviceContext context;
        const std::uint64_t flush_bytes = infernix::bench::default_cold_flush_bytes();
        bench::L2FlushBuffer flush(flush_bytes);
        std::printf("# gpu=%s sm=%d shared_per_sm=%zu timed=moe_experts graph resident "
                    "routing=untimed cold_flush_mib=%zu seed=20261015 repeat=%d\n",
                    context.props.name, context.props.multiProcessorCount,
                    context.props.sharedMemPerMultiprocessor,
                    static_cast<std::size_t>(flush_bytes >> 20), repeat);
        for (int columns : {1, 4, 8}) {
            if (tokens != 0 && tokens != columns) { continue; }
            for (bool split : {false, true}) { run(columns, split, repeat, context, flush); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "offloaded_moe_decode_bench: %s\n", error.what());
        return 1;
    }
}
