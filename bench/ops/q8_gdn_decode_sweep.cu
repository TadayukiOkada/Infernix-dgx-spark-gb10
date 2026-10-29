// Exact Q8 GDN decode schedule screening. No production dispatch changes.
#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"

#include <string>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {
constexpr int kRows = 16384;
constexpr int kK = 2560;

template <int Rows, int Warps, Q8StreamX Stage, int Residency = 2>
void candidate(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Keep KSplit=1, chunk/FMA/reduction order, and production column passes.
    if (x.ne[1] <= 4)
        launch_q8_a16_stream_linear<Q8A16StreamSchedule<kK, Rows, 1, Warps, 4, 4, Stage, Residency>>(
            x, w, out, stream);
    else
        launch_q8_a16_stream_linear<Q8A16StreamSchedule<kK, Rows, 1, Warps, 8, 8, Stage, Residency>>(
            x, w, out, stream);
}

struct Route { const char* name; Q8Launch launch; };
const Route routes[] = {
    {"production", [](const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
         ops::linear(x, w, out, s);
     }},
    {"global_r2w8b2", candidate<2, 8, Q8StreamX::Global>},
    {"global_r2w4b2", candidate<2, 4, Q8StreamX::Global>},
    {"global_r1w8b2", candidate<1, 8, Q8StreamX::Global>},
    {"global_r1w4b2", candidate<1, 4, Q8StreamX::Global>},
    {"global_r2w8b1", candidate<2, 8, Q8StreamX::Global, 1>},
    {"shared_r2w8b2", candidate<2, 8, Q8StreamX::Shared>},
    {"shared_r2w4b2", candidate<2, 4, Q8StreamX::Shared>},
    {"shared_r1w8b2", candidate<1, 8, Q8StreamX::Shared>},
    {"shared_r1w4b2", candidate<1, 4, Q8StreamX::Shared>},
};

std::vector<std::uint16_t> download(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<std::uint16_t> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), buffer.p, count * 2, cudaMemcpyDeviceToHost));
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        int repeat = 50, warmup = 5;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg != "--repeat" && arg != "--warmup") || i + 1 == argc)
                throw std::invalid_argument("usage: q8_gdn_decode_sweep [--repeat N] [--warmup N]");
            const std::string text = argv[++i];
            std::size_t used = 0;
            const int value = std::stoi(text, &used);
            if (used != text.size()) throw std::invalid_argument("invalid integer " + text);
            (arg == "--repeat" ? repeat : warmup) = value;
        }
        if (repeat <= 0 || warmup < 0) throw std::invalid_argument("invalid repeat/warmup");
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::fprintf(stderr,
                     "GPU=%s SMs=%d CC=%d.%d; cold eviction=%zu MiB; exact production oracle\n",
                     prop.name, prop.multiProcessorCount, prop.major, prop.minor,
                     static_cast<std::size_t>(infernix::bench::resolved_cold_flush_bytes() >> 20));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto weight = bench::make_row_split_weight(QType::Q8_G32_FP16, kRows, kK, kK, 20261015U);
        auto input = bench::make_bf16(kK * 8, 20261016U);
        DeviceBuffer output(kRows * 8 * 2), reference(kRows * 8 * 2);
        bench::L2FlushBuffer flush(infernix::bench::default_cold_flush_bytes());
        std::vector<std::uint16_t> column;
        std::printf("N,K,T,route,cache,exact,median_us,min_us,p95_us\n");
        for (int t = 1; t <= 8; ++t) {
            Tensor x(input.p, DType::BF16, {kK, t});
            Tensor out(output.p, DType::BF16, {kRows, t});
            Tensor ref(reference.p, DType::BF16, {kRows, t});
            ops::linear(x, weight.weight, ref, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto expected = download(reference, kRows * t);
            if (t == 1) column = expected;
            if (!std::equal(column.begin(), column.end(), expected.begin()))
                throw std::runtime_error("production column invariance failure");
            for (const auto& route : routes) {
                const auto check = [&] {
                    if (download(output, kRows * t) != expected)
                        throw std::runtime_error(std::string(route.name) + " bit mismatch T=" + std::to_string(t));
                };
                // Initialize launch attributes and check every output before capture.
                CUDA_CHECK(cudaMemsetAsync(output.p, 0xff, output.bytes, stream));
                route.launch(x, weight.weight, out, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                check();
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) { route.launch(x, weight.weight, out, s); });
                for (const bool cold : {false, true}) {
                    const auto timing = cold ? bench::measure_cold_graph(graph, flush, stream, warmup, repeat)
                                             : bench::measure_graph(graph, stream, warmup, repeat);
                    check();
                    std::printf("%d,%d,%d,%s,%s,yes,%.3f,%.3f,%.3f\n", kRows, kK, t, route.name,
                                cold ? "cold" : "warm", timing.median_us, timing.min_us, timing.p95_us);
                }
            }
            std::fflush(stdout);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q8_gdn_decode_sweep: %s\n", error.what());
        return 1;
    }
}
