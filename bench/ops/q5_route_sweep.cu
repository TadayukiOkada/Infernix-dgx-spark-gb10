// Q5 A16 linear route sweep benchmark for GB10 retuning.
//
// Times individual registered Q5 routes across token bands with cold-L2
// flushing and verifies numerical agreement against the production route.

#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {

struct RouteCandidate {
    const char* name;
    Q5Launch launch;
    int min_t    = 1;
    int max_t    = 0; // 0 = unbounded
    int static_k = 0; // 0 = any K
};

void launch_production(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    ops::linear(x, w, out, stream);
}

const RouteCandidate kCandidates[] = {
    {"production", launch_production, 1, 0, 0},

    // Direct SIMT (static-K & general)
    {"launch_q5_a16_direct_r1_t1_w4_k5120", launch_q5_a16_direct_r1_t1_w4_k5120, 1, 1, 5120},
    {"launch_q5_a16_direct_r1_t2_w4_k5120", launch_q5_a16_direct_r1_t2_w4_k5120, 2, 2, 5120},
    {"launch_q5_a16_direct_r1_t3_w4_k5120", launch_q5_a16_direct_r1_t3_w4_k5120, 3, 3, 5120},
    {"launch_q5_a16_direct_r1_t1_w4_k6144", launch_q5_a16_direct_r1_t1_w4_k6144, 1, 1, 6144},
    {"launch_q5_a16_direct_r1_t2_w2_k6144", launch_q5_a16_direct_r1_t2_w2_k6144, 2, 2, 6144},
    {"launch_q5_a16_direct_r1_t3_w2_k6144", launch_q5_a16_direct_r1_t3_w2_k6144, 3, 3, 6144},
    {"launch_q5_a16_direct_r1_t1_w4_k17408", launch_q5_a16_direct_r1_t1_w4_k17408, 1, 1, 17408},
    {"launch_q5_a16_direct_r1_t2_w2_k17408", launch_q5_a16_direct_r1_t2_w2_k17408, 2, 2, 17408},
    {"launch_q5_a16_direct_r1_t3_w2_k17408", launch_q5_a16_direct_r1_t3_w2_k17408, 3, 3, 17408},
    {"launch_q5_a16_direct_r1_t4_w2_k17408", launch_q5_a16_direct_r1_t4_w2_k17408, 4, 4, 17408},
    {"launch_q5_a16_direct_r2_t4_w2_g8_b4", launch_q5_a16_direct_r2_t4_w2_g8_b4, 1, 32, 0},
    {"launch_q5_a16_direct_r2_t4_w4_g4_b4", launch_q5_a16_direct_r2_t4_w4_g4_b4, 1, 32, 0},

    // Sliced-K MMA
    {"launch_q5_a16_sliced_r16_t8_capacity4", launch_q5_a16_sliced_r16_t8_capacity4, 1, 4, 0},
    {"launch_q5_a16_sliced_r16_t8_w4_s2", launch_q5_a16_sliced_r16_t8_w4_s2, 1, 64, 0},
    {"launch_q5_a16_sliced_r16_t16_w4_s2", launch_q5_a16_sliced_r16_t16_w4_s2, 1, 128, 0},
    {"launch_q5_a16_sliced_r16_t16_w2_s2", launch_q5_a16_sliced_r16_t16_w2_s2, 1, 128, 0},
    {"launch_q5_a16_sliced_r16_t24_w4_s2", launch_q5_a16_sliced_r16_t24_w4_s2, 1, 192, 0},
    {"launch_q5_a16_sliced_r16_t32_w4_s2", launch_q5_a16_sliced_r16_t32_w4_s2, 1, 256, 0},
    {"launch_q5_a16_sliced_r32_t16_w4_s2", launch_q5_a16_sliced_r32_t16_w4_s2, 1, 128, 0},
    {"launch_q5_a16_sliced_r32_t24_w4_s2_pairwise", launch_q5_a16_sliced_r32_t24_w4_s2_pairwise, 1, 192, 0},
    {"launch_q5_a16_sliced_r32_t32_w2_s2", launch_q5_a16_sliced_r32_t32_w2_s2, 1, 576, 0},
    {"launch_q5_a16_sliced_r32_t32_w4_s1", launch_q5_a16_sliced_r32_t32_w4_s1, 1, 576, 0},
    {"launch_q5_a16_sliced_r32_t32_w4_s2", launch_q5_a16_sliced_r32_t32_w4_s2, 1, 576, 0},
    {"launch_q5_a16_sliced_r32_t64_w2_s1", launch_q5_a16_sliced_r32_t64_w2_s1, 1, 576, 0},
    {"launch_q5_a16_sliced_r32_t64_w2_s2", launch_q5_a16_sliced_r32_t64_w2_s2, 1, 576, 0},

    // Full MMA (including new small/higher-occupancy GB10 BlockK=64 schedules)
    {"launch_q5_a16_mma_r32_t16_k64", launch_q5_a16_mma_r32_t16_k64, 4, 64, 0},
    {"launch_q5_a16_mma_r32_t32_k64", launch_q5_a16_mma_r32_t32_k64, 8, 128, 0},
    {"launch_q5_a16_mma_r32_t48_k64", launch_q5_a16_mma_r32_t48_k64, 16, 192, 0},
    {"launch_q5_a16_mma_r32_t64_k64", launch_q5_a16_mma_r32_t64_k64, 16, 0, 0},
    {"launch_q5_a16_mma_r64_t32_k64", launch_q5_a16_mma_r64_t32_k64, 8, 128, 0},
    {"launch_q5_a16_mma_r64_t48_k64", launch_q5_a16_mma_r64_t48_k64, 16, 192, 0},
    {"launch_q5_a16_mma_r64_t64_k64", launch_q5_a16_mma_r64_t64_k64, 16, 0, 0},
    {"launch_q5_a16_mma_r64_t96_k64", launch_q5_a16_mma_r64_t96_k64, 32, 0, 0},
    {"launch_q5_a16_mma_r64_t128_k64_m2", launch_q5_a16_mma_r64_t128_k64_m2, 32, 0, 0},
    {"launch_q5_a16_mma_r32_t32_k128_s2_a2", launch_q5_a16_mma_r32_t32_k128_s2_a2, 8, 0, 0},
    {"launch_q5_a16_mma_r32_t64_k128_s2_a1", launch_q5_a16_mma_r32_t64_k128_s2_a1, 16, 0, 0},
    {"launch_q5_a16_mma_r32_t128", launch_q5_a16_mma_r32_t128, 32, 0, 0},
    {"launch_q5_a16_mma_r32_t128_k128_s2_a1", launch_q5_a16_mma_r32_t128_k128_s2_a1, 32, 0, 0},
    {"launch_q5_a16_mma_r64_t64_k128_s1_a1", launch_q5_a16_mma_r64_t64_k128_s1_a1, 16, 0, 0},
    {"launch_q5_a16_mma_r64_t64_k128_s2_a1", launch_q5_a16_mma_r64_t64_k128_s2_a1, 16, 0, 0},
    {"launch_q5_a16_mma_r64_t80_k128_s1_a1", launch_q5_a16_mma_r64_t80_k128_s1_a1, 24, 0, 0},
    {"launch_q5_a16_mma_r64_t96_k128_s1_a1", launch_q5_a16_mma_r64_t96_k128_s1_a1, 32, 0, 0},
    {"launch_q5_a16_mma_r64_t128", launch_q5_a16_mma_r64_t128, 32, 0, 0},
    {"launch_q5_a16_mma_r64_t128_k128_s1_a1", launch_q5_a16_mma_r64_t128_k128_s1_a1, 32, 0, 0},
};

std::vector<int> parse_tokens(const std::string& spec) {
    std::vector<int> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const auto colon = item.find(':');
        if (colon == std::string::npos) {
            out.push_back(std::stoi(item));
        } else {
            const auto colon2 = item.find(':', colon + 1);
            const int start   = std::stoi(item.substr(0, colon));
            const int end     = std::stoi(item.substr(
                colon + 1, colon2 == std::string::npos ? std::string::npos : colon2 - colon - 1));
            const int step    = colon2 == std::string::npos ? 1 : std::stoi(item.substr(colon2 + 1));
            for (int t = start; t <= end; t += step) out.push_back(t);
        }
    }
    return out;
}

bool check_numeric_agreement(const std::vector<__nv_bfloat16>& got,
                             const std::vector<__nv_bfloat16>& ref) {
    if (got.size() != ref.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = __bfloat162float(got[i]);
        const float r = __bfloat162float(ref[i]);
        if (!std::isfinite(g) || !std::isfinite(r)) return false;
        const float diff = std::fabs(g - r);
        const float tol  = std::max(0.0625f, 0.03125f * std::fabs(r));
        if (diff > tol) return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int n                = 5120;
        int k                = 6144;
        std::string tok_spec = "1,2,4,8,16,24,32,48,64,96,128,192,256,384,512";
        int warmup           = 3;
        int repeat           = 15;
        int flush_mib        = 32;
        std::string csv_out;
        std::string filter;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
            else if (a == "--k" && i + 1 < argc) k = std::stoi(argv[++i]);
            else if (a == "--tokens" && i + 1 < argc) tok_spec = argv[++i];
            else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
            else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
            else if (a == "--flush-mib" && i + 1 < argc) flush_mib = std::stoi(argv[++i]);
            else if (a == "-o" && i + 1 < argc) csv_out = argv[++i];
            else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
        }

        const auto tokens_list = parse_tokens(tok_spec);
        int max_t              = 1;
        for (int t : tokens_list) max_t = std::max(max_t, t);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::size_t flush_bytes = static_cast<std::size_t>(flush_mib) << 20;
        bench::L2FlushBuffer flush(flush_bytes);

        auto qw = bench::make_row_split_weight(QType::Q5_G64_FP16, n, k, (k + 127) / 128 * 128,
                                               0x5eed1235ULL);
        const std::size_t out_count = static_cast<std::size_t>(n) * max_t;
        DeviceBuffer x_storage(static_cast<std::size_t>(k) * max_t * 2);
        DeviceBuffer out_storage(out_count * 2);
        DeviceBuffer ref_storage(out_count * 2);
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(x_storage.p),
                                               static_cast<std::size_t>(k) * max_t, 0xa5a5ULL,
                                               -0.5F, 0.5F));
        CUDA_CHECK(cudaDeviceSynchronize());

        const Tensor x   = Tensor(x_storage.p, DType::BF16, {k, max_t});
        const Tensor out = Tensor(out_storage.p, DType::BF16, {n, max_t});
        const Tensor ref = Tensor(ref_storage.p, DType::BF16, {n, max_t});

        cudaEvent_t ev_start{}, ev_stop{};
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        std::ofstream csv;
        if (!csv_out.empty()) {
            csv.open(csv_out);
            csv << "N,K,T,route,median_us,min_us,p95_us,gbs,tflops,bit_exact,t_invariant\n";
        }

        // Reference at minimum valid T (4 for Vision n=1152 shapes, 1 otherwise)
        const int ref_t = (n == 1152) ? 4 : 1;
        std::vector<__nv_bfloat16> t1_ref(static_cast<std::size_t>(n) * ref_t);
        if (max_t >= ref_t) {
            Tensor x1   = x;
            x1.ne[1]    = ref_t;
            Tensor y1   = ref;
            y1.ne[1]    = ref_t;
            launch_production(x1, qw.weight, y1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(t1_ref.data(), y1.data, n * ref_t * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToHost));
        }

        const std::size_t groups     = static_cast<std::size_t>(n) * ((k + 63) / 64);
        const std::size_t code_bytes = groups * Q5RowSplitStorage::kCodeBytesPerGroup;
        const std::size_t high_bytes = groups * Q5RowSplitStorage::kHighBytesPerGroup;

        std::printf("=== Q5 Route Sweep: N=%d K=%d (flush=%d MiB, warmup=%d, repeat=%d) ===\n", n,
                    k, flush_mib, warmup, repeat);

        for (int t : tokens_list) {
            if (n == 1152 && t % 4 != 0) continue;
            Tensor xs   = x;
            xs.ne[1]    = t;
            Tensor ys   = out;
            ys.ne[1]    = t;
            Tensor rs   = ref;
            rs.ne[1]    = t;
            const std::size_t out_elems = static_cast<std::size_t>(n) * t;
            const std::size_t out_bytes = out_elems * sizeof(__nv_bfloat16);

            launch_production(xs, qw.weight, rs, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<__nv_bfloat16> ref_host(out_elems);
            CUDA_CHECK(cudaMemcpy(ref_host.data(), rs.data, out_bytes, cudaMemcpyDeviceToHost));
            std::vector<__nv_bfloat16> got(out_elems);

            const double total_bytes =
                static_cast<double>(code_bytes + high_bytes) + static_cast<double>(groups) * 2.0 +
                (static_cast<double>(k) * t + static_cast<double>(n) * t) * 2.0;
            const double flops = 2.0 * static_cast<double>(n) * k * t;

            double best_us         = 1e30;
            const char* best_route = "";
            double prod_us         = 0.0;

            for (const auto& cand : kCandidates) {
                if (t < cand.min_t) continue;
                if (cand.max_t > 0 && t > cand.max_t) continue;
                if (cand.static_k > 0 && k != cand.static_k) continue;
                if (!filter.empty() && std::string(cand.name).find(filter) == std::string::npos &&
                    std::string(cand.name) != "production")
                    continue;

                bool ok = true;
                try {
                    CUDA_CHECK(cudaMemsetAsync(ys.data, 0, out_bytes, stream));
                    cand.launch(xs, qw.weight, ys, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } catch (...) {
                    cudaGetLastError();
                    ok = false;
                }
                if (!ok) continue;

                CUDA_CHECK(cudaMemcpy(got.data(), ys.data, out_bytes, cudaMemcpyDeviceToHost));
                const bool numeric_ok = check_numeric_agreement(got, ref_host);
                const bool t_invariant =
                    t >= ref_t &&
                    std::memcmp(got.data(), t1_ref.data(), n * sizeof(__nv_bfloat16)) == 0;
                if (!numeric_ok) continue;

                for (int w_i = 0; w_i < warmup; ++w_i) {
                    bench::flush_l2(flush, stream);
                    cand.launch(xs, qw.weight, ys, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<double> samples_us;
                samples_us.reserve(repeat);
                for (int r_i = 0; r_i < repeat; ++r_i) {
                    bench::flush_l2(flush, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaEventRecord(ev_start, stream));
                    cand.launch(xs, qw.weight, ys, stream);
                    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
                    CUDA_CHECK(cudaEventSynchronize(ev_stop));
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                    samples_us.push_back(static_cast<double>(ms) * 1000.0);
                }
                std::sort(samples_us.begin(), samples_us.end());
                const double med_us = samples_us[samples_us.size() / 2];
                const double min_us = samples_us.front();
                const double p95_us =
                    samples_us[std::min(samples_us.size() - 1, (samples_us.size() * 95) / 100)];
                const double gbs    = (total_bytes / (med_us * 1e-6)) / 1e9;
                const double tflops = (flops / (med_us * 1e-6)) / 1e12;

                if (std::strcmp(cand.name, "production") == 0) {
                    prod_us = med_us;
                } else if (med_us < best_us) {
                    best_us    = med_us;
                    best_route = cand.name;
                }

                if (csv.is_open()) {
                    csv << n << "," << k << "," << t << "," << cand.name << "," << med_us << ","
                        << min_us << "," << p95_us << "," << gbs << "," << tflops << ","
                        << (numeric_ok ? 1 : 0) << "," << (t_invariant ? 1 : 0) << "\n";
                }
            }

            const double gain_pct = prod_us > 0.0 ? (prod_us - best_us) / prod_us * 100.0 : 0.0;
            std::printf("  T=%4d | prod: %8.2f us | best: %8.2f us (%+6.2f%%) -> %s\n", t, prod_us,
                        best_us, -gain_pct, best_route);
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
