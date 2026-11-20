#include "ops/hyper_connection/hyper_connection_up_collapse.h"

#include "ops/linear/q4/q4_instances.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams = 4;

// The collapse's own expressions (hyper_connection.cu collapse_kernel), so the fused result is
// bit-identical to the composed route's.
__device__ __forceinline__ float hc_sigmoid(float x) { return 1.0F / (1.0F + expf(-x)); }

// A CTA's BlockRows rows are the 4 streams of Features consecutive features: local row
// s * Features + j is W_up row s * hidden + d0 + j, d0 = blockIdx.x * Features.
template <int Features>
struct HcStreamRows {
    static constexpr int kOutputRowsPerCta = Features;
    int hidden;

    int output_rows(int) const { return hidden; }

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return (local_row / Features) * hidden + row0 + local_row % Features;
    }
};

// Never stored to: the epilogue writes x itself.
struct HcNoOutput {};

struct HcCollapseEpilogue {
    const bf16* normalized; // [S*H, T]
    bf16* x;                // [H, T]
    int hidden;

    // The tile of BF16-rounded u values, token-major with a padded pitch: [BlockTokens][BlockRows + 8].
    template <class Cfg>
    static constexpr int kSharedBytes = Cfg::kBlockTokens * (Cfg::kBlockRows + 8) * 2;

    template <class Cfg, bool Full, class Tile, class Acc>
    __device__ __forceinline__ void finish_tile(const Tile&, unsigned char* scratch, const Acc& acc, int m0, int n0,
                                                int, int n) const {
        constexpr int BM = Cfg::kBlockRows, BN = Cfg::kBlockTokens, Pitch = BM + 8;
        constexpr int Features = BM / kStreams;
        static_assert(Features % 4 == 0);
        constexpr int Quads = Features / 4;
        auto* u = reinterpret_cast<bf16*>(scratch);
        const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;
        __syncthreads(); // the operand tiles are no longer read
#pragma unroll
        for (int mi = 0; mi < Cfg::kMmaRows; ++mi) {
            const int r0 = wm * Cfg::kWarpRows + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < Cfg::kMmaTokens; ++ni) {
                const int c0 = wn * Cfg::kWarpTokens + ni * 8 + 2 * lid;
                // u as the composed route stores it: one BF16 rounding of the FP32 accumulator.
                u[c0 * Pitch + r0]           = __float2bfloat16_rn(acc[mi][ni][0]);
                u[(c0 + 1) * Pitch + r0]     = __float2bfloat16_rn(acc[mi][ni][1]);
                u[c0 * Pitch + r0 + 8]       = __float2bfloat16_rn(acc[mi][ni][2]);
                u[(c0 + 1) * Pitch + r0 + 8] = __float2bfloat16_rn(acc[mi][ni][3]);
            }
        }
        __syncthreads();
        const bool full_tokens  = Full || (n0 + BN <= n);
        const std::size_t width = static_cast<std::size_t>(kStreams) * hidden;
        if constexpr (Features % 8 == 0) {
            constexpr int Octets = Features / 8;
            for (int item = tid; item < Octets * BN; item += Cfg::kThreads) {
                const int j = (item % Octets) * 8, c = item / Octets;
                const int t = n0 + c, d = m0 + j;
                if (!full_tokens && t >= n) { continue; }
                const std::size_t base_t = static_cast<std::size_t>(t) * width + d;
                uint4 nv[kStreams];
                uint4 uv[kStreams];
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    nv[s] = *reinterpret_cast<const uint4*>(&normalized[base_t + static_cast<std::size_t>(s) * hidden]);
                    uv[s] = *reinterpret_cast<const uint4*>(&u[c * Pitch + s * Features + j]);
                }
                float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const auto* up = reinterpret_cast<const __nv_bfloat162*>(&uv[s]);
                    const auto* np = reinterpret_cast<const __nv_bfloat162*>(&nv[s]);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        sum[2 * p] += hc_sigmoid(__bfloat162float(up[p].x)) * __bfloat162float(np[p].x);
                        sum[2 * p + 1] += hc_sigmoid(__bfloat162float(up[p].y)) * __bfloat162float(np[p].y);
                    }
                }
                constexpr float kInvS = 1.0F / static_cast<float>(kStreams);
                uint4 out_vec;
                auto* op = reinterpret_cast<__nv_bfloat162*>(&out_vec);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    op[p] = __floats2bfloat162_rn(sum[2 * p] * kInvS, sum[2 * p + 1] * kInvS);
                }
                *reinterpret_cast<uint4*>(&x[static_cast<std::size_t>(t) * hidden + d]) = out_vec;
            }
        } else {
            for (int item = tid; item < Quads * BN; item += Cfg::kThreads) {
                const int j = (item % Quads) * 4, c = item / Quads;
                const int t = n0 + c, d = m0 + j;
                if (!full_tokens && t >= n) { continue; }
                const std::size_t base_t = static_cast<std::size_t>(t) * width + d;
                uint2 nv[kStreams];
                uint2 uv[kStreams];
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    nv[s] = *reinterpret_cast<const uint2*>(&normalized[base_t + static_cast<std::size_t>(s) * hidden]);
                    uv[s] = *reinterpret_cast<const uint2*>(&u[c * Pitch + s * Features + j]);
                }
                float sum0 = 0.0F, sum1 = 0.0F, sum2 = 0.0F, sum3 = 0.0F;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const __nv_bfloat162 u01 = *reinterpret_cast<const __nv_bfloat162*>(&uv[s].x);
                    const __nv_bfloat162 u23 = *reinterpret_cast<const __nv_bfloat162*>(&uv[s].y);
                    const __nv_bfloat162 n01 = *reinterpret_cast<const __nv_bfloat162*>(&nv[s].x);
                    const __nv_bfloat162 n23 = *reinterpret_cast<const __nv_bfloat162*>(&nv[s].y);
                    sum0 += hc_sigmoid(__bfloat162float(u01.x)) * __bfloat162float(n01.x);
                    sum1 += hc_sigmoid(__bfloat162float(u01.y)) * __bfloat162float(n01.y);
                    sum2 += hc_sigmoid(__bfloat162float(u23.x)) * __bfloat162float(n23.x);
                    sum3 += hc_sigmoid(__bfloat162float(u23.y)) * __bfloat162float(n23.y);
                }
                const __nv_bfloat162 out01 =
                    __floats2bfloat162_rn(sum0 / static_cast<float>(kStreams), sum1 / static_cast<float>(kStreams));
                const __nv_bfloat162 out23 =
                    __floats2bfloat162_rn(sum2 / static_cast<float>(kStreams), sum3 / static_cast<float>(kStreams));
                *reinterpret_cast<uint2*>(&x[static_cast<std::size_t>(t) * hidden + d]) =
                    uint2{*reinterpret_cast<const unsigned*>(&out01), *reinterpret_cast<const unsigned*>(&out23)};
            }
        }
    }
};

template <class Schedule>
void launch(const Tensor& m, const Weight& up, const Tensor& normalized, Tensor& x, cudaStream_t stream) {
    constexpr int Features = Schedule::kBlockRows / kStreams;
    const int hidden       = x.ne[0];
    launch_q8_a16_mma<Schedule, HcStreamRows<Features>>(
        q8_linear_operands(m, up), HcNoOutput{},
        HcCollapseEpilogue{static_cast<const bf16*>(normalized.data), static_cast<bf16*>(x.data), hidden}, stream,
        HcStreamRows<Features>{hidden});
}

template <class Schedule>
void launch_q4(const Tensor& m, const Weight& up, const Tensor& normalized, Tensor& x, cudaStream_t stream) {
    constexpr int Features = Schedule::kBlockRows / kStreams;
    const int hidden       = x.ne[0];
    launch_q4_a16_mma<Schedule>(
        q4_linear_operands(m, up), HcNoOutput{},
        HcCollapseEpilogue{static_cast<const bf16*>(normalized.data), static_cast<bf16*>(x.data), hidden}, stream,
        HcStreamRows<Features>{hidden});
}

} // namespace

bool hc_up_collapse_supported(const Weight& up, std::int32_t hidden, std::int32_t streams,
                              std::int32_t columns) noexcept {
    // From 5 columns, fuse up + collapse so u stays in shared memory instead of round-tripping
    // [streams * hidden, T] through global memory. The composed mix keeps 1..4 (its own fused route);
    // 5..16 take the 16-token tile, whose measured win over the separate linear and collapse is what
    // moved the boundary from 17 (where the 48-token tile lost 2..4 % at 5 and 9..16 in the
    // 2026-10-30 GB10 retune) down to the fused mix's edge (infernix_hyper_connection_bench, GB10).
    if (streams != kStreams || hidden <= 0 || hidden % 16 != 0 || columns <= 4 || up.qdata == nullptr ||
        up.n != streams * hidden || up.layout != QuantLayout::RowSplit || up.scales == nullptr ||
        up.padded_shape[1] % 128 != 0 || up.padded_shape[1] < up.k) {
        return false;
    }
    if (up.qtype == QType::Q8_G32_FP16) return up.k % 8 == 0;
    if (up.qtype == QType::Q4_G64_FP16) return up.k % 64 == 0;
    return false;
}

void hc_up_collapse(const Tensor& m, const Weight& up, const Tensor& normalized, std::int32_t streams, Tensor& x,
                    cudaStream_t stream) {
    const std::int32_t T = m.ne[1];
    if (!hc_up_collapse_supported(up, x.ne[0], streams, T) || m.ne[0] != up.k || normalized.ne[1] != T ||
        normalized.ne[0] != up.n || x.ne[1] != T) {
        throw std::invalid_argument("hyper_connection: fused up + collapse shapes disagree");
    }
    const int hidden = x.ne[0];
    if (up.qtype == QType::Q4_G64_FP16) {
        if (T <= 32) return launch_q4<q4_instances::MmaR32T32K64Wr16Wt16S3A3B2>(m, up, normalized, x, stream);
        if (T <= 64) return launch_q4<q4_instances::MmaR32T64K64Wr16Wt32S2A2B2>(m, up, normalized, x, stream);
        if (hidden % 32 == 0) {
            using MmaR128T64 = Q4A16MmaSchedule<128, 64, 64, 32, 32, 2, 2, Q4MmaFragmentPipeline::Serial,
                                                Cache::cg, Cache::cg, Q4ScaleLoad::Pair32, 2>;
            return launch_q4<MmaR128T64>(m, up, normalized, x, stream);
        }
        return launch_q4<q4_instances::MmaR64T64K64Wr32Wt16S2A2B2>(m, up, normalized, x, stream);
    }
    // The linear family's measured argmin for this shape at 9..32 columns rides the same tile: the
    // 32-row K64 one, whose unpredicated K and 4 CTAs/SM the fused route needs to keep its bandwidth.
    // The 64-row K128 tile of the 17..48 band runs ~180 GB/s here and loses even to the composed
    // pair it replaces (infernix_hyper_connection_bench, GB10, 2026-11-05).
    if (T <= 16) return launch<q8_instances::MmaR32T16>(m, up, normalized, x, stream);
    if (T <= 24) return launch<q8_instances::MmaR32T24>(m, up, normalized, x, stream);
    if (T <= 32) return launch<q8_instances::MmaR32T16>(m, up, normalized, x, stream);
    if (T <= 48) return launch<q8_instances::MmaR32T24>(m, up, normalized, x, stream);
    if (T <= 64) {
        if (hidden % 32 == 0) {
            return launch<q8_instances::MmaR64x32T64K64A2>(m, up, normalized, x, stream);
        }
        return launch<q8_instances::MmaR32T64>(m, up, normalized, x, stream);
    }
    if (T <= 128) {
        if (hidden % 32 == 0) {
            return launch<q8_instances::MmaR128x32T64K64A2>(m, up, normalized, x, stream);
        }
        return launch<q8_instances::MmaR32T64>(m, up, normalized, x, stream);
    }
    if (T <= 512) {
        if (hidden % 32 == 0) {
            return launch<q8_instances::MmaR64x32T128K64A2>(m, up, normalized, x, stream);
        }
        return launch<q8_instances::MmaR64T128>(m, up, normalized, x, stream);
    }
    if (hidden % 32 == 0) {
        return launch<q8_instances::MmaR128x64T128K64A1>(m, up, normalized, x, stream);
    }
    return launch<q8_instances::MmaR64T128>(m, up, normalized, x, stream);
}

} // namespace infernix::ops::detail
