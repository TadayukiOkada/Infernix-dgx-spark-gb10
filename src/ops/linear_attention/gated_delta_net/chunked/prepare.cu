#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <stdexcept>

namespace infernix::ops::detail::gated_delta_net::chunked {
namespace {

// Fp16Packet (GdnChunkArithmetic::Fp16Accumulate): the packet's 16-bit elements hold FP16.

// One CTA per (value head, chunk): the historical route. It loads the group's shared Q/K rows
// redundantly; it still wins for group sizes <= 2 (measured on GB10 sm_121a at T=2048, cold L2:
// group 1 334.5 vs 373.6 us, group 2 492.2 vs 524.0 us), where the CTA count matters more than
// the redundant reads, which stay L2-resident.
struct PrepareShared {
    QkChunk qk;
    ControlChunk control;
    float prefix[kChunkSize];
    float beta[kChunkSize];
    float lower[kChunkSize * kChunkSize];
};

template <bool Normalize, bool Fp16Packet = false>
__global__
__launch_bounds__(256, 2) void prepare_head_kernel(Arguments args,
                                                   QkChunk* __restrict__ qk_workspace,
                                                   ControlChunk* __restrict__ control_workspace) {
    __shared__ PrepareShared sm;
    const int tid = threadIdx.x, lane = tid & 31;
    const int head = blockIdx.x % args.value_heads, chunk = blockIdx.x / args.value_heads;
    const int start = chunk * kChunkSize, length = min(kChunkSize, args.tokens - start);
    const int group = args.value_heads / args.qk_heads, qh = head / group;
    const int row = tid / 16, col = (tid & 15) * 8;
    const auto qo = (static_cast<std::int64_t>(start + row) * args.qk_heads + qh) * kStateDim;

    float q[8]{}, k[8]{};
    if (row < length) {
        const uint4 qp = load_vec<uint4>(args.q + qo + col);
        const uint4 kp = load_vec<uint4>(args.k + qo + col);
        const auto* qb = reinterpret_cast<const __nv_bfloat16*>(&qp);
        const auto* kb = reinterpret_cast<const __nv_bfloat16*>(&kp);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = __bfloat162float(qb[i]);
            k[i] = __bfloat162float(kb[i]);
        }
    }
    if constexpr (Normalize) {
        float qs = 0, ks = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qs = fmaf(q[i], q[i], qs);
            ks = fmaf(k[i], k[i], ks);
        }
        qs             = warp_sum<16>(qs);
        ks             = warp_sum<16>(ks);
        const float qi = rsqrtf(qs + kNormEps), ki = rsqrtf(ks + kNormEps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] *= qi;
            k[i] *= ki;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (Fp16Packet) {
            sm.qk.q[vector_index(row, col + i)] =
                __ushort_as_bfloat16(__half_as_ushort(__float2half_rn(q[i])));
            sm.qk.k[vector_index(row, col + i)] =
                __ushort_as_bfloat16(__half_as_ushort(__float2half_rn(k[i])));
        } else {
            sm.qk.q[vector_index(row, col + i)] = __float2bfloat16_rn(q[i]);
            sm.qk.k[vector_index(row, col + i)] = __float2bfloat16_rn(k[i]);
        }
    }
    if (tid < 32) sm.control.padding[tid] = 0;
    if (tid < kChunkSize) {
        const auto off = static_cast<std::int64_t>(start + tid) * args.value_heads + head;
        float g        = tid < length ? args.g[off] : 0.0F;
        sm.beta[tid]   = tid < length ? args.beta[off] : 0.0F;
#pragma unroll
        for (int offset = 1; offset < kChunkSize; offset <<= 1) {
            const float previous = __shfl_up_sync(0xffffU, g, offset, kChunkSize);
            if (tid >= offset) g += previous;
        }
        sm.prefix[tid]         = g;
        sm.control.prefix[tid] = exp2_approx(g * kLog2E);
        const float last       = __shfl_sync(0xffffU, g, kChunkSize - 1, kChunkSize);
        sm.control.suffix[tid] = exp2_approx((last - g) * kLog2E);
    }
    __syncthreads();

    // Both products consume the same represented Q/K that recurrence will load.
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.qk.k : sm.qk.q;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(a + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                   offset + (lane >> 4) * 8)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(sm.qk.k + vector_index(n * 8 + (lane & 7),
                                                             offset + ((lane >> 3) & 1) * 8)));
                if constexpr (Fp16Packet)
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2], af[3],
                            bf[0], bf[1]);
                else
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2],
                             af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8;
                const int c = n * 8 + (lane & 3) * 2 + (i & 1);
                // Do not evaluate upper-triangle exponentials: g can be strongly negative.
                const float value =
                    r >= c ? acc[n][i] * exp2_approx((sm.prefix[r] - sm.prefix[c]) * kLog2E) : 0.0F;
                if (tid < 32)
                    sm.lower[r * kChunkSize + c] = r > c ? sm.beta[r] * value : 0.0F;
                else
                    sm.control.mqk[square_index(r, c)] = value;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < kChunkSize; ++r) {
            float value = r == c ? 1.0F : 0.0F;
#pragma unroll
            for (int j = 0; j < r; ++j)
                value = fmaf(-sm.lower[r * kChunkSize + j], sm.control.solve[square_index(j, c)],
                             value);
            if (lane < 16) sm.control.solve[square_index(r, c)] = value;
            __syncwarp();
        }
    }
    __syncthreads();
    sm.control.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];
    __syncthreads();

    const int chunks = chunk_count(args.tokens);
    auto* ctrl       = reinterpret_cast<uint4*>(control_workspace +
                                                static_cast<std::int64_t>(head) * chunks + chunk);
    const auto* src  = reinterpret_cast<const uint4*>(&sm.control);
    if (tid < sizeof(ControlChunk) / 16) ctrl[tid] = src[tid];
    // Only the first value head in a group publishes Q/K. Other producers never read this
    // workspace; the following kernel supplies the only producer/consumer synchronization.
    if (head % group == 0) {
        auto* dst =
            reinterpret_cast<uint4*>(qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
        const auto* data = reinterpret_cast<const uint4*>(&sm.qk);
#pragma unroll
        for (int i = tid; i < sizeof(QkChunk) / 16; i += 256) dst[i] = data[i];
    }
}

// Warp 0 and warp 1 stage the shared K*K^T / Q*K^T products; then up to 4 warps each own one
// value head of the group and stage its whole ControlChunk without a cross-warp barrier.
// Unioning QkChunk (dead after the shared MMA + workspace publish) with the 4-head control
// staging keeps static SMEM at 15,872 B (<= 20 KiB), enabling 4-5 CTAs/SM on GB10.
inline constexpr int kHeadWarps = 4;

struct PrepareGroupShared {
    union {
        QkChunk qk;
        struct {
            ControlChunk control[kHeadWarps];
            float prefix[kHeadWarps][kChunkSize];
            float beta[kHeadWarps][kChunkSize];
            float lower[kHeadWarps][kChunkSize * kChunkSize];
        };
    };
    float kk[kChunkSize * kChunkSize];
    float qk_product[kChunkSize * kChunkSize];
};

// One CTA per (QK head, chunk). The GQA group's value heads share Q/K, their L2 normalization,
// and the K*K^T and Q*K^T products, so the shared work runs once; the per-head control (gates,
// scaling, forward substitution) then runs with one warp per group head. Bit-identical to
// prepare_head_kernel.
// Fp16Packet (GdnChunkArithmetic::Fp16Accumulate): the packet's 16-bit elements hold FP16.
template <bool Normalize, bool Fp16Packet = false>
__global__
__launch_bounds__(256, 4) void prepare_group_kernel(Arguments args,
                                                    QkChunk* __restrict__ qk_workspace,
                                                    ControlChunk* __restrict__ control_workspace) {
    __shared__ PrepareGroupShared sm;
    const int tid = threadIdx.x, lane = tid & 31;
    const int chunks    = chunk_count(args.tokens);
    const int qh        = blockIdx.x % args.qk_heads;
    const int raw_chunk = static_cast<int>(blockIdx.x / args.qk_heads);
    // Above 64 chunks (>1024 tokens) the prepare working set exceeds GB10's 32 MiB L2; walking
    // chunks in reverse leaves chunk 0..~79 resident in L2 where recurrence starts.
    const int chunk = chunks > 64 ? (chunks - 1 - raw_chunk) : raw_chunk;
    const int start = chunk * kChunkSize, length = min(kChunkSize, args.tokens - start);
    const int group = args.value_heads / args.qk_heads;
    const int row = tid / 16, col = (tid & 15) * 8;
    const auto qo = (static_cast<std::int64_t>(start + row) * args.qk_heads + qh) * kStateDim;

    float q[8]{}, k[8]{};
    if (row < length) {
        const uint4 qp = load_vec<uint4>(args.q + qo + col);
        const uint4 kp = load_vec<uint4>(args.k + qo + col);
        const auto* qb = reinterpret_cast<const __nv_bfloat16*>(&qp);
        const auto* kb = reinterpret_cast<const __nv_bfloat16*>(&kp);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = __bfloat162float(qb[i]);
            k[i] = __bfloat162float(kb[i]);
        }
    }
    if constexpr (Normalize) {
        float qs = 0, ks = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qs = fmaf(q[i], q[i], qs);
            ks = fmaf(k[i], k[i], ks);
        }
        qs             = warp_sum<16>(qs);
        ks             = warp_sum<16>(ks);
        const float qi = rsqrtf(qs + kNormEps), ki = rsqrtf(ks + kNormEps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] *= qi;
            k[i] *= ki;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (Fp16Packet) {
            sm.qk.q[vector_index(row, col + i)] =
                __ushort_as_bfloat16(__half_as_ushort(__float2half_rn(q[i])));
            sm.qk.k[vector_index(row, col + i)] =
                __ushort_as_bfloat16(__half_as_ushort(__float2half_rn(k[i])));
        } else {
            sm.qk.q[vector_index(row, col + i)] = __float2bfloat16_rn(q[i]);
            sm.qk.k[vector_index(row, col + i)] = __float2bfloat16_rn(k[i]);
        }
    }
    __syncthreads();

    // Both products consume the same represented Q/K that recurrence will load. Warp 0 stages
    // the unscaled K*K^T, warp 1 the unscaled Q*K^T; simultaneously, all threads publish the
    // shared QkChunk so its shared-memory union slot can be reused for per-head ControlChunks.
    {
        auto* dst =
            reinterpret_cast<uint4*>(qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
        const auto* data = reinterpret_cast<const uint4*>(&sm.qk);
#pragma unroll
        for (int i = tid; i < sizeof(QkChunk) / 16; i += 256) dst[i] = data[i];
    }
    if (tid < 64) {
        const auto* a = tid < 32 ? sm.qk.k : sm.qk.q;
        float* out    = tid < 32 ? sm.kk : sm.qk_product;
        float acc[2][4]{};
#pragma unroll
        for (int offset = 0; offset < kStateDim; offset += 16) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(a + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                   offset + (lane >> 4) * 8)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(sm.qk.k + vector_index(n * 8 + (lane & 7),
                                                             offset + ((lane >> 3) & 1) * 8)));
                if constexpr (Fp16Packet)
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2], af[3],
                            bf[0], bf[1]);
                else
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2],
                             af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8;
                const int c = n * 8 + (lane & 3) * 2 + (i & 1);
                // Do not publish upper-triangle products: g can be strongly negative and the
                // head warps must not evaluate the upper-triangle exponentials below.
                out[r * kChunkSize + c] = r >= c ? acc[n][i] : 0.0F;
            }
        }
    }
    __syncthreads();

    const int warp = tid / 32;
    for (int base = 0; base < group; base += kHeadWarps) {
        const int active = min(group - base, kHeadWarps);
        const int head   = qh * group + base + warp;
        if (warp < active) {
            sm.control[warp].padding[lane] = 0;
            if (lane < kChunkSize) {
                const auto off = static_cast<std::int64_t>(start + lane) * args.value_heads + head;
                float g        = lane < length ? args.g[off] : 0.0F;
                sm.beta[warp][lane] = lane < length ? args.beta[off] : 0.0F;
#pragma unroll
                for (int offset = 1; offset < kChunkSize; offset <<= 1) {
                    const float previous = __shfl_up_sync(0xffffU, g, offset, kChunkSize);
                    if (lane >= offset) g += previous;
                }
                sm.prefix[warp][lane]         = g;
                sm.control[warp].prefix[lane] = exp2_approx(g * kLog2E);
                const float last              = __shfl_sync(0xffffU, g, kChunkSize - 1, kChunkSize);
                sm.control[warp].suffix[lane] = exp2_approx((last - g) * kLog2E);
            }
        }
        __syncwarp();
        if (warp < active) {
            ControlChunk& control = sm.control[warp];
            const float* prefix   = sm.prefix[warp];
            const float* beta     = sm.beta[warp];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = lane * 8 + i; // each lane covers 8 of the 256 (row, column) pairs
                const int r = s / kChunkSize, c = s % kChunkSize;
                // Do not evaluate upper-triangle exponentials: g can be strongly negative.
                const float decay =
                    r >= c ? exp2_approx((prefix[r] - prefix[c]) * kLog2E) : 0.0F;
                control.mqk[square_index(r, c)] = sm.qk_product[r * kChunkSize + c] * decay;
                sm.lower[warp][r * kChunkSize + c] =
                    r > c ? beta[r] * (sm.kk[r * kChunkSize + c] * decay) : 0.0F;
            }
            __syncwarp();
            const int c = lane & 15;
#pragma unroll
            for (int r = 0; r < kChunkSize; ++r) {
                float value = r == c ? 1.0F : 0.0F;
#pragma unroll
                for (int j = 0; j < r; ++j)
                    value = fmaf(-sm.lower[warp][r * kChunkSize + j],
                                 control.solve[square_index(j, c)], value);
                if (lane < kChunkSize) control.solve[square_index(r, c)] = value;
                __syncwarp();
            }
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = lane * 8 + i;
                const int r = s / kChunkSize, cc = s % kChunkSize;
                control.solve[square_index(r, cc)] *= beta[cc];
            }
        }
        __syncthreads();
        // All warps publish the round's staged chunks: a small group would otherwise leave the
        // copy to one warp while the rest of the CTA waits at the next barrier.
        constexpr int kVectors = sizeof(ControlChunk) / 16;
        for (int i = tid; i < active * kVectors; i += 256) {
            const int w = i / kVectors, v = i % kVectors;
            reinterpret_cast<uint4*>(control_workspace +
                                     static_cast<std::int64_t>(qh * group + base + w) * chunks +
                                     chunk)[v] =
                reinterpret_cast<const uint4*>(&sm.control[w])[v];
        }
        if (base + kHeadWarps < group) __syncthreads();
    }
}

// Group size 3 is the first measured win for the shared route (Qwen3.8-Flash-Next and Qwen3.5
// checkpoints run group 3: value_heads 48, qk_heads 16).
inline constexpr int kGroupSharedMinGroup = 3;

template <bool Normalize, bool Fp16Packet>
void launch_pair(const Arguments& args, QkChunk* qk, ControlChunk* control, cudaStream_t stream) {
    const int group = args.value_heads / args.qk_heads;
    if (group >= kGroupSharedMinGroup) {
        const unsigned blocks = static_cast<unsigned>(static_cast<std::int64_t>(args.qk_heads) *
                                                      chunk_count(args.tokens));
        prepare_group_kernel<Normalize, Fp16Packet><<<blocks, 256, 0, stream>>>(args, qk, control);
    } else {
        const unsigned blocks = static_cast<unsigned>(static_cast<std::int64_t>(args.value_heads) *
                                                      chunk_count(args.tokens));
        prepare_head_kernel<Normalize, Fp16Packet><<<blocks, 256, 0, stream>>>(args, qk, control);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_prepare(const Arguments& args, QkChunk* qk, ControlChunk* control, bool normalize_qk,
                    cudaStream_t stream, bool fp16) {
    if (fp16) {
        if (!normalize_qk) {
            throw std::invalid_argument("gated_delta_net: FP16 Q/K packets require normalize_qk");
        }
        launch_pair<true, true>(args, qk, control, stream);
    } else if (normalize_qk)
        launch_pair<true, false>(args, qk, control, stream);
    else
        launch_pair<false, false>(args, qk, control, stream);
}
} // namespace infernix::ops::detail::gated_delta_net::chunked
