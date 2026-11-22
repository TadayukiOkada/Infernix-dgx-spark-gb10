// QSA indexer, pooled keys, block selection and block-sparse attention
// (docs/maintainer/qwen3_8-flash-next-design.md §8.10). KV storages: bf16 and int8 (INT8-G64 with
// Hadamard-rotated keys, as written by kv_cache_append).

#include "infernix/ops/qsa.h"
#include "infernix/ops/sigmoid_mul.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/speculative_tree.h"
#include "ops/common/math.cuh"
#include "ops/common/sincos_reduced.cuh"

#include "core/paged_kv_storage.h"
#include "core/pdl.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/sigmoid_gate_mul_pair.cuh"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/kv_window.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/qsa/page_spaces.cuh"
#include "ops/qsa/qsa_prompt.h"
#include "ops/qsa/qsa_select.h"
#include "ops/qsa/vq_window.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kHeadDim      = 256;
constexpr int kIndexDim     = 128;
constexpr int kMaxGroup     = 16; // query heads per KV head
constexpr int kTokenTile    = 64;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    return v;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w]; }
    __syncthreads();
    return total;
}

// Half-split RoPE angle of pair i (i < rotary/2) at the token whose axis a of its RoPE position is
// rope[a * stride], in FP32 like the reference. Interleaved M-RoPE: pair i takes axis i % 3, so a
// text token (three equal axes) rotates exactly as by its 1-D position.
__device__ __forceinline__ void rope_angle(const std::int32_t* rope, std::int64_t stride, int i, int rotary,
                                           float theta, const PreparedRope& yarn, float& c, float& s) {
    double angle;
    if (yarn.factor == 1.0F) {
        const float inv = 1.0F / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary));
        angle           = static_cast<double>(rope[(i % 3) * stride]) * static_cast<double>(inv);
    } else {
        // YaRN: the attention rotation's coefficients (ops/kernel/rope.cuh PreparedRopeCoefficients): the
        // angle in FP64, reduced to one turn, FP32 sincos, both scaled by the attention scale.
        angle = static_cast<double>(rope[(i % 3) * stride]) * yarn.inverse[i];
    }
    sincos_turns(angle, &s, &c);
    if (yarn.factor != 1.0F) {
        s *= yarn.attention_scale;
        c *= yarn.attention_scale;
    }
}

// ------------------------------------------------------------------------------- index query

// One CTA per column: warp `threadIdx.y` is index head h (blockDim = (32, heads)); the row's sum
// of squares is one warp reduction, so the kernel uses no shared memory and no barrier. Every
// rotary pair (i, i + half) and its angle live in one thread. For half <= 32 (every real QSA
// geometry) lane i < half owns exactly one pair, plus the unrotated pair (rotary + 2i, +1), and
// the lanes past half take four unrotated elements each: half * 2 + (32 - half) * 4 = 128 - 2 *
// half tiles [rotary, kIndexDim) exactly and the warp's critical path is one rope_angle. Only
// half > 32 (rotary > 64, no shipped geometry) has more pairs than lanes: there lanes take the
// two pairs (2l, 2l + half) and (2l + 1, 2l + half + 1) and lanes past half / 2 cover the
// unrotated region four elements each. rotary % 4 == 0 keeps half even, which aligns every
// contiguous pair load and store to a __nv_bfloat162.
__global__ void __launch_bounds__(32 * 4)
    index_query_kernel(bf16* __restrict__ q, const bf16* __restrict__ weight,
                       const std::int32_t* __restrict__ rope, int heads, int rotary,
                       float theta, PreparedRope yarn, float eps) {
    pdl::enter(); // launched as a programmatic dependent
    const int t = blockIdx.x, h = threadIdx.y, lane = threadIdx.x;
    const int half = rotary / 2;
    bf16* row = q + (static_cast<std::size_t>(t) * heads + h) * kIndexDim;
    const bool one_pair = half <= 32, pair_rotated = one_pair ? lane < half : lane < half / 2;
    int e0, e1, e2, e3;
    if (one_pair) { // pair (lane, lane + half) for lanes < half; e2..e3 are never rotary partners
        e0 = pair_rotated ? lane : rotary + 2 * half + 4 * (lane - half);
        e1 = pair_rotated ? lane + half : e0 + 1;
        e2 = pair_rotated ? rotary + 2 * lane : e0 + 2;
    } else { // two bf162 pairs per lane: (e0, e0 + 1) and (e2, e2 + 1)
        e0 = pair_rotated ? 2 * lane : rotary + (lane - half / 2) * 4;
        e2 = pair_rotated ? e0 + half : e0 + 2;
    }
    e3 = e2 + 1;
    float x0, x1, x2, x3;
    if (pair_rotated && one_pair) { // the pair straddles the half split: two scalar elements
        x0 = __bfloat162float(row[e0]);
        x1 = __bfloat162float(row[e1]);
    } else {
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(row + e0);
        x0                     = __low2float(a);
        x1                     = __high2float(a);
    }
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(row + e2);
    x2                     = __low2float(b);
    x3                     = __high2float(b);
    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3) / kIndexDim + eps);
    const auto normed   = [&](float x, int e) {
        return __bfloat162float(__float2bfloat16_rn(x * inv * (1.0F + __bfloat162float(weight[e]))));
    };
    const float n0 = normed(x0, e0), n1 = normed(x1, e1), n2 = normed(x2, e2), n3 = normed(x3, e3);
    float o0 = n0, o1 = n1, o2 = n2, o3 = n3;
    if (pair_rotated) {
        // Pair i rotates by the column's axis i % 3 angle; the low element takes -s and the high
        // element +s, exactly as the elementwise reference. one_pair holds the pair in (e0, e1),
        // the two-pair layout in (e0, e2) and (e1, e3).
        float c0, s0, c1, s1;
        rope_angle(rope + t, gridDim.x, one_pair ? lane : e0, rotary, theta, yarn, c0, s0);
        if (one_pair) {
            o0 = n0 * c0 - n1 * s0;
            o1 = n1 * c0 + n0 * s0;
        } else {
            rope_angle(rope + t, gridDim.x, e0 + 1, rotary, theta, yarn, c1, s1);
            o0 = n0 * c0 - n2 * s0;
            o2 = n2 * c0 + n0 * s0;
            o1 = n1 * c1 - n3 * s1;
            o3 = n3 * c1 + n1 * s1;
        }
    }
    if (pair_rotated && one_pair) {
        row[e0] = __float2bfloat16_rn(o0);
        row[e1] = __float2bfloat16_rn(o1);
    } else {
        *reinterpret_cast<__nv_bfloat162*>(row + e0) = __floats2bfloat162_rn(o0, o1);
    }
    *reinterpret_cast<__nv_bfloat162*>(row + e2) = __floats2bfloat162_rn(o2, o3);
}

// --------------------------------------------------------------------------------- pooled keys

// The pooled-key element of `position` (its page's space resolved: the pool's plane or the spaces').
__device__ __forceinline__ bf16* pooled_element(bf16* pool, const QsaPageSpaces& spaces, const std::int32_t* table,
                                                int position, int slot_width, int lane) {
    int page;
    bf16* plane = detail::qsa_pooled_plane(spaces, pool, table[position >> kPagedKVPageShift], page);
    return plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (position & kPagedKVPageMask) + lane;
}

// One CTA (128 threads) per column; columns that complete a block pool it. The other CTAs are
// otherwise idle and (when update_tails) write the raw key of their position to the sequence's
// tail slot: the tail positions p of the latest block satisfy p % ratio <= ratio - 2, so they are
// always among the non-completing columns, and the fused write covers exactly the positions the
// separate tail kernel used to copy. A block the call completes whose first position precedes the
// call reads the tails for those earlier positions; its tail slot q % ratio is exactly the slot a
// later position p' = q + ratio of the latest block would overwrite, so when such a straddling
// completion exists this CTA performs all of the sequence's tail writes instead, after its reads —
// the idle CTAs of that sequence skip their write and no CTA races another's tail read.
__global__ void __launch_bounds__(kIndexDim)
    pool_kernel(const bf16* __restrict__ raw, const bf16* __restrict__ weight,
                bf16* __restrict__ tails, const std::int32_t* __restrict__ tables,
                int table_stride, const std::int32_t* __restrict__ table_rows,
                const std::int32_t* __restrict__ positions,
                const std::int32_t* __restrict__ tail_slots,
                const std::int32_t* __restrict__ rope, const std::int32_t* __restrict__ block_rope,
                int batch, int width, int ratio, int rotary, float theta, PreparedRope yarn, float eps,
                bool update_tails, bf16* __restrict__ pooled, QsaPageSpaces spaces,
                const SpeculativeTreeRow* __restrict__ trees) {
    __shared__ float scratch[4];
    __shared__ float normed[kIndexDim];
    pdl::enter(); // launched as a programmatic dependent
    const int t = blockIdx.x, d = threadIdx.x;
    const int sequence = t / width, local = t % width;
    const int main_end =
        (trees != nullptr && trees[sequence].tree != 0) ? static_cast<int>(trees[sequence].main_depth) : (width - 1);
    if (local > main_end) { return; }
    const int p = positions[t];
    if ((p + 1) % ratio != 0) {
        if (update_tails) {
            const int start    = positions[sequence * width];
            const int last     = positions[sequence * width + main_end];
            const int shift    = start % ratio;
            const bool straddles = last >= 0 && shift != 0 && start + (ratio - 1 - shift) <= last;
            if (!straddles) {
                const int base = last - last % ratio;
                const int end  = min(last, base + ratio - 2);
                if (p >= base && p <= end) {
                    tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) +
                           (p - base)) * kIndexDim + d] = raw[static_cast<std::size_t>(t) * kIndexDim + d];
                }
            }
        }
        return;
    }
    const int start    = positions[sequence * width];
    const int first    = p - ratio + 1;
    float sum          = 0.0F;
    for (int q = first; q <= p; ++q) {
        const bf16 v = q >= start ? raw[static_cast<std::size_t>(t - (p - q)) * kIndexDim + d]
                                  : tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) +
                                           (q % ratio)) * kIndexDim + d];
        sum += __bfloat162float(v);
    }
    if (update_tails) {
        const int shift = start % ratio;
        if (shift != 0 && p == start + (ratio - 1 - shift)) { // this CTA straddles: it owns the writes
            const int last = positions[sequence * width + main_end];
            if (last >= 0) {
                const int base = last - last % ratio;
                const int end  = min(last, base + ratio - 2);
                for (int q = max(base, start); q <= end; ++q) {
                    tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) +
                           (q - base)) * kIndexDim + d] = raw[static_cast<std::size_t>(t + (q - p)) * kIndexDim + d];
                }
            }
        }
    }
    const float mean = __bfloat162float(__float2bfloat16_rn(sum / static_cast<float>(ratio)));
    const float inv  = rsqrtf(block_sum(mean * mean, scratch) / kIndexDim + eps);
    normed[d] = __bfloat162float(__float2bfloat16_rn(mean * inv * (1.0F + __bfloat162float(weight[d]))));
    __syncthreads();
    float out = normed[d];
    if (d < rotary) {
        const int half = rotary / 2, i = d % half;
        float c, s;
        // The block's first token: a column of this call, or the sequence's block start before it.
        const std::int32_t* rope_base = first >= start ? rope + (t - (p - first)) : block_rope + sequence;
        const std::int64_t rope_stride = first >= start ? static_cast<std::int64_t>(gridDim.x) : batch;
        rope_angle(rope_base, rope_stride, i, rotary, theta, yarn, c, s);
        out = d < half ? normed[d] * c - normed[d + half] * s : normed[d] * c + normed[d - half] * s;
    }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[sequence]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    const int token           = first + d / slot_width;
    *pooled_element(pooled, spaces, table, token, slot_width, d % slot_width) = __float2bfloat16_rn(out);
}

// One CTA per (sequence, layer): the tail after the first commit_columns[sequence] columns.
__global__ void commit_tail_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                                   const std::int32_t* __restrict__ commit_columns,
                                   const std::int32_t* __restrict__ tail_slots, int width, int batch, int ratio,
                                   int slots, bf16* __restrict__ tails) {
    const int sequence = blockIdx.x, layer = blockIdx.y, d = threadIdx.x;
    const int n = commit_columns[sequence];
    if (n <= 0) { return; }
    const int start = positions[sequence * width];
    const int last  = start + n - 1;
    const int base  = last - last % ratio;
    const int end   = min(last, base + ratio - 2);
    const bf16* layer_raw = raw + static_cast<std::size_t>(layer) * batch * width * kIndexDim;
    bf16* layer_tails     = tails + static_cast<std::size_t>(layer) * slots * (ratio - 1) * kIndexDim;
    for (int q = max(base, start); q <= end; ++q) {
        const int column = sequence * width + (q - start);
        layer_tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (q % ratio)) * kIndexDim + d] =
            layer_raw[static_cast<std::size_t>(column) * kIndexDim + d];
    }
}

__global__ void __launch_bounds__(32)
    commit_tail_vec8_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ commit_columns,
                            const std::int32_t* __restrict__ tail_slots, int width, int batch, int ratio,
                            int slots, bf16* __restrict__ tails) {
    const int sequence = blockIdx.x, layer = blockIdx.y, d = threadIdx.x * 8;
    const int n = commit_columns[sequence];
    if (n <= 0 || d >= kIndexDim) { return; }
    const int start = positions[sequence * width];
    const int last  = start + n - 1;
    const int base  = last - last % ratio;
    const int end   = min(last, base + ratio - 2);
    const bf16* layer_raw = raw + static_cast<std::size_t>(layer) * batch * width * kIndexDim;
    bf16* layer_tails     = tails + static_cast<std::size_t>(layer) * slots * (ratio - 1) * kIndexDim;
    for (int q = max(base, start); q <= end; ++q) {
        const int column = sequence * width + (q - start);
        *reinterpret_cast<uint4*>(layer_tails +
                                  (static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (q % ratio)) *
                                      kIndexDim +
                                  d) =
            *reinterpret_cast<const uint4*>(layer_raw + static_cast<std::size_t>(column) * kIndexDim + d);
    }
}

// ----------------------------------------------------------------------------------- selection
// Block selection (scores, top-k, ascending ids) is qsa_select (qsa_select.cu).

// ----------------------------------------------------------------------------------- attention

// The attended token at index i of column t's list.
__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* selected, int ratio, int p) {
    if (count < 0) { return i; } // dense: tokens 0..p
    const int block_tokens = count * ratio;
    if (i < block_tokens) { return selected[i / ratio] * ratio + i % ratio; }
    return (p + 1) / ratio * ratio + (i - block_tokens);
}

__device__ __forceinline__ int attended_count(int count, int ratio, int p) {
    return count < 0 ? p + 1 : count * ratio + (p + 1) % ratio;
}

__device__ __forceinline__ bf16 apply_qsa_gate(bf16 x, const bf16* gate, std::int64_t idx) {
    if (gate == nullptr) { return x; }
    return __float2bfloat16_rn(__bfloat162float(x) * sigmoid(__bfloat162float(gate[idx])));
}

__device__ __forceinline__ __nv_bfloat162 apply_qsa_gate_pair(__nv_bfloat162 x, const bf16* gate, std::int64_t idx) {
    if (gate == nullptr) { return x; }
    if ((idx & 1) == 0 && (reinterpret_cast<std::uintptr_t>(gate) & 3U) == 0U) {
        return sigmoid_gate_mul_pair(*reinterpret_cast<const __nv_bfloat162*>(gate + idx), x);
    }
    return __halves2bfloat162(apply_qsa_gate(__low2bfloat16(x), gate, idx),
                              apply_qsa_gate(__high2bfloat16(x), gate, idx + 1));
}

void resolve_qsa_gate(const Tensor* gate, int heads, int columns, const bf16*& gate_ptr,
                      std::int64_t& gate_col_stride) {
    gate_ptr        = nullptr;
    gate_col_stride = 0;
    if (gate == nullptr || gate->data == nullptr) { return; }
    require(gate->dtype == DType::BF16 && gate->nb[0] == 2, "gate must be BF16 with contiguous inner elements");
    if (gate->ne[0] == heads * kHeadDim && gate->ne[2] == 1 && gate->ne[3] == 1 &&
        (gate->ne[1] == columns || (columns == 1 && gate->ne[1] == 1))) {
        gate_ptr        = static_cast<const bf16*>(gate->data);
        gate_col_stride = columns == 1 ? 0 : static_cast<std::int64_t>(gate->nb[1] / 2);
        return;
    }
    if (gate->is_contiguous() && gate->ne[0] == kHeadDim && gate->ne[1] == heads && gate->ne[2] == columns &&
        gate->ne[3] == 1) {
        gate_ptr        = static_cast<const bf16*>(gate->data);
        gate_col_stride = static_cast<std::int64_t>(heads) * kHeadDim;
        return;
    }
    require(false, "gate shape disagrees with attention output");
}

// Score sums of one token for every query head of a group: part[h] is this lane's partial dot
// product for head h. A reduce-scatter in warp_sum's xor order (16, 8, then 4, 2, 1 over the
// remaining quarter): at each step a lane adds its partner's value of a head to its own, exactly
// as warp_sum's `v += shfl_xor(v, o)`, so head h's sum has warp_sum's bits; it ends in the lanes
// whose bits 4 and 3 select its quarter. Those lanes with bits 0-2 clear write probs[h][j].
template <int Group>
__device__ __forceinline__ void scatter_head_sums(const float (&part)[Group], int group, int lane, float scale,
                                                  float (*probs)[kTokenTile], int j) {
    static_assert(Group % 4 == 0, "the reduce-scatter halves the heads twice");
    constexpr int kHalf = Group / 2, kQuarter = Group / 4;
    const bool upper16 = (lane & 16) != 0, upper8 = (lane & 8) != 0;
    float half[kHalf];
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
        const float mine = upper16 ? part[i + kHalf] : part[i];
        const float send = upper16 ? part[i] : part[i + kHalf];
        half[i]          = mine + __shfl_xor_sync(0xFFFFFFFFU, send, 16);
    }
    float quarter[kQuarter];
#pragma unroll
    for (int i = 0; i < kQuarter; ++i) {
        const float mine = upper8 ? half[i + kQuarter] : half[i];
        const float send = upper8 ? half[i] : half[i + kQuarter];
        quarter[i]       = mine + __shfl_xor_sync(0xFFFFFFFFU, send, 8);
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < kQuarter; ++i) { quarter[i] += __shfl_xor_sync(0xFFFFFFFFU, quarter[i], o); }
    }
    if ((lane & 7) == 0) {
        const int first = (upper16 ? kHalf : 0) + (upper8 ? kQuarter : 0);
#pragma unroll
        for (int i = 0; i < kQuarter; ++i) {
            if (first + i < group) { probs[first + i][j] = quarter[i] * scale; }
        }
    }
}

// One stage of attended tokens in shared memory for non-VQ storages, gathered by cp.async from each
// token's page space: four planes [K rows | V rows | K scales | V scales] of kTokenTile tokens each, in
// the paged planes' own byte layout (paged_kv_storage_layout). When ShareKV is true (BFloat16 1-stage),
// K and V share a single 32 KiB plane sequentially so two CTAs fit per SM on GB10 (<= 50,176 B/CTA).
template <int KeyBytes, int ValueBytes, int KeyScaleBytes, int ValueScaleBytes, bool ShareKV = false>
struct StagePlanes {
    static constexpr int kKeyBytes        = KeyBytes;
    static constexpr int kValueBytes      = ValueBytes;
    static constexpr int kKeyScaleBytes   = KeyScaleBytes;
    static constexpr int kValueScaleBytes = ValueScaleBytes;
    static constexpr bool kVq             = false;
    static constexpr bool kShareKV        = ShareKV;
    static constexpr int kValues          = ShareKV ? 0 : kTokenTile * KeyBytes;
    static constexpr int kKeyScales       = kValues + kTokenTile * ValueBytes;
    static constexpr int kValueScales     = kKeyScales + (ShareKV ? 0 : kTokenTile * KeyScaleBytes);
    static constexpr int kStageBytes      = kValueScales + kTokenTile * ValueScaleBytes;
    static_assert(KeyBytes % 16 == 0 && ValueBytes % 16 == 0 && kKeyScales % 16 == 0 &&
                  kKeyScaleBytes >= 0 && kValueScaleBytes >= 0);
    __device__ static const unsigned char* key_row(const unsigned char* stage, int j) { return stage + j * KeyBytes; }
    __device__ static const unsigned char* value_row(const unsigned char* stage, int j) {
        return stage + kValues + j * ValueBytes;
    }
    __device__ static const unsigned char* key_scales(const unsigned char* stage, int j) {
        return stage + kKeyScales + j * KeyScaleBytes;
    }
    __device__ static const unsigned char* value_scales(const unsigned char* stage, int j) {
        return stage + kValueScales + j * ValueScaleBytes;
    }
};

__device__ __forceinline__ float bf16_at(const unsigned char* p, int i) {
    return __bfloat162float(reinterpret_cast<const bf16*>(p)[i]);
}

__device__ __forceinline__ float fp16_at(const unsigned char* p, int i) {
    return __half2float(reinterpret_cast<const __half*>(p)[i]);
}

__device__ __forceinline__ float e4m3_at(const unsigned char* p, int i) {
    __nv_fp8_e4m3 v;
    v.__x = p[i];
    return static_cast<float>(v);
}

// E2M1 element d of a packed row (dimension 2j in the low nibble of byte j).
__device__ __forceinline__ float e2m1_at(const unsigned char* p, int d) {
    __nv_fp4_e2m1 v;
    v.__x = static_cast<__nv_fp4_storage_t>((p[d >> 1] >> (4 * (d & 1))) & 0xFU);
    return static_cast<float>(v);
}

// FP8 E4M3 rows with one FP16 scale; NVFP4 rows (E2M1 codes) with one E4M3 scale per 16.
__device__ __forceinline__ float fp8_row_cell(const unsigned char* row, const unsigned char* scale, int d) {
    return e4m3_at(row, d) * fp16_at(scale, 0);
}
__device__ __forceinline__ float nvfp4_cell(const unsigned char* row, const unsigned char* scales, int d) {
    return e2m1_at(row, d) * e4m3_at(scales, d >> 4);
}

template <KvCacheStorage Storage, int Stages = 2>
struct KVTile;

// BF16 keys, FP16 values. Share K and V plane so 1-stage fits 2 CTAs/SM in 99 KiB SMEM.
template <int Stages>
struct KVTile<KvCacheStorage::BFloat16, Stages> : StagePlanes<kHeadDim * 2, kHeadDim * 2, 0, 0, true> {
    using Base = StagePlanes<kHeadDim * 2, kHeadDim * 2, 0, 0, true>;
    static constexpr bool kRotatedKeys = false, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) { return bf16_at(Base::key_row(stage, j), d); }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        return bf16_at(Base::key_row(stage, j) + 2 * lane, 32 * r);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return fp16_at(Base::value_row(stage, j), d);
    }
};

// INT8 codes with one FP16 scale per 64; keys rotated.
template <int Stages>
struct KVTile<KvCacheStorage::Int8Group64, Stages> : StagePlanes<kHeadDim, kHeadDim, 8, 8> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j))[d]) *
               fp16_at(key_scales(stage, j), d >> 6);
    }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j) + lane)[32 * r]) *
               fp16_at(key_scales(stage, j), r >> 1);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(value_row(stage, j))[d]) *
               fp16_at(value_scales(stage, j), d >> 6);
    }
};

// FP8 E4M3 rows with an FP16 row scale; keys rotated.
template <int Stages>
struct KVTile<KvCacheStorage::Fp8E4M3Row256, Stages> : StagePlanes<kHeadDim, kHeadDim, 2, 2> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        return fp8_row_cell(key_row(stage, j) + lane, key_scales(stage, j), 32 * r);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// NVFP4 keys and values, both rotated.
template <int Stages>
struct KVTile<KvCacheStorage::Nvfp4Group16, Stages> : StagePlanes<kHeadDim / 2, kHeadDim / 2, 16, 16> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = true;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        __nv_fp4_e2m1 v;
        v.__x = static_cast<__nv_fp4_storage_t>(((key_row(stage, j) + (lane >> 1))[16 * r] >> (4 * (lane & 1))) & 0xFU);
        return static_cast<float>(v) * e4m3_at(key_scales(stage, j) + (lane >> 4), 2 * r);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// FP8 keys, NVFP4 values, both rotated.
template <int Stages>
struct KVTile<KvCacheStorage::Fp8KeyNvfp4Value, Stages> : StagePlanes<kHeadDim, kHeadDim / 2, 2, 16> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = true;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        return fp8_row_cell(key_row(stage, j) + lane, key_scales(stage, j), 32 * r);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// Vector-quantized storages (vq2, k4v2; ops/kv_cache/vq2_codec.cuh, q4_lloyd_codec.cuh): every tile
// token's K and V become INT8-G64 rows of the rotated vectors in the INT8 tile's planes, so scores and
// values then read exactly like INT8. A row is its exact window row (a window slot whose tag matches
// the stored codes, or the call's staged row) when the query reads the key exactly
// (vq_exact_key), and its codes decoded otherwise: codebook entry c (VQ2) or level c (Q4) with the
// row scale S as all four group scales, S * c exactly. Behind the INT8 planes each stage holds the
// paged code rows, their FP16 row scales, the window tags and one keep flag per row.
enum class VqKey : unsigned char { Vq2, Q4 };

template <VqKey Key>
struct VqTile {
    static constexpr int kKeyBytes          = kHeadDim;
    static constexpr int kValueBytes        = kHeadDim;
    static constexpr int kKeyScaleBytes     = 8;
    static constexpr int kValueScaleBytes   = 8;
    static constexpr bool kVq               = true;
    static constexpr bool kShareKV          = false;
    static constexpr bool kRotatedKeys      = true, kRotatedValues = true;
    static constexpr VqKey kKey             = Key;
    static constexpr int kValues            = kTokenTile * kKeyBytes;
    static constexpr int kKeyScales         = kValues + kTokenTile * kValueBytes;
    static constexpr int kValueScales       = kKeyScales + kTokenTile * kKeyScaleBytes;
    static constexpr int kKeyCodeBytes      = Key == VqKey::Vq2 ? kKVCacheVq2CodeBytes : kKVCacheQ4CodeBytes;
    static constexpr int kValueCodeBytes    = kKVCacheVq2CodeBytes;
    static constexpr int kKeyCodes          = kValueScales + kTokenTile * kValueScaleBytes;
    static constexpr int kValueCodes        = kKeyCodes + kTokenTile * kKeyCodeBytes;
    static constexpr int kCodeScales        = kValueCodes + kTokenTile * kValueCodeBytes; // [token][role] FP16
    static constexpr int kTags              = kCodeScales + kTokenTile * 4;               // [token][role] I32
    static constexpr int kKeep              = kTags + kTokenTile * 8;                     // [token][role] u8
    static constexpr int kStageBytes        = kKeep + kTokenTile * 2;
    __device__ static const unsigned char* key_row(const unsigned char* stage, int j) { return stage + j * kKeyBytes; }
    __device__ static const unsigned char* value_row(const unsigned char* stage, int j) {
        return stage + kValues + j * kValueBytes;
    }
    __device__ static const unsigned char* key_scales(const unsigned char* stage, int j) {
        return stage + kKeyScales + j * kKeyScaleBytes;
    }
    __device__ static const unsigned char* value_scales(const unsigned char* stage, int j) {
        return stage + kValueScales + j * kValueScaleBytes;
    }
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j))[d]) *
               fp16_at(key_scales(stage, j), static_cast<unsigned>(d) >> 6);
    }
    __device__ static float key_lane(const unsigned char* stage, int j, int lane, int r) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j) + lane)[32 * r]) *
               fp16_at(key_scales(stage, j), r >> 1);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(value_row(stage, j))[d]) *
               fp16_at(value_scales(stage, j), static_cast<unsigned>(d) >> 6);
    }
};

template <int Stages>
struct KVTile<KvCacheStorage::Vq2, Stages> : VqTile<VqKey::Vq2> {};
template <int Stages>
struct KVTile<KvCacheStorage::Q4KeyVq2Value, Stages> : VqTile<VqKey::Q4> {};

// Copies one scale row of `Bytes` (2: a plain load, which cp.async cannot do; 8 or 16: cp.async).
template <int Bytes>
__device__ __forceinline__ void copy_scales(unsigned char* dst, const unsigned char* src) {
    if constexpr (Bytes == 2) {
        *reinterpret_cast<__half*>(dst) = *reinterpret_cast<const __half*>(src);
    } else {
        cp_async<Bytes>(dst, src);
    }
}

// Dynamic shared memory of attention_kernel<Storage, Group, Stages>, in bytes and offsets: Stages
// tile stages, the rotated query heads, the tile's probabilities, the running statistics and one
// token/page-id slot per stage.
template <KvCacheStorage Storage, int Group, int Stages>
struct AttentionShared {
    using Tile                           = KVTile<Storage, Stages>;
    static constexpr std::size_t kStage  = Tile::kStageBytes;
    static constexpr std::size_t kQ      = Stages * kStage;
    static constexpr std::size_t kProbs  = kQ + sizeof(float) * Group * kHeadDim;
    static constexpr std::size_t kStats  = kProbs + sizeof(float) * Group * kTokenTile;
    static constexpr std::size_t kTokens = kStats + sizeof(float) * 4 * ((3 * Group + 3) / 4);
    static constexpr std::size_t kCodebook = kTokens + sizeof(std::int32_t) * 2 * Stages * kTokenTile;
    static constexpr std::size_t kBytes    = kCodebook + (Tile::kVq ? sizeof(kKVCacheVq2Codebook) : 0);
    static_assert(kStage % 16 == 0 && kBytes <= 99 * 1024, "one CTA's shared memory on sm_120");
};

// Threads of attention_kernel: one warp per token in the score phase, and two threads per output
// dimension in the value phase, each folding half of the group's heads.
constexpr int kAttentionThreads = 512;
constexpr int kAttentionWarps   = kAttentionThreads / 32;

// One CTA per (split, KV head, column): the column's query heads of this KV head over one slice of
// its attended tokens. Partials (max, sum, unnormalized output) go to workspace when split > 1.
// `Group` bounds the query heads per KV head (group = heads / kv_heads <= Group; `Exact` when they
// are equal, so head loops need no bounds), so every head loop has a compile-time trip count and
// the accumulators stay in registers.
//
// Each 64-token tile's K/V rows are gathered into shared memory by cp.async in one round. With two
// stages the next tile streams under this one; one stage serves calls whose splits hold one tile. A
// lane reads its eight query values of a head as two 16-byte words ([head][r / 4][lane][r % 4] for
// dimension lane + 32 r) and the probabilities four tokens at a time. A head's arithmetic is the
// reference order and does not depend on Group, the staging or the thread mapping: the score's fma
// order over r and the butterfly, the tile's online-softmax update (an exact maximum, then l summed
// serially in token order) and the in-order value fold, under the fixed split partition.
template <KvCacheStorage Storage, int Group, bool Exact, int Stages>
__global__ void __launch_bounds__(kAttentionThreads,
                                  (AttentionShared<Storage, Group, Stages>::kBytes <= 50176 &&
                                   Storage != KvCacheStorage::Vq2)
                                      ? 2
                                      : 1)
    attention_kernel(const bf16* __restrict__ q, int heads, int kv_heads, PagedKVLayerView kv, QsaPageSpaces spaces,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     int width, int ratio, const std::int32_t* __restrict__ selected,
                     const std::int32_t* __restrict__ counts, int top_blocks, int splits, float scale,
                     float* __restrict__ partial, bf16* __restrict__ out, detail::QsaVqWindow vq,
                     const SpeculativeTreeRow* __restrict__ trees) {
    using Tile   = KVTile<Storage, Stages>;
    using Shared = AttentionShared<Storage, Group, Stages>;
    static_assert(Group % 2 == 0, "two threads per dimension split the heads");
    constexpr int Half = Group / 2;
    pdl::enter();
    const int split = blockIdx.x, kv_head = blockIdx.y, t = blockIdx.z;
    const int group     = Exact ? Group : heads / kv_heads;
    const int seq       = t / width, local = t % width;
    const bool use_tree = trees != nullptr && trees[seq].tree != 0;
    const int base_p    = use_tree ? positions[seq * width] : 0;
    const bool is_live  = !use_tree || local < trees[seq].nodes;
    const bool is_side  = use_tree && local > trees[seq].main_depth && local < trees[seq].nodes;
    const int p         = !is_live ? base_p
                                   : (is_side ? (base_p + static_cast<int>(trees[seq].depth[local])) : positions[t]);
    const int count     = counts[t];
    const int total     = attended_count(count, ratio, p);
    const int per_split = (total + splits - 1) / splits;
    const int begin = split * per_split, end = min(total, begin + per_split);
    const int tiles = begin < end ? (end - begin + kTokenTile - 1) / kTokenTile : 0;
    if (tiles == 0) {
        if (splits == 1) {
            for (int i = threadIdx.x; i < group * kHeadDim; i += kAttentionThreads) {
                const int h = i / kHeadDim, d = i % kHeadDim;
                out[(static_cast<std::size_t>(t) * heads + kv_head * group + h) * kHeadDim + d] =
                    __float2bfloat16_rn(0.0F);
            }
        } else if (threadIdx.x < group) {
            const int head = kv_head * group + threadIdx.x;
            float* slot = partial + ((static_cast<std::size_t>(t) * splits + split) * heads + head) * (kHeadDim + 2);
            slot[kHeadDim]     = -INFINITY;
            slot[kHeadDim + 1] = 0.0F;
        }
        return;
    }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    extern __shared__ __align__(16) unsigned char smem[];
    float* qs      = reinterpret_cast<float*>(smem + Shared::kQ); // [Group][2][32][4]
    auto* probs    = reinterpret_cast<float(*)[kTokenTile]>(smem + Shared::kProbs);
    float* m_run   = reinterpret_cast<float*>(smem + Shared::kStats);
    float* l_run   = m_run + Group;
    float* rescale = l_run + Group;
    auto* tokens   = reinterpret_cast<std::int32_t(*)[kTokenTile]>(smem + Shared::kTokens); // [Stages]
    auto* ids      = tokens + Stages;                                                       // [Stages]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // Query element (h, lane + 32 r) in the lane-contiguous layout.
    const auto q_at = [](int h, int lane_, int r) { return (h * 64 + (r >> 2) * 32 + lane_) * 4 + (r & 3); };

    // Tile i's tokens and their block-table page ids into slot `buf`.
    const auto prepare = [&](int i, int buf) {
        const int first = begin + i * kTokenTile;
        if (threadIdx.x < min(kTokenTile, end - first)) {
            int token = attended_token(first + threadIdx.x, count, list, ratio, p);
            if (is_side && token > base_p) {
                const int d = token - base_p;
                int cur     = local;
                while (cur > 0 && static_cast<int>(trees[seq].depth[cur]) > d) {
                    cur = static_cast<int>(trees[seq].parent[cur]);
                }
                token = base_p + cur;
            }
            tokens[buf][threadIdx.x] = token;
            ids[buf][threadIdx.x]    = table[token >> kPagedKVPageShift];
        }
    };
    // Vector-quantized storages: the paged code rows and their scales, and for every key the query
    // reads exactly its exact row (window slot with its tags, or the call's staged row).
    const int call_first = positions[(t / width) * width];
    const int window_row = vq.slots != nullptr ? vq.slots[t / width] : 0;
    const auto vq_issue  = [&](unsigned char* stage, int buf, int n) {
        if constexpr (Tile::kVq) {
            constexpr int KC = Tile::kKeyCodeBytes / 16, VC = Tile::kValueCodeBytes / 16;
            const auto row_at = [&](int j, int page) {
                return static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize +
                       (tokens[buf][j] & kPagedKVPageMask);
            };
            for (int c = threadIdx.x; c < n * (KC + VC); c += kAttentionThreads) {
                const int j = c / (KC + VC), r = c % (KC + VC);
                const bool is_key = r < KC;
                const int chunk   = is_key ? r : r - KC;
                int page;
                const void* base = is_key ? detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k,
                                                                    spaces.lent_k, ids[buf][j], page)
                                          : detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v,
                                                                    spaces.lent_v, ids[buf][j], page);
                const int bytes = is_key ? Tile::kKeyCodeBytes : Tile::kValueCodeBytes;
                cp_async<16, Cache::cg>(stage + (is_key ? Tile::kKeyCodes : Tile::kValueCodes) + j * bytes + 16 * chunk,
                                        static_cast<const unsigned char*>(base) + row_at(j, page) * bytes + 16 * chunk);
            }
            for (int c = threadIdx.x; c < n * 32; c += kAttentionThreads) {
                const int j = c / 32, role = (c / 16) % 2, chunk = c % 16;
                const int key  = tokens[buf][j];
                const int mode = detail::vq_read_mode(vq, key, p, call_first);
                if (mode == 0) { continue; }
                const std::int8_t* src =
                    mode == 2 ? (role ? vq.staged_v_codes : vq.staged_k_codes) +
                                    (static_cast<std::int64_t>(kv_head) * vq.staged_width + (key - call_first)) * kHeadDim
                              : (role ? vq.v_codes : vq.k_codes) +
                                    detail::vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(key)) * kHeadDim;
                cp_async<16, Cache::cg>(stage + (role ? Tile::kValues : 0) + j * kHeadDim + 16 * chunk, src + 16 * chunk);
            }
            if (const int c = threadIdx.x; c < n * 2) {
                const int j = c / 2, role = c % 2;
                const int key = tokens[buf][j];
                int page;
                const void* base = role ? detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                                  spaces.lent_v_scale, ids[buf][j], page)
                                        : detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                                  spaces.lent_k_scale, ids[buf][j], page);
                copy_scales<2>(stage + Tile::kCodeScales + (2 * j + role) * 2,
                               static_cast<const unsigned char*>(base) + row_at(j, page) * 2);
                const int mode           = detail::vq_read_mode(vq, key, p, call_first);
                unsigned char* group_dst = stage + (role ? Tile::kValueScales : Tile::kKeyScales) + j * 8;
                if (mode == 2) {
                    const std::int64_t staged = static_cast<std::int64_t>(kv_head) * vq.staged_width + (key - call_first);
                    cp_async<8>(group_dst, (role ? vq.staged_v_scales : vq.staged_k_scales) + staged * kKVWindowGroups);
                } else if (mode == 1) {
                    const std::int64_t at = detail::vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(key));
                    cp_async<8>(group_dst, (role ? vq.v_scales : vq.k_scales) + at * kKVWindowGroups);
                    cp_async<4>(stage + Tile::kTags + (2 * j + role) * 4, vq.tags + at * 2 + role);
                }
            }
        }
    };
    // VQ2 codebook for code decoding (in the first tile's copy group).
    const std::int8_t* codebook = reinterpret_cast<const std::int8_t*>(smem + Shared::kCodebook);
    if constexpr (Tile::kVq) {
        if (const int i = threadIdx.x; i < static_cast<int>(sizeof(kKVCacheVq2Codebook)) / 16) {
            cp_async<16>(smem + Shared::kCodebook + 16 * i, g_kv_cache_vq2_codebook + 16 * i);
        }
    }
    // Byte offset of token j's row of `bytes` in its page's plane.
    const auto offset = [&](int buf, int j, int page, int bytes) {
        return (static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize +
                (tokens[buf][j] & kPagedKVPageMask)) *
               bytes;
    };
    // Gathers the K and V rows and scales of tile i's tokens (slot `buf`) into stage s.
    const auto issue = [&](int i, int buf, int s) {
        unsigned char* stage = smem + s * Shared::kStage;
        const int n          = min(kTokenTile, end - (begin + i * kTokenTile));
        if constexpr (Tile::kVq) {
            vq_issue(stage, buf, n);
            cp_commit();
        } else if constexpr (Tile::kShareKV) {
            constexpr int KC = Tile::kKeyBytes / 16;
            for (int c = threadIdx.x; c < n * KC; c += kAttentionThreads) {
                const int j = c / KC, chunk = c % KC;
                int page;
                const void* base = detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k, spaces.lent_k,
                                                           ids[buf][j], page);
                cp_async<16, Cache::cg>(stage + j * Tile::kKeyBytes + 16 * chunk,
                                        static_cast<const unsigned char*>(base) + offset(buf, j, page, Tile::kKeyBytes) +
                                            16 * chunk);
            }
            cp_commit();
        } else {
            constexpr int KC = Tile::kKeyBytes / 16, VC = Tile::kValueBytes / 16;
            if constexpr (KC == VC) {
                for (int c = threadIdx.x; c < n * KC; c += kAttentionThreads) {
                    const int j = c / KC, chunk = c % KC;
                    const int id = ids[buf][j];
                    int k_page, v_page;
                    const void* k_base = detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k,
                                                                 spaces.lent_k, id, k_page);
                    const void* v_base = detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v,
                                                                 spaces.lent_v, id, v_page);
                    cp_async<16, Cache::cg>(stage + j * Tile::kKeyBytes + 16 * chunk,
                                            static_cast<const unsigned char*>(k_base) +
                                                offset(buf, j, k_page, Tile::kKeyBytes) + 16 * chunk);
                    cp_async<16, Cache::cg>(stage + Tile::kValues + j * Tile::kValueBytes + 16 * chunk,
                                            static_cast<const unsigned char*>(v_base) +
                                                offset(buf, j, v_page, Tile::kValueBytes) + 16 * chunk);
                }
            } else {
                for (int c = threadIdx.x; c < n * (KC + VC); c += kAttentionThreads) {
                    const int j = c / (KC + VC), r = c % (KC + VC);
                    const bool is_key = r < KC;
                    const int chunk   = is_key ? r : r - KC;
                    int page;
                    const void* base = is_key ? detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k,
                                                                        spaces.lent_k, ids[buf][j], page)
                                              : detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v,
                                                                        spaces.lent_v, ids[buf][j], page);
                    const int bytes = is_key ? Tile::kKeyBytes : Tile::kValueBytes;
                    cp_async<16, Cache::cg>(stage + (is_key ? 0 : Tile::kValues) + j * bytes + 16 * chunk,
                                            static_cast<const unsigned char*>(base) + offset(buf, j, page, bytes) +
                                                16 * chunk);
                }
            }
            if constexpr (Tile::kKeyScaleBytes > 2) {
                if (const int j = threadIdx.x; j < n) {
                    int page;
                    const void* base = detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                               spaces.lent_k_scale, ids[buf][j], page);
                    copy_scales<Tile::kKeyScaleBytes>(stage + Tile::kKeyScales + j * Tile::kKeyScaleBytes,
                                                      static_cast<const unsigned char*>(base) +
                                                          offset(buf, j, page, Tile::kKeyScaleBytes));
                }
            }
            if constexpr (Tile::kValueScaleBytes > 2) {
                if (const int j = threadIdx.x - kTokenTile; j >= 0 && j < n) {
                    int page;
                    const void* base = detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                               spaces.lent_v_scale, ids[buf][j], page);
                    copy_scales<Tile::kValueScaleBytes>(stage + Tile::kValueScales + j * Tile::kValueScaleBytes,
                                                        static_cast<const unsigned char*>(base) +
                                                            offset(buf, j, page, Tile::kValueScaleBytes));
                }
            }
            cp_commit();
            if constexpr (Tile::kKeyScaleBytes == 2) {
                if (const int j = threadIdx.x; j < n) {
                    int page;
                    const void* base = detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                               spaces.lent_k_scale, ids[buf][j], page);
                    copy_scales<2>(stage + Tile::kKeyScales + j * 2,
                                   static_cast<const unsigned char*>(base) + offset(buf, j, page, 2));
                }
            }
            if constexpr (Tile::kValueScaleBytes == 2) {
                if (const int j = threadIdx.x - kTokenTile; j >= 0 && j < n) {
                    int page;
                    const void* base = detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                               spaces.lent_v_scale, ids[buf][j], page);
                    copy_scales<2>(stage + Tile::kValueScales + j * 2,
                                   static_cast<const unsigned char*>(base) + offset(buf, j, page, 2));
                }
            }
        }
    };
    const auto issue_shared_v = [&](int n, int buf, int s) {
        if constexpr (Tile::kShareKV) {
            unsigned char* stage = smem + s * Shared::kStage;
            constexpr int VC     = Tile::kValueBytes / 16;
            for (int c = threadIdx.x; c < n * VC; c += kAttentionThreads) {
                const int j = c / VC, chunk = c % VC;
                int page;
                const void* base = detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v, spaces.lent_v,
                                                           ids[buf][j], page);
                cp_async<16, Cache::cg>(stage + j * Tile::kValueBytes + 16 * chunk,
                                        static_cast<const unsigned char*>(base) + offset(buf, j, page, Tile::kValueBytes) +
                                            16 * chunk);
            }
            cp_commit();
        }
    };

    // The first tile streams while the query heads load and rotate.
    prepare(0, 0);
    __syncthreads();
    issue(0, 0, 0);
    if constexpr (Tile::kRotatedKeys) {
        // Keys are stored in the normalized Hadamard domain; load and rotate each query head in registers.
        for (int h = warp; h < group; h += kAttentionWarps) {
            const bf16* q_head = q + (static_cast<std::size_t>(t) * heads + kv_head * group + h) * kHeadDim + lane;
            float v[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = __bfloat162float(q_head[32 * r]); }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { qs[q_at(h, lane, r)] = v[r]; }
        }
    } else {
        // The query heads: one 16-byte load of eight elements per thread.
        for (int i = threadIdx.x; i < group * kHeadDim / 8; i += kAttentionThreads) {
            const int h = i / (kHeadDim / 8), d0 = 8 * (i % (kHeadDim / 8));
            const uint4 raw = *reinterpret_cast<const uint4*>(
                q + (static_cast<std::size_t>(t) * heads + kv_head * group + h) * kHeadDim + d0);
            const auto* values = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = d0 + e;
                qs[q_at(h, d & 31, d >> 5)] = __bfloat162float(values[e]);
            }
        }
    }
    float acc[Half];
    constexpr bool kSingleTile = Stages == 1 && !Tile::kVq;
    for (int i = 0; i < (kSingleTile ? 1 : tiles); ++i) {
        int n = kSingleTile ? (end - begin) : min(kTokenTile, end - (begin + i * kTokenTile));
        asm("" : "+r"(n));
        const int s = Stages == 2 ? i % 2 : 0;
        bool ahead  = false;
        if constexpr (!kSingleTile) {
            if (Stages == 1 && i > 0) {
                prepare(i, 0);
                __syncthreads();
                issue(i, 0, 0);
            } else if (Stages == 2 && i + 1 < tiles) {
                prepare(i + 1, (i + 1) % 2);
                __syncthreads();
                issue(i + 1, (i + 1) % 2, 1 - s);
                ahead = true;
            }
        }
        if (ahead) {
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        if constexpr (Tile::kVq) {
            // Keep a row's exact copy when its tag matches the stored codes (staged rows always), and
            // decode the codes of every other row into the INT8 planes, the row scale as each group's.
            unsigned char* st = smem + s * Shared::kStage;
            if (const int c = threadIdx.x; c < n * 2) {
                const int j = c / 2, role = c % 2;
                const int key  = tokens[s][j];
                const int mode = detail::vq_read_mode(vq, key, p, call_first);
                bool keep      = mode == 2;
                if (mode == 1) {
                    const std::uint32_t scale_bits =
                        reinterpret_cast<const std::uint16_t*>(st + Tile::kCodeScales)[2 * j + role];
                    std::uint32_t tag;
                    if (role == 0 && Tile::kKeyCodeBytes == 128) {
                        tag = kv_window_tag<32>(key, reinterpret_cast<const std::uint32_t*>(st + Tile::kKeyCodes + j * 128),
                                                scale_bits);
                    } else {
                        const unsigned char* codes =
                            role ? st + Tile::kValueCodes + j * 64 : st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes;
                        tag = kv_window_tag<16>(key, reinterpret_cast<const std::uint32_t*>(codes), scale_bits);
                    }
                    keep = static_cast<std::uint32_t>(reinterpret_cast<const std::int32_t*>(st + Tile::kTags)[2 * j + role]) ==
                           tag;
                }
                st[Tile::kKeep + 2 * j + role] = keep ? 1 : 0;
            }
            __syncthreads();
            for (int c = threadIdx.x; c < n * 64; c += kAttentionThreads) {
                const int j = c / 64, role = (c / 32) % 2, w = c % 32;
                if (st[Tile::kKeep + 2 * j + role] != 0) { continue; }
                uint2 codes8;
                if (role == 0 && Tile::kKey == VqKey::Q4) {
                    codes8 = kv_cache_q4_decode_word(
                        reinterpret_cast<const std::uint32_t*>(st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes)[w]);
                } else {
                    const unsigned char* codes =
                        role ? st + Tile::kValueCodes + j * 64 : st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes;
                    codes8 = kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(codes)[w], codebook);
                }
                *reinterpret_cast<uint2*>(st + (role ? Tile::kValues : 0) + j * kHeadDim + 8 * w) = codes8;
                if (w == 0) {
                    const __half scale = reinterpret_cast<const __half*>(st + Tile::kCodeScales)[2 * j + role];
                    __half* groups = reinterpret_cast<__half*>(st + (role ? Tile::kValueScales : Tile::kKeyScales) + j * 8);
#pragma unroll
                    for (int g = 0; g < kKVWindowGroups; ++g) { groups[g] = scale; }
                }
            }
            __syncthreads();
        }
        const unsigned char* stage = smem + s * Shared::kStage;
        const auto ld_q4           = [](const float4* p) {
            float4 v;
            asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                         : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(p))));
            return v;
        };
        // Scores: one warp per token pair, all query heads of the group.
        for (int j0 = warp * 2; j0 < n; j0 += kAttentionWarps * 2) {
            const int j1    = j0 + 1;
            const bool has1 = j1 < n;
            float k0[8], k1[8];
            if constexpr (!Tile::kShareKV) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    k0[r] = Tile::key_lane(stage, j0, lane, r);
                    k1[r] = 0.0F;
                }
                if (has1) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) { k1[r] = Tile::key_lane(stage, j1, lane, r); }
                }
            } else {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    k0[r] = Tile::key_lane(stage, j0, lane, r);
                    k1[r] = has1 ? Tile::key_lane(stage, j1, lane, r) : 0.0F;
                }
            }
            float part0[Group], part1[Group];
#pragma unroll
            for (int h = 0; h < Group; ++h) {
                float dot0 = 0.0F, dot1 = 0.0F;
                if (h < group) {
                    const float4 lo   = ld_q4(reinterpret_cast<const float4*>(qs) + h * 64 + lane);
                    const float4 hi   = ld_q4(reinterpret_cast<const float4*>(qs) + h * 64 + 32 + lane);
                    const float qv[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        dot0 += qv[r] * k0[r];
                        dot1 += qv[r] * k1[r];
                    }
                }
                part0[h] = dot0;
                part1[h] = dot1;
            }
            scatter_head_sums<Group>(part0, group, lane, scale, probs, j0);
            if (has1) { scatter_head_sums<Group>(part1, group, lane, scale, probs, j1); }
        }
        __syncthreads();
        if constexpr (Tile::kShareKV) { issue_shared_v(n, s, s); }
        // Online softmax per head: the tile's exact maximum, the exponentials in parallel, then l
        // in token order.
        for (int h = warp; h < group; h += kAttentionWarps) {
            const float a    = lane < n ? probs[h][lane] : -INFINITY;
            const float b    = lane + 32 < n ? probs[h][lane + 32] : -INFINITY;
            const float m_in = (kSingleTile || i == 0) ? -INFINITY : m_run[h];
            const float m    = fmaxf(m_in, warp_max(fmaxf(a, b)));
            if (lane < n) { probs[h][lane] = expf(a - m); }
            if (lane + 32 < n) { probs[h][lane + 32] = expf(b - m); }
            __syncwarp();
            if (lane == 0) {
                if (!kSingleTile && i > 0) { rescale[h] = expf(m_in - m); }
                m_run[h] = m;
            }
        }
        if constexpr (Tile::kShareKV) { cp_wait<0>(); }
        __syncthreads();
        int h_tid;
        asm("mov.u32 %0, %%tid.x;" : "=r"(h_tid));
        if (h_tid < group) {
            const int h = h_tid;
            float l     = (kSingleTile || i == 0) ? 0.0F : l_run[h] * rescale[h];
            int j       = 0;
            for (; j + 3 < n; j += 4) {
                const float4 pr = *reinterpret_cast<const float4*>(&probs[h][j]);
                l               = (((l + pr.x) + pr.y) + pr.z) + pr.w;
            }
            for (; j < n; ++j) { l += probs[h][j]; }
            l_run[h] = l;
        }
        const int d_own = threadIdx.x % kHeadDim, h_base = (threadIdx.x / kHeadDim) * Half;
        const auto mine = [&](int u) { return Exact || h_base + u < group; };
#pragma unroll
        for (int u = 0; u < Half; ++u) {
            if (kSingleTile || i == 0) {
                acc[u] = 0.0F;
            } else if (mine(u)) {
                acc[u] *= rescale[h_base + u];
            }
        }
        // Values, folded in token order, four tokens' probabilities a load.
        int j0 = 0;
        for (; j0 + 3 < n; j0 += 4) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { v[u] = Tile::value(stage, j0 + u, d_own); }
#pragma unroll
            for (int u = 0; u < Half; ++u) {
                if (mine(u)) {
                    const float4 pr = ld_q4(reinterpret_cast<const float4*>(&probs[h_base + u][j0]));
                    acc[u] += pr.x * v[0];
                    acc[u] += pr.y * v[1];
                    acc[u] += pr.z * v[2];
                    acc[u] += pr.w * v[3];
                }
            }
        }
        if (j0 < n) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { v[u] = j0 + u < n ? Tile::value(stage, j0 + u, d_own) : 0.0F; }
#pragma unroll
            for (int u = 0; u < Half; ++u) {
                if (mine(u)) {
                    const float4 pr = ld_q4(reinterpret_cast<const float4*>(&probs[h_base + u][j0]));
                    acc[u] += pr.x * v[0];
                    if (j0 + 1 < n) { acc[u] += pr.y * v[1]; }
                    if (j0 + 2 < n) { acc[u] += pr.z * v[2]; }
                    if (j0 + 3 < n) { acc[u] += pr.w * v[3]; }
                }
            }
        }
        __syncthreads();
    }
    const int d_own = threadIdx.x % kHeadDim, h_base = (threadIdx.x / kHeadDim) * Half;
    const auto mine = [&](int i) { return Exact || h_base + i < group; };
    int split_e, kv_head_e, t_e;
    asm volatile("mov.u32 %0, %%ctaid.x; mov.u32 %1, %%ctaid.y; mov.u32 %2, %%ctaid.z;"
                 : "=r"(split_e), "=r"(kv_head_e), "=r"(t_e));
    if constexpr (Tile::kRotatedValues) {
        if (splits == 1) {
            // Values are stored in the normalized Hadamard domain (its own inverse): rotate back when
            // writing directly, or once after merging the unrotated partials when split > 1.
            __syncthreads(); // the query heads are no longer read
            float* rows = qs; // [Group][256]
#pragma unroll
            for (int i = 0; i < Half; ++i) {
                if (mine(i)) { rows[(h_base + i) * kHeadDim + d_own] = acc[i]; }
            }
            __syncthreads();
            for (int h = warp; h < group; h += kAttentionWarps) {
                float v[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) { v[r] = rows[h * kHeadDim + lane + 32 * r]; }
                normalized_hadamard_d256_inplace(v, lane);
                const float l  = l_run[h];
                const int head = kv_head_e * group + h;
                bf16* dst           = out + (static_cast<std::size_t>(t_e) * heads + head) * kHeadDim + lane;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    dst[32 * r] = __float2bfloat16_rn(l > 0.0F ? v[r] / l : 0.0F);
                }
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        if (!mine(i)) { continue; }
        const int h = h_base + i, head = kv_head_e * group + h;
        if (splits == 1) {
            const float l = l_run[h];
            out[(static_cast<std::size_t>(t_e) * heads + head) * kHeadDim + d_own] =
                __float2bfloat16_rn(l > 0.0F ? acc[i] / l : 0.0F);
        } else {
            float* slot =
                partial + ((static_cast<std::size_t>(t_e) * splits + split_e) * heads + head) * (kHeadDim + 2);
            slot[d_own] = acc[i];
            if (d_own == 0) {
                slot[kHeadDim]     = m_run[h];
                slot[kHeadDim + 1] = l_run[h];
            }
        }
    }
}

template <bool RotateValues>
__global__ void __launch_bounds__(kHeadDim / 2)
    merge_kernel(const float* __restrict__ partial, int heads, int splits, const bf16* __restrict__ gate,
                 std::int64_t gate_col_stride, bf16* __restrict__ out) {
    pdl::enter();
    const int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        m = fmaxf(m, partial[((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2) + kHeadDim]);
    }
    float l = 0.0F, acc0 = 0.0F, acc1 = 0.0F;
    for (int s = 0; s < splits; ++s) {
        const float* slot = partial + ((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2);
        const float ms = slot[kHeadDim];
        if (ms == -INFINITY) { continue; }
        const float w     = expf(ms - m);
        const float2 pair = reinterpret_cast<const float2*>(slot)[d];
        l += w * slot[kHeadDim + 1];
        acc0 += w * pair.x;
        acc1 += w * pair.y;
    }
    const float val0 = l > 0.0F ? acc0 / l : 0.0F;
    const float val1 = l > 0.0F ? acc1 / l : 0.0F;
    if constexpr (!RotateValues) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(val0, val1);
        const std::int64_t g_idx  = static_cast<std::int64_t>(t) * gate_col_stride + head * kHeadDim + 2 * d;
        reinterpret_cast<__nv_bfloat162*>(out + (static_cast<std::size_t>(t) * heads + head) * kHeadDim)[d] =
            apply_qsa_gate_pair(pair, gate, g_idx);
    } else {
        __shared__ float smem[kHeadDim];
        reinterpret_cast<float2*>(smem)[d] = make_float2(val0, val1);
        __syncthreads();
        const int lane = d & 31, warp = d >> 5;
        float v[2] = {smem[lane + 64 * warp], smem[lane + 64 * warp + 32]};
        hadamard_d64_fragment_inplace(v, lane);
        smem[lane + 64 * warp]      = v[0];
        smem[lane + 64 * warp + 32] = v[1];
        __syncthreads();
        bf16* dst = out + (static_cast<std::size_t>(t) * heads + head) * kHeadDim;
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
            const float x0 = smem[lane + 32 * offset];
            const float x1 = smem[lane + 32 * (offset + 2)];
            const float x2 = smem[lane + 32 * (offset + 4)];
            const float x3 = smem[lane + 32 * (offset + 6)];
            const float y  = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, warp);
            const int elem = lane + 64 * warp + 32 * offset;
            const std::int64_t g_idx = static_cast<std::int64_t>(t) * gate_col_stride + head * kHeadDim + elem;
            dst[elem]                = apply_qsa_gate(__float2bfloat16_rn(y), gate, g_idx);
        }
    }
}

// ------------------------------------------------ split-KV Tensor Core MMA decode & verify (columns <= 8)

enum class ColumnsCode { Bf16, Fp16, Int8, Fp8, Nvfp4 };

template <ColumnsCode C>
struct ColumnsRow;
template <>
struct ColumnsRow<ColumnsCode::Bf16> {
    static constexpr int Bytes = 2 * kHeadDim, ScaleSlot = 0;
};
template <>
struct ColumnsRow<ColumnsCode::Fp16> {
    static constexpr int Bytes = 2 * kHeadDim, ScaleSlot = 0;
};
template <>
struct ColumnsRow<ColumnsCode::Int8> {
    static constexpr int Bytes = kHeadDim, ScaleSlot = 8;
    static constexpr float MaxScale = 256.0f;
};
template <>
struct ColumnsRow<ColumnsCode::Fp8> {
    static constexpr int Bytes = kHeadDim, ScaleSlot = 4;
    static constexpr float MaxScale = 64.0f;
};
template <>
struct ColumnsRow<ColumnsCode::Nvfp4> {
    static constexpr int Bytes = kHeadDim / 2, ScaleSlot = 16;
};

template <KvCacheStorage Storage>
struct ColumnsShape;
template <>
struct ColumnsShape<KvCacheStorage::BFloat16> {
    static constexpr ColumnsCode Key = ColumnsCode::Bf16, Value = ColumnsCode::Fp16;
};
template <>
struct ColumnsShape<KvCacheStorage::Int8Group64> {
    static constexpr ColumnsCode Key = ColumnsCode::Int8, Value = ColumnsCode::Int8;
};
template <>
struct ColumnsShape<KvCacheStorage::Fp8E4M3Row256> {
    static constexpr ColumnsCode Key = ColumnsCode::Fp8, Value = ColumnsCode::Fp8;
};
template <>
struct ColumnsShape<KvCacheStorage::Nvfp4Group16> {
    static constexpr ColumnsCode Key = ColumnsCode::Nvfp4, Value = ColumnsCode::Nvfp4;
};
template <>
struct ColumnsShape<KvCacheStorage::Fp8KeyNvfp4Value> {
    static constexpr ColumnsCode Key = ColumnsCode::Fp8, Value = ColumnsCode::Nvfp4;
};

__device__ __forceinline__ void col_widen_int8(unsigned codes, unsigned& lo, unsigned& hi) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x5140);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 l       = __hsub2(load_vec<__half2>(&e), offset);
    const __half2 h       = __hsub2(load_vec<__half2>(&o), offset);
    lo                    = load_vec<unsigned>(&l);
    hi                    = load_vec<unsigned>(&h);
}

__device__ __forceinline__ unsigned col_e4m3x2_to_f16x2(unsigned pair) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    return static_cast<unsigned>(raw.x) | (static_cast<unsigned>(raw.y) << 16);
}

__device__ __forceinline__ void col_widen_e4m3(unsigned codes, unsigned& lo, unsigned& hi) {
    lo = col_e4m3x2_to_f16x2(codes & 0xffffu);
    hi = col_e4m3x2_to_f16x2(codes >> 16);
}

__device__ __forceinline__ unsigned col_e2m1x2_times(unsigned byte, __half2 scale) {
    __nv_fp4x2_e2m1 codes;
    codes.__x       = static_cast<__nv_fp4x2_storage_t>(byte);
    const __half2 v = __hmul2(static_cast<__half2>(codes), scale);
    return load_vec<unsigned>(&v);
}

__device__ __forceinline__ __half col_e4m3_to_half(unsigned char code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<__half>(v);
}

struct ColumnsOnlineSoftmax {
    float maximum        = -INFINITY;
    float sum            = 0.0f;
    float scaled_maximum = 0.0f;

    __device__ __forceinline__ float update(float tile_max, float scale_l2) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_max);
        scaled_maximum       = maximum == -INFINITY ? 0.0f : maximum * scale_l2;
        return previous == -INFINITY ? 0.0f : exp2_approx(__fmaf_rn(previous, scale_l2, -scaled_maximum));
    }

    __device__ __forceinline__ float probability(float score, float scale_l2) const {
        return exp2_approx(__fmaf_rn(score, scale_l2, -scaled_maximum));
    }

    __device__ __forceinline__ void accumulate(float alpha, float tile_sum) {
        sum = __fmaf_rn(sum, alpha, tile_sum);
    }

    __device__ __forceinline__ void finish() { sum = ops::warp_sum<4>(sum, 0xffffffffu); }
};

template <KvCacheStorage Storage, int N>
__host__ __device__ constexpr std::size_t sparse_gqa_columns_smem_bytes() {
    using Shape              = ColumnsShape<Storage>;
    using K                  = ColumnsRow<Shape::Key>;
    using V                  = ColumnsRow<Shape::Value>;
    constexpr bool Quantized = Storage != KvCacheStorage::BFloat16;
    constexpr int MaxCode    = K::Bytes > V::Bytes ? K::Bytes : V::Bytes;
    constexpr int MaxScale   = K::ScaleSlot > V::ScaleSlot ? K::ScaleSlot : V::ScaleSlot;
    constexpr int KvBytes    = 2 * N * kHeadDim * 2;
    constexpr int RawBytes   = Quantized ? (N * MaxCode + N * MaxScale + 16) : 0;
    constexpr int Total      = KvBytes + RawBytes;
    constexpr int Partial    = 16 * kHeadDim * 4; // 16,384 B (only used when splits == 1 && RotatedValue)
    return static_cast<std::size_t>(Total > Partial ? Total : Partial);
}

template <KvCacheStorage Storage, int N, int Parts, int ClusterSplits = 1>
__global__ void __launch_bounds__(Parts * 32)
    sparse_gqa_columns_kernel(const bf16* __restrict__ q, int heads, int kv_heads, PagedKVLayerView kv,
                              QsaPageSpaces spaces, const std::int32_t* __restrict__ tables, int table_stride,
                              const std::int32_t* __restrict__ table_rows,
                              const std::int32_t* __restrict__ positions, int width, int ratio,
                              const std::int32_t* __restrict__ selected, const std::int32_t* __restrict__ counts,
                              int top_blocks, int splits, float scale, const bf16* __restrict__ gate,
                              std::int64_t gate_col_stride, float* __restrict__ partial, bf16* __restrict__ out,
                              const SpeculativeTreeRow* __restrict__ trees) {
    static_assert((N == 16 || N == 32) && (Parts == 2 || Parts == 4));
    static_assert(ClusterSplits == 1 || ClusterSplits == 2 || ClusterSplits == 4);
    using Shape                           = ColumnsShape<Storage>;
    constexpr bool Quantized              = Storage != KvCacheStorage::BFloat16;
    constexpr bool RotatedValue           = Shape::Value == ColumnsCode::Nvfp4;
    constexpr int Threads                 = Parts * 32;
    constexpr int GroupHeads              = 12;
    constexpr int OutCols                 = kHeadDim / Parts / 8;
    constexpr int kClusterPartialStride   = kHeadDim + 4;
    constexpr float Log2E                 = 1.4426950408889634074f;
    constexpr unsigned FullMask           = 0xffffffffu;

    pdl::enter();
    const int split   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int t       = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;

    const int seq       = t / width, local = t % width;
    const bool use_tree = trees != nullptr && trees[seq].tree != 0;
    const int base_p    = use_tree ? positions[seq * width] : 0;
    const bool is_live  = !use_tree || local < trees[seq].nodes;
    const bool is_side  = use_tree && local > trees[seq].main_depth && local < trees[seq].nodes;
    const int p         = !is_live ? base_p
                                   : (is_side ? (base_p + static_cast<int>(trees[seq].depth[local])) : positions[t]);
    const int count     = counts[t];
    const int total     = attended_count(count, ratio, p);
    if (total <= 0) {
        if (splits == 1 || (ClusterSplits > 1 && split == 0)) {
            for (int i = tid; i < GroupHeads * (kHeadDim / 2); i += Threads) {
                const int h = i / (kHeadDim / 2), d2 = i % (kHeadDim / 2);
                reinterpret_cast<__nv_bfloat162*>(
                    out + (static_cast<std::size_t>(t) * heads + kv_head * GroupHeads + h) * kHeadDim)[d2] =
                    __floats2bfloat162_rn(0.0f, 0.0f);
            }
        } else if (ClusterSplits == 1 && tid < GroupHeads) {
            float* slot =
                partial + ((static_cast<std::size_t>(t) * splits + split) * heads + kv_head * GroupHeads + tid) *
                              (kHeadDim + 2);
            slot[kHeadDim]     = -CUDART_INF_F;
            slot[kHeadDim + 1] = 0.0f;
        }
        return;
    }

    const int total_tiles = (total + N - 1) / N;
    const int start       = (split * total_tiles / splits) * N;
    const int end         = min(total, ((split + 1) * total_tiles / splits) * N);
    if constexpr (ClusterSplits == 1) {
        if (start >= end) {
            if (splits > 1 && tid < GroupHeads) {
                float* slot =
                    partial +
                    ((static_cast<std::size_t>(t) * splits + split) * heads + kv_head * GroupHeads + tid) *
                        (kHeadDim + 2);
                slot[kHeadDim]     = -CUDART_INF_F;
                slot[kHeadDim + 1] = 0.0f;
            }
            return;
        }
    }

    extern __shared__ __align__(16) unsigned char smem[];
    if (ClusterSplits == 1 || start < end) {
    auto* ks         = reinterpret_cast<std::uint16_t*>(smem);                 // [N][256]
    auto* vs         = ks + N * kHeadDim;                                      // [N][256]
    using KRow       = ColumnsRow<Shape::Key>;
    using VRow       = ColumnsRow<Shape::Value>;
    constexpr int MaxCode = KRow::Bytes > VRow::Bytes ? KRow::Bytes : VRow::Bytes;
    auto* raw_codes  = smem + 2 * N * kHeadDim * 2;                            // [N][MaxCode]
    auto* raw_scales = raw_codes + N * MaxCode;                                // [N][MaxScale]
    auto* meta_words = reinterpret_cast<unsigned*>(
        raw_scales + N * (KRow::ScaleSlot > VRow::ScaleSlot ? KRow::ScaleSlot : VRow::ScaleSlot));

    // Stage 12 query heads (padded to 16 MMA rows) into ks, load into registers qf[16][4], then free ks.
    if constexpr (Quantized) {
        for (int r = warp; r < 16; r += Parts) {
            float vals[8]{};
            if (r < GroupHeads) {
                const bf16* q_src = q + (static_cast<std::size_t>(t) * heads + kv_head * GroupHeads + r) * kHeadDim;
#pragma unroll
                for (int i = 0; i < 8; ++i) { vals[i] = __bfloat162float(q_src[lane + 32 * i]); }
                normalized_hadamard_d256_inplace(vals, lane);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d                         = lane + 32 * i;
                ks[r * kHeadDim + (d ^ ((r & 7) << 3))] = __half_as_ushort(__float2half_rn(vals[i]));
            }
        }
    } else {
        for (int i = tid; i < 16 * 32; i += Threads) {
            const int r           = i >> 5;
            const int d           = (i & 31) << 3;
            std::uint16_t* q_dst  = ks + r * kHeadDim + (d ^ ((r & 7) << 3));
            if (r < GroupHeads) {
                const bf16* q_src =
                    q + (static_cast<std::size_t>(t) * heads + kv_head * GroupHeads + r) * kHeadDim + d;
                cp_async<16>(q_dst, q_src);
            } else {
                store_vec(q_dst, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
    }
    __syncthreads();

    unsigned qf[16][4];
    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        ldmatrix_x4(qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                    smem_addr(ks + ar * kHeadDim + (d ^ ((ar & 7) << 3))));
    }
    __syncthreads();

    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    auto stage_plane = [&]<bool IsKey>(int k0) {
        constexpr ColumnsCode C  = IsKey ? Shape::Key : Shape::Value;
        using R                  = ColumnsRow<C>;
        const bool own_valid     = lane < N && (k0 + lane) < end;
        int own_token = own_valid ? attended_token(k0 + lane, count, list, ratio, p) : 0;
        if (own_valid && is_side && own_token > base_p) {
            const int d = own_token - base_p;
            int cur     = local;
            while (cur > 0 && static_cast<int>(trees[seq].depth[cur]) > d) {
                cur = static_cast<int>(trees[seq].parent[cur]);
            }
            own_token = base_p + cur;
        }
        const int own_id = own_valid ? table[own_token >> kPagedKVPageShift] : 0;
        int own_page             = 0;
        const auto* data_plane   = static_cast<const unsigned char*>(
            IsKey ? detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k, spaces.lent_k, own_id, own_page)
                  : detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v, spaces.lent_v, own_id, own_page));
        const std::size_t row_at =
            static_cast<std::size_t>(kv_head + kv_heads * own_page) * kPagedKVPageSize +
            (own_token & kPagedKVPageMask);
        const auto own_ptr       = reinterpret_cast<unsigned long long>(data_plane + row_at * R::Bytes);

        constexpr int Chunks = R::Bytes / 16;
        static_assert((N * Chunks) % Threads == 0);
        if constexpr (!Quantized) {
            std::uint16_t* dst_plane = IsKey ? ks : vs;
#pragma unroll
            for (int step = 0; step < (N * Chunks) / Threads; ++step) {
                const int idx          = tid + step * Threads;
                const int r            = idx / Chunks;
                const int c            = idx % Chunks;
                const int d            = c << 3;
                const auto* src_row    = reinterpret_cast<const unsigned char*>(__shfl_sync(FullMask, own_ptr, r));
                const bool ok          = __shfl_sync(FullMask, static_cast<int>(own_valid), r) != 0;
                std::uint16_t* dst_ptr = dst_plane + r * kHeadDim + (d ^ ((r & 7) << 3));
                cp_async_zfill<16, Cache::cg>(dst_ptr, src_row + 16 * c, ok ? 16 : 0);
            }
        } else {
#pragma unroll
            for (int step = 0; step < (N * Chunks) / Threads; ++step) {
                const int idx       = tid + step * Threads;
                const int r         = idx / Chunks;
                const int c         = idx % Chunks;
                const auto* src_row = reinterpret_cast<const unsigned char*>(__shfl_sync(FullMask, own_ptr, r));
                const bool ok       = __shfl_sync(FullMask, static_cast<int>(own_valid), r) != 0;
                cp_async_zfill<16, Cache::cg>(raw_codes + r * R::Bytes + 16 * c, src_row + 16 * c, ok ? 16 : 0);
            }
            if (warp == 0) {
                if (lane < N) {
                    int scale_page          = 0;
                    const auto* scale_plane = static_cast<const unsigned char*>(
                        IsKey ? detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                        spaces.lent_k_scale, own_id, scale_page)
                              : detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                        spaces.lent_v_scale, own_id, scale_page));
                    if constexpr (R::ScaleSlot == 4) {
                        cp_async_zfill<4>(raw_scales + lane * 4, scale_plane + (2 * row_at & ~std::size_t{3}),
                                          own_valid ? 4 : 0);
                    } else if constexpr (R::ScaleSlot == 8) {
                        cp_async_zfill<8>(raw_scales + lane * 8, scale_plane + 8 * row_at, own_valid ? 8 : 0);
                    } else {
                        cp_async_zfill<16>(raw_scales + lane * 16, scale_plane + 16 * row_at, own_valid ? 16 : 0);
                    }
                }
                if constexpr (R::ScaleSlot == 4) {
                    const unsigned odd = __ballot_sync(FullMask, own_valid && (row_at & 1U) != 0U);
                    if (lane == 0) { meta_words[0] = odd; }
                }
            }
        }
    };

    auto expand_plane = [&]<ColumnsCode C>(std::uint16_t* dst) -> float {
        if constexpr (!Quantized) {
            return 1.0f;
        } else if constexpr (C == ColumnsCode::Nvfp4) {
            for (int i = tid; i < N * 32; i += Threads) {
                const int r          = i >> 5;
                const int c_idx      = i & 31;
                const int d          = c_idx << 3;
                const __half sc      = col_e4m3_to_half(raw_scales[r * 16 + (c_idx >> 1)]);
                const __half2 s2     = __halves2half2(sc, sc);
                const unsigned pack4 = reinterpret_cast<const unsigned*>(raw_codes + r * 128)[c_idx];
                uint4 decoded;
                decoded.x = col_e2m1x2_times(pack4 & 0xffu, s2);
                decoded.y = col_e2m1x2_times((pack4 >> 8) & 0xffu, s2);
                decoded.z = col_e2m1x2_times((pack4 >> 16) & 0xffu, s2);
                decoded.w = col_e2m1x2_times(pack4 >> 24, s2);
                store_vec(dst + r * kHeadDim + (d ^ ((r & 7) << 3)), decoded);
            }
            __syncthreads();
            return 1.0f;
        } else {
            using R               = ColumnsRow<C>;
            const unsigned parity = (C == ColumnsCode::Fp8) ? meta_words[0] : 0u;
            if (warp == 0) {
                float own_max = 0.0f;
                if (lane < N) {
                    if constexpr (C == ColumnsCode::Fp8) {
                        const __half s_h =
                            reinterpret_cast<const __half*>(raw_scales)[2 * lane + ((parity >> lane) & 1u)];
                        own_max = fabsf(__half2float(s_h));
                    } else {
                        const __half* row_s = reinterpret_cast<const __half*>(raw_scales + lane * 8);
#pragma unroll
                        for (int g = 0; g < 4; ++g) { own_max = fmaxf(own_max, fabsf(__half2float(row_s[g]))); }
                    }
                }
                const float smax = ops::warp_max(own_max, FullMask);
                if (lane == 0) {
                    int shift = 0;
                    while (shift < 14 && ldexpf(smax, -shift) > R::MaxScale) { ++shift; }
                    meta_words[1] = static_cast<unsigned>(shift);
                }
            }
            __syncthreads();
            const int shift  = static_cast<int>(meta_words[1]);
            const float down = ldexpf(1.0f, -shift);
            for (int i = tid; i < N * 32; i += Threads) {
                const int r     = i >> 5;
                const int c_idx = i & 31;
                const int d     = c_idx << 3;
                __half s_orig;
                if constexpr (C == ColumnsCode::Fp8) {
                    s_orig = reinterpret_cast<const __half*>(raw_scales)[2 * r + ((parity >> r) & 1u)];
                } else {
                    s_orig = reinterpret_cast<const __half*>(raw_scales + r * 8)[c_idx >> 3];
                }
                const __half s_scaled = __float2half_rn(__half2float(s_orig) * down);
                const __half2 s2      = __halves2half2(s_scaled, s_scaled);
                const uint2 raw8      = reinterpret_cast<const uint2*>(raw_codes + r * 256)[c_idx];
                unsigned lo0, hi0, lo1, hi1;
                if constexpr (C == ColumnsCode::Fp8) {
                    col_widen_e4m3(raw8.x, lo0, hi0);
                    col_widen_e4m3(raw8.y, lo1, hi1);
                } else {
                    col_widen_int8(raw8.x, lo0, hi0);
                    col_widen_int8(raw8.y, lo1, hi1);
                }
                const __half2 v0 = __hmul2(load_vec<__half2>(&lo0), s2);
                const __half2 v1 = __hmul2(load_vec<__half2>(&hi0), s2);
                const __half2 v2 = __hmul2(load_vec<__half2>(&lo1), s2);
                const __half2 v3 = __hmul2(load_vec<__half2>(&hi1), s2);
                store_vec(dst + r * kHeadDim + (d ^ ((r & 7) << 3)),
                          make_uint4(load_vec<unsigned>(&v0), load_vec<unsigned>(&v1), load_vec<unsigned>(&v2),
                                     load_vec<unsigned>(&v3)));
            }
            __syncthreads();
            return ldexpf(1.0f, shift);
        }
    };

    float acc[OutCols][4]{};
    ColumnsOnlineSoftmax state[2];
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const float scale_l2 = scale * Log2E;

    stage_plane.template operator()<true>(start);
    cp_commit();

    for (int k0 = start; k0 < end; k0 += N) {
        cp_wait<0>();
        __syncthreads();
        const float k_up = expand_plane.template operator()<Shape::Key>(ks);
        stage_plane.template operator()<false>(k0);
        cp_commit();

        float score[N / 8][4]{};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned bf[2];
                const int r = n * 8 + (lane & 7);
                const int d = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(ks + r * kHeadDim + (d ^ ((r & 7) << 3))));
                if constexpr (!Quantized) {
                    mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], qf[k][0], qf[k][1], qf[k][2],
                             qf[k][3], bf[0], bf[1]);
                } else {
                    mma_f16(score[n][0], score[n][1], score[n][2], score[n][3], qf[k][0], qf[k][1], qf[k][2],
                            qf[k][3], bf[0], bf[1]);
                }
            }
        }
        if constexpr (Shape::Key == ColumnsCode::Int8 || Shape::Key == ColumnsCode::Fp8) {
#pragma unroll
            for (int n = 0; n < N / 8; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) { score[n][j] *= k_up; }
            }
        }

        cp_wait<0>();
        __syncthreads();
        const float v_up = expand_plane.template operator()<Shape::Value>(vs);
        if (k0 + N < end) {
            stage_plane.template operator()<true>(k0 + N);
            cp_commit();
        }

        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (k0 + n * 8 + lid * 2 + (j & 1) >= end) { score[n][j] = -CUDART_INF_F; }
                maximum[j >> 1] = fmaxf(maximum[j >> 1], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(ops::warp_max<4>(maximum[0], FullMask), scale_l2),
                                state[1].update(ops::warp_max<4>(maximum[1], FullMask), scale_l2)};
        unsigned pf[N / 16][4];
        float tile_sum[2]{};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_l2);
            const float p1 = state[0].probability(score[n][1], scale_l2);
            const float p2 = state[1].probability(score[n][2], scale_l2);
            const float p3 = state[1].probability(score[n][3], scale_l2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            if constexpr (Shape::Value == ColumnsCode::Int8 || Shape::Value == ColumnsCode::Fp8) {
                pf[n >> 1][(n & 1) * 2]     = pack_f16x2(p0 * v_up, p1 * v_up);
                pf[n >> 1][(n & 1) * 2 + 1] = pack_f16x2(p2 * v_up, p3 * v_up);
            } else {
                pf[n >> 1][(n & 1) * 2]     = pack_f16x2(p0, p1);
                pf[n >> 1][(n & 1) * 2 + 1] = pack_f16x2(p2, p3);
            }
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < OutCols; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[n][j] *= alpha[j >> 1]; }
#pragma unroll
            for (int k = 0; k < N / 16; ++k) {
                unsigned vf[2];
                const int r = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d = warp * (kHeadDim / Parts) + n * 8;
                ldmatrix_x2_t(vf[0], vf[1], smem_addr(vs + r * kHeadDim + (d ^ ((r & 7) << 3))));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2], pf[k][3],
                        vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    state[0].finish();
    state[1].finish();
    float* partial_rows = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int h = gid + j * 8;
        if (h >= GroupHeads) { continue; }
        const int head_idx = kv_head * GroupHeads + h;
        const bool direct  = ClusterSplits == 1 && splits == 1;
        const float inv    = direct ? (state[j].sum > 0.0f ? (1.0f / state[j].sum) : 0.0f) : 1.0f;
        float* slot        = ClusterSplits > 1
                                 ? (partial_rows + h * kClusterPartialStride)
                                 : (direct ? nullptr
                                           : partial + ((static_cast<std::size_t>(t) * splits + split) * heads +
                                                        head_idx) *
                                                           (kHeadDim + 2));
#pragma unroll
        for (int n = 0; n < OutCols; ++n) {
            const int d   = warp * (kHeadDim / Parts) + n * 8 + lid * 2;
            const float x = acc[n][2 * j] * inv;
            const float y = acc[n][2 * j + 1] * inv;
            if (direct) {
                if constexpr (RotatedValue) {
                    *reinterpret_cast<float2*>(partial_rows + h * kHeadDim + d) = make_float2(x, y);
                } else {
                    const __nv_bfloat162 pair = __floats2bfloat162_rn(x, y);
                    const std::int64_t g_idx = static_cast<std::int64_t>(t) * gate_col_stride + head_idx * kHeadDim + d;
                    *reinterpret_cast<__nv_bfloat162*>(
                        out + (static_cast<std::size_t>(t) * heads + head_idx) * kHeadDim + d) =
                        apply_qsa_gate_pair(pair, gate, g_idx);
                }
            } else {
                *reinterpret_cast<float2*>(slot + d) = make_float2(x, y);
            }
        }
        if (!direct && warp == 0 && lid == 0) {
            slot[kHeadDim]     = state[j].maximum == -CUDART_INF_F ? -CUDART_INF_F : state[j].scaled_maximum;
            slot[kHeadDim + 1] = state[j].sum;
        }
    }
    if constexpr (RotatedValue && ClusterSplits == 1) {
        if (splits == 1) {
            __syncthreads();
            for (int h = warp; h < GroupHeads; h += Parts) {
                float vals[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { vals[i] = partial_rows[h * kHeadDim + lane * 8 + i]; }
                normalized_hadamard_d256_contiguous_inplace(vals, lane);
                const int head_idx = kv_head * GroupHeads + h;
                bf16* dst          = out + (static_cast<std::size_t>(t) * heads + head_idx) * kHeadDim + lane * 8;
                const std::int64_t g_base =
                    static_cast<std::int64_t>(t) * gate_col_stride + head_idx * kHeadDim + lane * 8;
#pragma unroll
                for (int p_i = 0; p_i < 4; ++p_i) {
                    const __nv_bfloat162 pair = __floats2bfloat162_rn(vals[2 * p_i], vals[2 * p_i + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(dst + 2 * p_i) =
                        apply_qsa_gate_pair(pair, gate, g_base + 2 * p_i);
                }
            }
        }
    }
    } else if constexpr (ClusterSplits > 1) {
        if (tid < GroupHeads) {
            float* slot        = reinterpret_cast<float*>(smem) + tid * kClusterPartialStride;
            slot[kHeadDim]     = -CUDART_INF_F;
            slot[kHeadDim + 1] = 0.0f;
        }
    }
    if constexpr (ClusterSplits > 1) {
        __syncthreads();
        auto cluster = cooperative_groups::this_cluster();
        cluster.sync();
        if (split == 0) {
            float* partial_rows = reinterpret_cast<float*>(smem);
            for (int h = warp; h < GroupHeads; h += Parts) {
                float ms[ClusterSplits];
                float ls[ClusterSplits];
                float m = -CUDART_INF_F;
#pragma unroll
                for (int s = 0; s < ClusterSplits; ++s) {
                    const float* rslot = (s == 0) ? (partial_rows + h * kClusterPartialStride)
                                                  : cluster.map_shared_rank(partial_rows + h * kClusterPartialStride, s);
                    const float2 ml    = *reinterpret_cast<const float2*>(rslot + kHeadDim);
                    ms[s]              = ml.x;
                    ls[s]              = ml.y;
                    m                  = fmaxf(m, ml.x);
                }
                float w[ClusterSplits];
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < ClusterSplits; ++s) {
                    w[s] = (ms[s] == -CUDART_INF_F || ls[s] <= 0.0f) ? 0.0f : exp2_approx(ms[s] - m);
                    sum += w[s] * ls[s];
                }
                float vals[8]{};
                if (sum > 0.0f) {
#pragma unroll
                    for (int s = 0; s < ClusterSplits; ++s) {
                        if (w[s] == 0.0f) { continue; }
                        const float* rslot =
                            (s == 0) ? (partial_rows + h * kClusterPartialStride)
                                     : cluster.map_shared_rank(partial_rows + h * kClusterPartialStride, s);
                        const float* src = rslot + lane * 8;
                        const float2 a   = load_vec<float2>(src);
                        const float2 b   = load_vec<float2>(src + 2);
                        const float2 c   = load_vec<float2>(src + 4);
                        const float2 d   = load_vec<float2>(src + 6);
                        vals[0] += w[s] * a.x;
                        vals[1] += w[s] * a.y;
                        vals[2] += w[s] * b.x;
                        vals[3] += w[s] * b.y;
                        vals[4] += w[s] * c.x;
                        vals[5] += w[s] * c.y;
                        vals[6] += w[s] * d.x;
                        vals[7] += w[s] * d.y;
                    }
                    const float inv = 1.0f / sum;
#pragma unroll
                    for (int i = 0; i < 8; ++i) { vals[i] *= inv; }
                    if constexpr (RotatedValue) { normalized_hadamard_d256_contiguous_inplace(vals, lane); }
                }
                const int head_idx = kv_head * GroupHeads + h;
                bf16* dst          = out + (static_cast<std::size_t>(t) * heads + head_idx) * kHeadDim + lane * 8;
                const std::int64_t g_base =
                    static_cast<std::int64_t>(t) * gate_col_stride + head_idx * kHeadDim + lane * 8;
#pragma unroll
                for (int p_i = 0; p_i < 4; ++p_i) {
                    const __nv_bfloat162 pair = __floats2bfloat162_rn(vals[2 * p_i], vals[2 * p_i + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(dst + 2 * p_i) =
                        apply_qsa_gate_pair(pair, gate, g_base + 2 * p_i);
                }
            }
        }
        cluster.sync();
    }
}

template <bool RotateValues>
__global__ void __launch_bounds__(128)
    sparse_gqa_merge_warp_kernel(const float* __restrict__ partial, int heads, int splits,
                                 const bf16* __restrict__ gate, std::int64_t gate_col_stride,
                                 bf16* __restrict__ out) {
    pdl::enter();
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int head = static_cast<int>(blockIdx.x) * 4 + warp;
    const int t    = static_cast<int>(blockIdx.y);
    if (head >= heads) { return; }

    __shared__ float weights[4][48];
    float m = -CUDART_INF_F;
    for (int s = lane; s < splits; s += 32) {
        const float* slot = partial + ((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2);
        m                 = fmaxf(m, slot[kHeadDim]);
    }
    m = ops::warp_max(m, 0xffffffffu);
    float sum = 0.0f;
    for (int s = lane; s < splits; s += 32) {
        const float* slot = partial + ((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2);
        const float ms    = slot[kHeadDim];
        const float ls    = slot[kHeadDim + 1];
        const float w     = (ms == -CUDART_INF_F || ls <= 0.0f) ? 0.0f : exp2_approx(ms - m);
        weights[warp][s]  = w;
        sum += w * ls;
    }
    sum = ops::warp_sum(sum, 0xffffffffu);
    __syncwarp();

    float vals[8]{};
    if (sum > 0.0f) {
        for (int s = 0; s < splits; ++s) {
            const float w = weights[warp][s];
            if (w == 0.0f) { continue; }
            const float* slot =
                partial + ((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2) + lane * 8;
            const float2 a = load_vec<float2>(slot);
            const float2 b = load_vec<float2>(slot + 2);
            const float2 c = load_vec<float2>(slot + 4);
            const float2 d = load_vec<float2>(slot + 6);
            vals[0] += w * a.x;
            vals[1] += w * a.y;
            vals[2] += w * b.x;
            vals[3] += w * b.y;
            vals[4] += w * c.x;
            vals[5] += w * c.y;
            vals[6] += w * d.x;
            vals[7] += w * d.y;
        }
        const float inv = 1.0f / sum;
#pragma unroll
        for (int i = 0; i < 8; ++i) { vals[i] *= inv; }
        if constexpr (RotateValues) { normalized_hadamard_d256_contiguous_inplace(vals, lane); }
    }

    bf16* dst                 = out + (static_cast<std::size_t>(t) * heads + head) * kHeadDim + lane * 8;
    const std::int64_t g_base = static_cast<std::int64_t>(t) * gate_col_stride + head * kHeadDim + lane * 8;
#pragma unroll
    for (int p_i = 0; p_i < 4; ++p_i) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(vals[2 * p_i], vals[2 * p_i + 1]);
        *reinterpret_cast<__nv_bfloat162*>(dst + 2 * p_i) = apply_qsa_gate_pair(pair, gate, g_base + 2 * p_i);
    }
}

// Calls of at most this many columns: decode, one row's verification, up to eight plain lanes.
constexpr int kSmallCallColumns = 8;

// Splits per (column, KV head).
// - Calls of at most kSmallCallColumns columns: one 64-token tile per split of a column's attended
//   bound (33 at budget 2048, ratio 4), whatever the width. A column's bits then do not depend on how
//   many columns share the call, so concurrent sequences decode as they do alone (the preemption and
//   prefix-cache resumes rely on it; test_qsa checks it). Against one wave per call (the rule below)
//   this costs ~2 us at W = 5 and ~10 us at W = 8 per call, under 0.2 % of a round; W = 1-2 are equal.
// - Wider calls: one wave of the decode kernel's CTAs (one 512-thread CTA per SM), at most 48 and at
//   least 4 once split. When a 4-split decode kernel would exceed a single wave (splits < 4, i.e.
//   columns * kv_heads > sms / 4; columns >= 9 on 48 SMs), the call runs unsplit on the Tensor Core
//   prompt route (qsa_prompt_attention). Against the earlier 340-CTA / 33-split rule: 42-73 % faster
//   at 43-85 columns (design §19.3.15), and avoids a 2-wave decode cliff at 9-12 columns on GB10.
int attention_splits(int columns, const QsaGeometry& g) {
    const int attended = g.budget + g.ratio - 1;
    if (columns <= kSmallCallColumns) { return (attended + kTokenTile - 1) / kTokenTile; }
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
            throw std::runtime_error("qsa: cannot query the multiprocessor count");
        }
        return count;
    }();
    const int splits = std::clamp(sms / (columns * g.kv_heads), 1, 48);
    return splits < 4 ? 1 : splits;
}

struct Workspace {
    std::int32_t* selected;
    std::int32_t* counts;
    void* select;             // qsa_select scratch
    std::size_t select_bytes;
    float* partial;
    detail::KVCacheVqStaging staging; // a wide vector-quantized call's window rows
};

bool vector_quantized(KvCacheStorage storage) {
    return storage == KvCacheStorage::Vq2 || storage == KvCacheStorage::Q4KeyVq2Value;
}

// Whether a call of `columns` columns stages its window rows: vector-quantized storage with a window,
// wider than the calls whose append may write their slots before attention.
bool staged_call(KvCacheStorage storage, bool window, int columns) {
    return vector_quantized(storage) && window && columns > kKVWindowInlineWidth;
}

Workspace carve(void* base, const QsaGeometry& g, int columns, int max_context, bool staging,
                std::size_t& bytes) {
    auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const int top_blocks = g.budget / g.ratio;
    const std::size_t selected = align(sizeof(std::int32_t) * top_blocks * static_cast<std::size_t>(columns));
    const std::size_t counts   = align(sizeof(std::int32_t) * static_cast<std::size_t>(columns));
    const std::size_t select   = align(detail::qsa_select_scratch_bytes(g, columns, max_context));
    const int splits           = attention_splits(columns, g);
    const std::size_t partial  = splits > 1 ? align(sizeof(float) * static_cast<std::size_t>(columns) * splits *
                                                    g.heads * (kHeadDim + 2))
                                            : 0;
    // Staging rows [256, columns, Hkv] per role, [4, ...] FP16 scales per role, [2, ...] tags.
    const std::size_t rows   = static_cast<std::size_t>(columns) * g.kv_heads;
    const std::size_t codes  = staging ? align(kHeadDim * rows) : 0;
    const std::size_t scales = staging ? align(sizeof(__half) * kKVWindowGroups * rows) : 0;
    const std::size_t tags   = staging ? align(sizeof(std::int32_t) * 2 * rows) : 0;
    bytes = selected + counts + select + partial + 2 * codes + 2 * scales + tags;
    auto* p = static_cast<unsigned char*>(base);
    Workspace out{};
    if (p != nullptr) {
        out.selected     = reinterpret_cast<std::int32_t*>(p);
        out.counts       = reinterpret_cast<std::int32_t*>(p + selected);
        out.select       = p + selected + counts;
        out.select_bytes = select;
        out.partial      = partial ? reinterpret_cast<float*>(p + selected + counts + select) : nullptr;
        if (staging) {
            unsigned char* at = p + selected + counts + select + partial;
            out.staging.k_codes  = reinterpret_cast<std::int8_t*>(at);
            out.staging.v_codes  = reinterpret_cast<std::int8_t*>(at + codes);
            out.staging.k_scales = reinterpret_cast<__half*>(at + 2 * codes);
            out.staging.v_scales = reinterpret_cast<__half*>(at + 2 * codes + scales);
            out.staging.tags     = reinterpret_cast<std::int32_t*>(at + 2 * codes + 2 * scales);
        }
    }
    return out;
}

void require_geometry(const QsaGeometry& g) {
    require(g.head_dim == kHeadDim && g.index_head_dim == kIndexDim && g.kv_heads > 0 && g.heads % g.kv_heads == 0 &&
                g.heads / g.kv_heads <= kMaxGroup && g.index_heads > 0 && g.index_heads <= 4 && g.ratio > 1 &&
                kIndexDim % g.ratio == 0 && g.budget % g.ratio == 0 && g.rotary_dim > 0 &&
                g.rotary_dim <= kIndexDim && g.rotary_dim % 2 == 0 && g.theta > 0 && g.eps > 0,
            "unsupported geometry");
    require(std::isfinite(g.yarn_factor) && g.yarn_factor >= 1.0F && g.yarn_factor <= 4.0F &&
                (g.yarn_factor == 1.0F || (g.original_positions > 0 && g.theta > 1.0F)),
            "YaRN needs a factor in [1,4], original positions and theta above 1");
}

// The index rotation's YaRN preparation (factor 1: the plain rotation).
PreparedRope index_rope(const QsaGeometry& g) {
    if (g.yarn_factor == 1.0F) { return PreparedRope{}; }
    return prepare_rope(g.rotary_dim, g.theta,
                        {g.yarn_factor, static_cast<std::uint32_t>(g.original_positions)});
}

void require_batch(const QsaBatch& b, int columns) {
    require(b.batch > 0 && b.width > 0 && b.batch * b.width == columns && contiguous(b.positions, DType::I32) &&
                b.positions.numel() == columns && contiguous(b.table_rows, DType::I32) &&
                b.table_rows.numel() == b.batch && contiguous(b.block_tables, DType::I32) &&
                contiguous(b.tail_slots, DType::I32) && b.tail_slots.numel() == b.batch,
            "batch description is invalid");
}

} // namespace

void qsa_index_query(Tensor& q, const Tensor& norm_weight, const Tensor& rope_positions,
                     const QsaGeometry& geometry, cudaStream_t stream) {
    require_geometry(geometry);
    require(contiguous(q, DType::BF16) && contiguous(norm_weight, DType::BF16) &&
                contiguous(rope_positions, DType::I32),
            "index query needs BF16 q, BF16 weight and I32 RoPE positions");
    const int columns = q.ne[2];
    require(q.ne[0] == kIndexDim && q.ne[1] == geometry.index_heads && rope_positions.ne[0] == columns &&
                rope_positions.numel() == 3 * static_cast<std::int64_t>(columns) &&
                norm_weight.numel() == kIndexDim && columns > 0,
            "index query shapes disagree (RoPE positions are [T, 3])");
    require(geometry.rotary_dim % 4 == 0, "index query needs a rotary width divisible by 4");
    require(pdl::launch_consumer({dim3(columns), dim3(32, geometry.index_heads), 0, stream}, index_query_kernel,
        static_cast<bf16*>(q.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const std::int32_t*>(rope_positions.data), geometry.index_heads, geometry.rotary_dim,
        geometry.theta, index_rope(geometry), geometry.eps) == cudaSuccess,
            "index query launch failed");
    check_launch("index query");
}

void qsa_pool_keys(const Tensor& raw_keys, const Tensor& norm_weight, Tensor& tails,
                   const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& geometry,
                   cudaStream_t stream, const Tensor* tree_rows) {
    require_geometry(geometry);
    const int columns = raw_keys.ne[1];
    require_batch(batch, columns);
    const SpeculativeTreeRow* trees_p = nullptr;
    if (tree_rows != nullptr) {
        validate_speculative_tree_rows(*tree_rows, batch.batch, "qsa_pool_keys");
        trees_p = static_cast<const SpeculativeTreeRow*>(tree_rows->data);
    }
    require(contiguous(raw_keys, DType::BF16) && raw_keys.ne[0] == kIndexDim && contiguous(tails, DType::BF16) &&
                tails.ne[0] == kIndexDim && tails.ne[1] == geometry.ratio - 1 &&
                contiguous(layer.pooled_pages, DType::BF16) &&
                layer.pooled_pages.ne[0] == kIndexDim / geometry.ratio && contiguous(norm_weight, DType::BF16),
            "pool shapes disagree");
    require(contiguous(batch.rope_positions, DType::I32) && batch.rope_positions.ne[0] == columns &&
                batch.rope_positions.numel() == 3 * static_cast<std::int64_t>(columns) &&
                contiguous(batch.block_start_rope, DType::I32) && batch.block_start_rope.ne[0] == batch.batch &&
                batch.block_start_rope.numel() == 3 * static_cast<std::int64_t>(batch.batch),
            "pool RoPE positions are [T, 3] and block starts [batch, 3]");
    // The tail write (when update_tails) is fused into pool_kernel's otherwise-idle CTAs.
    require(pdl::launch_consumer({dim3(columns), dim3(kIndexDim), 0, stream}, pool_kernel,
        static_cast<const bf16*>(raw_keys.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<bf16*>(tails.data), static_cast<const std::int32_t*>(batch.block_tables.data),
        batch.block_tables.ne[0], static_cast<const std::int32_t*>(batch.table_rows.data),
        static_cast<const std::int32_t*>(batch.positions.data), static_cast<const std::int32_t*>(batch.tail_slots.data),
        static_cast<const std::int32_t*>(batch.rope_positions.data),
        static_cast<const std::int32_t*>(batch.block_start_rope.data), batch.batch, batch.width, geometry.ratio,
        geometry.rotary_dim, geometry.theta, index_rope(geometry), geometry.eps, batch.update_tails,
        static_cast<bf16*>(layer.pooled_pages.data), layer.spaces, trees_p) == cudaSuccess,
            "pool launch failed");
    check_launch("pool");
}

void qsa_commit_tails(const Tensor& raw_keys, const Tensor& positions, const Tensor& commit_columns,
                      Tensor& tails, const Tensor& tail_slots, const QsaGeometry& geometry,
                      cudaStream_t stream) {
    require_geometry(geometry);
    const int width = raw_keys.ne[1], batch = raw_keys.ne[2], layers = raw_keys.ne[3];
    require(contiguous(raw_keys, DType::BF16) && raw_keys.ne[0] == kIndexDim && width > 0 && batch > 0 &&
                layers > 0 && contiguous(positions, DType::I32) && positions.numel() == std::int64_t(width) * batch &&
                contiguous(commit_columns, DType::I32) && commit_columns.ne[0] == batch &&
                contiguous(tail_slots, DType::I32) && tail_slots.ne[0] == batch && contiguous(tails, DType::BF16) &&
                tails.ne[0] == kIndexDim && tails.ne[1] == geometry.ratio - 1 && tails.ne[3] == layers,
            "commit tail shapes disagree");
    const bool vec8 = ((reinterpret_cast<std::uintptr_t>(raw_keys.data) |
                        reinterpret_cast<std::uintptr_t>(tails.data)) &
                       15U) == 0U;
    if (vec8) {
        commit_tail_vec8_kernel<<<dim3(batch, layers), 32, 0, stream>>>(
            static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(commit_columns.data), static_cast<const std::int32_t*>(tail_slots.data),
            width, batch, geometry.ratio, tails.ne[2], static_cast<bf16*>(tails.data));
    } else {
        commit_tail_kernel<<<dim3(batch, layers), kIndexDim, 0, stream>>>(
            static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(commit_columns.data), static_cast<const std::int32_t*>(tail_slots.data),
            width, batch, geometry.ratio, tails.ne[2], static_cast<bf16*>(tails.data));
    }
    check_launch("commit tails");
}

std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry, std::int32_t columns,
                                          std::int32_t max_context, KvCacheStorage storage) {
    require_geometry(geometry);
    // Bounded by the staging a windowed call of this width would need. Covers every narrower call
    // too: a call of kSmallCallColumns columns keeps more split partials than wider calls up to ~21
    // columns, and every other term grows with the width.
    std::size_t bytes = 0, small = 0;
    (void)carve(nullptr, geometry, columns, max_context, staged_call(storage, true, columns), bytes);
    const int narrow = std::min<int>(columns, kSmallCallColumns);
    (void)carve(nullptr, geometry, narrow, max_context, staged_call(storage, true, narrow), small);
    return std::max(bytes, small);
}

void qsa_attention(const Tensor& q, const Tensor& index_q, const QsaKVLayer& layer,
                   const QsaBatch& batch, const QsaGeometry& geometry, float scale,
                   std::int32_t max_context, void* workspace, std::size_t workspace_bytes,
                   Tensor& out, cudaStream_t stream, const QsaAppend* append, const Tensor* gate,
                   bool allow_fast_paths, const Tensor* tree_rows) {
    require_geometry(geometry);
    const int columns = q.ne[2];
    require_batch(batch, columns);
    require(contiguous(q, DType::BF16) && q.ne[0] == kHeadDim && q.ne[1] == geometry.heads &&
                contiguous(index_q, DType::BF16) && index_q.ne[0] == kIndexDim &&
                index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns && contiguous(out, DType::BF16) &&
                out.ne[0] == kHeadDim && out.ne[1] == geometry.heads && out.ne[2] == columns,
            "attention shapes disagree");
    require(layer.kv.num_kv_heads == geometry.kv_heads && layer.kv.head_dim == kHeadDim, "KV geometry differs");
    const bf16* gate_ptr         = nullptr;
    std::int64_t gate_col_stride = 0;
    resolve_qsa_gate(gate, geometry.heads, columns, gate_ptr, gate_col_stride);
    const bool vq     = vector_quantized(layer.kv.storage);
    const SpeculativeTreeRow* trees_p = nullptr;
    if (tree_rows != nullptr) {
        require(!vq, "qsa attention: speculative trees require scalar KV cache storage");
        validate_speculative_tree_rows(*tree_rows, batch.batch, "qsa_attention");
        trees_p = static_cast<const SpeculativeTreeRow*>(tree_rows->data);
    }
    const bool window = vq && layer.kv.window.present();
    const bool staged = staged_call(layer.kv.storage, window, columns);
    require(vq == (append != nullptr), "the vector-quantized storages, and only they, take the call's K and V");
    require(!staged || batch.batch == 1, "a call wider than the inline window width takes one sequence");
    std::size_t need = 0;
    Workspace ws     = carve(workspace, geometry, columns, max_context, staged, need);
    require(workspace != nullptr && workspace_bytes >= need, "workspace is too small");
    // The vector-quantized append: paged codes, and the window slots of a narrow call or the staged
    // rows of a wide one (committed after attention).
    PagedKVBatchLayerView cache{.k_pages       = layer.kv.k_pages,
                                .v_pages       = layer.kv.v_pages,
                                .k_scale_pages = layer.kv.k_scale_pages,
                                .v_scale_pages = layer.kv.v_scale_pages,
                                .block_tables  = batch.block_tables,
                                .head_dim      = layer.kv.head_dim,
                                .num_kv_heads  = layer.kv.num_kv_heads,
                                .storage       = layer.kv.storage,
                                .window        = layer.kv.window};
    const Tensor positions2 = batch.positions.view({batch.width, batch.batch});
    if (vq) {
        require(contiguous(append->k, DType::BF16) && contiguous(append->v, DType::BF16) &&
                    append->k.ne[0] == kHeadDim && append->k.ne[1] == geometry.kv_heads &&
                    append->k.ne[2] == batch.width && append->k.ne[3] == batch.batch &&
                    std::equal(append->v.ne, append->v.ne + 4, append->k.ne),
                "append K/V are BF16 [head_dim, kv_heads, width, batch]");
        require(!window || (layer.kv.window.slots.data != nullptr && layer.kv.window.slots.dtype == DType::I32 &&
                            layer.kv.window.slots.numel() >= batch.batch),
                "the window's slots give each sequence's window row");
        detail::kv_cache_append_vq_batch_launch(append->k, append->v, positions2, Tensor{}, batch.table_rows, cache,
                                                staged ? &ws.staging : nullptr, stream);
    }
    detail::QsaVqWindow vq_args;
    if (window) {
        vq_args.k_codes  = static_cast<const std::int8_t*>(layer.kv.window.k_codes.data);
        vq_args.v_codes  = static_cast<const std::int8_t*>(layer.kv.window.v_codes.data);
        vq_args.k_scales = static_cast<const __half*>(layer.kv.window.k_scales.data);
        vq_args.v_scales = static_cast<const __half*>(layer.kv.window.v_scales.data);
        vq_args.tags     = static_cast<const std::int32_t*>(layer.kv.window.tags.data);
        vq_args.slots    = static_cast<const std::int32_t*>(layer.kv.window.slots.data);
        if (staged) {
            vq_args.staged_k_codes  = ws.staging.k_codes;
            vq_args.staged_v_codes  = ws.staging.v_codes;
            vq_args.staged_k_scales = ws.staging.k_scales;
            vq_args.staged_v_scales = ws.staging.v_scales;
            vq_args.staged_width    = columns;
        }
    }
    const auto commit = [&] {
        if (staged) {
            detail::kv_cache_vq_window_commit_launch(ws.staging, positions2, Tensor{}, cache, columns, stream);
        }
    };
    const int top_blocks = geometry.budget / geometry.ratio;
    const auto* tables   = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int stride     = batch.block_tables.ne[0];
    const auto* rows     = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos      = static_cast<const std::int32_t*>(batch.positions.data);
    detail::qsa_select(index_q, layer.pooled_pages, batch, geometry, max_context, ws.select, ws.select_bytes,
                       {ws.selected, ws.counts}, stream, layer.spaces, allow_fast_paths, trees_p);
    // Calls this kernel would run unsplit take the Tensor Core prompt route (qsa_prompt.h): the same
    // width-invariance class, so a column's bits still do not depend on its call's width there.
    const int splits = attention_splits(columns, geometry);
    if (trees_p == nullptr && splits == 1 && detail::qsa_prompt_supported(layer, geometry)) {
        detail::qsa_prompt_attention(q, layer, batch, geometry, scale, ws.selected, ws.counts, out, stream,
                                     vq_args, gate, allow_fast_paths);
        commit();
        return;
    }

    const bool scalar_kv = !vq && (layer.kv.storage == KvCacheStorage::BFloat16 ||
                                   layer.kv.storage == KvCacheStorage::Int8Group64 ||
                                   layer.kv.storage == KvCacheStorage::Fp8E4M3Row256 ||
                                   layer.kv.storage == KvCacheStorage::Nvfp4Group16 ||
                                   layer.kv.storage == KvCacheStorage::Fp8KeyNvfp4Value);
    const bool tiny_simt = (batch.width == 1 && max_context <= 64) || max_context <= 8;
    if (columns <= kSmallCallColumns && geometry.heads == 24 && geometry.kv_heads == 2 && scalar_kv && !tiny_simt) {
        static const int sms = [] {
            int device = 0, count = 0;
            if (cudaGetDevice(&device) != cudaSuccess ||
                cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
                throw std::runtime_error("qsa: cannot query the multiprocessor count");
            }
            return count;
        }();
        const int max_attended = std::min<int>(max_context, geometry.budget + geometry.ratio - 1);
        int key_rows           = 32;
        int parts              = (batch.width >= 2 && batch.width <= 6) ? 4 : 2;
        int mma_splits         = std::clamp((max_attended + 31) / 32, 1, std::min(32, splits));
        if (allow_fast_paths) {
            const int target = std::max(1, (sms * 3 / (2 * geometry.kv_heads) + columns - 1) / columns);
            mma_splits       = 1;
            while (mma_splits < target && mma_splits < 32 && mma_splits * 2 <= splits) { mma_splits *= 2; }
            const std::size_t tok_bytes =
                paged_kv_storage_layout(layer.kv.storage, kHeadDim).physical_bytes_per_token_head();
            const bool streaming = static_cast<std::uint64_t>(max_context) * batch.batch * geometry.kv_heads *
                                       tok_bytes >
                                   24ULL * 1024 * 1024;
            const bool expand = layer.kv.storage != KvCacheStorage::BFloat16 && columns > 6 &&
                                columns * geometry.kv_heads * mma_splits <= 2 * sms && streaming;
            key_rows = (expand || columns == 4) ? 16 : 32;
            if (expand && mma_splits * 2 <= splits) { mma_splits = std::min(32, mma_splits * 2); }
            parts = 2;
            if (key_rows == 32 && max_context <= 64) {
                parts = 4;
            } else if (key_rows == 32 && columns >= 2 && columns <= 6 && columns != 4) {
                parts        = 4;
                const int t2 = std::max(1, 2 * sms / (geometry.kv_heads * columns));
                int s2       = 1;
                while (s2 * 2 <= t2 && s2 < 32 && s2 * 2 <= splits) { s2 *= 2; }
                mma_splits = std::min(mma_splits, s2);
            }
            const int tiles_bound = std::max(1, (max_attended + key_rows - 1) / key_rows);
            mma_splits            = std::min(mma_splits, tiles_bound);
        }
        const dim3 mma_grid(static_cast<unsigned>(mma_splits), static_cast<unsigned>(geometry.kv_heads),
                            static_cast<unsigned>(columns));
        const bool use_cluster_merge = (mma_splits == 2 || mma_splits == 4);
        const auto launch_cols = [&]<KvCacheStorage Storage, int N, int Parts>() {
            constexpr std::size_t smem_bytes = sparse_gqa_columns_smem_bytes<Storage, N>();
            static_assert(smem_bytes <= 48 * 1024, "columns tile fits in default 48 KiB SMEM");
            const auto launch_cluster = [&]<int ClusterSplits>() {
                cudaLaunchAttribute attributes[2]{};
                attributes[0].id               = cudaLaunchAttributeClusterDimension;
                attributes[0].val.clusterDim.x = static_cast<unsigned>(ClusterSplits);
                attributes[0].val.clusterDim.y = 1;
                attributes[0].val.clusterDim.z = 1;
                unsigned count                 = 1;
                if (pdl::graph_dependent(stream)) {
                    attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
                    attributes[1].val.programmaticStreamSerializationAllowed = 1;
                    count                                                    = 2;
                }
                cudaLaunchConfig_t config{};
                config.gridDim          = mma_grid;
                config.blockDim         = dim3(Parts * 32);
                config.dynamicSmemBytes = smem_bytes;
                config.stream           = stream;
                config.attrs            = attributes;
                config.numAttrs         = count;
                return cudaLaunchKernelEx(
                    &config, sparse_gqa_columns_kernel<Storage, N, Parts, ClusterSplits>,
                    static_cast<const bf16*>(q.data), geometry.heads, geometry.kv_heads, layer.kv, layer.spaces,
                    tables, stride, rows, pos, batch.width, geometry.ratio,
                    static_cast<const std::int32_t*>(ws.selected), static_cast<const std::int32_t*>(ws.counts),
                    top_blocks, mma_splits, scale, gate_ptr, gate_col_stride, ws.partial,
                    static_cast<bf16*>(out.data), trees_p);
            };
            const cudaError_t error =
                mma_splits == 2
                    ? launch_cluster.template operator()<2>()
                    : (mma_splits == 4
                           ? launch_cluster.template operator()<4>()
                           : pdl::launch_consumer(
                                 {mma_grid, dim3(Parts * 32), smem_bytes, stream},
                                 sparse_gqa_columns_kernel<Storage, N, Parts, 1>, static_cast<const bf16*>(q.data),
                                 geometry.heads, geometry.kv_heads, layer.kv, layer.spaces, tables, stride, rows, pos,
                                 batch.width, geometry.ratio, static_cast<const std::int32_t*>(ws.selected),
                                 static_cast<const std::int32_t*>(ws.counts), top_blocks, mma_splits, scale, gate_ptr,
                                 gate_col_stride, ws.partial, static_cast<bf16*>(out.data), trees_p));
            if (error != cudaSuccess) {
                throw std::runtime_error(std::string("qsa mma columns: ") + cudaGetErrorString(error));
            }
        };
        const auto dispatch_cols = [&]<KvCacheStorage Storage>() {
            if (key_rows == 16) {
                launch_cols.template operator()<Storage, 16, 2>();
            } else if (parts == 4) {
                launch_cols.template operator()<Storage, 32, 4>();
            } else {
                launch_cols.template operator()<Storage, 32, 2>();
            }
        };
        switch (layer.kv.storage) {
        case KvCacheStorage::Int8Group64:
            dispatch_cols.template operator()<KvCacheStorage::Int8Group64>();
            break;
        case KvCacheStorage::Fp8E4M3Row256:
            dispatch_cols.template operator()<KvCacheStorage::Fp8E4M3Row256>();
            break;
        case KvCacheStorage::Nvfp4Group16:
            dispatch_cols.template operator()<KvCacheStorage::Nvfp4Group16>();
            break;
        case KvCacheStorage::Fp8KeyNvfp4Value:
            dispatch_cols.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
            break;
        default:
            dispatch_cols.template operator()<KvCacheStorage::BFloat16>();
            break;
        }
        check_launch("mma columns");
        if (mma_splits > 1 && !use_cluster_merge) {
            const bool rotate_values = layer.kv.storage == KvCacheStorage::Nvfp4Group16 ||
                                       layer.kv.storage == KvCacheStorage::Fp8KeyNvfp4Value;
            const dim3 merge_grid(static_cast<unsigned>((geometry.heads + 3) / 4), static_cast<unsigned>(columns));
            const cudaError_t error =
                rotate_values
                    ? pdl::launch_consumer({merge_grid, dim3(128), 0, stream}, sparse_gqa_merge_warp_kernel<true>,
                                           static_cast<const float*>(ws.partial), geometry.heads, mma_splits,
                                           gate_ptr, gate_col_stride, static_cast<bf16*>(out.data))
                    : pdl::launch_consumer({merge_grid, dim3(128), 0, stream}, sparse_gqa_merge_warp_kernel<false>,
                                           static_cast<const float*>(ws.partial), geometry.heads, mma_splits,
                                           gate_ptr, gate_col_stride, static_cast<bf16*>(out.data));
            if (error != cudaSuccess) {
                throw std::runtime_error(std::string("qsa mma merge: ") + cudaGetErrorString(error));
            }
        }
        commit();
        return;
    }

    const dim3 grid(splits, geometry.kv_heads, columns);
    // A column attends to at most budget + ratio - 1 tokens (dense below the budget, selected blocks
    // and the open block above it), so this bounds every split's tiles.
    const int split_tokens = (geometry.budget + geometry.ratio - 1 + splits - 1) / splits;
    const auto launch = [&]<KvCacheStorage Storage, int Group, bool Exact, int Stages>() {
        using Shared                = AttentionShared<Storage, Group, Stages>;
        constexpr std::size_t bytes = Shared::kBytes;
        constexpr auto kernel       = attention_kernel<Storage, Group, Exact, Stages>;
        static const bool reserved  = [] {
            return cudaFuncSetAttribute(attention_kernel<Storage, Group, Exact, Stages>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(Shared::kBytes)) == cudaSuccess;
        }();
        require(reserved, "attention shared memory cannot be reserved");
        const cudaError_t error = pdl::launch_consumer(
            {grid, dim3(kAttentionThreads), bytes, stream}, kernel, static_cast<const bf16*>(q.data), geometry.heads,
            geometry.kv_heads, layer.kv, layer.spaces, tables, stride, rows, pos, batch.width, geometry.ratio,
            static_cast<const std::int32_t*>(ws.selected), static_cast<const std::int32_t*>(ws.counts), top_blocks,
            splits, scale, ws.partial, static_cast<bf16*>(out.data), vq_args, trees_p);
        if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa attention: ") + cudaGetErrorString(error)); }
    };
    // Two stages only where a split holds more than one tile and both fit; one stage otherwise
    // (BF16's shared K/V stage, and every decode and verification call).
    const auto launch_stages = [&]<KvCacheStorage Storage, int Group, bool Exact>() {
        using One = AttentionShared<Storage, Group, 1>; // two stages add a stage and a token slot
        if constexpr (One::kBytes + One::kStage + sizeof(std::int32_t) * 2 * kTokenTile <= 99 * 1024) {
            if (split_tokens > kTokenTile) {
                launch.template operator()<Storage, Group, Exact, 2>();
                return;
            }
        }
        launch.template operator()<Storage, Group, Exact, 1>();
    };
    // Qwen4Exp's 24 / 2 = 12 query heads per KV head take the exact instance; any other group up to
    // 16 the bounded one.
    const auto launch_storage = [&]<KvCacheStorage Storage>() {
        if (geometry.heads / geometry.kv_heads == 12) {
            launch_stages.template operator()<Storage, 12, true>();
        } else {
            launch_stages.template operator()<Storage, kMaxGroup, false>();
        }
    };
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
        launch_storage.template operator()<KvCacheStorage::BFloat16>();
        break;
    case KvCacheStorage::Int8Group64:
        launch_storage.template operator()<KvCacheStorage::Int8Group64>();
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch_storage.template operator()<KvCacheStorage::Fp8E4M3Row256>();
        break;
    case KvCacheStorage::Nvfp4Group16:
        launch_storage.template operator()<KvCacheStorage::Nvfp4Group16>();
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch_storage.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
        break;
    case KvCacheStorage::Vq2:
        launch_storage.template operator()<KvCacheStorage::Vq2>();
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        launch_storage.template operator()<KvCacheStorage::Q4KeyVq2Value>();
        break;
    default:
        throw std::invalid_argument("qsa: this KV storage is not implemented yet");
    }
    check_launch("attention");
    if (splits == 1 && gate_ptr != nullptr) {
        Tensor out_view = gate->ne[0] == kHeadDim ? out : out.view({geometry.heads * kHeadDim, columns});
        sigmoid_mul(*gate, out_view, stream);
    }
    if (splits > 1) {
        const bool rotate_values = layer.kv.storage == KvCacheStorage::Nvfp4Group16 ||
                                   layer.kv.storage == KvCacheStorage::Fp8KeyNvfp4Value ||
                                   vector_quantized(layer.kv.storage);
        const cudaError_t error =
            rotate_values
                ? pdl::launch_consumer({dim3(geometry.heads, columns), dim3(kHeadDim / 2), 0, stream},
                                       merge_kernel<true>, static_cast<const float*>(ws.partial), geometry.heads,
                                       splits, gate_ptr, gate_col_stride, static_cast<bf16*>(out.data))
                : pdl::launch_consumer({dim3(geometry.heads, columns), dim3(kHeadDim / 2), 0, stream},
                                       merge_kernel<false>, static_cast<const float*>(ws.partial), geometry.heads,
                                       splits, gate_ptr, gate_col_stride, static_cast<bf16*>(out.data));
        if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa merge: ") + cudaGetErrorString(error)); }
    }
    commit();
}

} // namespace infernix::ops
