#include "infernix/ops/ple.h"

#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
// The widest block a gate CTA uses: one V-wide slice per thread up to 32 warps, then K slices
// per thread.
constexpr int kGateMaxThreads = 1024;
// A decode convolution folds its state shift into conv_kernel only up to this history length, so
// the shift stays a compile-time-indexed register array (a runtime index would spill it to local
// memory). Longer histories keep the separate state_kernel.
constexpr int kFusedConvMaxSpan = 16;
// Below one CTA per SM, an NVFP4 embedding column is split into warp-sized CTAs so a short chunk
// occupies several SMs instead of one.
constexpr int kEmbedSpreadColumns = 48;
constexpr int kEmbedCtaTasks      = 32;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ple: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ple ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

bool column_strided_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.ne[2] == 1 && t.ne[3] == 1 &&
           t.nb[0] == static_cast<std::int64_t>(dtype_size(dtype)) &&
           t.nb[1] >= static_cast<std::int64_t>(t.ne[0]) * t.nb[0] &&
           t.nb[1] % t.nb[0] == 0;
}

template <std::size_t Alignment>
bool aligned(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % Alignment == 0;
}

// The gate geometry ple_gate and ple_gate_conv_inject share: everything the kernels index except
// the gated/normalized outputs, which only the composed route materializes.
struct GateGeometry {
    int hidden, width, columns, key_stride, value_stride;
    DType norm;
};

GateGeometry gate_geometry(const Tensor& key, const Tensor& value, const Tensor& residual,
                           const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
                           std::int32_t streams, float eps, int width, int columns) {
    require(column_strided_2d(key, DType::BF16) && column_strided_2d(value, DType::BF16),
            "gate requires column-strided 2D BF16 key and value tensors");
    require(contiguous(residual, DType::BF16), "gate requires contiguous BF16 tensors");
    const DType norm = key_norm.dtype;
    require((norm == DType::BF16 || norm == DType::FP32) && contiguous(key_norm, norm) &&
                contiguous(query_norm, norm) && contiguous(conv_norm, norm),
            "gate norms must be contiguous, all BF16 or all FP32");
    require(streams > 0 && width % streams == 0, "gate width must hold whole streams");
    const int hidden = width / streams;
    require(value.ne[0] == hidden && value.ne[1] == columns && residual.ne[0] == width &&
                residual.ne[1] == columns && key_norm.numel() == width && query_norm.numel() == width &&
                conv_norm.numel() == width && columns > 0 && eps > 0,
            "gate shapes disagree");
    return {hidden, width, columns, static_cast<int>(key.nb[1] / static_cast<std::int64_t>(sizeof(bf16))),
            static_cast<int>(value.nb[1] / static_cast<std::int64_t>(sizeof(bf16))), norm};
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// One block reduction per barrier pair. The two operands of the gate's key/query statistics share
// a pair, which is why the gate needs three barrier pairs rather than four.
__device__ __forceinline__ float block_sum(float value, float2* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp].x = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w].x; }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float2 block_sum2(float a, float b, float2* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_xor_sync(0xFFFFFFFFU, a, offset);
        b += __shfl_xor_sync(0xFFFFFFFFU, b, offset);
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = make_float2(a, b); }
    __syncthreads();
    float2 total = make_float2(0.0F, 0.0F);
    for (int w = 0; w < blockDim.x / 32; ++w) {
        total.x += scratch[w].x;
        total.y += scratch[w].y;
    }
    __syncthreads();
    return total;
}

__global__ void embed_kernel(const std::uint8_t* __restrict__ rows, const bf16* __restrict__ scale,
                             int elements, bf16* __restrict__ out) {
    pdl::enter();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= elements) { return; }
    __nv_fp8_e4m3 code;
    code.__x      = rows[static_cast<std::size_t>(t) * elements + i];
    const float v = static_cast<float>(code) * __bfloat162float(scale[0]);
    out[static_cast<std::size_t>(t) * elements + i] = __float2bfloat16_rn(v);
}

// bf16_rn(p * m) of the exact product: hi = fl32(p * m) and its error lo (an FMA residual, exact)
// round together, so lo decides only a hi that lies exactly halfway between two BF16 values.
__device__ __forceinline__ std::uint16_t bf16_rn_product(float p, float m) {
    const float hi         = p * m;
    const float lo         = fmaf(p, m, -hi);
    const std::uint32_t b  = __float_as_uint(hi);
    const std::uint32_t lo16 = b & 0xFFFFU;
    std::uint32_t up;
    if (lo16 == 0x8000U && lo != 0.0F) {
        up = (lo > 0.0F) == (hi > 0.0F) ? 1U : 0U; // the exact magnitude lies above or below the tie
    } else {
        up = lo16 > 0x8000U || (lo16 == 0x8000U && (b & 0x10000U) != 0);
    }
    return static_cast<std::uint16_t>((b >> 16) + up);
}

// The eight packed E2M1 codes of one 16-element block. An NVFP4 row is nine bytes per 16 elements,
// so a block's codes are generally not 8-byte aligned; two aligned 8-byte loads and a 64-bit
// funnel shift replace eight single-byte loads. The out-of-range ends fall back to byte loads
// rather than reading past the tensor.
__device__ __forceinline__ std::uint64_t load_packed_codes(const std::uint8_t* p, const std::uint8_t* first,
                                                           const std::uint8_t* last) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t base    = address & ~std::uintptr_t(7U);
    if (base >= reinterpret_cast<std::uintptr_t>(first) && base + 16 <= reinterpret_cast<std::uintptr_t>(last)) {
        const uint2 low  = *reinterpret_cast<const uint2*>(base);
        const uint2 high = *reinterpret_cast<const uint2*>(base + 8);
        const std::uint64_t l = static_cast<std::uint64_t>(low.x) | (static_cast<std::uint64_t>(low.y) << 32);
        const std::uint64_t h = static_cast<std::uint64_t>(high.x) | (static_cast<std::uint64_t>(high.y) << 32);
        const unsigned shift  = static_cast<unsigned>(address & 7U) * 8U;
        return shift == 0U ? l : ((l >> shift) | (h << (64U - shift)));
    }
    std::uint64_t words = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { words |= static_cast<std::uint64_t>(p[i]) << (8 * i); }
    return words;
}

// E2M1 is a four-bit float: sign, a two-bit exponent with bias one, and one mantissa bit, whose
// subnormal is 0.5 — so the eight magnitudes are {0, 1/2, 1, 3/2, 2, 3, 4, 6}. Decoding it with
// integer arithmetic (rather than a sixteen-entry table indexed by the code nibble) keeps the
// table out of local memory, which a dynamically indexed array would cost a 64-byte stack frame.
__device__ __forceinline__ float e2m1_value(unsigned code) {
    const unsigned sign     = (code & 8U) << 28;
    const unsigned exponent = (code >> 1) & 3U, mantissa = code & 1U;
    const unsigned bits = exponent == 0U ? mantissa * 0x3F000000U : ((exponent + 126U) << 23) | (mantissa << 22);
    return __uint_as_float(bits | sign);
}

// One CTA per (task group, column); each task decodes one 16-element block of one head's row.
// A task's 16 outputs stay two-per-word in eight registers, and the store width follows the
// output tensor's alignment: every block's output starts at (h * width + 16 * b) elements, so a
// 16-byte-aligned tensor keeps every one of them 16-byte aligned.
template <bool VectorStore>
__global__ void __launch_bounds__(kThreads)
    embed_nvfp4_kernel(const std::uint8_t* __restrict__ rows, const float* __restrict__ multiplier, int row_bytes,
                       int heads, int width, int columns, int tasks_per_cta, std::uint16_t* __restrict__ out) {
    pdl::enter();
    const int t      = blockIdx.y;
    const int blocks = width / 16;
    const int tasks  = heads * blocks;
    const float m    = multiplier[0];
    const std::uint8_t* const first = rows;
    const std::uint8_t* const last  = rows + static_cast<std::size_t>(columns) * heads * row_bytes;
    const int begin = blockIdx.x * tasks_per_cta;
    const int end   = begin + tasks_per_cta < tasks ? begin + tasks_per_cta : tasks;
    for (int task = begin + threadIdx.x; task < end; task += blockDim.x) {
        const int h = task / blocks, b = task % blocks;
        const std::uint8_t* row = rows + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
        __nv_fp8_e4m3 code;
        code.__x = row[width / 2 + b];
        const float s           = static_cast<float>(code);
        const std::uint64_t codes = load_packed_codes(row + 8 * b, first, last);
        std::uint32_t words[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t pair = static_cast<std::uint8_t>(codes >> (8 * i));
            const std::uint32_t low  = bf16_rn_product(e2m1_value(pair & 15U) * s, m); // E2M1 times E4M3: exact in FP32
            const std::uint32_t high = bf16_rn_product(e2m1_value(pair >> 4U) * s, m);
            words[i]                 = low | (high << 16);
        }
        std::uint16_t* o =
            out + static_cast<std::size_t>(t) * heads * width + static_cast<std::size_t>(h) * width + 16 * b;
        if constexpr (VectorStore) {
            *reinterpret_cast<uint4*>(o)     = make_uint4(words[0], words[1], words[2], words[3]);
            *reinterpret_cast<uint4*>(o + 8) = make_uint4(words[4], words[5], words[6], words[7]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[2 * i]     = static_cast<std::uint16_t>(words[i] & 0xFFFFU);
                o[2 * i + 1] = static_cast<std::uint16_t>(words[i] >> 16);
            }
        }
    }
}

__device__ __forceinline__ float norm_value(bf16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float norm_value(float w) { return w; }

// V consecutive BF16 as floats: one 16-byte load when V is 8, which the launcher only selects for
// an aligned operand whose whole geometry is a multiple of 8 elements.
template <int V>
__device__ __forceinline__ void load_bf16x(const bf16* __restrict__ p, float* out) {
    static_assert(V == 1 || V == 8, "unsupported vector width");
    if constexpr (V == 8) {
        union { uint4 bits; __nv_bfloat162 half[4]; } raw;
        raw.bits = *reinterpret_cast<const uint4*>(p);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 pair = __bfloat1622float2(raw.half[j]);
            out[2 * j]        = pair.x;
            out[2 * j + 1]    = pair.y;
        }
    } else {
        out[0] = __bfloat162float(p[0]);
    }
}

template <int V>
__device__ __forceinline__ void store_bf16x(bf16* __restrict__ p, const float* in) {
    static_assert(V == 1 || V == 8, "unsupported vector width");
    if constexpr (V == 8) {
        union { uint4 bits; __nv_bfloat162 half[4]; } raw;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            raw.half[j] = __floats2bfloat162_rn(in[2 * j], in[2 * j + 1]);
        }
        *reinterpret_cast<uint4*>(p) = raw.bits;
    } else {
        p[0] = __float2bfloat16_rn(in[0]);
    }
}

template <class Norm, int V>
__device__ __forceinline__ void load_normx(const Norm* __restrict__ p, float* out) {
    static_assert(V == 1 || V == 8, "unsupported vector width");
    if constexpr (V == 8 && sizeof(Norm) == 4) {
        const float4* const f = reinterpret_cast<const float4*>(p);
        const float4 low      = f[0];
        const float4 high     = f[1];
        out[0]                = low.x;
        out[1]                = low.y;
        out[2]                = low.z;
        out[3]                = low.w;
        out[4]                = high.x;
        out[5]                = high.y;
        out[6]                = high.z;
        out[7]                = high.w;
    } else if constexpr (V == 8) {
        load_bf16x<8>(reinterpret_cast<const bf16*>(p), out);
    } else {
        out[0] = norm_value(p[0]);
    }
}

// One CTA per (stream, column). Each thread holds its whole slice of key, residual and value in
// registers, so every operand is read from memory once; the two statistics reductions share a
// barrier pair. V is the elements per access and K the maximum number of accesses per thread.
template <class Norm, int V, int K>
__global__ void __launch_bounds__(kGateMaxThreads)
    gate_kernel(const bf16* __restrict__ key, const bf16* __restrict__ value,
                const bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm, int hidden,
                int width, int key_stride, int value_stride, float eps, bf16* __restrict__ gated,
                bf16* __restrict__ normalized) {
    pdl::enter();
    __shared__ float2 scratch[kGateMaxThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base     = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const std::size_t key_base = static_cast<std::size_t>(t) * key_stride + static_cast<std::size_t>(s) * hidden;
    const std::size_t val_base = static_cast<std::size_t>(t) * value_stride;
    float kf[K][V], qf[K][V], vf[K][V];
    float key_sq = 0.0F, query_sq = 0.0F;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        load_bf16x<V>(key + key_base + d, &kf[i][0]);
        load_bf16x<V>(residual + base + d, &qf[i][0]);
        load_bf16x<V>(value + val_base + d, &vf[i][0]);
#pragma unroll
        for (int j = 0; j < V; ++j) {
            key_sq += kf[i][j] * kf[i][j];
            query_sq += qf[i][j] * qf[i][j];
        }
    }
    const float2 squares = block_sum2(key_sq, query_sq, scratch);
    const float key_inv   = rsqrtf(squares.x / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(squares.y / static_cast<float>(hidden) + eps);
    float dot = 0.0F;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        float kn[V], qn[V];
        load_normx<Norm, V>(key_norm + c, kn);
        load_normx<Norm, V>(query_norm + c, qn);
#pragma unroll
        for (int j = 0; j < V; ++j) {
            dot += (kf[i][j] * key_inv * (1.0F + kn[j])) * (qf[i][j] * query_inv * (1.0F + qn[j]));
        }
    }
    float g = block_sum(dot, scratch) / sqrtf(static_cast<float>(hidden));
    g       = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate = sigmoidf(g);
    float gated_sq   = 0.0F;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        float scaled[V];
#pragma unroll
        for (int j = 0; j < V; ++j) {
            scaled[j] = gate * vf[i][j];
            gated_sq += scaled[j] * scaled[j];
        }
        store_bf16x<V>(gated + base + d, scaled);
    }
    const float gated_inv = rsqrtf(block_sum(gated_sq, scratch) / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        float cn[V], out[V];
        load_normx<Norm, V>(conv_norm + c, cn);
#pragma unroll
        for (int j = 0; j < V; ++j) { out[j] = (gate * vf[i][j]) * gated_inv * (1.0F + cn[j]); }
        store_bf16x<V>(normalized + base + d, out);
    }
}

// The fused decode gate + state convolution (one column per sequence, an in-place or moved state
// update): one CTA per (stream, sequence). The convolution is channel-independent and its channels
// are exactly the gate's, so each thread folds the convolution of its own 8 channels into the
// gate's registers: the gated and normalized values are rounded to BF16 exactly as gate_kernel
// stores them, stay unpacked only into floats, and never reach global memory. The history shift is
// streamed through one uint4 in increasing order (as state_kernel does), so an in-place update
// (source == destination) never reads an element it already overwrote and the register array never
// grows with SPAN. The accumulation order (history taps ascending, then the latest column), the
// rounding boundaries and the residual expression are conv_state_vec8_kernel's, so the result is
// bitwise-identical to the composed gate + conv_state pair.
template <class Norm, int SPAN>
__global__ void __launch_bounds__(kGateMaxThreads)
    gate_conv_vec8_kernel(const bf16* __restrict__ key, const bf16* __restrict__ value,
                          bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                          const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm,
                          int hidden, int width, int key_stride, int value_stride, float eps,
                          const bf16* __restrict__ weight, int taps, int dilation, bf16* states,
                          const std::int32_t* __restrict__ source_slots,
                          const std::int32_t* __restrict__ destination_slots) {
    pdl::enter();
    __shared__ float2 scratch[kGateMaxThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base     = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const std::size_t key_base = static_cast<std::size_t>(t) * key_stride + static_cast<std::size_t>(s) * hidden;
    const std::size_t val_base = static_cast<std::size_t>(t) * value_stride;
    const int d                 = threadIdx.x * 8;
    const bool active           = d < hidden;
    float kf[8], qf[8], vf[8];
    float key_sq = 0.0F, query_sq = 0.0F;
    if (active) {
        load_bf16x<8>(key + key_base + d, kf);
        load_bf16x<8>(residual + base + d, qf);
        load_bf16x<8>(value + val_base + d, vf);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            key_sq += kf[j] * kf[j];
            query_sq += qf[j] * qf[j];
        }
    }
    const float2 squares = block_sum2(key_sq, query_sq, scratch);
    const float key_inv   = rsqrtf(squares.x / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(squares.y / static_cast<float>(hidden) + eps);
    float dot = 0.0F;
    if (active) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        float kn[8], qn[8];
        load_normx<Norm, 8>(key_norm + c, kn);
        load_normx<Norm, 8>(query_norm + c, qn);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dot += (kf[j] * key_inv * (1.0F + kn[j])) * (qf[j] * query_inv * (1.0F + qn[j]));
        }
    }
    float g = block_sum(dot, scratch) / sqrtf(static_cast<float>(hidden));
    g       = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate = sigmoidf(g);
    float gated_sq   = 0.0F;
    if (active) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = gate * vf[j];
            gated_sq += v * v;
        }
    }
    const float gated_inv = rsqrtf(block_sum(gated_sq, scratch) / static_cast<float>(hidden) + eps);
    uint4 gated_bits = make_uint4(0U, 0U, 0U, 0U), norm_bits = make_uint4(0U, 0U, 0U, 0U);
    if (active) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        float cn[8];
        load_normx<Norm, 8>(conv_norm + c, cn);
        union { uint4 bits; __nv_bfloat162 half[4]; } gb, nb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 scaled = make_float2(gate * vf[2 * j], gate * vf[2 * j + 1]);
            const float2 norm   = make_float2(scaled.x * gated_inv * (1.0F + cn[2 * j]),
                                              scaled.y * gated_inv * (1.0F + cn[2 * j + 1]));
            gb.half[j]          = __floats2bfloat162_rn(scaled.x, scaled.y);
            nb.half[j]          = __floats2bfloat162_rn(norm.x, norm.y);
        }
        gated_bits = gb.bits;
        norm_bits  = nb.bits;
    }
    if (active) {
        const std::size_t channels = static_cast<std::size_t>(width); // the conv's channels are the gate's width
        const std::size_t words    = channels / 8;                     // uint4-strided history
        const std::size_t word     = (static_cast<std::size_t>(s) * hidden + d) / 8;
        const uint4* const history = reinterpret_cast<const uint4*>(states) +
                                     static_cast<std::size_t>(source_slots[t]) * SPAN * words + word;
        uint4* const shifted = reinterpret_cast<uint4*>(states) +
                               static_cast<std::size_t>(destination_slots[t]) * SPAN * words + word;
        const bf16* const channel = weight + static_cast<std::size_t>(s) * hidden + d;
        float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        uint4 v      = history[0];
#pragma unroll
        for (int k = 0; k + 1 < SPAN; ++k) {
            const int tap = k / dilation;
            if (tap * dilation == k && tap + 1 < taps) {
                float w[8], x[8];
                load_bf16x<8>(channel + static_cast<std::size_t>(tap) * channels, w);
                load_bf16x<8>(reinterpret_cast<const bf16*>(&v), x);
#pragma unroll
                for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
            }
            const uint4 next = history[static_cast<std::size_t>(k + 1) * words];
            shifted[static_cast<std::size_t>(k) * words] = next;
            v                                            = next;
        }
        {
            const int tap = (SPAN - 1) / dilation;
            if (tap * dilation == SPAN - 1 && tap + 1 < taps) {
                float w[8], x[8];
                load_bf16x<8>(channel + static_cast<std::size_t>(tap) * channels, w);
                load_bf16x<8>(reinterpret_cast<const bf16*>(&v), x);
#pragma unroll
                for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
            }
            float w[8], x[8];
            load_bf16x<8>(channel + static_cast<std::size_t>(taps - 1) * channels, w);
            load_bf16x<8>(reinterpret_cast<const bf16*>(&norm_bits), x);
#pragma unroll
            for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
            shifted[static_cast<std::size_t>(SPAN - 1) * words] = norm_bits;
        }
        float g8[8];
        load_bf16x<8>(reinterpret_cast<const bf16*>(&gated_bits), g8);
        float r[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { r[u] = qf[u] + (g8[u] + acc[u] * sigmoidf(acc[u])); }
        store_bf16x<8>(residual + base + d, r);
    }
}

// The gate for a geometry or alignment the vectorized kernel does not cover: the strided scalar
// walk, still with the shared statistics barrier pair.
template <class Norm>
__global__ void __launch_bounds__(kThreads)
    gate_kernel_scalar(const bf16* __restrict__ key, const bf16* __restrict__ value,
                       const bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                       const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm, int hidden,
                       int width, int key_stride, int value_stride, float eps, bf16* __restrict__ gated,
                       bf16* __restrict__ normalized) {
    __shared__ float2 scratch[kThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base     = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const std::size_t key_base = static_cast<std::size_t>(t) * key_stride + static_cast<std::size_t>(s) * hidden;
    const std::size_t val_base = static_cast<std::size_t>(t) * value_stride;
    float key_sq = 0.0F, query_sq = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float k = __bfloat162float(key[key_base + d]);
        const float q = __bfloat162float(residual[base + d]);
        key_sq += k * k;
        query_sq += q * q;
    }
    const float2 squares  = block_sum2(key_sq, query_sq, scratch);
    const float key_inv   = rsqrtf(squares.x / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(squares.y / static_cast<float>(hidden) + eps);
    float dot = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        const float k = __bfloat162float(key[key_base + d]) * key_inv * (1.0F + norm_value(key_norm[c]));
        const float q = __bfloat162float(residual[base + d]) * query_inv * (1.0F + norm_value(query_norm[c]));
        dot += k * q;
    }
    float g = block_sum(dot, scratch) / sqrtf(static_cast<float>(hidden));
    g       = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate = sigmoidf(g);
    float gated_sq   = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float v = gate * __bfloat162float(value[val_base + d]);
        gated[base + d] = __float2bfloat16_rn(v);
        gated_sq += v * v;
    }
    const float gated_inv = rsqrtf(block_sum(gated_sq, scratch) / static_cast<float>(hidden) + eps);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        const float v = gate * __bfloat162float(value[val_base + d]);
        normalized[base + d] = __float2bfloat16_rn(v * gated_inv * (1.0F + norm_value(conv_norm[c])));
    }
}

// One thread per (channel, column): reads the pre-update history.
__global__ void conv_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                            const bf16* __restrict__ weight, int taps, int dilation, int span,
                            int channels, int width, const bf16* __restrict__ states,
                            const std::int32_t* __restrict__ source_slots, int columns,
                            bf16* __restrict__ residual) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= channels || t >= columns) { return; }
    const int sequence = t / width, local = t % width;
    const int slot     = source_slots[sequence];
    float acc          = 0.0F;
    for (int j = 0; j < taps; ++j) {
        const int source = local - (taps - 1 - j) * dilation;
        float x;
        if (source >= 0) {
            x = __bfloat162float(normalized[static_cast<std::size_t>(sequence * width + source) * channels + c]);
        } else {
            x = __bfloat162float(states[(static_cast<std::size_t>(slot) * span + (span + source)) * channels + c]);
        }
        acc += __bfloat162float(weight[static_cast<std::size_t>(j) * channels + c]) * x;
    }
    const std::size_t i = static_cast<std::size_t>(t) * channels + c;
    const float out     = __bfloat162float(gated[i]) + acc * sigmoidf(acc);
    residual[i]         = __float2bfloat16_rn(__bfloat162float(residual[i]) + out);
}

__global__ void __launch_bounds__(kThreads)
    conv_vec8_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                     const bf16* __restrict__ weight, int taps, int dilation, int span, int channels,
                     int width, const bf16* __restrict__ states,
                     const std::int32_t* __restrict__ source_slots, int columns,
                     bf16* __restrict__ residual) {
    pdl::enter();
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int t = blockIdx.y;
    if (c >= channels || t >= columns) { return; }
    const int sequence = t / width, local = t % width;
    const int slot     = source_slots[sequence];
    float acc[8]       = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int j = 0; j < taps; ++j) {
        const int source = local - (taps - 1 - j) * dilation;
        float x[8], w[8];
        if (source >= 0) {
            load_bf16x<8>(normalized + static_cast<std::size_t>(sequence * width + source) * channels + c, x);
        } else {
            load_bf16x<8>(states + (static_cast<std::size_t>(slot) * span + (span + source)) * channels + c, x);
        }
        load_bf16x<8>(weight + static_cast<std::size_t>(j) * channels + c, w);
#pragma unroll
        for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
    }
    const std::size_t i = static_cast<std::size_t>(t) * channels + c;
    float g[8], r[8];
    load_bf16x<8>(gated + i, g);
    load_bf16x<8>(residual + i, r);
#pragma unroll
    for (int u = 0; u < 8; ++u) { r[u] += g[u] + acc[u] * sigmoidf(acc[u]); }
    store_bf16x<8>(residual + i, r);
}

// The decode convolution: one column per sequence, so every tap but the last reads the history and
// the shift moves it forward by exactly one column. One thread per (channel, sequence) reads the
// span once, injects the residual and rewrites the slot, which is what conv_kernel and a separate
// state_kernel used to do in two passes over the history. SPAN is (taps - 1) * dilation, so the
// history taps are the multiples of `dilation` below it and the register array stays indexed by
// compile-time constants. As in state_kernel, a sequence's threads own their source and
// destination slot, so an in-place update (source == destination) is safe.
template <int SPAN>
__global__ void __launch_bounds__(kThreads)
    conv_state_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                      const bf16* __restrict__ weight, int taps, int dilation, int channels,
                      bf16* __restrict__ states, const std::int32_t* __restrict__ source_slots,
                      const std::int32_t* __restrict__ destination_slots, bf16* __restrict__ residual) {
    const int c        = blockIdx.x * blockDim.x + threadIdx.x;
    const int sequence = blockIdx.y;
    if (c >= channels) { return; }
    const int src = source_slots[sequence], dst = destination_slots[sequence];
    const std::size_t column = static_cast<std::size_t>(sequence) * channels; // width == 1: t == sequence
    const bf16* const history = states + static_cast<std::size_t>(src) * SPAN * channels + c;
    bf16* const shifted       = states + static_cast<std::size_t>(dst) * SPAN * channels + c;
    bf16 past[SPAN];
#pragma unroll
    for (int k = 0; k < SPAN; ++k) { past[k] = history[static_cast<std::size_t>(k) * channels]; }
    const bf16 latest = normalized[column + c];
    float acc         = 0.0F;
#pragma unroll
    for (int k = 0; k < SPAN; ++k) {
        const int tap = k / dilation;
        if (tap * dilation == k && tap + 1 < taps) {
            acc += __bfloat162float(weight[static_cast<std::size_t>(tap) * channels + c]) * __bfloat162float(past[k]);
        }
    }
    acc += __bfloat162float(weight[static_cast<std::size_t>(taps - 1) * channels + c]) * __bfloat162float(latest);
    const float out = __bfloat162float(gated[column + c]) + acc * sigmoidf(acc);
    residual[column + c] = __float2bfloat16_rn(__bfloat162float(residual[column + c]) + out);
#pragma unroll
    for (int k = 0; k + 1 < SPAN; ++k) { shifted[static_cast<std::size_t>(k) * channels] = past[k + 1]; }
    if (SPAN > 0) { shifted[static_cast<std::size_t>(SPAN - 1) * channels] = latest; }
}

template <int SPAN>
__global__ void __launch_bounds__(kThreads)
    conv_state_vec8_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                           const bf16* __restrict__ weight, int taps, int dilation, int channels,
                           bf16* __restrict__ states, const std::int32_t* __restrict__ source_slots,
                           const std::int32_t* __restrict__ destination_slots, bf16* __restrict__ residual) {
    pdl::enter();
    const int c        = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int sequence = blockIdx.y;
    if (c >= channels) { return; }
    const int src             = source_slots[sequence], dst = destination_slots[sequence];
    const std::size_t column  = static_cast<std::size_t>(sequence) * channels;
    const bf16* const history = states + static_cast<std::size_t>(src) * SPAN * channels + c;
    bf16* const shifted       = states + static_cast<std::size_t>(dst) * SPAN * channels + c;
    uint4 past[SPAN];
#pragma unroll
    for (int k = 0; k < SPAN; ++k) {
        past[k] = *reinterpret_cast<const uint4*>(history + static_cast<std::size_t>(k) * channels);
    }
    const uint4 latest = *reinterpret_cast<const uint4*>(normalized + column + c);
    float acc[8]       = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int k = 0; k < SPAN; ++k) {
        const int tap = k / dilation;
        if (tap * dilation == k && tap + 1 < taps) {
            float w[8], x[8];
            load_bf16x<8>(weight + static_cast<std::size_t>(tap) * channels + c, w);
            load_bf16x<8>(reinterpret_cast<const bf16*>(&past[k]), x);
#pragma unroll
            for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
        }
    }
    {
        float w[8], x[8];
        load_bf16x<8>(weight + static_cast<std::size_t>(taps - 1) * channels + c, w);
        load_bf16x<8>(reinterpret_cast<const bf16*>(&latest), x);
#pragma unroll
        for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
    }
    float g[8], r[8];
    load_bf16x<8>(gated + column + c, g);
    load_bf16x<8>(residual + column + c, r);
#pragma unroll
    for (int u = 0; u < 8; ++u) { r[u] += g[u] + acc[u] * sigmoidf(acc[u]); }
    store_bf16x<8>(residual + column + c, r);
#pragma unroll
    for (int k = 0; k + 1 < SPAN; ++k) {
        *reinterpret_cast<uint4*>(shifted + static_cast<std::size_t>(k) * channels) = past[k + 1];
    }
    if (SPAN > 0) { *reinterpret_cast<uint4*>(shifted + static_cast<std::size_t>(SPAN - 1) * channels) = latest; }
}

// One thread per (channel, sequence) shifts the history in increasing order, so an in-place
// update (source == destination slot) never reads an element it already overwrote.
__global__ void state_kernel(const bf16* __restrict__ normalized, int span, int channels, int width,
                             int sequences, bf16* __restrict__ states,
                             const std::int32_t* __restrict__ source_slots,
                             const std::int32_t* __restrict__ destination_slots) {
    const int c        = blockIdx.x * blockDim.x + threadIdx.x;
    const int sequence = blockIdx.y;
    if (c >= channels || sequence >= sequences) { return; }
    const int src = source_slots[sequence], dst = destination_slots[sequence];
    for (int k = 0; k < span; ++k) {
        const int index = width - span + k; // column relative to the chunk start
        bf16 v;
        if (index >= 0) {
            v = normalized[static_cast<std::size_t>(sequence * width + index) * channels + c];
        } else {
            v = states[(static_cast<std::size_t>(src) * span + (span + index)) * channels + c];
        }
        states[(static_cast<std::size_t>(dst) * span + k) * channels + c] = v;
    }
}

__global__ void __launch_bounds__(kThreads)
    state_vec8_kernel(const bf16* __restrict__ normalized, int span, int channels, int width, int sequences,
                      bf16* __restrict__ states, const std::int32_t* __restrict__ source_slots,
                      const std::int32_t* __restrict__ destination_slots) {
    pdl::enter();
    const int c        = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int sequence = blockIdx.y;
    if (c >= channels || sequence >= sequences) { return; }
    const int src = source_slots[sequence], dst = destination_slots[sequence];
    for (int k = 0; k < span; ++k) {
        const int index = width - span + k;
        uint4 v;
        if (index >= 0) {
            v = *reinterpret_cast<const uint4*>(normalized +
                                                static_cast<std::size_t>(sequence * width + index) * channels + c);
        } else {
            v = *reinterpret_cast<const uint4*>(
                states + (static_cast<std::size_t>(src) * span + (span + index)) * channels + c);
        }
        *reinterpret_cast<uint4*>(states + (static_cast<std::size_t>(dst) * span + k) * channels + c) = v;
    }
}

// ple_conv_commit: the state_kernel shift with a per-row column count, reading row b's columns
// at b * stride.
__global__ void commit_kernel(const bf16* __restrict__ normalized, int span, int channels, int stride,
                              const std::int32_t* __restrict__ commit_columns, bf16* __restrict__ states,
                              const std::int32_t* __restrict__ slots) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels) { return; }
    const int n = commit_columns[b];
    if (n <= 0) { return; }
    const int slot = slots[b];
    for (int k = 0; k < span; ++k) {
        const int index = n - span + k;
        bf16 v;
        if (index >= 0) {
            v = normalized[static_cast<std::size_t>(b * stride + index) * channels + c];
        } else {
            v = states[(static_cast<std::size_t>(slot) * span + (span + index)) * channels + c];
        }
        states[(static_cast<std::size_t>(slot) * span + k) * channels + c] = v;
    }
}

__global__ void __launch_bounds__(kThreads)
    commit_vec8_kernel(const bf16* __restrict__ normalized, int span, int channels, int stride,
                       const std::int32_t* __restrict__ commit_columns, bf16* __restrict__ states,
                       const std::int32_t* __restrict__ slots) {
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int b = blockIdx.y;
    if (c >= channels) { return; }
    const int n = commit_columns[b];
    if (n <= 0) { return; }
    const int slot = slots[b];
    for (int k = 0; k < span; ++k) {
        const int index = n - span + k;
        uint4 v;
        if (index >= 0) {
            v = *reinterpret_cast<const uint4*>(normalized + static_cast<std::size_t>(b * stride + index) * channels + c);
        } else {
            v = *reinterpret_cast<const uint4*>(
                states + (static_cast<std::size_t>(slot) * span + (span + index)) * channels + c);
        }
        *reinterpret_cast<uint4*>(states + (static_cast<std::size_t>(slot) * span + k) * channels + c) = v;
    }
}

void launch_embed_nvfp4(const std::uint8_t* rows, const float* multiplier, int row_bytes, int heads, int width,
                        int columns, std::uint16_t* out, cudaStream_t stream) {
    const int tasks = heads * (width / 16);
    int tasks_per_cta = tasks < kThreads ? tasks : kThreads;
    if (columns < kEmbedSpreadColumns && tasks > kEmbedCtaTasks) { tasks_per_cta = kEmbedCtaTasks; }
    const dim3 grid((tasks + tasks_per_cta - 1) / tasks_per_cta, columns);
    if (aligned<16>(out)) {
        (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream}, embed_nvfp4_kernel<true>,
                                   rows, multiplier, row_bytes, heads, width, columns, tasks_per_cta, out);
    } else {
        (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream}, embed_nvfp4_kernel<false>,
                                   rows, multiplier, row_bytes, heads, width, columns, tasks_per_cta, out);
    }
}

} // namespace

void ple_embed(const Tensor& rows, const Tensor& scale, Tensor& out, cudaStream_t stream) {
    require(contiguous(rows, DType::U8) && contiguous(out, DType::BF16),
            "embed requires contiguous U8 rows and BF16 output");
    const int heads   = rows.ne[1];
    const int columns = rows.ne[2];
    const int row_bytes = rows.ne[0];
    require(columns > 0 && heads > 0, "embed shapes disagree");
    if (scale.dtype == DType::FP32) {
        require(contiguous(scale, DType::FP32), "NVFP4 embed requires a contiguous FP32 scale");
        const int embed_dim = out.ne[0];
        require(scale.numel() == 1 && out.ne[1] == columns && embed_dim % heads == 0,
                "embed shapes disagree");
        const int head_width = embed_dim / heads;
        require(head_width % 16 == 0 && row_bytes == head_width / 2 + head_width / 16,
                "NVFP4 row bytes disagree with the head width");
        launch_embed_nvfp4(static_cast<const std::uint8_t*>(rows.data), static_cast<const float*>(scale.data),
                           row_bytes, heads, head_width, columns, static_cast<std::uint16_t*>(out.data), stream);
        check_launch("embed");
        return;
    }
    require(contiguous(scale, DType::BF16), "embed requires a BF16 or FP32 scale");
    const int elements = row_bytes * heads;
    require(scale.numel() == 1 && out.ne[0] == elements && out.ne[1] == columns,
            "embed shapes disagree");
    (void)pdl::launch_consumer(
        pdl::LaunchConfig{dim3((elements + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream},
        embed_kernel, static_cast<const std::uint8_t*>(rows.data), static_cast<const bf16*>(scale.data), elements,
        static_cast<bf16*>(out.data));
    check_launch("embed");
}

void ple_embed_nvfp4(const Tensor& rows, const Tensor& multiplier, Tensor& out, cudaStream_t stream) {
    require(contiguous(rows, DType::U8) && contiguous(multiplier, DType::FP32) && contiguous(out, DType::BF16),
            "NVFP4 embed requires contiguous U8 rows, an FP32 multiplier and BF16 output");
    const int row_bytes = rows.ne[0], heads = rows.ne[1], columns = rows.ne[2];
    require(row_bytes % 9 == 0 && heads > 0 && columns > 0 && multiplier.numel() == 1,
            "NVFP4 rows must be whole 16-element blocks (8 code bytes and a scale each)");
    const int width = row_bytes / 9 * 16;
    require(out.ne[0] == heads * width && out.ne[1] == columns, "NVFP4 embed shapes disagree");
    launch_embed_nvfp4(static_cast<const std::uint8_t*>(rows.data), static_cast<const float*>(multiplier.data),
                       row_bytes, heads, width, columns, static_cast<std::uint16_t*>(out.data), stream);
    check_launch("embed nvfp4");
}

void ple_gate(const Tensor& key, const Tensor& value, const Tensor& residual,
              const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
              std::int32_t streams, float eps, Tensor& gated, Tensor& normalized,
              cudaStream_t stream) {
    const int width = key.ne[0], columns = key.ne[1];
    const GateGeometry geometry =
        gate_geometry(key, value, residual, key_norm, query_norm, conv_norm, streams, eps, width, columns);
    for (const Tensor* t : {static_cast<const Tensor*>(&gated), static_cast<const Tensor*>(&normalized)}) {
        require(contiguous(*t, DType::BF16) && t->ne[0] == width && t->ne[1] == columns,
                "gate requires contiguous BF16 tensors");
    }
    const int hidden       = geometry.hidden;
    const int key_stride   = geometry.key_stride;
    const int value_stride = geometry.value_stride;
    const auto launch = [&]<class Norm>() {
        const auto* key_p   = static_cast<const bf16*>(key.data);
        const auto* value_p = static_cast<const bf16*>(value.data);
        const auto* residual_p = static_cast<const bf16*>(residual.data);
        const auto* key_norm_p  = static_cast<const Norm*>(key_norm.data);
        const auto* query_norm_p = static_cast<const Norm*>(query_norm.data);
        const auto* conv_norm_p  = static_cast<const Norm*>(conv_norm.data);
        auto* gated_p      = static_cast<bf16*>(gated.data);
        auto* normalized_p = static_cast<bf16*>(normalized.data);
        const dim3 grid(streams, columns);
        // Eight elements per access needs every operand 16-byte aligned and every index multiple
        // of 8. A cached slice is at most one access per thread at that width and at most eight at
        // the scalar width, which keeps the three cached operands inside the 64 registers a
        // 1,024-thread block allows; anything larger takes the scalar walk, which caches nothing.
        const bool vector8 =
            hidden % 8 == 0 && width % 8 == 0 && key_stride % 8 == 0 && value_stride % 8 == 0 &&
            aligned<16>(key_p) && aligned<16>(value_p) && aligned<16>(residual_p) && aligned<16>(gated_p) &&
            aligned<16>(normalized_p) && aligned<16>(key_norm_p) && aligned<16>(query_norm_p) &&
            aligned<16>(conv_norm_p) && hidden <= kGateMaxThreads * 8;
        const int vector     = vector8 ? 8 : 1;
        const int slices     = hidden / vector;
        int block            = ((slices + 31) / 32) * 32;
        if (block > kGateMaxThreads) { block = kGateMaxThreads; }
        const int per_thread = (slices + block - 1) / block;
        const auto run = [&]<int V, int K>() {
            // The vectorized gate is the PLE chain's main path: a programmatic dependent when the
            // stream captures a graph, an ordinary launch otherwise (pdl::launch_consumer).
            (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(block), 0, stream}, gate_kernel<Norm, V, K>,
                                       key_p, value_p, residual_p, key_norm_p, query_norm_p, conv_norm_p, hidden,
                                       width, key_stride, value_stride, eps, gated_p, normalized_p);
        };
        const auto run_scalar = [&]() {
            gate_kernel_scalar<Norm><<<grid, kThreads, 0, stream>>>(key_p, value_p, residual_p, key_norm_p,
                                                                    query_norm_p, conv_norm_p, hidden, width,
                                                                    key_stride, value_stride, eps, gated_p,
                                                                    normalized_p);
        };
        if (vector8) {
            run.template operator()<8, 1>();
        } else if (per_thread <= 1) {
            run.template operator()<1, 1>();
        } else if (per_thread <= 2) {
            run.template operator()<1, 2>();
        } else if (per_thread <= 4) {
            run.template operator()<1, 4>();
        } else if (per_thread <= 8) {
            run.template operator()<1, 8>();
        } else {
            run_scalar();
        }
    };
    if (geometry.norm == DType::FP32) {
        launch.template operator()<float>();
    } else {
        launch.template operator()<bf16>();
    }
    check_launch("gate");
}

void ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                     std::int32_t dilation, Tensor& states, const Tensor& source_slots,
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream) {
    require(contiguous(gated, DType::BF16) && contiguous(normalized, DType::BF16) &&
                contiguous(weight, DType::BF16) && contiguous(states, DType::BF16) &&
                contiguous(residual, DType::BF16) && contiguous(source_slots, DType::I32) &&
                (destination_slots.data == nullptr || contiguous(destination_slots, DType::I32)),
            "conv requires contiguous tensors");
    const int channels = gated.ne[0], columns = gated.ne[1], taps = weight.ne[1];
    const int sequences = source_slots.ne[0];
    const bool update   = destination_slots.data != nullptr;
    require(dilation > 0 && taps > 1 && weight.ne[0] == channels && normalized.ne[0] == channels &&
                normalized.ne[1] == columns && residual.ne[0] == channels && residual.ne[1] == columns &&
                sequences > 0 && (!update || destination_slots.ne[0] == sequences) && columns % sequences == 0,
            "conv shapes disagree");
    const int span = (taps - 1) * dilation;
    require(states.ne[0] == channels && states.ne[1] == span, "conv state must be [C, span, slots]");
    const int width = columns / sequences;
    const auto* gated_p      = static_cast<const bf16*>(gated.data);
    const auto* normalized_p = static_cast<const bf16*>(normalized.data);
    const auto* weight_p     = static_cast<const bf16*>(weight.data);
    const auto* source_p     = static_cast<const std::int32_t*>(source_slots.data);
    const auto* destination_p = static_cast<const std::int32_t*>(destination_slots.data);
    auto* states_p   = static_cast<bf16*>(states.data);
    auto* residual_p = static_cast<bf16*>(residual.data);
    const bool vec8 = channels % 8 == 0 && aligned<16>(gated_p) && aligned<16>(normalized_p) &&
                      aligned<16>(weight_p) && aligned<16>(states_p) && aligned<16>(residual_p);
    if (update && width == 1 && span >= 1 && span <= kFusedConvMaxSpan) {
        const int tasks = vec8 ? (channels / 8) : channels;
        const dim3 grid((tasks + kThreads - 1) / kThreads, sequences);
        const auto run = [&]<int SPAN>() {
            if (vec8) {
                (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(kThreads), 0, stream},
                                           conv_state_vec8_kernel<SPAN>, gated_p, normalized_p, weight_p, taps,
                                           dilation, channels, states_p, source_p, destination_p, residual_p);
            } else {
                conv_state_kernel<SPAN><<<grid, kThreads, 0, stream>>>(
                    gated_p, normalized_p, weight_p, taps, dilation, channels, states_p, source_p, destination_p,
                    residual_p);
            }
        };
        switch (span) {
#define INFERNIX_PLE_FUSED_STATE(n)                                                                        \
    case n:                                                                                                \
        run.template operator()<n>();                                                                      \
        break;
            INFERNIX_PLE_FUSED_STATE(1)
            INFERNIX_PLE_FUSED_STATE(2)
            INFERNIX_PLE_FUSED_STATE(3)
            INFERNIX_PLE_FUSED_STATE(4)
            INFERNIX_PLE_FUSED_STATE(5)
            INFERNIX_PLE_FUSED_STATE(6)
            INFERNIX_PLE_FUSED_STATE(7)
            INFERNIX_PLE_FUSED_STATE(8)
            INFERNIX_PLE_FUSED_STATE(9)
            INFERNIX_PLE_FUSED_STATE(10)
            INFERNIX_PLE_FUSED_STATE(11)
            INFERNIX_PLE_FUSED_STATE(12)
            INFERNIX_PLE_FUSED_STATE(13)
            INFERNIX_PLE_FUSED_STATE(14)
            INFERNIX_PLE_FUSED_STATE(15)
            INFERNIX_PLE_FUSED_STATE(16)
#undef INFERNIX_PLE_FUSED_STATE
        }
        check_launch("conv state");
        return;
    }
    if (vec8) {
        (void)pdl::launch_consumer(pdl::LaunchConfig{dim3((channels / 8 + kThreads - 1) / kThreads, columns),
                                                     dim3(kThreads), 0, stream},
                                   conv_vec8_kernel, gated_p, normalized_p, weight_p, taps, dilation, span, channels,
                                   width, states_p, source_p, columns, residual_p);
    } else {
        conv_kernel<<<dim3((channels + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
            gated_p, normalized_p, weight_p, taps, dilation, span, channels, width, states_p, source_p, columns,
            residual_p);
    }
    check_launch("conv");
    if (!update) { return; }
    if (vec8) {
        (void)pdl::launch_consumer(pdl::LaunchConfig{dim3((channels / 8 + kThreads - 1) / kThreads, sequences),
                                                     dim3(kThreads), 0, stream},
                                   state_vec8_kernel, normalized_p, span, channels, width, sequences, states_p,
                                   source_p, destination_p);
    } else {
        state_kernel<<<dim3((channels + kThreads - 1) / kThreads, sequences), kThreads, 0, stream>>>(
            normalized_p, span, channels, width, sequences, states_p, source_p, destination_p);
    }
    check_launch("state");
}

void ple_gate_conv_inject(const Tensor& key, const Tensor& value, Tensor& residual,
                          const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
                          std::int32_t streams, float eps, const Tensor& weight, std::int32_t dilation,
                          Tensor& states, const Tensor& source_slots, const Tensor& destination_slots,
                          Tensor& gated, Tensor& normalized, cudaStream_t stream) {
    const int width   = key.ne[0], columns = key.ne[1];
    const int taps    = weight.ne[1];
    const int span    = (taps - 1) * static_cast<int>(dilation);
    const bool update = destination_slots.data != nullptr;
    // The fused kernel's preconditions, everything the composed pair would validate except the
    // scratch it does not use; an unmet condition (or an invalid shape, which throws below or in
    // the composed fallback) leaves the exact composed route.
    const bool fused =
        update && source_slots.data != nullptr && contiguous(source_slots, DType::I32) &&
        contiguous(destination_slots, DType::I32) && source_slots.ne[0] == columns &&
        destination_slots.ne[0] == columns && taps > 1 && span >= 1 && span <= kFusedConvMaxSpan &&
        dilation > 0 && contiguous(weight, DType::BF16) && weight.ne[0] == width &&
        contiguous(states, DType::BF16) && states.ne[0] == width && states.ne[1] == span;
    if (fused) {
        const GateGeometry geometry =
            gate_geometry(key, value, residual, key_norm, query_norm, conv_norm, streams, eps, width, columns);
        const auto* key_p    = static_cast<const bf16*>(key.data);
        const auto* value_p  = static_cast<const bf16*>(value.data);
        auto* residual_p     = static_cast<bf16*>(residual.data);
        const auto* weight_p = static_cast<const bf16*>(weight.data);
        auto* states_p       = static_cast<bf16*>(states.data);
        const auto* source_p = static_cast<const std::int32_t*>(source_slots.data);
        const auto* dest_p   = static_cast<const std::int32_t*>(destination_slots.data);
        const bool vector8   = geometry.hidden % 8 == 0 && geometry.key_stride % 8 == 0 &&
                              geometry.value_stride % 8 == 0 && geometry.hidden <= kGateMaxThreads * 8 &&
                              aligned<16>(key_p) && aligned<16>(value_p) && aligned<16>(residual_p) &&
                              aligned<16>(weight_p) && aligned<16>(states_p) && aligned<16>(key_norm.data) &&
                              aligned<16>(query_norm.data) && aligned<16>(conv_norm.data);
        if (vector8) {
            const auto launch = [&]<class Norm>() {
                const auto* key_norm_p   = static_cast<const Norm*>(key_norm.data);
                const auto* query_norm_p = static_cast<const Norm*>(query_norm.data);
                const auto* conv_norm_p  = static_cast<const Norm*>(conv_norm.data);
                int block = ((geometry.hidden / 8 + 31) / 32) * 32;
                if (block > kGateMaxThreads) { block = kGateMaxThreads; }
                const pdl::LaunchConfig cfg{dim3(streams, columns), dim3(block), 0, stream};
                const auto run = [&]<int SPAN>() {
                    (void)pdl::launch_consumer(cfg, gate_conv_vec8_kernel<Norm, SPAN>, key_p, value_p, residual_p,
                                               key_norm_p, query_norm_p, conv_norm_p, geometry.hidden, width,
                                               geometry.key_stride, geometry.value_stride, eps, weight_p, taps,
                                               static_cast<int>(dilation), states_p, source_p, dest_p);
                };
                switch (span) {
#define INFERNIX_PLE_GATE_CONV(n)                                                                            \
    case n:                                                                                                  \
        run.template operator()<n>();                                                                        \
        break;
                    INFERNIX_PLE_GATE_CONV(1)
                    INFERNIX_PLE_GATE_CONV(2)
                    INFERNIX_PLE_GATE_CONV(3)
                    INFERNIX_PLE_GATE_CONV(4)
                    INFERNIX_PLE_GATE_CONV(5)
                    INFERNIX_PLE_GATE_CONV(6)
                    INFERNIX_PLE_GATE_CONV(7)
                    INFERNIX_PLE_GATE_CONV(8)
                    INFERNIX_PLE_GATE_CONV(9)
                    INFERNIX_PLE_GATE_CONV(10)
                    INFERNIX_PLE_GATE_CONV(11)
                    INFERNIX_PLE_GATE_CONV(12)
                    INFERNIX_PLE_GATE_CONV(13)
                    INFERNIX_PLE_GATE_CONV(14)
                    INFERNIX_PLE_GATE_CONV(15)
                    INFERNIX_PLE_GATE_CONV(16)
#undef INFERNIX_PLE_GATE_CONV
                }
            };
            if (geometry.norm == DType::FP32) {
                launch.template operator()<float>();
            } else {
                launch.template operator()<bf16>();
            }
            check_launch("gate conv");
            return;
        }
    }
    ple_gate(key, value, residual, key_norm, query_norm, conv_norm, streams, eps, gated, normalized, stream);
    ple_conv_inject(gated, normalized, weight, dilation, states, source_slots, destination_slots, residual, stream);
}

void ple_conv_commit(const Tensor& normalized, const Tensor& commit_columns, Tensor& states,
                     const Tensor& slots, cudaStream_t stream) {
    require(contiguous(normalized, DType::BF16) && contiguous(commit_columns, DType::I32) &&
                contiguous(states, DType::BF16) && contiguous(slots, DType::I32),
            "commit requires contiguous tensors");
    const int channels = normalized.ne[0], width = normalized.ne[1], rows = normalized.ne[2];
    require(rows > 0 && width > 0 && commit_columns.ne[0] == rows && slots.ne[0] == rows &&
                states.ne[0] == channels && states.ne[1] > 0,
            "commit shapes disagree");
    const auto* normalized_p = static_cast<const bf16*>(normalized.data);
    auto* states_p           = static_cast<bf16*>(states.data);
    if (channels % 8 == 0 && aligned<16>(normalized_p) && aligned<16>(states_p)) {
        commit_vec8_kernel<<<dim3((channels / 8 + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(
            normalized_p, states.ne[1], channels, width,
            static_cast<const std::int32_t*>(commit_columns.data), states_p,
            static_cast<const std::int32_t*>(slots.data));
    } else {
        commit_kernel<<<dim3((channels + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(
            normalized_p, states.ne[1], channels, width,
            static_cast<const std::int32_t*>(commit_columns.data), states_p,
            static_cast<const std::int32_t*>(slots.data));
    }
    check_launch("commit");
}

} // namespace infernix::ops
