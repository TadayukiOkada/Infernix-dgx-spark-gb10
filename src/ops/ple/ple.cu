#include "infernix/ops/ple.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
constexpr int kMaxPleHashHeads    = 64;

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

// One block reduction per barrier pair (scalar fallback) or single-barrier scratch (vectorized gate).
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

__device__ __forceinline__ float block_sum1_once(float value, float* scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w]; }
    return total;
}

__device__ __forceinline__ float3 block_sum3_once(float a, float b, float c, float (*scratch)[kGateMaxThreads / 32]) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_xor_sync(0xFFFFFFFFU, a, offset);
        b += __shfl_xor_sync(0xFFFFFFFFU, b, offset);
        c += __shfl_xor_sync(0xFFFFFFFFU, c, offset);
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        scratch[0][warp] = a;
        scratch[1][warp] = b;
        scratch[2][warp] = c;
    }
    __syncthreads();
    float3 total = make_float3(0.0F, 0.0F, 0.0F);
    for (int w = 0; w < blockDim.x / 32; ++w) {
        total.x += scratch[0][w];
        total.y += scratch[1][w];
        total.z += scratch[2][w];
    }
    return total;
}

enum class PleNextNorm : std::uint8_t { kNone = 0, kBf16 = 1, kFp32 = 2 };

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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1000
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
#endif

__device__ __forceinline__ float2 decode_e2m1_pair_f32(std::uint8_t pair) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    union {
        unsigned bits;
        __half2 h2;
    } out;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.f16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(out.bits)
        : "r"(static_cast<unsigned>(pair)));
    return __half22float2(out.h2);
#else
    return make_float2(e2m1_value(pair & 15U), e2m1_value(pair >> 4U));
#endif
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
            const std::uint8_t pair  = static_cast<std::uint8_t>(codes >> (8 * i));
            const float2 vals        = decode_e2m1_pair_f32(pair);
            const std::uint32_t low  = bf16_rn_product(vals.x * s, m); // E2M1 times E4M3: exact in FP32
            const std::uint32_t high = bf16_rn_product(vals.y * s, m);
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

template <bool VectorStore>
__global__ void __launch_bounds__(kThreads)
    hash_embed_nvfp4_kernel(const std::int32_t* __restrict__ windows, const std::uint64_t* __restrict__ multipliers,
                            const std::uint64_t* __restrict__ sizes, const std::uint64_t* __restrict__ offsets,
                            const std::uint8_t* __restrict__ resident_blocks, const float* __restrict__ multiplier,
                            int ngram_size, int heads_per_ngram, int eos_token_id, int heads, int width, int row_bytes,
                            int rows_per_block, int block_bytes, int columns, int tasks_per_cta,
                            std::uint16_t* __restrict__ out) {
    pdl::enter();
    __shared__ std::uint32_t sh_rows[kMaxPleHashHeads];
    const int t = blockIdx.y;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const int n_idx             = h / heads_per_ngram + 2;
        const std::int32_t* col_win = windows + static_cast<std::size_t>(t) * ngram_size;
        bool closed                 = false;
        std::uint64_t mixed         = static_cast<std::uint64_t>(col_win[0]) * multipliers[0];
        for (int s = 1; s < n_idx; ++s) {
            const std::int32_t w = col_win[s];
            if (w == eos_token_id) { closed = true; }
            const std::uint64_t tok = static_cast<std::uint64_t>(closed ? eos_token_id : w);
            mixed ^= tok * multipliers[s];
        }
        sh_rows[h] = static_cast<std::uint32_t>(mixed % sizes[h] + offsets[h]);
    }
    __syncthreads();

    const int blocks = width / 16;
    const int tasks  = heads * blocks;
    const float m    = multiplier[0];
    const int begin  = blockIdx.x * tasks_per_cta;
    const int end    = begin + tasks_per_cta < tasks ? begin + tasks_per_cta : tasks;
    for (int task = begin + threadIdx.x; task < end; task += blockDim.x) {
        const int h                  = task / blocks;
        const int b                  = task % blocks;
        const std::uint32_t row_id   = sh_rows[h];
        const std::size_t block      = static_cast<std::size_t>(row_id) / static_cast<std::size_t>(rows_per_block);
        const std::size_t in_block   = static_cast<std::size_t>(row_id) % static_cast<std::size_t>(rows_per_block);
        const std::uint8_t* blk_base = resident_blocks + block * static_cast<std::size_t>(block_bytes);
        const std::uint8_t* row      = blk_base + in_block * static_cast<std::size_t>(row_bytes);
        __nv_fp8_e4m3 code;
        code.__x                  = row[width / 2 + b];
        const float s             = static_cast<float>(code);
        const std::uint64_t codes = load_packed_codes(row + 8 * b, blk_base, blk_base + block_bytes);
        std::uint32_t words[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t pair  = static_cast<std::uint8_t>(codes >> (8 * i));
            const float2 vals        = decode_e2m1_pair_f32(pair);
            const std::uint32_t low  = bf16_rn_product(vals.x * s, m);
            const std::uint32_t high = bf16_rn_product(vals.y * s, m);
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

// One CTA per (stream, column). Key, residual, key_norm and query_norm are read once in a single
// pass that accumulates (key_sq, query_sq, raw_dot) together and reduces all three across one
// barrier; value is then read once after the gate scalar is resolved.
template <class Norm, int V, int K>
__global__ void __launch_bounds__(kGateMaxThreads)
    gate_kernel(const bf16* __restrict__ key, const bf16* __restrict__ value,
                const bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm, int hidden,
                int width, int key_stride, int value_stride, float eps, bf16* __restrict__ gated,
                bf16* __restrict__ normalized) {
    pdl::enter();
    __shared__ float scratch3[3][kGateMaxThreads / 32];
    __shared__ float scratch1[kGateMaxThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base     = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const std::size_t key_base = static_cast<std::size_t>(t) * key_stride + static_cast<std::size_t>(s) * hidden;
    const std::size_t val_base = static_cast<std::size_t>(t) * value_stride;
    float key_sq = 0.0F, query_sq = 0.0F, raw_dot = 0.0F;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        float kf[V], qf[V], kn[V], qn[V];
        load_bf16x<V>(key + key_base + d, kf);
        load_bf16x<V>(residual + base + d, qf);
        load_normx<Norm, V>(key_norm + c, kn);
        load_normx<Norm, V>(query_norm + c, qn);
#pragma unroll
        for (int j = 0; j < V; ++j) {
            key_sq += kf[j] * kf[j];
            query_sq += qf[j] * qf[j];
            raw_dot += (kf[j] * (1.0F + kn[j])) * (qf[j] * (1.0F + qn[j]));
        }
    }
    const float3 stats    = block_sum3_once(key_sq, query_sq, raw_dot, scratch3);
    const float key_inv   = rsqrtf(stats.x / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(stats.y / static_cast<float>(hidden) + eps);
    float g               = (stats.z * key_inv * query_inv) / sqrtf(static_cast<float>(hidden));
    g                     = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate      = sigmoidf(g);
    float vf[K][V];
    float gated_sq = 0.0F;
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int d = (i * blockDim.x + threadIdx.x) * V;
        if (d >= hidden) { break; }
        load_bf16x<V>(value + val_base + d, &vf[i][0]);
        float scaled[V];
#pragma unroll
        for (int j = 0; j < V; ++j) {
            scaled[j] = gate * vf[i][j];
            gated_sq += scaled[j] * scaled[j];
        }
        store_bf16x<V>(gated + base + d, scaled);
    }
    const float gated_inv = rsqrtf(block_sum1_once(gated_sq, scratch1) / static_cast<float>(hidden) + eps);
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

template <PleNextNorm kNextNorm>
__device__ __forceinline__ void store_ple_next_norm(const float (&r_rounded)[8], bool active,
                                                    const void* __restrict__ next_norm_weight,
                                                    std::size_t c, int hidden, float next_norm_eps,
                                                    float* scratch_next, bf16* __restrict__ next_out_ptr) {
    float sum_sq = 0.0F;
    if (active) {
#pragma unroll
        for (int u = 0; u < 8; ++u) { sum_sq += r_rounded[u] * r_rounded[u]; }
    }
    const float total = block_sum1_once(sum_sq, scratch_next);
    if (!active) { return; }
    const float inv = rsqrtf(total / static_cast<float>(hidden) + next_norm_eps);
    float wf[8];
    if constexpr (kNextNorm == PleNextNorm::kFp32) {
        load_normx<float, 8>(static_cast<const float*>(next_norm_weight) + c, wf);
    } else {
        load_normx<bf16, 8>(static_cast<const bf16*>(next_norm_weight) + c, wf);
    }
    float out[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) { out[u] = r_rounded[u] * inv * (1.0F + wf[u]); }
    store_bf16x<8>(next_out_ptr, out);
}

// The fused decode gate + state convolution (one column per sequence, an in-place or moved state
// update): one CTA per (stream, sequence). Single-pass (key_sq, query_sq, raw_dot) matches
// gate_kernel bit for bit, and optional kNextNorm fuses the following block's HC stream RMSNorm.
template <class Norm, int SPAN, PleNextNorm kNextNorm = PleNextNorm::kNone>
__global__ void __launch_bounds__(kGateMaxThreads)
    gate_conv_vec8_kernel(const bf16* __restrict__ key, const bf16* __restrict__ value,
                          bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                          const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm,
                          int hidden, int width, int key_stride, int value_stride, float eps,
                          const bf16* __restrict__ weight, int taps, int dilation, bf16* states,
                          const std::int32_t* __restrict__ source_slots,
                          const std::int32_t* __restrict__ destination_slots,
                          const void* __restrict__ next_norm_weight = nullptr,
                          bf16* __restrict__ next_norm_out = nullptr, float next_norm_eps = 0.0F) {
    pdl::enter();
    __shared__ float scratch3[3][kGateMaxThreads / 32];
    __shared__ float scratch1[kGateMaxThreads / 32];
    __shared__ float scratch_next[kNextNorm != PleNextNorm::kNone ? (kGateMaxThreads / 32) : 1];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base     = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const std::size_t key_base = static_cast<std::size_t>(t) * key_stride + static_cast<std::size_t>(s) * hidden;
    const std::size_t val_base = static_cast<std::size_t>(t) * value_stride;
    const int d                = threadIdx.x * 8;
    const bool active          = d < hidden;
    const std::size_t c        = static_cast<std::size_t>(s) * hidden + d;
    float qf[8];
    float key_sq = 0.0F, query_sq = 0.0F, raw_dot = 0.0F;
    if (active) {
        float kf[8], kn[8], qn[8];
        load_bf16x<8>(key + key_base + d, kf);
        load_bf16x<8>(residual + base + d, qf);
        load_normx<Norm, 8>(key_norm + c, kn);
        load_normx<Norm, 8>(query_norm + c, qn);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            key_sq += kf[j] * kf[j];
            query_sq += qf[j] * qf[j];
            raw_dot += (kf[j] * (1.0F + kn[j])) * (qf[j] * (1.0F + qn[j]));
        }
    }
    const float3 stats    = block_sum3_once(key_sq, query_sq, raw_dot, scratch3);
    const float key_inv   = rsqrtf(stats.x / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(stats.y / static_cast<float>(hidden) + eps);
    float g               = (stats.z * key_inv * query_inv) / sqrtf(static_cast<float>(hidden));
    g                     = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate      = sigmoidf(g);
    float vf[8];
    float gated_sq = 0.0F;
    if (active) {
        load_bf16x<8>(value + val_base + d, vf);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = gate * vf[j];
            gated_sq += v * v;
        }
    }
    const float gated_inv = rsqrtf(block_sum1_once(gated_sq, scratch1) / static_cast<float>(hidden) + eps);
    uint4 gated_bits = make_uint4(0U, 0U, 0U, 0U), norm_bits = make_uint4(0U, 0U, 0U, 0U);
    if (active) {
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
    float r_rounded[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    if (active) {
        const std::size_t channels = static_cast<std::size_t>(width); // the conv's channels are the gate's width
        const std::size_t words    = channels / 8;                     // uint4-strided history
        const std::size_t word     = c / 8;
        const uint4* const history = reinterpret_cast<const uint4*>(states) +
                                     static_cast<std::size_t>(source_slots[t]) * SPAN * words + word;
        uint4* const shifted = reinterpret_cast<uint4*>(states) +
                               static_cast<std::size_t>(destination_slots[t]) * SPAN * words + word;
        const bf16* const channel = weight + c;
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
        union { uint4 bits; __nv_bfloat162 half[4]; } rb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float r0 = qf[2 * j] + (g8[2 * j] + acc[2 * j] * sigmoidf(acc[2 * j]));
            const float r1 = qf[2 * j + 1] + (g8[2 * j + 1] + acc[2 * j + 1] * sigmoidf(acc[2 * j + 1]));
            rb.half[j]     = __floats2bfloat162_rn(r0, r1);
            if constexpr (kNextNorm != PleNextNorm::kNone) {
                const float2 rf      = __bfloat1622float2(rb.half[j]);
                r_rounded[2 * j]     = rf.x;
                r_rounded[2 * j + 1] = rf.y;
            }
        }
        *reinterpret_cast<uint4*>(residual + base + d) = rb.bits;
    }
    if constexpr (kNextNorm != PleNextNorm::kNone) {
        store_ple_next_norm<kNextNorm>(r_rounded, active, next_norm_weight, c, hidden, next_norm_eps,
                                       scratch_next, next_norm_out + base + d);
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

__device__ __forceinline__ int ple_conv_source(const SpeculativeTreeRow* __restrict__ trees, int sequence, int local,
                                               int lag) {
    if (lag == 0) { return local; }
    if (trees == nullptr || trees[sequence].tree == 0) { return local - lag; }
    const SpeculativeTreeRow& tree = trees[sequence];
    if (local >= tree.nodes) { return local - lag; }
    int col   = local;
    int steps = 0;
    while (steps < lag && col > 0) {
        col = static_cast<int>(tree.parent[col]);
        ++steps;
    }
    if (col < 0) { return col - (lag - steps); }
    return steps == lag ? col : -(lag - steps);
}

// One thread per (channel, column): reads the pre-update history.
__global__ void conv_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                            const bf16* __restrict__ weight, int taps, int dilation, int span,
                            int channels, int width, const bf16* __restrict__ states,
                            const std::int32_t* __restrict__ source_slots, const SpeculativeTreeRow* __restrict__ trees,
                            int columns, bf16* __restrict__ residual) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= channels || t >= columns) { return; }
    const int sequence = t / width, local = t % width;
    const int slot     = source_slots[sequence];
    float acc          = 0.0F;
    for (int j = 0; j < taps; ++j) {
        const int source = ple_conv_source(trees, sequence, local, (taps - 1 - j) * dilation);
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
                     const std::int32_t* __restrict__ source_slots, const SpeculativeTreeRow* __restrict__ trees,
                     int columns, bf16* __restrict__ residual) {
    pdl::enter();
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int t = blockIdx.y;
    if (c >= channels || t >= columns) { return; }
    const int sequence = t / width, local = t % width;
    const int slot     = source_slots[sequence];
    float acc[8]       = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int j = 0; j < taps; ++j) {
        const int source = ple_conv_source(trees, sequence, local, (taps - 1 - j) * dilation);
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

// 8-step temporal-tiled prefill convolution for Qwen4Exp's 4-tap dilation-3 (span-9) geometry:
// each thread owns a channel pair (__nv_bfloat162) across Tile = 8 consecutive columns of one
// sequence, rotating a 9-column register ring so weights and history are loaded once per 8 columns.
template <int Block, int Tile>
__global__ void __launch_bounds__(Block)
    conv_add_tiled_kernel(const __nv_bfloat162* __restrict__ gated,
                          const __nv_bfloat162* __restrict__ normalized,
                          const __nv_bfloat162* __restrict__ weight, int pairs, int width,
                          const __nv_bfloat162* __restrict__ states,
                          const std::int32_t* __restrict__ source_slots,
                          __nv_bfloat162* __restrict__ residual) {
    pdl::enter();
    constexpr int kSpan = 9;
    const int p         = blockIdx.x * Block + threadIdx.x;
    if (p >= pairs) { return; }
    const int first    = blockIdx.y * Tile;
    const int sequence = blockIdx.z;
    if (first >= width) { return; }
    const int slot = source_slots[sequence];

    float history[kSpan][2];
    float w[4][2];
#pragma unroll
    for (int j = 0; j < kSpan; ++j) {
        const int src = first - kSpan + j;
        const __nv_bfloat162 raw =
            src >= 0 ? normalized[static_cast<std::size_t>(sequence * width + src) * pairs + p]
                     : states[(static_cast<std::size_t>(slot) * kSpan + (kSpan + src)) * pairs + p];
        const float2 f = __bfloat1622float2(raw);
        history[j][0]  = f.x;
        history[j][1]  = f.y;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(weight[static_cast<std::size_t>(j) * pairs + p]);
        w[j][0]        = f.x;
        w[j][1]        = f.y;
    }
#pragma unroll
    for (int i = 0; i < Tile; ++i) {
        const int local = first + i;
        if (local >= width) { break; }
        const std::size_t off = static_cast<std::size_t>(sequence * width + local) * pairs + p;
        const float2 n        = __bfloat1622float2(normalized[off]);
        const float2 g        = __bfloat1622float2(gated[off]);
        const float2 r        = __bfloat1622float2(residual[off]);
        float sum0            = w[0][0] * history[i % kSpan][0];
        float sum1            = w[0][1] * history[i % kSpan][1];
        sum0                  = fmaf(w[1][0], history[(i + 3) % kSpan][0], sum0);
        sum1                  = fmaf(w[1][1], history[(i + 3) % kSpan][1], sum1);
        sum0                  = fmaf(w[2][0], history[(i + 6) % kSpan][0], sum0);
        sum1                  = fmaf(w[2][1], history[(i + 6) % kSpan][1], sum1);
        sum0                  = fmaf(w[3][0], n.x, sum0);
        sum1                  = fmaf(w[3][1], n.y, sum1);
        residual[off] = __floats2bfloat162_rn(r.x + (g.x + sum0 * sigmoidf(sum0)),
                                              r.y + (g.y + sum1 * sigmoidf(sum1)));
        history[i % kSpan][0] = n.x;
        history[i % kSpan][1] = n.y;
    }
}

// One CTA per (stream, column): computes the convolution + residual update for one stream of column
// t, stores the rounded BF16 residual, and immediately reduces and writes the next block's HC
// per-stream unit-offset RMSNorm from registers without re-reading residual from global memory.
template <PleNextNorm kNextNorm>
__global__ void __launch_bounds__(kGateMaxThreads)
    conv_norm_vec8_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                          const bf16* __restrict__ weight, int taps, int dilation, int span,
                          int hidden, int channels, int width, const bf16* __restrict__ states,
                          const std::int32_t* __restrict__ source_slots,
                          const SpeculativeTreeRow* __restrict__ trees, bf16* __restrict__ residual,
                          const void* __restrict__ next_norm_weight,
                          bf16* __restrict__ next_norm_out, float next_norm_eps) {
    pdl::enter();
    __shared__ float scratch_next[kGateMaxThreads / 32];
    const int s          = blockIdx.x;
    const int t          = blockIdx.y;
    const int d          = threadIdx.x * 8;
    const bool active    = d < hidden;
    const std::size_t c  = static_cast<std::size_t>(s) * hidden + d;
    const std::size_t base = static_cast<std::size_t>(t) * channels + c;
    float r_rounded[8]   = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    if (active) {
        const int sequence = t / width, local = t % width;
        const int slot     = source_slots[sequence];
        float acc[8]       = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        for (int j = 0; j < taps; ++j) {
            const int source = ple_conv_source(trees, sequence, local, (taps - 1 - j) * dilation);
            float x[8], w[8];
            if (source >= 0) {
                load_bf16x<8>(
                    normalized + static_cast<std::size_t>(sequence * width + source) * channels + c,
                    x);
            } else {
                load_bf16x<8>(
                    states + (static_cast<std::size_t>(slot) * span + (span + source)) * channels + c,
                    x);
            }
            load_bf16x<8>(weight + static_cast<std::size_t>(j) * channels + c, w);
#pragma unroll
            for (int u = 0; u < 8; ++u) { acc[u] += w[u] * x[u]; }
        }
        float g[8], r[8];
        load_bf16x<8>(gated + base, g);
        load_bf16x<8>(residual + base, r);
        union { uint4 bits; __nv_bfloat162 half[4]; } rb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float r0 = r[2 * j] + (g[2 * j] + acc[2 * j] * sigmoidf(acc[2 * j]));
            const float r1 = r[2 * j + 1] + (g[2 * j + 1] + acc[2 * j + 1] * sigmoidf(acc[2 * j + 1]));
            rb.half[j]     = __floats2bfloat162_rn(r0, r1);
            const float2 rf      = __bfloat1622float2(rb.half[j]);
            r_rounded[2 * j]     = rf.x;
            r_rounded[2 * j + 1] = rf.y;
        }
        *reinterpret_cast<uint4*>(residual + base) = rb.bits;
    }
    store_ple_next_norm<kNextNorm>(r_rounded, active, next_norm_weight, c, hidden, next_norm_eps,
                                   scratch_next, next_norm_out + base);
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

template <bool VectorStore>
__global__ void __launch_bounds__(kThreads)
    gather_embed_nvfp4_kernel(const std::uint8_t* __restrict__ cache_base,
                              const std::uint8_t* __restrict__ fallback_rows,
                              const std::uint32_t* __restrict__ slots, const float* __restrict__ multiplier,
                              std::uint32_t cache_slots, int heads, int width, int row_bytes, int columns,
                              int tasks_per_cta, std::uint16_t* __restrict__ out) {
    pdl::enter();
    __shared__ std::uint32_t sh_slots[kMaxPleHashHeads];
    const int t = blockIdx.y;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        sh_slots[h] = slots[static_cast<std::size_t>(t) * heads + h];
    }
    __syncthreads();

    const int blocks = width / 16;
    const int tasks  = heads * blocks;
    const float m    = multiplier[0];
    const int begin  = blockIdx.x * tasks_per_cta;
    const int end    = begin + tasks_per_cta < tasks ? begin + tasks_per_cta : tasks;
    for (int task = begin + threadIdx.x; task < end; task += blockDim.x) {
        const int h                  = task / blocks;
        const int b                  = task % blocks;
        const std::uint32_t slot_raw = sh_slots[h];
        const std::uint8_t* row;
        const std::uint8_t* first;
        const std::uint8_t* last;
        if ((slot_raw & kPleCacheStagingBit) != 0U) {
            if (fallback_rows == nullptr) { continue; }
            first = fallback_rows;
            last  = fallback_rows + static_cast<std::size_t>(columns) * heads * row_bytes;
            row   = fallback_rows + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
        } else {
            if (slot_raw >= cache_slots) { continue; }
            first = cache_base;
            last  = cache_base + static_cast<std::size_t>(cache_slots) * row_bytes;
            row   = cache_base + static_cast<std::size_t>(slot_raw) * row_bytes;
        }
        __nv_fp8_e4m3 code;
        code.__x                  = row[width / 2 + b];
        const float s             = static_cast<float>(code);
        const std::uint64_t codes = load_packed_codes(row + 8 * b, first, last);
        std::uint32_t words[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t pair  = static_cast<std::uint8_t>(codes >> (8 * i));
            const float2 vals        = decode_e2m1_pair_f32(pair);
            const std::uint32_t low  = bf16_rn_product(vals.x * s, m);
            const std::uint32_t high = bf16_rn_product(vals.y * s, m);
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

__device__ __forceinline__ uint4 decode_fp8x8_to_bf16x8(std::uint32_t lo, std::uint32_t hi, float s) {
    union {
        uint4 bits;
        __nv_bfloat162 half[4];
    } raw;
    __nv_fp8_e4m3 c0, c1, c2, c3, c4, c5, c6, c7;
    c0.__x      = static_cast<std::uint8_t>(lo & 0xFFU);
    c1.__x      = static_cast<std::uint8_t>((lo >> 8) & 0xFFU);
    c2.__x      = static_cast<std::uint8_t>((lo >> 16) & 0xFFU);
    c3.__x      = static_cast<std::uint8_t>((lo >> 24) & 0xFFU);
    c4.__x      = static_cast<std::uint8_t>(hi & 0xFFU);
    c5.__x      = static_cast<std::uint8_t>((hi >> 8) & 0xFFU);
    c6.__x      = static_cast<std::uint8_t>((hi >> 16) & 0xFFU);
    c7.__x      = static_cast<std::uint8_t>((hi >> 24) & 0xFFU);
    raw.half[0] = __floats2bfloat162_rn(static_cast<float>(c0) * s, static_cast<float>(c1) * s);
    raw.half[1] = __floats2bfloat162_rn(static_cast<float>(c2) * s, static_cast<float>(c3) * s);
    raw.half[2] = __floats2bfloat162_rn(static_cast<float>(c4) * s, static_cast<float>(c5) * s);
    raw.half[3] = __floats2bfloat162_rn(static_cast<float>(c6) * s, static_cast<float>(c7) * s);
    return raw.bits;
}

__global__ void __launch_bounds__(kThreads)
    gather_embed_fp8_vec16_kernel(const std::uint8_t* __restrict__ cache_base,
                                  const std::uint8_t* __restrict__ fallback_rows,
                                  const std::uint32_t* __restrict__ slots, const bf16* __restrict__ scale,
                                  std::uint32_t cache_slots, int heads, int width, int tasks_per_cta,
                                  bf16* __restrict__ out) {
    pdl::enter();
    __shared__ std::uint32_t sh_slots[kMaxPleHashHeads];
    const int t = blockIdx.y;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        sh_slots[h] = slots[static_cast<std::size_t>(t) * heads + h];
    }
    __syncthreads();

    const int blocks = width / 16;
    const int tasks  = heads * blocks;
    const float s    = __bfloat162float(scale[0]);
    const int begin  = blockIdx.x * tasks_per_cta;
    const int end    = begin + tasks_per_cta < tasks ? begin + tasks_per_cta : tasks;
    for (int task = begin + threadIdx.x; task < end; task += blockDim.x) {
        const int h                  = task / blocks;
        const int b                  = task % blocks;
        const std::uint32_t slot_raw = sh_slots[h];
        const std::uint8_t* row;
        if ((slot_raw & kPleCacheStagingBit) != 0U) {
            if (fallback_rows == nullptr) { continue; }
            row = fallback_rows + (static_cast<std::size_t>(t) * heads + h) * width;
        } else {
            if (slot_raw >= cache_slots) { continue; }
            row = cache_base + static_cast<std::size_t>(slot_raw) * width;
        }
        const uint4 packed = *reinterpret_cast<const uint4*>(row + 16 * b);
        bf16* o =
            out + static_cast<std::size_t>(t) * heads * width + static_cast<std::size_t>(h) * width + 16 * b;
        *reinterpret_cast<uint4*>(o)     = decode_fp8x8_to_bf16x8(packed.x, packed.y, s);
        *reinterpret_cast<uint4*>(o + 8) = decode_fp8x8_to_bf16x8(packed.z, packed.w, s);
    }
}

__global__ void __launch_bounds__(kThreads)
    gather_embed_fp8_kernel(const std::uint8_t* __restrict__ cache_base,
                            const std::uint8_t* __restrict__ fallback_rows,
                            const std::uint32_t* __restrict__ slots, const bf16* __restrict__ scale,
                            std::uint32_t cache_slots, int heads, int head_width, bf16* __restrict__ out) {
    pdl::enter();
    __shared__ std::uint32_t sh_slots[kMaxPleHashHeads];
    const int t = blockIdx.y;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        sh_slots[h] = slots[static_cast<std::size_t>(t) * heads + h];
    }
    __syncthreads();

    const int i        = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = heads * head_width;
    if (i >= elements) { return; }
    const int h                  = i / head_width;
    const int d                  = i % head_width;
    const std::uint32_t slot_raw = sh_slots[h];
    const std::uint8_t* row;
    if ((slot_raw & kPleCacheStagingBit) != 0U) {
        if (fallback_rows == nullptr) { return; }
        row = fallback_rows + (static_cast<std::size_t>(t) * heads + h) * head_width;
    } else {
        if (slot_raw >= cache_slots) { return; }
        row = cache_base + static_cast<std::size_t>(slot_raw) * head_width;
    }
    __nv_fp8_e4m3 code;
    code.__x      = row[d];
    const float v = static_cast<float>(code) * __bfloat162float(scale[0]);
    out[static_cast<std::size_t>(t) * elements + i] = __float2bfloat16_rn(v);
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

void launch_hash_embed_nvfp4(const std::int32_t* windows, const std::uint64_t* multipliers,
                             const std::uint64_t* sizes, const std::uint64_t* offsets,
                             const std::uint8_t* resident_blocks, const float* multiplier, int ngram_size,
                             int heads_per_ngram, int eos_token_id, int heads, int width, int row_bytes,
                             int rows_per_block, int block_bytes, int columns, std::uint16_t* out,
                             cudaStream_t stream) {
    const int tasks   = heads * (width / 16);
    int tasks_per_cta = tasks < kThreads ? tasks : kThreads;
    if (columns < kEmbedSpreadColumns && tasks > kEmbedCtaTasks) { tasks_per_cta = kEmbedCtaTasks; }
    const dim3 grid((tasks + tasks_per_cta - 1) / tasks_per_cta, columns);
    if (aligned<16>(out)) {
        (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream},
                                   hash_embed_nvfp4_kernel<true>, windows, multipliers, sizes, offsets,
                                   resident_blocks, multiplier, ngram_size, heads_per_ngram, eos_token_id, heads, width,
                                   row_bytes, rows_per_block, block_bytes, columns, tasks_per_cta, out);
    } else {
        (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream},
                                   hash_embed_nvfp4_kernel<false>, windows, multipliers, sizes, offsets,
                                   resident_blocks, multiplier, ngram_size, heads_per_ngram, eos_token_id, heads, width,
                                   row_bytes, rows_per_block, block_bytes, columns, tasks_per_cta, out);
    }
}

void launch_gather_embed_nvfp4(const std::uint8_t* cache_base, const std::uint8_t* fallback_rows,
                               const std::uint32_t* slots, const float* multiplier, std::uint32_t cache_slots,
                               int heads, int width, int row_bytes, int columns, const std::uint32_t* ready,
                               const std::uint32_t* expected, std::uint64_t* wait_stats,
                               const std::int32_t* wait_row, std::uint16_t* out, cudaStream_t stream) {
    const bool gated = ready != nullptr;
    if (gated) { wait_pinned_word(ready, expected, wait_stats, wait_row, stream); }
    const int tasks   = heads * (width / 16);
    int tasks_per_cta = tasks < kThreads ? tasks : kThreads;
    if (columns < kEmbedSpreadColumns && tasks > kEmbedCtaTasks) { tasks_per_cta = kEmbedCtaTasks; }
    const dim3 grid((tasks + tasks_per_cta - 1) / tasks_per_cta, columns);
    const auto launch = [&](auto kernel) {
        if (gated) {
            kernel<<<grid, dim3(tasks_per_cta), 0, stream>>>(cache_base, fallback_rows, slots, multiplier, cache_slots,
                                                             heads, width, row_bytes, columns, tasks_per_cta, out);
        } else {
            (void)pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream}, kernel, cache_base,
                                       fallback_rows, slots, multiplier, cache_slots, heads, width, row_bytes, columns,
                                       tasks_per_cta, out);
        }
    };
    if (aligned<16>(out)) {
        launch(gather_embed_nvfp4_kernel<true>);
    } else {
        launch(gather_embed_nvfp4_kernel<false>);
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

void ple_hash_embed_nvfp4(const Tensor& windows, const PleNgramHashTable& table, Tensor& out, cudaStream_t stream) {
    require(table.resident_blocks != nullptr && table.multipliers != nullptr && table.sizes != nullptr &&
                table.offsets != nullptr && table.multiplier != nullptr,
            "hash embed requires a resident NVFP4 table and hash descriptors");
    require(contiguous(windows, DType::I32) && contiguous(out, DType::BF16) &&
                contiguous(*table.multipliers, DType::I64) && contiguous(*table.sizes, DType::I64) &&
                contiguous(*table.offsets, DType::I64) && contiguous(*table.multiplier, DType::FP32),
            "hash embed requires contiguous tensors");
    require(table.ngram_size >= 2 && table.ngram_size <= 32 && table.heads_per_ngram > 0,
            "hash embed n-gram geometry out of range");
    const int heads = (table.ngram_size - 1) * table.heads_per_ngram;
    require(heads > 0 && heads <= kMaxPleHashHeads, "hash embed head count exceeds kernel limit");
    const int columns = windows.ne[1];
    require(columns > 0 && windows.ne[0] == table.ngram_size && windows.ne[2] == 1 && windows.ne[3] == 1,
            "hash embed windows shape disagrees");
    require(table.multipliers->numel() == table.ngram_size && table.sizes->numel() == heads &&
                table.offsets->numel() == heads && table.multiplier->numel() == 1,
            "hash embed table tensor sizes disagree");
    require(table.head_width > 0 && table.head_width % 16 == 0 &&
                table.row_bytes == table.head_width / 2 + table.head_width / 16 && table.rows_per_block > 0 &&
                table.block_bytes >= table.rows_per_block * table.row_bytes,
            "hash embed NVFP4 block geometry disagrees");
    require(out.ne[0] == heads * table.head_width && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "hash embed output shape disagrees");
    launch_hash_embed_nvfp4(static_cast<const std::int32_t*>(windows.data),
                            reinterpret_cast<const std::uint64_t*>(table.multipliers->data),
                            reinterpret_cast<const std::uint64_t*>(table.sizes->data),
                            reinterpret_cast<const std::uint64_t*>(table.offsets->data),
                            static_cast<const std::uint8_t*>(table.resident_blocks),
                            static_cast<const float*>(table.multiplier->data), table.ngram_size, table.heads_per_ngram,
                            table.eos_token_id, heads, table.head_width, table.row_bytes, table.rows_per_block,
                            table.block_bytes, columns, static_cast<std::uint16_t*>(out.data), stream);
    check_launch("hash embed");
}

void ple_gather_embed_nvfp4(const PleCacheGather& gather, Tensor& out, cudaStream_t stream) {
    require(gather.cache_base != nullptr && gather.slots != nullptr && gather.scale != nullptr,
            "gather embed nvfp4 requires mapped cache, slots and scale");
    require(gather.cache_slots > 0 && gather.heads > 0 && gather.heads <= kMaxPleHashHeads,
            "gather embed nvfp4 cache slots or head count out of range");
    require((gather.ready == nullptr) == (gather.expected == nullptr),
            "gather embed nvfp4 gate requires both ready and expected pointers");
    require(contiguous(out, DType::BF16) && contiguous(*gather.scale, DType::FP32) &&
                gather.scale->numel() == 1,
            "gather embed nvfp4 requires contiguous FP32 scale and BF16 output");
    const int width   = gather.head_width;
    const int columns = out.ne[1];
    require(columns > 0 && width > 0 && width % 16 == 0 &&
                gather.row_bytes == width / 2 + width / 16 &&
                out.ne[0] == gather.heads * width && out.ne[2] == 1 && out.ne[3] == 1,
            "gather embed nvfp4 shapes disagree");
    launch_gather_embed_nvfp4(
        static_cast<const std::uint8_t*>(gather.cache_base),
        static_cast<const std::uint8_t*>(gather.fallback_rows), gather.slots,
        static_cast<const float*>(gather.scale->data), gather.cache_slots, gather.heads, width,
        gather.row_bytes, columns, gather.ready, gather.expected, gather.wait_stats, gather.wait_row,
        static_cast<std::uint16_t*>(out.data), stream);
    check_launch("gather embed nvfp4");
}

void ple_gather_embed_fp8(const PleCacheGather& gather, Tensor& out, cudaStream_t stream) {
    require(gather.cache_base != nullptr && gather.slots != nullptr && gather.scale != nullptr,
            "gather embed fp8 requires mapped cache, slots and scale");
    require(gather.cache_slots > 0 && gather.heads > 0 && gather.heads <= kMaxPleHashHeads,
            "gather embed fp8 cache slots or head count out of range");
    require((gather.ready == nullptr) == (gather.expected == nullptr),
            "gather embed fp8 gate requires both ready and expected pointers");
    require(contiguous(out, DType::BF16) && contiguous(*gather.scale, DType::BF16) &&
                gather.scale->numel() == 1,
            "gather embed fp8 requires contiguous BF16 scale and BF16 output");
    const int width    = gather.head_width;
    const int columns  = out.ne[1];
    const int elements = gather.heads * width;
    require(columns > 0 && width > 0 && gather.row_bytes == width &&
                out.ne[0] == elements && out.ne[2] == 1 && out.ne[3] == 1,
            "gather embed fp8 shapes disagree");
    const bool gated = gather.ready != nullptr;
    if (gated) { wait_pinned_word(gather.ready, gather.expected, gather.wait_stats, gather.wait_row, stream); }
    const bool vec16 = (width % 16 == 0) && aligned<16>(gather.cache_base) &&
                       (gather.fallback_rows == nullptr || aligned<16>(gather.fallback_rows)) &&
                       aligned<16>(out.data);
    if (vec16) {
        const int tasks   = gather.heads * (width / 16);
        int tasks_per_cta = tasks < kThreads ? tasks : kThreads;
        if (columns < kEmbedSpreadColumns && tasks > kEmbedCtaTasks) { tasks_per_cta = kEmbedCtaTasks; }
        const dim3 grid((tasks + tasks_per_cta - 1) / tasks_per_cta, columns);
        if (gated) {
            gather_embed_fp8_vec16_kernel<<<grid, dim3(tasks_per_cta), 0, stream>>>(
                static_cast<const std::uint8_t*>(gather.cache_base),
                static_cast<const std::uint8_t*>(gather.fallback_rows), gather.slots,
                static_cast<const bf16*>(gather.scale->data), gather.cache_slots, gather.heads, width, tasks_per_cta,
                static_cast<bf16*>(out.data));
        } else {
            (void)pdl::launch_consumer(
                pdl::LaunchConfig{grid, dim3(tasks_per_cta), 0, stream}, gather_embed_fp8_vec16_kernel,
                static_cast<const std::uint8_t*>(gather.cache_base),
                static_cast<const std::uint8_t*>(gather.fallback_rows), gather.slots,
                static_cast<const bf16*>(gather.scale->data), gather.cache_slots, gather.heads, width, tasks_per_cta,
                static_cast<bf16*>(out.data));
        }
    } else {
        const dim3 grid((elements + kThreads - 1) / kThreads, columns);
        if (gated) {
            gather_embed_fp8_kernel<<<grid, dim3(kThreads), 0, stream>>>(
                static_cast<const std::uint8_t*>(gather.cache_base),
                static_cast<const std::uint8_t*>(gather.fallback_rows), gather.slots,
                static_cast<const bf16*>(gather.scale->data), gather.cache_slots, gather.heads, width,
                static_cast<bf16*>(out.data));
        } else {
            (void)pdl::launch_consumer(
                pdl::LaunchConfig{grid, dim3(kThreads), 0, stream}, gather_embed_fp8_kernel,
                static_cast<const std::uint8_t*>(gather.cache_base),
                static_cast<const std::uint8_t*>(gather.fallback_rows), gather.slots,
                static_cast<const bf16*>(gather.scale->data), gather.cache_slots, gather.heads, width,
                static_cast<bf16*>(out.data));
        }
    }
    check_launch("gather embed fp8");
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
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream,
                     const Tensor* next_norm_weight, Tensor* next_norm_out, float next_norm_eps,
                     std::int32_t streams, const Tensor* tree_rows) {
    require(contiguous(gated, DType::BF16) && contiguous(normalized, DType::BF16) &&
                contiguous(weight, DType::BF16) && contiguous(states, DType::BF16) &&
                contiguous(residual, DType::BF16) && contiguous(source_slots, DType::I32) &&
                (destination_slots.data == nullptr || contiguous(destination_slots, DType::I32)),
            "conv requires contiguous tensors");
    const int channels = gated.ne[0], columns = gated.ne[1], taps = weight.ne[1];
    const int sequences = source_slots.ne[0];
    const bool update   = destination_slots.data != nullptr;
    const SpeculativeTreeRow* trees_p = nullptr;
    if (tree_rows != nullptr && tree_rows->data != nullptr) {
        validate_speculative_tree_rows(*tree_rows, sequences, "ple_conv_inject");
        require(!update, "conv tree_rows cannot update states in place");
        trees_p = reinterpret_cast<const SpeculativeTreeRow*>(tree_rows->data);
    }
    require(dilation > 0 && taps > 1 && weight.ne[0] == channels && normalized.ne[0] == channels &&
                normalized.ne[1] == columns && residual.ne[0] == channels && residual.ne[1] == columns &&
                sequences > 0 && (!update || destination_slots.ne[0] == sequences) && columns % sequences == 0,
            "conv shapes disagree");
    const int span = (taps - 1) * dilation;
    require(states.ne[0] == channels && states.ne[1] == span, "conv state must be [C, span, slots]");
    const bool want_next_norm = next_norm_weight != nullptr || next_norm_out != nullptr;
    if (want_next_norm) {
        require(next_norm_weight != nullptr && next_norm_out != nullptr,
                "conv next norm requires both weight and output");
        require(contiguous(*next_norm_out, DType::BF16) &&
                    (contiguous(*next_norm_weight, DType::BF16) || contiguous(*next_norm_weight, DType::FP32)),
                "conv next norm requires BF16 output and BF16 or FP32 weight");
        require(streams > 0 && channels % streams == 0 &&
                    next_norm_weight->numel() == channels &&
                    next_norm_out->ne[0] == channels && next_norm_out->ne[1] == columns,
                "conv next norm shapes disagree");
        require(next_norm_eps > 0.0F, "conv next norm epsilon must be positive");
    }
    const int width = columns / sequences;
    const auto* gated_p       = static_cast<const bf16*>(gated.data);
    const auto* normalized_p  = static_cast<const bf16*>(normalized.data);
    const auto* weight_p      = static_cast<const bf16*>(weight.data);
    const auto* source_p      = static_cast<const std::int32_t*>(source_slots.data);
    const auto* destination_p = static_cast<const std::int32_t*>(destination_slots.data);
    auto* states_p   = static_cast<bf16*>(states.data);
    auto* residual_p = static_cast<bf16*>(residual.data);
    const bool vec8 = channels % 8 == 0 && aligned<16>(gated_p) && aligned<16>(normalized_p) &&
                      aligned<16>(weight_p) && aligned<16>(states_p) && aligned<16>(residual_p);
    const bool pair2 = channels % 2 == 0 && aligned<4>(gated_p) && aligned<4>(normalized_p) &&
                       aligned<4>(weight_p) && aligned<4>(states_p) && aligned<4>(residual_p);
    if (!want_next_norm && update && width == 1 && span >= 1 && span <= kFusedConvMaxSpan) {
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
    bool fused_next_norm_done = false;
    if (want_next_norm && vec8) {
        const int hidden = channels / streams;
        if (hidden % 8 == 0 && hidden <= kGateMaxThreads * 8 &&
            aligned<16>(next_norm_weight->data) && aligned<16>(next_norm_out->data)) {
            int block = ((hidden / 8 + 31) / 32) * 32;
            if (block > kGateMaxThreads) { block = kGateMaxThreads; }
            const pdl::LaunchConfig cfg{dim3(streams, columns), dim3(block), 0, stream};
            auto* nn_out = static_cast<bf16*>(next_norm_out->data);
            if (next_norm_weight->dtype == DType::FP32) {
                (void)pdl::launch_consumer(cfg, conv_norm_vec8_kernel<PleNextNorm::kFp32>, gated_p, normalized_p,
                                           weight_p, taps, static_cast<int>(dilation), span, hidden, channels, width,
                                           states_p, source_p, trees_p, residual_p, next_norm_weight->data, nn_out,
                                           next_norm_eps);
            } else {
                (void)pdl::launch_consumer(cfg, conv_norm_vec8_kernel<PleNextNorm::kBf16>, gated_p, normalized_p,
                                           weight_p, taps, static_cast<int>(dilation), span, hidden, channels, width,
                                           states_p, source_p, trees_p, residual_p, next_norm_weight->data, nn_out,
                                           next_norm_eps);
            }
            check_launch("conv norm");
            fused_next_norm_done = true;
        }
    }
    if (!fused_next_norm_done) {
        constexpr int kTiledBlock = 128;
        constexpr int kTileSteps  = 8;
        const int time_tiles      = (width + kTileSteps - 1) / kTileSteps;
        if (trees_p == nullptr && pair2 && taps == 4 && dilation == 3 && width >= kTileSteps &&
            time_tiles <= 65535 && sequences <= 65535) {
            const int pairs = channels / 2;
            const dim3 grid((pairs + kTiledBlock - 1) / kTiledBlock, time_tiles, sequences);
            (void)pdl::launch_consumer(
                pdl::LaunchConfig{grid, dim3(kTiledBlock), 0, stream},
                conv_add_tiled_kernel<kTiledBlock, kTileSteps>,
                reinterpret_cast<const __nv_bfloat162*>(gated_p),
                reinterpret_cast<const __nv_bfloat162*>(normalized_p),
                reinterpret_cast<const __nv_bfloat162*>(weight_p), pairs, width,
                reinterpret_cast<const __nv_bfloat162*>(states_p), source_p,
                reinterpret_cast<__nv_bfloat162*>(residual_p));
        } else if (vec8) {
            (void)pdl::launch_consumer(pdl::LaunchConfig{dim3((channels / 8 + kThreads - 1) / kThreads, columns),
                                                         dim3(kThreads), 0, stream},
                                       conv_vec8_kernel, gated_p, normalized_p, weight_p, taps, dilation, span,
                                       channels, width, states_p, source_p, trees_p, columns, residual_p);
        } else {
            conv_kernel<<<dim3((channels + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
                gated_p, normalized_p, weight_p, taps, dilation, span, channels, width, states_p, source_p, trees_p,
                columns, residual_p);
        }
        check_launch("conv");
        if (want_next_norm) {
            hyper_connection_norm(residual, *next_norm_weight, streams, next_norm_eps, *next_norm_out, stream);
        }
    }
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
                          Tensor& gated, Tensor& normalized, cudaStream_t stream,
                          const Tensor* next_norm_weight, Tensor* next_norm_out, float next_norm_eps,
                          const Tensor* tree_rows) {
    const int width   = key.ne[0], columns = key.ne[1];
    const int taps    = weight.ne[1];
    const int span    = (taps - 1) * static_cast<int>(dilation);
    const bool update = destination_slots.data != nullptr;
    const bool want_next_norm = next_norm_weight != nullptr || next_norm_out != nullptr;
    if (want_next_norm) {
        require(next_norm_weight != nullptr && next_norm_out != nullptr,
                "gate conv next norm requires both weight and output");
        require(contiguous(*next_norm_out, DType::BF16) &&
                    (contiguous(*next_norm_weight, DType::BF16) || contiguous(*next_norm_weight, DType::FP32)),
                "gate conv next norm requires BF16 output and BF16 or FP32 weight");
        require(next_norm_weight->numel() == width &&
                    next_norm_out->ne[0] == width && next_norm_out->ne[1] == columns,
                "gate conv next norm shapes disagree");
        require(next_norm_eps > 0.0F, "gate conv next norm epsilon must be positive");
    }
    // The fused kernel's preconditions, everything the composed pair would validate except the
    // scratch it does not use; an unmet condition (or an invalid shape, which throws below or in
    // the composed fallback) leaves the exact composed route.
    const bool fused =
        (tree_rows == nullptr || tree_rows->data == nullptr) &&
        update && source_slots.data != nullptr && contiguous(source_slots, DType::I32) &&
        contiguous(destination_slots, DType::I32) && source_slots.ne[0] == columns &&
        destination_slots.ne[0] == columns && taps > 1 && span >= 1 && span <= kFusedConvMaxSpan &&
        dilation > 0 && contiguous(weight, DType::BF16) && weight.ne[0] == width &&
        contiguous(states, DType::BF16) && states.ne[0] == width && states.ne[1] == span &&
        (!want_next_norm || (aligned<16>(next_norm_weight->data) && aligned<16>(next_norm_out->data)));
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
            const void* nn_w   = want_next_norm ? next_norm_weight->data : nullptr;
            auto*       nn_out = want_next_norm ? static_cast<bf16*>(next_norm_out->data) : nullptr;
            const auto launch = [&]<class Norm, PleNextNorm kNextNorm>() {
                const auto* key_norm_p   = static_cast<const Norm*>(key_norm.data);
                const auto* query_norm_p = static_cast<const Norm*>(query_norm.data);
                const auto* conv_norm_p  = static_cast<const Norm*>(conv_norm.data);
                int block = ((geometry.hidden / 8 + 31) / 32) * 32;
                if (block > kGateMaxThreads) { block = kGateMaxThreads; }
                const pdl::LaunchConfig cfg{dim3(streams, columns), dim3(block), 0, stream};
                const auto run = [&]<int SPAN>() {
                    (void)pdl::launch_consumer(cfg, gate_conv_vec8_kernel<Norm, SPAN, kNextNorm>, key_p, value_p,
                                               residual_p, key_norm_p, query_norm_p, conv_norm_p, geometry.hidden,
                                               width, geometry.key_stride, geometry.value_stride, eps, weight_p, taps,
                                               static_cast<int>(dilation), states_p, source_p, dest_p, nn_w, nn_out,
                                               next_norm_eps);
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
            const auto dispatch_norm = [&]<class Norm>() {
                if (!want_next_norm) {
                    launch.template operator()<Norm, PleNextNorm::kNone>();
                } else if (next_norm_weight->dtype == DType::FP32) {
                    launch.template operator()<Norm, PleNextNorm::kFp32>();
                } else {
                    launch.template operator()<Norm, PleNextNorm::kBf16>();
                }
            };
            if (geometry.norm == DType::FP32) {
                dispatch_norm.template operator()<float>();
            } else {
                dispatch_norm.template operator()<bf16>();
            }
            check_launch("gate conv");
            return;
        }
    }
    ple_gate(key, value, residual, key_norm, query_norm, conv_norm, streams, eps, gated, normalized, stream);
    ple_conv_inject(gated, normalized, weight, dilation, states, source_slots, destination_slots, residual, stream,
                    next_norm_weight, next_norm_out, next_norm_eps, streams, tree_rows);
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
