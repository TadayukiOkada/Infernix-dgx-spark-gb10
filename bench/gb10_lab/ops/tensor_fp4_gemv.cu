// tensor_fp4_gemv.cu — faithful port of dgpp's NVFP4 decode GEMV core
// (dgpp/src/kernels/fp4_gemv.cuh) into the gb10_lab harness, measured at real
// Qwen3.8-Flash-Next decode shapes against the Q8 baseline in dense_width.
//
// Roadmap R1 (docs/performance/dgpp-gap-roadmap.md): does decoding the dense
// projection family at 4-bit NVFP4 beat Infernix's Q8 dense decode on single
// stream? The earlier lab `dense_width` FP4 GEMV was ALU-bound (~60 GB/s)
// because it was a NAIVE transcription: scattered global activation reads, a
// shift/mask nibble decode per code, no load pipelining. dgpp's real kernel
// dodges that with four techniques this port reproduces verbatim:
//   1. hardware e2m1->f16 decode (__nv_cvt_fp4x2_to_halfraw2: one F2FP for two
//      codes) + ONE __hmul2 by the group's e4m3 scale -> exact in f16.
//   2. issue-before-consume: every weight chunk + scale of a pass is loaded
//      into registers before any is consumed (bytes in flight).
//   3. activations staged once into shared memory as bf16 and re-read from
//      there across the warp's row steps (no scattered global activation reads).
//   4. fixed xor-shuffle reduction tree across each row's lane group, launch
//      shape independent.
// Geometry is a compile-time function of K (a lane owns up to 4 chunks of a
// row, a row spans K/128 lanes ...), exactly as in fp4_gemv.cuh. This op runs
// the decode case m = 1 (single stream).
//
// Storage contract matches true NVFP4 (so the GB/s number is comparable to
// Q8's): payload e2m1 codes packed two-per-byte (low nibble = even element)
// [out, K/2]; scales e4m3 bytes [out, K/16]; one f32 global scale g per matrix;
// out[row] = dot(row, x) / g. Scale storage is 1 B/group (e4m3), NOT the fp16
// (2 B/group) used by dense_width's simplified encoding, so this op measures the
// real NVFP4 DRAM traffic.
#include "gb10_lab/common.h"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <type_traits>

namespace {

// ---------------------------------------------------------------------------
// dgpp gemv_common constants (verbatim).
// ---------------------------------------------------------------------------
constexpr int kWarps         = 8;             // weight rows per block
constexpr int kThreads       = kWarps * 32;
constexpr int kChunkBytes    = 16;            // one uint4 per lane per step
constexpr int kCodesPerChunk = 2 * kChunkBytes; // 32
constexpr int kGroup         = 16;            // codes per e4m3 scale (NVFP4)
constexpr int kSteps         = 1;             // row steps per warp
constexpr int kMaxChunksPerLane = 4;          // loads in flight per lane
constexpr int kMaxPasses     = 4;

// ---------------------------------------------------------------------------
// Row geometry as a compile-time function of K (dgpp Geom<K>, verbatim logic).
// ---------------------------------------------------------------------------
__host__ __device__ constexpr int pow2_divisor_le32(int r) {
    int l = 1;
    while (l < 32 && r % (l * 2) == 0) l *= 2;
    return l;
}
__host__ __device__ constexpr int lanes_per_row_of(int k) {
    const int r  = k / kCodesPerChunk;
    const int c0 = r < kMaxChunksPerLane ? r : kMaxChunksPerLane;
    if (c0 > 0 && r % c0 == 0) {
        const int l = r / c0;
        if (l >= 1 && l <= 32 && (l & (l - 1)) == 0) return l;
    }
    return pow2_divisor_le32(r);
}
__host__ __device__ constexpr int chunks_of(int k) { return (k / kCodesPerChunk) / lanes_per_row_of(k); }
__host__ __device__ constexpr bool k_supported(int k) {
    return k >= kCodesPerChunk && k % kCodesPerChunk == 0 &&
           chunks_of(k) <= kMaxChunksPerLane * kMaxPasses;
}

template <int K>
struct Geom {
    static_assert(k_supported(K), "tensor_fp4_gemv: K unsupported");
    static constexpr int row_chunks     = K / kCodesPerChunk;
    static constexpr int lanes_per_row  = lanes_per_row_of(K);
    static constexpr int chunks         = row_chunks / lanes_per_row;
    static constexpr int passes         = (chunks + kMaxChunksPerLane - 1) / kMaxChunksPerLane;
    static constexpr int rows_per_step  = 32 / lanes_per_row;
    static constexpr int rows_per_warp  = kSteps * rows_per_step;
    static constexpr int rows_per_block = kWarps * rows_per_warp;
    static constexpr int row_bytes      = K / 2;
    __host__ __device__ static constexpr int pass_chunks(int p) {
        const int left = chunks - p * kMaxChunksPerLane;
        return left < 1 ? 1 : (left < kMaxChunksPerLane ? left : kMaxChunksPerLane);
    }
};

__host__ __device__ constexpr int rows_per_block_of(int k) { return kWarps * (kSteps * (32 / lanes_per_row_of(k))); }

// ---------------------------------------------------------------------------
// Decode primitives (dgpp, verbatim).
// ---------------------------------------------------------------------------
__device__ __forceinline__ __half2 scales_f16(uint16_t packed) {
    return __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3));
}
__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

// Stage kRows bf16 activation rows into shared memory as [kRows][K] (dgpp
// stage_activations, specialised for the decode case where x is 16-byte aligned).
template <int kRows, int K>
__device__ __forceinline__ void stage_activations(const uint16_t* __restrict__ x, uint16_t* __restrict__ sx) {
    constexpr int vecs_per_row = K / 8;
#pragma unroll
    for (int i = threadIdx.x; i < kRows * vecs_per_row; i += blockDim.x) {
        const int r = i / vecs_per_row, c = i - r * vecs_per_row;
        reinterpret_cast<uint4*>(sx + static_cast<size_t>(r) * K)[c] =
            reinterpret_cast<const uint4*>(x + static_cast<size_t>(r) * K)[c];
    }
}

// The activation window: four uint4 out of shared memory, loaded once per chunk
// column and reused across the warp's row steps (dgpp load_window).
template <int kRows>
__device__ __forceinline__ void load_window(const uint16_t* __restrict__ sx, int k, int e0,
                                            uint4 (&xv)[kRows][4]) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const uint4* xp = reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * k + e0);
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[r][q] = xp[q];
    }
}

// Hardware NVFP4 decode of one 16-byte chunk (32 codes) against staged
// activations: e2m1x2 -> f16x2 (one F2FP), one __hmul2 by the group's e4m3
// scale (exact in f16), then the fp32 FMA chain (dgpp consume_chunk).
template <int kRows>
__device__ __forceinline__ void consume_chunk(const uint4& wchunk, __half2 s01,
                                              const uint4 (&xv)[kRows][4], float (&acc)[kRows]) {
    const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
    const __half2 s00 = __half2half2(__low2half(s01)); // codes 0-15
    const __half2 s11 = __half2half2(__high2half(s01)); // codes 16-31
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t word = words[q];
        const __half2 sq = q < 2 ? s00 : s11;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t byte = (word >> (8 * j)) & 0xFFu;
            const __half2 pair(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1));
            const __half2 p = __hmul2(pair, sq); // {lo,hi} x s, exact
            const float w0 = __low2float(p);
            const float w1 = __high2float(p);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const uint32_t xw = j == 0 ? xv[r][q].x : j == 1 ? xv[r][q].y : j == 2 ? xv[r][q].z : xv[r][q].w;
                const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw & 0xFFFFu));
                const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw >> 16));
                acc[r] = __fmaf_rn(w0, x0, acc[r]);
                acc[r] = __fmaf_rn(w1, x1, acc[r]);
            }
        }
    }
}

// Fixed xor tree across a lane group of Lanes lanes (dgpp group_reduce).
template <int kRows, int Lanes>
__device__ __forceinline__ void group_reduce(float (&acc)[kRows]) {
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int off = Lanes >> 1; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
}

__device__ __forceinline__ void store_dot(float* out, float v) { *out = v; }

// The chunk's scale word: two consecutive e4m3 bytes at scale_cols = K/16.
template <int K>
__device__ __forceinline__ uint16_t load_chunk_scale(const uint8_t* __restrict__ scales, int row, int boff) {
    return *reinterpret_cast<const uint16_t*>(scales + static_cast<size_t>(row) * (K / kGroup) + boff / 8);
}

// Loads held in registers between issue and consumption (dgpp PassLoads).
template <int NC>
struct PassLoads {
    uint4 wv[kSteps * NC];
    uint16_t sv[kSteps * NC];
};

// Streaming (evict-first) weight load: decode weights are read once per step.
__device__ __forceinline__ uint4 load_chunk_stream(const uint4* p) { return __ldcs(p); }

// Issue every chunk + scale of a pass before any is consumed (dgpp issue_pass).
template <int K, int C0, int NC>
__device__ __forceinline__ void issue_pass(const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
                                           int n0, int n, PassLoads<NC>& L) {
    using G   = Geom<K>;
    const int warp      = threadIdx.x / 32;
    const int lane      = threadIdx.x % 32;
    const int group     = lane / G::lanes_per_row;
    const int lig       = lane % G::lanes_per_row;
    const int row_base  = n0 + warp * G::rows_per_warp;
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
        const int row = row_base + st * G::rows_per_step + group;
        const bool ok = row < n;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int boff = ((C0 + c) * G::lanes_per_row + lig) * kChunkBytes;
            const int i    = st * NC + c;
            L.wv[i] = ok ? load_chunk_stream(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * G::row_bytes + boff))
                         : make_uint4(0u, 0u, 0u, 0u);
            L.sv[i] = ok ? load_chunk_scale<K>(scales, row, boff) : static_cast<uint16_t>(0);
        }
    }
}

// Consume a pass column by column: activation window once, every row step against it.
template <int K, int kRows, int C0, int NC>
__device__ __forceinline__ void consume_pass(const PassLoads<NC>& L, const uint16_t* __restrict__ sx, int n0, int n,
                                             float (&acc)[kSteps][kRows]) {
    using G  = Geom<K>;
    const int warp     = threadIdx.x / 32;
    const int lane     = threadIdx.x % 32;
    const int group    = lane / G::lanes_per_row;
    const int lig      = lane % G::lanes_per_row;
    const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int e0 = ((C0 + c) * G::lanes_per_row + lig) * kCodesPerChunk;
        uint4 xv[kRows][4];
        load_window<kRows>(sx, K, e0, xv);
#pragma unroll
        for (int st = 0; st < kSteps; ++st) {
            const int row = row_base + st * G::rows_per_step + group;
            if (row >= n) continue;
            const int i = st * NC + c;
            consume_chunk<kRows>(L.wv[i], scales_f16(L.sv[i]), xv, acc[st]);
        }
    }
}

template <int K, int kRows, int C0, int NC>
__device__ __forceinline__ void pass_row_dots(const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx, int n0, int n, float (&acc)[kSteps][kRows]) {
    PassLoads<NC> L;
    issue_pass<K, C0, NC>(w, scales, n0, n, L);
    consume_pass<K, kRows, C0, NC>(L, sx, n0, n, acc);
}

// A warp's rows against kRows staged activations across every pass, then the xor tree.
template <int K, int kRows>
__device__ __forceinline__ void warp_row_dots(const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx, int n0, int n, float (&acc)[kSteps][kRows]) {
    using G = Geom<K>;
#pragma unroll
    for (int st = 0; st < kSteps; ++st)
#pragma unroll
        for (int a = 0; a < kRows; ++a) acc[st][a] = 0.f;
    pass_row_dots<K, kRows, 0, G::pass_chunks(0)>(w, scales, sx, n0, n, acc);
    if constexpr (G::passes > 1) pass_row_dots<K, kRows, kMaxChunksPerLane, G::pass_chunks(1)>(w, scales, sx, n0, n, acc);
    if constexpr (G::passes > 2) pass_row_dots<K, kRows, 2 * kMaxChunksPerLane, G::pass_chunks(2)>(w, scales, sx, n0, n, acc);
    if constexpr (G::passes > 3) pass_row_dots<K, kRows, 3 * kMaxChunksPerLane, G::pass_chunks(3)>(w, scales, sx, n0, n, acc);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) group_reduce<kRows, G::lanes_per_row>(acc[st]);
}

// Which lane owns the reduced dot of the warp's step-st row.
template <int K>
__device__ __forceinline__ int owned_row(int n0, int st, bool& mine) {
    using G   = Geom<K>;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    mine = (lane % G::lanes_per_row) == 0;
    return n0 + warp * G::rows_per_warp + st * G::rows_per_step + lane / G::lanes_per_row;
}

// The decode kernel: m = 1 activation row, out[row] = dot(row, x) / g.
template <int K>
__global__ __launch_bounds__(kThreads, 2) void fp4_gemv_kernel(const uint8_t* __restrict__ w,
                                                               const uint8_t* __restrict__ scales, float g,
                                                               const uint16_t* __restrict__ x, float* __restrict__ out,
                                                               int n) {
    using G      = Geom<K>;
    const int n0 = blockIdx.x * G::rows_per_block;
    if (n0 >= n) return;
    extern __shared__ uint16_t sx[]; // [1][K]
    stage_activations<1, K>(x, sx);
    __syncthreads();

    float acc[kSteps][1];
    warp_row_dots<K, 1>(w, scales, sx, n0, n, acc);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
        bool mine = false;
        const int row = owned_row<K>(n0, st, mine);
        if (mine && row < n) store_dot(out + row, __fdiv_rn(acc[st][0], g));
    }
}

// ---------------------------------------------------------------------------
// Host quantizer: true NVFP4 two-level scaling.
//   global amax A -> g = A / (448 * 6) so every block scale fits e4m3 (<=448).
//   per block (16 elems): s_b = round_e4m3(amax_b * g / 6); code = round_e2m1(w*g / s_b_val).
//   reconstructed weight = code_val * s_b_val / g.
// ---------------------------------------------------------------------------
static const float kE2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

struct Nvfp4Encoded {
    std::vector<std::uint8_t> codes;  // [out, K/2], two e2m1 codes/byte (low nibble even)
    std::vector<std::uint8_t> scales; // [out, K/16], e4m3 bytes
    float g = 1.f;
};

static inline uint8_t round_e2m1(float v) {
    const uint32_t sign = (v < 0.f) ? 8u : 0u;
    const float a = std::fabs(v);
    int best = 0; float bestd = std::numeric_limits<float>::infinity();
    for (int m = 0; m < 8; ++m) { const float d = std::fabs(a - kE2M1[m]); if (d < bestd) { bestd = d; best = m; } }
    return static_cast<uint8_t>(best | sign);
}
static inline float e2m1_val(uint8_t code) { return (code & 8u) ? -kE2M1[code & 7u] : kE2M1[code & 7u]; }

Nvfp4Encoded quantize_nvfp4(const std::vector<float>& w, int out, int k) {
    Nvfp4Encoded e;
    double amax_all = 0.0;
    for (float v : w) amax_all = std::max(amax_all, std::fabs(static_cast<double>(v)));
    e.g = amax_all > 0.0 ? static_cast<float>(amax_all / (448.0 * 6.0)) : 1.f;
    const int groups = k / kGroup;
    e.codes.assign(static_cast<size_t>(out) * (k / 2), 0);
    e.scales.assign(static_cast<size_t>(out) * groups, 0);
    for (int r = 0; r < out; ++r) {
        for (int gb = 0; gb < groups; ++gb) {
            const float* row = w.data() + static_cast<size_t>(r) * k + gb * kGroup;
            float amax_b = 0.f;
            for (int i = 0; i < kGroup; ++i) amax_b = std::max(amax_b, std::fabs(row[i]));
            float s_f32 = amax_b > 0.f ? amax_b * e.g / 6.f : 0.f;
            if (s_f32 > 448.f) s_f32 = 448.f;
            __nv_fp8_e4m3 s8(s_f32);
            const uint8_t sbyte = s8.__x;
            const float s_val = static_cast<float>(s8);
            e.scales[static_cast<size_t>(r) * groups + gb] = sbyte;
            const float denom = s_val > 0.f ? s_val : 1.f;
            for (int i = 0; i < kGroup; ++i) {
                const uint8_t code = (s_val > 0.f) ? round_e2m1(row[i] * e.g / denom) : 0;
                const size_t elem = static_cast<size_t>(r) * k + gb * kGroup + i;
                uint8_t& byte = e.codes[elem >> 1];
                if ((elem & 1) == 0) byte = static_cast<uint8_t>((byte & 0xF0u) | (code & 0x0Fu));
                else byte = static_cast<uint8_t>((byte & 0x0Fu) | ((code & 0x0Fu) << 4));
            }
        }
    }
    return e;
}

void ref_gemv_nvfp4(const Nvfp4Encoded& e, const std::vector<__nv_bfloat16>& x, std::vector<float>& y, int out, int k) {
    const int groups = k / kGroup;
    for (int r = 0; r < out; ++r) {
        double acc = 0.0;
        for (int gb = 0; gb < groups; ++gb) {
            const float s_val = static_cast<float>(__nv_fp8_e4m3{e.scales[static_cast<size_t>(r) * groups + gb]});
            for (int i = 0; i < kGroup; ++i) {
                const size_t elem = static_cast<size_t>(r) * k + gb * kGroup + i;
                const uint8_t byte = e.codes[elem >> 1];
                const uint8_t code = (elem & 1) ? (byte >> 4) : (byte & 0x0F);
                const double w = static_cast<double>(e2m1_val(code)) * s_val / e.g;
                acc += w * static_cast<double>(__bfloat162float(x[elem - static_cast<size_t>(r) * k]));
            }
        }
        y[r] = static_cast<float>(acc);
    }
}

struct Case { const char* label; int out; int k; };

template <int K>
void launch_fp4(const uint8_t* dW, const uint8_t* dS, float g, const uint16_t* dX, float* dY, int n, cudaStream_t s) {
    using G = Geom<K>;
    const int blocks = (n + G::rows_per_block - 1) / G::rows_per_block;
    const size_t smem = static_cast<size_t>(K) * 2; // [1][K] bf16
    fp4_gemv_kernel<K><<<blocks, kThreads, smem, s>>>(dW, dS, g, dX, dY, n);
}

void run_case(const LabArgs& args, const Case& c) {
    const int O = c.out, K = c.k;
    std::vector<float> hW(static_cast<size_t>(O) * K), hx(K);
    {
        std::uint64_t st = 0x9E3779B97F4A7C15ull;
        for (auto& v : hW) { st = st * 6364136223846793005ull + 1442695040888963407ull; v = ((st >> 40) & 0xFFFFF) / 1048576.f * 2.f - 1.f; }
        st = 0xBF58476D1CE4E5B9ull;
        for (auto& v : hx) { st = st * 6364136223846793005ull + 1442695040888963407ull; v = ((st >> 40) & 0xFFFFF) / 1048576.f * 2.f - 1.f; }
    }
    std::vector<__nv_bfloat16> hxb(K);
    for (int i = 0; i < K; ++i) hxb[i] = __float2bfloat16_rn(hx[i]);

    Nvfp4Encoded e = quantize_nvfp4(hW, O, K);
    std::vector<float> ref(O);
    ref_gemv_nvfp4(e, hxb, ref, O, K);

    uint8_t* dW = nullptr; LAB_CHECK(cudaMalloc(&dW, static_cast<size_t>(O) * (K / 2)));
    LAB_CHECK(cudaMemcpy(dW, e.codes.data(), static_cast<size_t>(O) * (K / 2), cudaMemcpyHostToDevice));
    uint8_t* dS = nullptr; LAB_CHECK(cudaMalloc(&dS, static_cast<size_t>(O) * (K / 16)));
    LAB_CHECK(cudaMemcpy(dS, e.scales.data(), static_cast<size_t>(O) * (K / 16), cudaMemcpyHostToDevice));
    uint16_t* dX = nullptr; LAB_CHECK(cudaMalloc(&dX, static_cast<size_t>(K) * 2));
    LAB_CHECK(cudaMemcpy(dX, hxb.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice));
    float* dY = nullptr; LAB_CHECK(cudaMalloc(&dY, static_cast<size_t>(O) * sizeof(float)));

    auto body = [&](cudaStream_t s) {
        if (K == 2560) launch_fp4<2560>(dW, dS, e.g, dX, dY, O, s);
        else if (K == 640) launch_fp4<640>(dW, dS, e.g, dX, dY, O, s);
        else throw std::invalid_argument("tensor_fp4_gemv: K not instantiated");
    };
    auto us = timed_iterations(body, args.warmup, args.iters);
    auto t = summarize(us);

    std::vector<float> got(O);
    LAB_CHECK(cudaMemcpy(got.data(), dY, static_cast<size_t>(O) * sizeof(float), cudaMemcpyDeviceToHost));
    ErrStats err = compare_f32(got.data(), ref.data(), static_cast<size_t>(O), 5e-2, 5e-2);

    const double bytes_moved = static_cast<double>(O) * K * 0.5 + static_cast<double>(O) * (K / 16) * 1.0;
    TimingResult tr{t.median_us, t.median_us, t.median_us, t.median_us, t.median_us, args.iters};
    emit_timing(args, "tensor_fp4_gemv", tr, err, bytes_moved, 2.0 * O * K,
                ",\"label\":\"" + std::string(c.label) + "\",\"out\":" + fmt_num(O) +
                ",\"k\":" + fmt_num(K) + ",\"lanes_per_row\":" + fmt_num(lanes_per_row_of(K)) +
                ",\"rows_per_block\":" + fmt_num(rows_per_block_of(K)) + ",\"format\":\"nvfp4\"");
    const double gbps = bytes_moved / (t.median_us * 1e-6) / 1e9;
    std::printf("  nvfp4 %-10s out=%-7d k=%-5d %9.2f us  enc %6.1f GB/s  correct=%s\n", c.label, O, K, t.median_us,
                gbps, err.pass ? "yes" : "NO");

    cudaFree(dW); cudaFree(dS); cudaFree(dX); cudaFree(dY);
}

} // namespace

void run_tensor_fp4_gemv(const LabArgs& args) {
    // Same decode shapes as dense_width so the numbers line up against its Q8 column.
    std::vector<Case> cases = {
        {"lm_head",   248320, 2560},
        {"gdn_out",     6144, 2560},
        {"hc_mix",      5120, 2560},
        {"shared_mlp",  1920,  640},
        {"proj_pack",  51200, 2560},
    };
    if (args.quick) cases = {{"proj_pack", 25600, 2560}};

    // dense_width q8_g32_fp16 medians (us) at these exact shapes, from
    // bench/gb10_lab/results/dense_width.jsonl — the baseline R1 must beat.
    std::printf("  (Q8 baseline from dense_width.jsonl: lm_head 3014.2, gdn_out 73.66, hc_mix 61.43,\n"
                "   shared_mlp 10.28, proj_pack 636.5 us)\n\n");
    for (const auto& c : cases) run_case(args, c);
}
