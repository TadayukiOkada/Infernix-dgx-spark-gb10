// dense_width.cu — Layer 3 extension: dense decode weight bit-width A/B.
//
// Question this op answers (roadmap R1, docs/performance/dgpp-gap-roadmap.md):
//   dgpp's single-stream edge is hypothesised to come from decoding the dense
//   projection family at 4-bit ("w4") while Infernix Dense8 decodes it at 8-bit
//   (q8_g32_fp16). Halving the weight bytes only helps if the nibble-dequant ALU
//   work does not destroy the bandwidth win. This op measures exactly that: the
//   same logical weight matrix streamed as BF16, q8_g32_fp16, and q4_g64_fp16,
//   at real Qwen3.8-Flash-Next decode shapes, reporting per-format latency AND
//   achieved encoded-bytes bandwidth against the ~246 GB/s GB10 ceiling.
//
// It is deliberately independent of infernix_ops: its own codecs, its own
// allocations, matching the storage-layout contracts in
// docs/maintainer/tensor-formats.md and the row-split codecs in
// src/ops/resident_moe/resident_moe.cu, so the numbers describe the format, not
// product dispatch quirks.
#include "gb10_lab/common.h"

namespace {

// ---------------------------------------------------------------------------
// Quantization (host) using the exact stored-code contracts.
//   q8_g32_fp16 : int8 codes [-127,127], one fp16 scale per group of 32.
//   q4_g64_fp16 : 4-bit codes [-8,7] packed two-per-byte (low nibble first),
//                 one fp16 scale per group of 64. Signed nibble u maps to
//                 (u >= 8 ? u - 16 : u), matching resident_moe::Q4.
// ---------------------------------------------------------------------------
struct Encoded {
    std::vector<std::uint8_t> codes;   // q8: 1 byte/weight ; q4: 2 weights/byte
    std::vector<__nv_bfloat16> scales; // fp16-as-bf16 storage width irrelevant; we store __half below
    std::vector<__half> half_scales;
    int groups;
};

Encoded quantize_q8(const std::vector<float>& w, int out, int k) {
    Encoded e;
    e.groups = k / 32;
    e.codes.assign((std::size_t)out * k, 0);
    e.half_scales.assign((std::size_t)out * e.groups, __half(0));
    for (int r = 0; r < out; ++r) {
        for (int g = 0; g < e.groups; ++g) {
            float amax = 0.f;
            const float* row = w.data() + (std::size_t)r * k + g * 32;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(row[i]));
            const float scale = amax > 0.f ? amax / 127.f : 0.f;
            e.half_scales[(std::size_t)r * e.groups + g] = __float2half_rn(scale);
            for (int i = 0; i < 32; ++i) {
                int code = (int)std::lround(row[i] / (scale > 0.f ? scale : 1.f));
                if (code > 127) code = 127;
                if (code < -127) code = -127;
                e.codes[(std::size_t)r * k + g * 32 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
            }
        }
    }
    return e;
}

// NVFP4-style benchmark encoding isolating dgpp's DECODE PRIMITIVE:
//   payload : e2m1 codes packed two-per-byte (low nibble = even element),
//             magnitudes {0,.5,1,1.5,2,3,4,6} via 3-bit index + sign bit 3.
//   scales  : one fp16 scale per group of 16 codes, pre-multiplied by 2^14
//             so the device folds it with the e2m1 value's 2^-14 exactly
//             (matches dgpp fp4_gemv scales_x16384 / e2m1_scaled).
// Scale storage here is fp16 (2 B/group) rather than NVFP4's e4m3 (1 B/group);
// that adds ~1 B/weight vs true NVFP4 but is immaterial to the decode-throughput
// question, which is about the code unpacking, not the scale width.
static const float kE2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
Encoded quantize_fp4(const std::vector<float>& w, int out, int k) {
    Encoded e;
    e.groups = k / 16;
    e.codes.assign((std::size_t)out * k / 2, 0);
    e.half_scales.assign((std::size_t)out * e.groups, __half(0));
    for (int r = 0; r < out; ++r) {
        for (int g = 0; g < e.groups; ++g) {
            float amax = 0.f;
            const float* row = w.data() + (std::size_t)r * k + g * 16;
            for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(row[i]));
            const float scale = amax > 0.f ? amax / 6.f : 0.f; // largest magnitude -> code 6
            e.half_scales[(std::size_t)r * e.groups + g] = __float2half_rn(scale * 16384.f);
            for (int i = 0; i < 16; ++i) {
                const float ratio = scale > 0.f ? row[i] / scale : 0.f;
                int best = 0; float bestd = 1e30f;
                for (int m = 0; m < 8; ++m) { const float d = std::fabs(std::fabs(ratio) - kE2M1[m]); if (d < bestd) { bestd = d; best = m; } }
                const int nibble = best | (ratio < 0.f ? 8 : 0);
                const int global = g * 16 + i;
                std::uint8_t& byte = e.codes[(std::size_t)r * k / 2 + global / 2];
                if (global % 2 == 0) byte |= static_cast<std::uint8_t>(nibble & 0xF);
                else                 byte |= static_cast<std::uint8_t>((nibble & 0xF) << 4);
            }
        }
    }
    return e;
}
void ref_gemv_deq_fp4(const Encoded& e, const std::vector<float>& x, std::vector<float>& y, int out, int k) {
    y.assign(out, 0.f);
    for (int r = 0; r < out; ++r)
        for (int g = 0; g < e.groups; ++g) {
            const float s = __half2float(e.half_scales[(std::size_t)r * e.groups + g]) / 16384.f;
            double acc = 0.0;
            for (int i = 0; i < 16; ++i) {
                const int global = g * 16 + i;
                const std::uint8_t byte = e.codes[(std::size_t)r * k / 2 + global / 2];
                const int nibble = (global % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                const float mag = kE2M1[nibble & 7];
                const float code = (nibble & 8) ? -mag : mag;
                acc += (double)code * s * x[global];
            }
            y[r] += (float)acc;
        }
}

Encoded quantize_q4(const std::vector<float>& w, int out, int k) {
    Encoded e;
    e.groups = k / 64;
    e.codes.assign((std::size_t)out * k / 2, 0); // two nibbles per byte
    e.half_scales.assign((std::size_t)out * e.groups, __half(0));
    for (int r = 0; r < out; ++r) {
        for (int g = 0; g < e.groups; ++g) {
            float amax = 0.f;
            const float* row = w.data() + (std::size_t)r * k + g * 64;
            for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(row[i]));
            const float scale = amax > 0.f ? amax / 8.f : 0.f; // symmetric target so codes fit [-8,7]
            e.half_scales[(std::size_t)r * e.groups + g] = __float2half_rn(scale);
            for (int i = 0; i < 64; ++i) {
                int code = (int)std::lround(row[i] / (scale > 0.f ? scale : 1.f));
                if (code > 7) code = 7;
                if (code < -8) code = -8;
                const int nibble = code < 0 ? code + 16 : code; // [0,15], signed via u>=8?u-16:u
                const int global = g * 64 + i;                  // code index within the row
                std::uint8_t& byte = e.codes[(std::size_t)r * k / 2 + global / 2];
                if (global % 2 == 0) byte |= static_cast<std::uint8_t>(nibble & 0xF);          // low nibble
                else                 byte |= static_cast<std::uint8_t>((nibble & 0xF) << 4);   // high nibble
            }
        }
    }
    return e;
}

// Host reference y[r] = sum_j dequant(code)[r][j] * x[j] — uses the SAME decoded
// values the device kernel sees, so tolerance reflects only reduction differences.
void ref_gemv_deq_q8(const Encoded& e, const std::vector<float>& x, std::vector<float>& y, int out, int k) {
    y.assign(out, 0.f);
    for (int r = 0; r < out; ++r)
        for (int g = 0; g < e.groups; ++g) {
            const float s = __half2float(e.half_scales[(std::size_t)r * e.groups + g]);
            double acc = 0.0;
            for (int i = 0; i < 32; ++i) {
                const int c = static_cast<std::int8_t>(e.codes[(std::size_t)r * k + g * 32 + i]);
                acc += (double)c * s * x[g * 32 + i];
            }
            y[r] += (float)acc;
        }
}
void ref_gemv_deq_q4(const Encoded& e, const std::vector<float>& x, std::vector<float>& y, int out, int k) {
    y.assign(out, 0.f);
    for (int r = 0; r < out; ++r)
        for (int g = 0; g < e.groups; ++g) {
            const float s = __half2float(e.half_scales[(std::size_t)r * e.groups + g]);
            double acc = 0.0;
            for (int i = 0; i < 64; ++i) {
                const int global = g * 64 + i;
                const std::uint8_t byte = e.codes[(std::size_t)r * k / 2 + global / 2];
                const int u = (global % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                const int code = u >= 8 ? u - 16 : u;
                acc += (double)code * s * x[global];
            }
            y[r] += (float)acc;
        }
}

// ---------------------------------------------------------------------------
// Device GEMV kernels. Activation x is FP32 and shared across all three formats
// (A16 activations in production too), so the ONLY variable is the weight stream.
// One block per output row; uint4 vectorised inner product; warp+shared reduce.
// ---------------------------------------------------------------------------
constexpr int BLOCK = 256;

__global__ void k_gemv_bf16(const uint4* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int out, int nv /* = k/8 */) {
    const int r = blockIdx.x;
    if (r >= out) return;
    const uint4* Wr = W + (std::size_t)r * nv;
    float local = 0.f;
    for (int g = threadIdx.x; g < nv; g += BLOCK) {
        BF16x8 w = u4_to_bf16(Wr[g]);
        #pragma unroll
        for (int e = 0; e < 8; ++e) local += bf16_f(w.x[e]) * x[g * 8 + e];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    __shared__ float smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) acc += smem[i];
        y[r] = acc;
    }
}

// q8: one uint4 chunk = 16 codes = half a 32-group; scale index = chunk/2.
__global__ void k_gemv_q8(const uint4* __restrict__ codes, const __half* __restrict__ scales,
                          const float* __restrict__ x, float* __restrict__ y, int out,
                          int chunks /* = k/16 */, int groups /* = k/32 */) {
    const int r = blockIdx.x;
    if (r >= out) return;
    const uint4* crow = codes + (std::size_t)r * chunks;
    const __half* srow = scales + (std::size_t)r * groups;
    float local = 0.f;
    for (int c = threadIdx.x; c < chunks; c += BLOCK) {
        const uint4 cw = crow[c]; // one uint4 = 16 int8 codes = half a 32-group
        const float s = __half2float(srow[c / 2]);
        const std::int8_t* cc = reinterpret_cast<const std::int8_t*>(&cw);
        #pragma unroll
        for (int i = 0; i < 16; ++i) local += (float)cc[i] * s * x[c * 16 + i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    __shared__ float smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) acc += smem[i];
        y[r] = acc;
    }
}

// q4: one uint4 chunk = 32 codes = half a 64-group; scale index = chunk/2.
// Nibble decode matches resident_moe::Q4 (u>=8?u-16:u), low nibble first.
__global__ void k_gemv_q4(const uint4* __restrict__ codes, const __half* __restrict__ scales,
                          const float* __restrict__ x, float* __restrict__ y, int out,
                          int chunks /* = k/32 */, int groups /* = k/64 */) {
    const int r = blockIdx.x;
    if (r >= out) return;
    const uint4* crow = codes + (std::size_t)r * chunks;
    const __half* srow = scales + (std::size_t)r * groups;
    float local = 0.f;
    for (int c = threadIdx.x; c < chunks; c += BLOCK) {
        const uint4 cw = crow[c];
        const std::uint32_t words[4] = {cw.x, cw.y, cw.z, cw.w};
        const float s = __half2float(srow[c / 2]);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int u = (int)((words[i] >> (4 * j)) & 0xFU);
                local += (float)(u >= 8 ? u - 16 : u) * s * x[c * 32 + i * 8 + j];
            }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    __shared__ float smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) acc += smem[i];
        y[r] = acc;
    }
}

// dgpp-style FP4 GEMV: branchless e2m1 bit-placement decode. One uint4 chunk =
// 32 codes spanning two 16-code groups; scales are fp16 pre-scaled by 2^14.
__device__ __forceinline__ float e2m1_scaled(uint32_t code) {
    const uint32_t bits = ((code & 7u) << 9) | ((code & 8u) << 12);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}
__global__ void k_gemv_fp4(const uint4* __restrict__ codes, const __half* __restrict__ scales,
                           const float* __restrict__ x, float* __restrict__ y, int out,
                           int chunks /* = k/32 */, int groups /* = k/16 */) {
    const int r = blockIdx.x;
    if (r >= out) return;
    const uint4* crow = codes + (std::size_t)r * chunks;
    const __half* srow = scales + (std::size_t)r * groups;
    float local = 0.f;
    for (int c = threadIdx.x; c < chunks; c += BLOCK) {
        const uint4 cw = crow[c];
        const std::uint32_t words[4] = {cw.x, cw.y, cw.z, cw.w};
        // chunk c covers codes [c*32, c*32+32) -> groups (c*2) and (c*2+1), each 16 codes.
        const float s0 = __half2float(srow[c * 2]);
        const float s1 = __half2float(srow[c * 2 + 1]);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t w = words[i];
                const int lo = i * 16 + j * 2, hi = lo + 1;
                const float s = lo < 16 ? s0 : s1;
                local = fmaf(e2m1_scaled(w & 0xFu), s * x[c * 32 + lo], local);
                local = fmaf(e2m1_scaled((w >> 4) & 0xFu), s * x[c * 32 + hi], local);
            }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    __shared__ float smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) acc += smem[i];
        y[r] = acc;
    }
}

struct Case { const char* label; int out; int k; };

void run_case(const LabArgs& args, const Case& c, double& q8_median) {
    const int O = c.out, K = c.k;
    std::vector<float> hW((std::size_t)O * K), hx(K);
    { // deterministic logical weights in [-1,1] and activation in [-1,1]
        std::uint64_t st = 0x9E3779B97F4A7C15ull;
        for (auto& v : hW) { st = st * 6364136223846793005ull + 1442695040888963407ull; v = ((st >> 40) & 0xFFFFF) / 1048576.f * 2.f - 1.f; }
        st = 0xBF58476D1CE4E5B9ull;
        for (auto& v : hx) { st = st * 6364136223846793005ull + 1442695040888963407ull; v = ((st >> 40) & 0xFFFFF) / 1048576.f * 2.f - 1.f; }
    }

    Encoded e8 = quantize_q8(hW, O, K);
    Encoded e4 = quantize_q4(hW, O, K);
    Encoded ef = quantize_fp4(hW, O, K);

    std::vector<float> ref8, ref4, reff;
    ref_gemv_deq_q8(e8, hx, ref8, O, K);
    ref_gemv_deq_q4(e4, hx, ref4, O, K);
    ref_gemv_deq_fp4(ef, hx, reff, O, K);

    // Device buffers.
    __nv_bfloat16* dWbf = nullptr; LAB_CHECK(cudaMalloc(&dWbf, (std::size_t)O * K * sizeof(__nv_bfloat16)));
    std::vector<__nv_bfloat16> hbf((std::size_t)O * K);
    for (std::size_t i = 0; i < hbf.size(); ++i) hbf[i] = f_bf16(hW[i]);
    LAB_CHECK(cudaMemcpy(dWbf, hbf.data(), hbf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    uint4* dC8 = nullptr; LAB_CHECK(cudaMalloc(&dC8, (std::size_t)O * K));
    LAB_CHECK(cudaMemcpy(dC8, e8.codes.data(), (std::size_t)O * K, cudaMemcpyHostToDevice));
    __half* dS8 = nullptr; LAB_CHECK(cudaMalloc(&dS8, (std::size_t)O * e8.groups * sizeof(__half)));
    LAB_CHECK(cudaMemcpy(dS8, e8.half_scales.data(), (std::size_t)O * e8.groups * sizeof(__half), cudaMemcpyHostToDevice));

    uint4* dC4 = nullptr; LAB_CHECK(cudaMalloc(&dC4, (std::size_t)O * K / 2));
    LAB_CHECK(cudaMemcpy(dC4, e4.codes.data(), (std::size_t)O * K / 2, cudaMemcpyHostToDevice));
    __half* dS4 = nullptr; LAB_CHECK(cudaMalloc(&dS4, (std::size_t)O * e4.groups * sizeof(__half)));
    LAB_CHECK(cudaMemcpy(dS4, e4.half_scales.data(), (std::size_t)O * e4.groups * sizeof(__half), cudaMemcpyHostToDevice));

    uint4* dCf = nullptr; LAB_CHECK(cudaMalloc(&dCf, (std::size_t)O * K / 2));
    LAB_CHECK(cudaMemcpy(dCf, ef.codes.data(), (std::size_t)O * K / 2, cudaMemcpyHostToDevice));
    __half* dSf = nullptr; LAB_CHECK(cudaMalloc(&dSf, (std::size_t)O * ef.groups * sizeof(__half)));
    LAB_CHECK(cudaMemcpy(dSf, ef.half_scales.data(), (std::size_t)O * ef.groups * sizeof(__half), cudaMemcpyHostToDevice));

    float* dx = nullptr; LAB_CHECK(cudaMalloc(&dx, (std::size_t)K * sizeof(float)));
    LAB_CHECK(cudaMemcpy(dx, hx.data(), (std::size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    float* dy = nullptr; LAB_CHECK(cudaMalloc(&dy, (std::size_t)O * sizeof(float)));

    auto report = [&](const char* fmt, double median_us, const ErrStats& err, double bytes_moved) {
        TimingResult t{median_us, median_us, median_us, median_us, median_us, args.iters};
        emit_timing(args, "dense_width", t, err, bytes_moved, 2.0 * O * K,
                    ",\"label\":\"" + std::string(c.label) + "\",\"out\":" + fmt_num(O) +
                    ",\"k\":" + fmt_num(K) + ",\"format\":\"" + fmt + "\"");
        double gbps = bytes_moved / (median_us * 1e-6) / 1e9;
        std::printf("  dense %-10s out=%-7d k=%-5d %-4s %9.2f us  enc %6.1f GB/s  correct=%s\n",
                    c.label, O, K, fmt, median_us, gbps, err.pass ? "yes" : "NO");
    };

    // BF16 stream.
    {
        auto body = [&](cudaStream_t s){ k_gemv_bf16<<<O, BLOCK, 0, s>>>((uint4*)dWbf, dx, dy, O, K / 8); };
        auto us = timed_iterations(body, args.warmup, args.iters);
        auto t = summarize(us);
        std::vector<float> got(O); LAB_CHECK(cudaMemcpy(got.data(), dy, (std::size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
        // bf16 reference: dequantize logical weights to bf16 then dot fp32.
        std::vector<float> refb(O, 0.f);
        for (int r = 0; r < O; ++r) { double acc = 0; for (int j = 0; j < K; ++j) acc += (double)bf16_f(hbf[(std::size_t)r*K+j]) * hx[j]; refb[r]=(float)acc; }
        ErrStats err = compare_f32(got.data(), refb.data(), (std::size_t)O, 5e-2, 5e-2);
        report("bf16", t.median_us, err, (double)O * K * 2 + (double)K * 4);
    }
    // Q8 stream.
    {
        auto body = [&](cudaStream_t s){ k_gemv_q8<<<O, BLOCK, 0, s>>>(dC8, dS8, dx, dy, O, K / 16, e8.groups); };
        auto us = timed_iterations(body, args.warmup, args.iters);
        auto t = summarize(us);
        std::vector<float> got(O); LAB_CHECK(cudaMemcpy(got.data(), dy, (std::size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
        ErrStats err = compare_f32(got.data(), ref8.data(), (std::size_t)O, 5e-2, 5e-2);
        q8_median = t.median_us;
        report("q8", t.median_us, err, (double)O * K * 1 + (double)O * e8.groups * 2);
    }
    // Q4 stream.
    {
        auto body = [&](cudaStream_t s){ k_gemv_q4<<<O, BLOCK, 0, s>>>(dC4, dS4, dx, dy, O, K / 32, e4.groups); };
        auto us = timed_iterations(body, args.warmup, args.iters);
        auto t = summarize(us);
        std::vector<float> got(O); LAB_CHECK(cudaMemcpy(got.data(), dy, (std::size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
        ErrStats err = compare_f32(got.data(), ref4.data(), (std::size_t)O, 5e-2, 5e-2);
        report("q4", t.median_us, err, (double)O * K * 0.5 + (double)O * e4.groups * 2);
        std::printf("     -> q4 vs q8 latency ratio: %.3fx  (1.0x would mean the halved stream cost nothing)\n",
                    t.median_us / std::max(q8_median, 1e-9));
    }
    // FP4 stream (dgpp branchless e2m1 bit-placement decode).
    {
        auto body = [&](cudaStream_t s){ k_gemv_fp4<<<O, BLOCK, 0, s>>>(dCf, dSf, dx, dy, O, K / 32, ef.groups); };
        auto us = timed_iterations(body, args.warmup, args.iters);
        auto t = summarize(us);
        std::vector<float> got(O); LAB_CHECK(cudaMemcpy(got.data(), dy, (std::size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
        ErrStats err = compare_f32(got.data(), reff.data(), (std::size_t)O, 8e-2, 8e-2);
        report("fp4", t.median_us, err, (double)O * K * 0.5 + (double)O * ef.groups * 2);
        std::printf("     -> fp4 vs q8 latency ratio: %.3fx  (<1.0 means dgpp-style FP4 beats Q8)\n",
                    t.median_us / std::max(q8_median, 1e-9));
    }

    cudaFree(dWbf); cudaFree(dC8); cudaFree(dS8); cudaFree(dC4); cudaFree(dS4); cudaFree(dCf); cudaFree(dSf); cudaFree(dx); cudaFree(dy);
}

} // namespace

void run_dense_width(const LabArgs& args) {
    // Real Qwen3.8-Flash-Next geometry: hidden=2560, vocab=248320, moe_inter=640.
    // Shapes chosen so the weight stream dominates DRAM (the decode regime).
    std::vector<Case> cases = {
        {"lm_head",   248320, 2560},  // output head: the single biggest dense read
        {"gdn_out",     6144, 2560},  // GDN value-head output projection proxy
        {"hc_mix",      5120, 2560},  // hyper-connection mixer proxy (rank*hidden-ish)
        {"shared_mlp",  1920,  640},  // shared-expert gate/up/down family proxy
        {"proj_pack",  51200, 2560},  // many [2560,2560] projections packed: Category-C aggregate
    };
    if (args.quick) cases = {{"proj_pack", 25600, 2560}};

    for (const auto& c : cases) {
        double q8 = 0.0;
        run_case(args, c, q8);
    }
}
