// int_dot.cu — Integer / packed-integer dot-product candidate families.
//
// Mission focus: find the FASTEST correct way to compute a dot product on GB10
// for each shape and numerical contract. Every candidate is judged against an
// EXACT integer host reference (no float tolerance) before any timing counts.
//
// Candidate families (same mathematical work, same INT8 inputs, INT32 accumulate):
//   1. scalar      : one int8 multiply + add per element (baseline SIMT)
//   2. unroll      : four independent INT32 accumulators, unrolled element loop
//   3. dp4a        : __dp4a packed signed-byte dot-accumulate (IDP.4A on sm_121)
//   4. dp4a_vec    : dp4a over uint4-staged words + warp/shared block reduction
//                    (used for long-K reductions where one thread per row starves)
//   5. binary_popc : popcount(a & b) over packed bits (binary-valued inputs only)
//
// Reduction strategies (large single dot products): warp-shuffle, hierarchical
// shared-memory, and split-K across blocks.
//
// Workload classes map to the mission matrix:
//   Tiny   K=1..32, many rows   -> thread-per-row, exposes launch + instruction cost
//   Medium K=64..1024           -> thread-per-row / small blocks
//   Large  K=4096,16384         -> block-per-row and split-K
//   Irregular odd K             -> tail padding measured explicitly
//   Batched many tiny dots      -> large row counts, launch amortisation
//
// Operation-count convention (stated in every record): a length-K dot product is
// counted as 2*K operations (K multiplies + K adds). Bytes moved counts BOTH
// operand streams (A and B), i.e. 2*B*K for B pairwise rows of length K, because
// these are independent dot products with no cross-row operand reuse. INT8 inputs
// => 1 byte/element. Padding to a 16-byte row stride is real extra read traffic
// and is included in kernel time but NOT in the logical byte count.
#include "gb10_lab/common.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Round a row byte-length up to a multiple of `m` so uint4/dp4a loads stay
// aligned without reading past the allocation. Padding lanes are zero, so their
// products contribute nothing to the result.
inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// Deterministic INT8 fill in [-127,127] (avoids the -128 asymmetric edge so the
// accumulator range is unambiguous; the extreme-value case is tested separately).
inline void fill_i8(std::vector<int8_t>& h, std::uint64_t seed) {
    std::uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (auto& v : h) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        int x = static_cast<int>((s >> 33) & 0xFF);       // 0..255
        v = static_cast<int8_t>(x > 127 ? x - 256 : x);   // -128..127, clamp away -128 below
        if (v == -128) v = 127;                            // exclude the asymmetric edge here
    }
}

// Exact integer comparison: device INT32 output vs host INT64 reference. Any
// nonzero difference is a hard failure (integer arithmetic is exact).
ErrStats compare_int(const int32_t* got, const int64_t* ref, std::size_t n) {
    ErrStats s; double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)got[i] - (double)ref[i]);
        worst = std::max(worst, d);
    }
    s.max_abs_err = worst;
    s.max_rel_err = worst;
    s.pass = (worst == 0.0);
    return s;
}

// ---------------------------------------------------------------------------
// Host references (exact, independent of the kernels).
// ---------------------------------------------------------------------------
void ref_dot_pairwise(std::vector<int64_t>& out, const std::vector<int8_t>& A,
                      const std::vector<int8_t>& B, int rows, int K, int stride) {
    out.assign(rows, 0);
    for (int r = 0; r < rows; ++r) {
        int64_t acc = 0;
        const int8_t* a = A.data() + (size_t)r * stride;
        const int8_t* b = B.data() + (size_t)r * stride;
        for (int j = 0; j < K; ++j) acc += (int64_t)a[j] * (int64_t)b[j];
        out[r] = acc;
    }
}

// Binary inner product for bit-packed 0/1 vectors: number of positions where
// both bits are 1 == popcount(A_word & B_word) summed over words. This is the
// mathematically valid mapping for a binary dot product. (XNOR would additionally
// count both-zero agreements, which is agreement, not an inner product.)
void ref_binary_dot(std::vector<int64_t>& out, const std::vector<uint32_t>& A,
                    const std::vector<uint32_t>& B, int rows, int words) {
    out.assign(rows, 0);
    for (int r = 0; r < rows; ++r) {
        int64_t acc = 0;
        for (int w = 0; w < words; ++w)
            acc += __builtin_popcount(A[(size_t)r*words+w] & B[(size_t)r*words+w]);
        out[r] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel family 1-3: thread-per-row (each thread owns one whole dot product).
// Best for tiny/medium K with many rows; exposes pure instruction cost.
// ---------------------------------------------------------------------------
__global__ void k_dot_scalar_tpr(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 int32_t* __restrict__ out, int rows, int K, int stride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int8_t* a = A + (size_t)r * stride;
    const int8_t* b = B + (size_t)r * stride;
    int32_t acc = 0;
    for (int j = 0; j < K; ++j) acc += (int32_t)a[j] * (int32_t)b[j];
    out[r] = acc;
}

__global__ void k_dot_unroll_tpr(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                 int32_t* __restrict__ out, int rows, int K, int stride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int8_t* a = A + (size_t)r * stride;
    const int8_t* b = B + (size_t)r * stride;
    int32_t c0=0,c1=0,c2=0,c3=0;
    int j = 0;
    for (; j + 4 <= K; j += 4) {
        c0 += (int32_t)a[j  ] * (int32_t)b[j  ];
        c1 += (int32_t)a[j+1] * (int32_t)b[j+1];
        c2 += (int32_t)a[j+2] * (int32_t)b[j+2];
        c3 += (int32_t)a[j+3] * (int32_t)b[j+3];
    }
    for (; j < K; ++j) c0 += (int32_t)a[j] * (int32_t)b[j];
    out[r] = c0 + c1 + c2 + c3;
}

// dp4a thread-per-row. Reads ceil(K/4) aligned int words from a 4-byte-strided
// buffer (stride already rounded to 16 by the host), so every word is in-bounds;
// zero padding lanes make unused products contribute 0. Signed __dp4a.
__global__ void k_dot_dp4a_tpr(const int32_t* __restrict__ A, const int32_t* __restrict__ B,
                               int32_t* __restrict__ out, int rows, int words, int wstride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int32_t* a = A + (size_t)r * wstride;
    const int32_t* b = B + (size_t)r * wstride;
    int32_t acc = 0;
    #pragma unroll 4
    for (int w = 0; w < words; ++w) acc = __dp4a(a[w], b[w], acc);
    out[r] = acc;
}

// dp4a with 4 independent accumulators to hide IDP latency on longer rows.
__global__ void k_dot_dp4a4_tpr(const int32_t* __restrict__ A, const int32_t* __restrict__ B,
                                int32_t* __restrict__ out, int rows, int words, int wstride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int32_t* a = A + (size_t)r * wstride;
    const int32_t* b = B + (size_t)r * wstride;
    int32_t c0=0,c1=0,c2=0,c3=0;
    int w = 0;
    for (; w + 4 <= words; w += 4) {
        c0 = __dp4a(a[w  ], b[w  ], c0);
        c1 = __dp4a(a[w+1], b[w+1], c1);
        c2 = __dp4a(a[w+2], b[w+2], c2);
        c3 = __dp4a(a[w+3], b[w+3], c3);
    }
    for (; w < words; ++w) c0 = __dp4a(a[w], b[w], c0);
    out[r] = c0 + c1 + c2 + c3;
}

// ---------------------------------------------------------------------------
// Kernel family 4: block-per-row dp4a with vectorised uint4 staging + warp
// shuffle + hierarchical shared-memory reduction. Used when K is large enough
// that a single thread per row under-parallelises the 48 SMs.
// ---------------------------------------------------------------------------
template <int BLOCK>
__global__ void k_dot_dp4a_block(const int32_t* __restrict__ A, const int32_t* __restrict__ B,
                                 int32_t* __restrict__ out, int rows, int words, int wstride) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int32_t* a = A + (size_t)r * wstride;
    const int32_t* b = B + (size_t)r * wstride;
    int32_t acc = 0;
    // uint4 load = 4 words = 16 int8 per iteration, coalesced across threads.
    for (int g = threadIdx.x; g < words; g += BLOCK) {
        int aw = a[g], bw = b[g];
        acc = __dp4a(aw, bw, acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ int32_t smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t tot = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) tot += smem[i];
        out[r] = tot;
    }
}

// ---------------------------------------------------------------------------
// Kernel family 5: binary inner product via popcount(A & B) over packed bits.
// Only valid for 0/1 inputs; separate numerical contract from INT8 dot.
// ---------------------------------------------------------------------------
__global__ void k_binary_popc_tpr(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
                                  int32_t* __restrict__ out, int rows, int words, int wstride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint32_t* a = A + (size_t)r * wstride;
    const uint32_t* b = B + (size_t)r * wstride;
    int32_t acc = 0;
    #pragma unroll 4
    for (int w = 0; w < words; ++w) acc += __popc(a[w] & b[w]);
    out[r] = acc;
}

// ---------------------------------------------------------------------------
// Split-K reduction: one giant dot product (K large, few outputs) spread across
// blocks; partial sums combined by a second kernel. Measures whether split-K
// beats a single block-per-row when output parallelism is scarce.
// ---------------------------------------------------------------------------
template <int BLOCK>
__global__ void k_splitk_partial(const int32_t* __restrict__ A, const int32_t* __restrict__ B,
                                 int32_t* __restrict__ partials, int words, int splits) {
    const int split = blockIdx.x % splits;
    const int words_per_split = (words + splits - 1) / splits;
    const int w0 = split * words_per_split;
    const int w1 = min(words, w0 + words_per_split);
    int32_t acc = 0;
    for (int w = w0 + threadIdx.x; w < w1; w += BLOCK)
        acc = __dp4a(A[w], B[w], acc);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ int32_t smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t tot = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) tot += smem[i];
        partials[split] = tot;
    }
}

// Cross-warp block reduction. blockDim here is 8 warps (256), so a bare warp
// shuffle would drop every warp but warp 0 — that was the original defect. We
// reduce each warp, stage lane-0 values in shared memory, then reduce those.
template <int BLOCK>
__global__ void k_splitk_combine(const int32_t* __restrict__ partials, int32_t* __restrict__ out,
                                 int splits) {
    int32_t acc = 0;
    for (int i = threadIdx.x; i < splits; i += BLOCK) acc += partials[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ int32_t smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t tot = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) tot += smem[i];
        out[0] = tot;
    }
}

// ---------------------------------------------------------------------------
// Single-array reduction strategies (INT8 -> INT32 sum). Same input, three
// communication mechanisms, so the cost of each is isolated from arithmetic.
// ---------------------------------------------------------------------------
// Warp-shuffle only: each thread grid-strides, reduce within warp, then a
// device-wide combine via one extra block is avoided by atomicAdd to out[0].
__global__ void k_sum_warp(const int8_t* __restrict__ x, int32_t* __restrict__ out, int n) {
    int32_t acc = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += (int32_t)x[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

// Hierarchical shared-memory block reduction, then one atomic per block.
template <int BLOCK>
__global__ void k_sum_smem(const int8_t* __restrict__ x, int32_t* __restrict__ out, int n) {
    int32_t acc = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += (int32_t)x[i];
    __shared__ int32_t smem[BLOCK / 32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t tot = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) tot += smem[i];
        atomicAdd(out, tot);
    }
}

// Split-K partial sums into a small buffer (no atomics), combined by one block.
template <int BLOCK>
__global__ void k_sum_splitk_partial(const int8_t* __restrict__ x, int32_t* __restrict__ partials,
                                     int n, int splits) {
    const int split = blockIdx.x % splits;
    const int per = (n + splits - 1) / splits;
    const int i0 = split * per, i1 = min(n, i0 + per);
    int32_t acc = 0;
    for (int i = i0 + threadIdx.x; i < i1; i += BLOCK) acc += (int32_t)x[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ int32_t smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t tot = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK / 32; ++i) tot += smem[i];
        partials[split] = tot;
    }
}

// ---------------------------------------------------------------------------
// Runner helpers.
// ---------------------------------------------------------------------------
struct CandResult { double median_us; bool correct; double max_err; };

// Allocate byte buffers padded to a 16-byte row stride and copy host data in.
void upload_padded(const std::vector<int8_t>& h, int rows, int K, int& stride,
                   int8_t** dA_bytes, int32_t** dA_words) {
    stride = round_up(K, 16);
    std::vector<int8_t> pad((size_t)rows * stride, 0);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < K; ++j) pad[(size_t)r*stride+j] = h[(size_t)r*K+j];
    LAB_CHECK(cudaMalloc(dA_bytes, (size_t)rows * stride));
    LAB_CHECK(cudaMemcpy(*dA_bytes, pad.data(), (size_t)rows * stride, cudaMemcpyHostToDevice));
    *dA_words = reinterpret_cast<int32_t*>(*dA_bytes);
}

// ---------------------------------------------------------------------------
// Cold-vs-warm L2 measurement. The default timed_iterations() runs the kernel
// back-to-back on the same buffers, so after the first pass the working set is
// L2-resident and effective bandwidth exceeds the DRAM ceiling — that is real
// L2-reuse performance, not a bug, but it must be labelled. A cold measurement
// flushes L2 before every timed launch by streaming a larger scratch buffer.
// ---------------------------------------------------------------------------
__global__ void k_flush(int32_t* __restrict__ scratch, size_t words) {
    // Grid-stride over a buffer far larger than L2; touches every line once.
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
        scratch[i] = (int32_t)(i & 0xff);
}

// Time `body` with L2 flushed before each iteration (cold), returning per-iter us.
template <class Body>
std::vector<double> timed_cold(Body&& body, int warmup, int iters,
                               int32_t* flush_buf, size_t flush_words) {
    cudaEvent_t s, e; LAB_CHECK(cudaEventCreate(&s)); LAB_CHECK(cudaEventCreate(&e));
    const int fb = 256; const int fg = (int)((flush_words + fb - 1) / fb);
    if (fg > 100000) {} // guard against absurd grids; flush_words chosen ~4x L2
    for (int i = 0; i < warmup; ++i) { k_flush<<<fg, fb>>>(flush_buf, flush_words); body(0); }
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        k_flush<<<fg, fb>>>(flush_buf, flush_words);   // evict L2
        LAB_CHECK(cudaEventRecord(s, 0));
        body(0);
        LAB_CHECK(cudaEventRecord(e, 0));
        LAB_CHECK(cudaEventSynchronize(e));
        float ms = 0.f; LAB_CHECK(cudaEventElapsedTime(&ms, s, e));
        us[i] = (double)ms * 1000.0;
    }
    LAB_CHECK(cudaEventDestroy(s)); LAB_CHECK(cudaEventDestroy(e));
    return us;
}

} // namespace

void run_int_dot(const LabArgs& args) {
    const DeviceInfo& dev = DeviceInfo::get();
    const int sm = dev.sm_count;

    // ---- Workload classes -------------------------------------------------
    struct Case { const char* cls; int K; int rows; };
    std::vector<Case> cases = {
        {"tiny_K8",   8,    8192},
        {"tiny_K32",  32,   8192},
        {"med_K128",  128,  8192},
        {"med_K256",  256,  8192},
        {"med_K1024", 1024, 4096},
        {"large_K4096",  4096, 2048},
        {"stream_K4096", 4096, 8192},   // 64 MB working set > 24 MiB L2
        {"streambig_K4096", 4096, 32768}, // 256 MB working set: L2 reuse negligible -> honest DRAM
        {"large_K16384", 16384, 1024},
        {"odd_K127",  127,  8192},   // irregular tail
        {"odd_K1023", 1023, 4096},   // irregular tail, medium
    };
    if (args.quick) cases = {{"med_K256", 256, 2048}, {"large_K4096", 4096, 1024}};

    std::printf("\n== INT8 dot-product candidate families (2K-op convention) ==\n");

    for (const auto& c : cases) {
        const int K = c.K, rows = c.rows;
        const int words = (K + 3) / 4;              // dp4a words covering real data
        const int wstride_bytes = round_up(K, 16);
        const int wstride = wstride_bytes / 4;      // words per padded row
        const double working_set = 2.0 * rows * K;  // both operands, bytes
        const char* regime = working_set <= (double)dev.l2_bytes ? "l2_resident" : "dram_stream";

        std::vector<int8_t> hA((size_t)rows*K), hB((size_t)rows*K);
        fill_i8(hA, 11ull + K); fill_i8(hB, 23ull + K);

        // Padded device buffers (both operands).
        int stride; int8_t *dA_b,*dB_b; int32_t *dA_w,*dB_w;
        upload_padded(hA, rows, K, stride, &dA_b, &dA_w);
        upload_padded(hB, rows, K, stride, &dB_b, &dB_w);
        int32_t* dOut = nullptr; LAB_CHECK(cudaMalloc(&dOut, (size_t)rows*sizeof(int32_t)));

        std::vector<int64_t> ref; ref_dot_pairwise(ref, hA, hB, rows, K, K);
        const double bytes = 2.0 * rows * K;   // both operand streams, INT8
        const double ops   = 2.0 * rows * K;   // 2K-op convention

        auto bench = [&](const char* impl, std::function<void(cudaStream_t)> body) {
            // correctness: run once, read back, exact compare
            body(0); LAB_CHECK(cudaDeviceSynchronize());
            std::vector<int32_t> got(rows);
            LAB_CHECK(cudaMemcpy(got.data(), dOut, (size_t)rows*sizeof(int32_t), cudaMemcpyDeviceToHost));
            ErrStats err = compare_int(got.data(), ref.data(), (size_t)rows);
            std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
            TimingResult t = summarize(us);
            emit_timing(args, "int_dot", t, err, bytes, ops,
                        ",\"impl\":\"" + std::string(impl) + "\"" +
                        ",\"cls\":\"" + std::string(c.cls) + "\"" +
                        ",\"K\":" + fmt_num(K) + ",\"rows\":" + fmt_num(rows) +
                        ",\"words\":" + fmt_num(words) +
                        ",\"working_set_bytes\":" + fmt_num(working_set) +
                        ",\"cache_regime\":\"" + std::string(regime) + "\"" +
                        ",\"in_dtype\":\"int8\",\"acc_dtype\":\"int32\",\"conv\":\"2K_ops\"");
            std::printf("  %-12s K=%-6d rows=%-6d %-10s %8.2f us  %7.1f GB/s  %s\n",
                        c.cls, K, rows, impl, t.median_us, bytes/(t.median_us*1e-6)/1e9,
                        err.pass?"ok":(std::string("FAIL err=")+std::to_string((long long)err.max_abs_err)).c_str());
        };

        // thread-per-row grid
        const int TPR = 256;
        const int tpr_blocks = (rows + TPR - 1) / TPR;

        bench("scalar", [&](cudaStream_t st){
            k_dot_scalar_tpr<<<tpr_blocks, TPR, 0, st>>>(dA_b, dB_b, dOut, rows, K, stride);
        });
        bench("unroll4", [&](cudaStream_t st){
            k_dot_unroll_tpr<<<tpr_blocks, TPR, 0, st>>>(dA_b, dB_b, dOut, rows, K, stride);
        });
        bench("dp4a", [&](cudaStream_t st){
            k_dot_dp4a_tpr<<<tpr_blocks, TPR, 0, st>>>(dA_w, dB_w, dOut, rows, words, wstride);
        });
        bench("dp4a4", [&](cudaStream_t st){
            k_dot_dp4a4_tpr<<<tpr_blocks, TPR, 0, st>>>(dA_w, dB_w, dOut, rows, words, wstride);
        });

        // block-per-row only worthwhile when K is large (thread-per-row starves).
        if (K >= 1024) {
            constexpr int BLOCK = 256;
            bench("dp4a_block", [&](cudaStream_t st){
                k_dot_dp4a_block<BLOCK><<<rows, BLOCK, 0, st>>>(dA_w, dB_w, dOut, rows, words, wstride);
            });
        }

        LAB_CHECK(cudaFree(dA_b)); LAB_CHECK(cudaFree(dB_b)); LAB_CHECK(cudaFree(dOut));
    }

    // ---- Split-K single giant reduction (few outputs, huge K) --------------
    std::printf("\n== split-K single-reduction (K=4096/16384, 1 output) ==\n");
    for (int K : {4096, 16384}) {
        const int words = (K+3)/4;
        const int wstride_bytes = round_up(K,16); const int wstride = wstride_bytes/4;
        std::vector<int8_t> hA(K), hB(K); fill_i8(hA, 101ull+K); fill_i8(hB, 211ull+K);
        std::vector<int8_t> padA(wstride_bytes,0), padB(wstride_bytes,0);
        for (int j=0;j<K;++j){ padA[j]=hA[j]; padB[j]=hB[j]; }
        int32_t *dA_w,*dB_w,*dPart,*dOut;
        LAB_CHECK(cudaMalloc(&dA_w, wstride_bytes)); LAB_CHECK(cudaMalloc(&dB_w, wstride_bytes));
        LAB_CHECK(cudaMemcpy(dA_w, padA.data(), wstride_bytes, cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMemcpy(dB_w, padB.data(), wstride_bytes, cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMalloc(&dPart, 256*sizeof(int32_t))); LAB_CHECK(cudaMalloc(&dOut, sizeof(int32_t)));

        int64_t acc=0; for(int j=0;j<K;++j) acc += (int64_t)hA[j]*(int64_t)hB[j];
        std::vector<int64_t> ref{acc};
        const double bytes = 2.0*K, ops = 2.0*K;

        auto bench_split = [&](int splits){
            constexpr int BLOCK = 256;
            auto body = [&](cudaStream_t st){
                k_splitk_partial<BLOCK><<<splits, BLOCK, 0, st>>>(dA_w, dB_w, dPart, words, splits);
                k_splitk_combine<BLOCK><<<1, BLOCK, 0, st>>>(dPart, dOut, splits);
            };
            body(0); LAB_CHECK(cudaDeviceSynchronize());
            int32_t got; LAB_CHECK(cudaMemcpy(&got, dOut, sizeof(int32_t), cudaMemcpyDeviceToHost));
            ErrStats err = compare_int(&got, ref.data(), 1);
            std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
            TimingResult t = summarize(us);
            emit_timing(args, "int_dot", t, err, bytes, ops,
                        ",\"impl\":\"splitk\"" + std::string(",\"cls\":\"single_K\"") +
                        ",\"K\":" + fmt_num(K) + ",\"splits\":" + fmt_num(splits) +
                        ",\"in_dtype\":\"int8\",\"acc_dtype\":\"int32\",\"conv\":\"2K_ops\"");
            std::printf("  split-K K=%-6d splits=%-4d %8.2f us  %7.1f GB/s  %s\n",
                        K, splits, t.median_us, bytes/(t.median_us*1e-6)/1e9,
                        err.pass?"ok":(std::string("FAIL err=")+std::to_string((long long)err.max_abs_err)).c_str());
        };
        bench_split(sm);      // one partial per SM
        bench_split(sm*4);    // oversubscribe to hide tail
        LAB_CHECK(cudaFree(dA_w)); LAB_CHECK(cudaFree(dB_w)); LAB_CHECK(cudaFree(dPart)); LAB_CHECK(cudaFree(dOut));
    }

    // ---- Reduction-strategy comparison: single big INT8->INT32 SUM ---------
    // Isolates communication/synchronization cost from the multiply. Same data,
    // four reduction mechanisms, exact integer oracle.
    std::printf("\n== reduction strategies: single-array sum (isolates comm cost) ==\n");
    for (int N : {16384, 65536, 1048576}) {
        std::vector<int8_t> h(N); fill_i8(h, 313ull + N);
        int32_t* dX = nullptr; LAB_CHECK(cudaMalloc(&dX, (size_t)N));
        LAB_CHECK(cudaMemcpy(dX, h.data(), (size_t)N, cudaMemcpyHostToDevice));
        int32_t* dOut = nullptr; LAB_CHECK(cudaMalloc(&dOut, sizeof(int32_t)));
        int32_t* dPart = nullptr; LAB_CHECK(cudaMalloc(&dPart, 256*sizeof(int32_t)));
        int64_t ref = 0; for (auto v : h) ref += v;
        std::vector<int64_t> refo{ref};
        const double bytes = (double)N, ops = (double)N;
        auto bench_red = [&](const char* impl, std::function<void(cudaStream_t)> body) {
            LAB_CHECK(cudaMemset(dOut, 0, sizeof(int32_t)));   // atomic reducers accumulate into out[0]
            body(0); LAB_CHECK(cudaDeviceSynchronize());
            int32_t got; LAB_CHECK(cudaMemcpy(&got, dOut, sizeof(int32_t), cudaMemcpyDeviceToHost));
            ErrStats err = compare_int(&got, refo.data(), 1);
            std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
            TimingResult t = summarize(us);
            emit_timing(args, "int_reduce", t, err, bytes, ops,
                        ",\"impl\":\"" + std::string(impl) + "\"" +
                        ",\"N\":" + fmt_num(N) +
                        ",\"working_set_bytes\":" + fmt_num(bytes) +
                        ",\"cache_regime\":\"" + (bytes<=(double)dev.l2_bytes?"l2_resident":"dram_stream") + "\"" +
                        ",\"in_dtype\":\"int8\",\"acc_dtype\":\"int32\"");
            std::printf("  %-14s N=%-8d %8.2f us  %7.1f GB/s  %s\n", impl, N, t.median_us,
                        bytes/(t.median_us*1e-6)/1e9, err.pass?"ok":(std::string("FAIL err=")+std::to_string((long long)err.max_abs_err)).c_str());
        };
        bench_red("warp_shuffle", [&](cudaStream_t st){ k_sum_warp<<<(N+255)/256,256,0,st>>>((const int8_t*)dX, dOut, N); });
        bench_red("smem_hier",    [&](cudaStream_t st){ k_sum_smem<256><<<(N+255)/256,256,0,st>>>((const int8_t*)dX, dOut, N); });
        // dedicated correct single-array split-K reducer:
        bench_red("split_k", [&](cudaStream_t st){
            k_sum_splitk_partial<256><<<sm*4,256,0,st>>>((const int8_t*)dX, dPart, N, sm*4);
            k_splitk_combine<256><<<1,256,0,st>>>(dPart, dOut, sm*4);
        });
        LAB_CHECK(cudaFree(dX)); LAB_CHECK(cudaFree(dOut)); LAB_CHECK(cudaFree(dPart));
    }

    // ---- Cold vs warm L2 sweep ---------------------------------------------
    // The mission explicitly asks to capture cases where L2 reuse is possible
    // and produces effective bandwidth ABOVE the DRAM ceiling, then separate it
    // from true DRAM streaming. We sweep a working set that straddles the 24 MiB
    // L2 and report each shape under BOTH regimes:
    //   warm = back-to-back launches on the same buffers (L2-resident after the
    //          first pass -> L2-assisted, can exceed 246 GB/s sustained read)
    //   cold = L2 flushed before every timed launch (true first-touch DRAM read)
    // Same kernel, same data, same correctness oracle; only the cache state differs.
    std::printf("\n== cold-vs-warm L2 sweep (dp4a_block, INT8 dot) ==\n");
    {
        // Flush scratch: ~4x L2 so a single pass evicts every L2 line.
        const size_t l2_bytes = dev.l2_bytes;
        const size_t flush_words = (4 * l2_bytes) / sizeof(int32_t);
        int32_t* dFlush = nullptr; LAB_CHECK(cudaMalloc(&dFlush, flush_words * sizeof(int32_t)));

        const int K = 4096;
        struct WS { int rows; };
        std::vector<WS> wss = {
            {256},     // 2 MB  -> fully L2-resident
            {1024},    // 8 MB  -> L2-resident
            {3072},    // 24 MB -> at the L2 boundary
            {8192},    // 64 MB -> exceeds L2
            {32768},   // 256 MB-> far exceeds L2
        };
        if (args.quick) wss = {{1024}, {8192}};

        for (auto& ws : wss) {
            const int rows = ws.rows;
            const int words = (K+3)/4; const int wstride_bytes = round_up(K,16); const int wstride = wstride_bytes/4;
            const double working_set = 2.0 * rows * K;   // both operands
            std::vector<int8_t> hA((size_t)rows*K), hB((size_t)rows*K);
            fill_i8(hA, 501ull+rows); fill_i8(hB, 601ull+rows);
            int stride; int8_t *dA_b,*dB_b; int32_t *dA_w,*dB_w;
            upload_padded(hA, rows, K, stride, &dA_b, &dA_w);
            upload_padded(hB, rows, K, stride, &dB_b, &dB_w);
            int32_t* dOut = nullptr; LAB_CHECK(cudaMalloc(&dOut, (size_t)rows*sizeof(int32_t)));
            std::vector<int64_t> ref; ref_dot_pairwise(ref, hA, hB, rows, K, K);
            const double bytes = 2.0 * rows * K, ops = 2.0 * rows * K;
            constexpr int BLOCK = 256;
            auto body = [&](cudaStream_t st){
                k_dot_dp4a_block<BLOCK><<<rows, BLOCK, 0, st>>>(dA_w, dB_w, dOut, rows, words, wstride);
            };
            // correctness once
            body(0); LAB_CHECK(cudaDeviceSynchronize());
            std::vector<int32_t> got(rows); LAB_CHECK(cudaMemcpy(got.data(), dOut, (size_t)rows*4, cudaMemcpyDeviceToHost));
            ErrStats err = compare_int(got.data(), ref.data(), (size_t)rows);

            auto emit = [&](const char* regime, const TimingResult& t) {
                emit_timing(args, "int_dot", t, err, bytes, ops,
                            ",\"impl\":\"dp4a_block\",\"cls\":\"l2_sweep\"" +
                            std::string(",\"K\":") + fmt_num(K) + ",\"rows\":" + fmt_num(rows) +
                            ",\"working_set_bytes\":" + fmt_num(working_set) +
                            ",\"cache_regime\":\"" + std::string(regime) + "\"" +
                            ",\"l2_bytes\":" + fmt_num((double)l2_bytes) +
                            ",\"in_dtype\":\"int8\",\"acc_dtype\":\"int32\",\"conv\":\"2K_ops\"");
                std::printf("  %-10s rows=%-6d ws=%7.1fMB %-4s %8.2f us  %7.1f GB/s  %s\n",
                            "l2_sweep", rows, working_set/1e6, regime, t.median_us,
                            bytes/(t.median_us*1e-6)/1e9, err.pass?"ok":(std::string("FAIL err=")+std::to_string((long long)err.max_abs_err)).c_str());
            };
            TimingResult warm = summarize(timed_iterations(body, args.warmup, args.iters));
            TimingResult cold = summarize(timed_cold(body, args.warmup, args.iters, dFlush, flush_words));
            emit("warm_l2", warm);
            emit("cold_dram", cold);
            LAB_CHECK(cudaFree(dA_b)); LAB_CHECK(cudaFree(dB_b)); LAB_CHECK(cudaFree(dOut));
        }
        LAB_CHECK(cudaFree(dFlush));
    }

    // ---- Binary inner product (separate contract) --------------------------
    std::printf("\n== binary dot: popcount(A & B), 0/1 inputs only ==\n");
    {
        const int Kbits = 256, rows = 8192;
        const int words = (Kbits+31)/32; const int wstride = round_up(words,4);
        std::vector<uint32_t> hA((size_t)rows*wstride,0), hB((size_t)rows*wstride,0);
        std::uint64_t s=7; for (int r=0;r<rows;++r) for (int w=0;w<words;++w){
            s=s*6364136223846793005ull+1442695040888963407ull; hA[(size_t)r*wstride+w]=(uint32_t)(s>>32);
            s=s*6364136223846793005ull+1442695040888963407ull; hB[(size_t)r*wstride+w]=(uint32_t)(s>>32);
        }
        uint32_t *dA,*dB; int32_t* dOut;
        LAB_CHECK(cudaMalloc(&dA,(size_t)rows*wstride*4)); LAB_CHECK(cudaMalloc(&dB,(size_t)rows*wstride*4));
        LAB_CHECK(cudaMalloc(&dOut,(size_t)rows*4));
        LAB_CHECK(cudaMemcpy(dA,hA.data(),(size_t)rows*wstride*4,cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMemcpy(dB,hB.data(),(size_t)rows*wstride*4,cudaMemcpyHostToDevice));
        std::vector<int64_t> ref; ref_binary_dot(ref,hA,hB,rows,words);
        const double bytes = 2.0*rows*words*4;   // bit-packed operand bytes actually read
        const double ops   = 2.0*rows*Kbits;     // conceptual 2K over the K bits
        auto body=[&](cudaStream_t st){ k_binary_popc_tpr<<<(rows+255)/256,256,0,st>>>(dA,dB,dOut,rows,words,wstride); };
        body(0); LAB_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> got(rows); LAB_CHECK(cudaMemcpy(got.data(),dOut,(size_t)rows*4,cudaMemcpyDeviceToHost));
        ErrStats err=compare_int(got.data(),ref.data(),(size_t)rows);
        std::vector<double> us=timed_iterations(body,args.warmup,args.iters); TimingResult t=summarize(us);
        emit_timing(args,"int_dot",t,err,bytes,ops,
                    ",\"impl\":\"binary_popc\",\"cls\":\"binary\",\"K_bits\":"+fmt_num(Kbits)+
                    ",\"rows\":"+fmt_num(rows)+",\"in_dtype\":\"binary\",\"acc_dtype\":\"int32\",\"conv\":\"2K_ops\"");
        std::printf("  binary_popc Kbits=%d rows=%d %8.2f us  %7.1f GB/s  %s\n",
                    Kbits,rows,t.median_us,bytes/(t.median_us*1e-6)/1e9,err.pass?"ok":"FAIL");
        LAB_CHECK(cudaFree(dA)); LAB_CHECK(cudaFree(dB)); LAB_CHECK(cudaFree(dOut));
    }
}
