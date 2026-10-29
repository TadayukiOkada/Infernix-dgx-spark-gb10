// rmsnorm.cu — Layer 3: RMSNorm over model hidden dimensions.
//
// RMSNorm(x)_i = x_i / sqrt(mean(x^2) + eps) * gamma_i. One block per row keeps
// the reduction intra-block (no atomics, no second launch). Pass 1 streams the
// row through L1 while accumulating the sum of squares in a register; a warp-
// shuffle + one-shared-float block reduction yields the normaliser; pass 2
// re-reads the same row (now an L1 hit) and writes scaled output. uint4 loads
// give 8 bf16 elements per transaction. This isolates how expensive
// normalization is on GB10 at the exact H values the production models use,
// including tails that are not multiples of 8.
#include "gb10_lab/common.h"

namespace {

constexpr float kEps = 1e-6f;

template <int BLOCK>
__global__ void k_rmsnorm(const uint4* __restrict__ x_in,   // [rows, H_vec]
                          const float* __restrict__ gamma,  // [H]
                          uint4* __restrict__ y_out,         // [rows, H_vec]
                          int rows, int H, int H_vec) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const uint4* xr = x_in + static_cast<size_t>(row) * H_vec;
    uint4* yr      = y_out + static_cast<size_t>(row) * H_vec;

    extern __shared__ float sreduce[];     // BLOCK floats: one partial per thread
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;

    // ---- Pass 1: accumulate sum-of-squares --------------------------------
    double ssq = 0.0;                       // double for accumulation stability
    for (int c = tid; c < H_vec; c += nthr) {
        BF16x8 b = u4_to_bf16(xr[c]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Padding tail is zero-filled on the host and contributes nothing.
            float f = bf16_f(b.x[j]);
            ssq += static_cast<double>(f) * f;
        }
    }

    // ---- Block reduction of ssq -------------------------------------------
    // Warp-level shuffle first, then one value per warp reduced by warp 0.
    float v = static_cast<float>(ssq);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) sreduce[warp] = v;
    __syncthreads();
    // Fold the per-warp partials with a single thread. A cross-warp shuffle
    // here would need a mask covering only the active lanes; using 0xffffffff
    // while only `nw` lanes run is undefined on Volta+ and can hang the kernel.
    if (tid == 0) {
        float tot = sreduce[0];
        const int nw = nthr / 32;           // <= 8 for BLOCK <= 256
        for (int w = 1; w < nw; ++w) tot += sreduce[w];
        sreduce[0] = tot;
    }
    __syncthreads();
    const float total_ssq = sreduce[0];
    const float inv_rms   = rsqrtf(total_ssq / (float)H + kEps);

    // ---- Pass 2: normalise, apply gamma, write ----------------------------
    for (int c = tid; c < H_vec; c += nthr) {
        BF16x8 ob = u4_to_bf16(xr[c]);       // L1 hit from pass 1
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int elem = c * 8 + j;
            float xv = bf16_f(ob.x[j]);
            // Keep the padding tail exactly zero to match the reference.
            ob.x[j] = (elem < H) ? f_bf16(xv * inv_rms * gamma[elem]) : f_bf16(0.f);
        }
        yr[c] = bf16_to_u4(ob);
    }
}

// Reference implementation in fp32 for correctness checking. Operates on a
// contiguous [rows, H] view (padding stripped).
void ref_rmsnorm(const std::vector<__nv_bfloat16>& x, const std::vector<float>& gamma,
                 std::vector<float>& out, int rows, int H, float eps) {
    out.assign((size_t)rows * H, 0.f);
    for (int r = 0; r < rows; ++r) {
        double ssq = 0.0;
        for (int i = 0; i < H; ++i) {
            float xv = bf16_f(x[(size_t)r * H + i]);
            ssq += (double)xv * xv;
        }
        float inv = rsqrtf((float)(ssq / H) + eps);
        for (int i = 0; i < H; ++i) {
            float xv = bf16_f(x[(size_t)r * H + i]);
            out[(size_t)r * H + i] = xv * inv * gamma[i];
        }
    }
}

} // namespace

void run_rmsnorm(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    const int H = p.hidden;
    const int H_pad = ((H + 7) / 8) * 8;      // pad row to a multiple of 8
    const int H_vec = H_pad / 8;              // uint4 slots per row

    // Row counts: decode-scale (few tokens) through prefill scale.
    std::vector<int> rows_cases = {1, 8, 128, 1024};
    if (args.quick) rows_cases = {1, 128};

    constexpr int BLOCK = 256;
    size_t shmem = BLOCK * sizeof(float);     // one partial sum per thread
    LAB_CHECK(cudaFuncSetAttribute(k_rmsnorm<BLOCK>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shmem));

    // gamma ~ small deterministic values; same seed every run.
    std::vector<float> h_gamma(H);
    fill_f32(h_gamma, 1234ull, 1.0f);
    float* d_gamma = nullptr;
    LAB_CHECK(cudaMalloc(&d_gamma, H * sizeof(float)));
    LAB_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), H * sizeof(float), cudaMemcpyHostToDevice));

    for (int rows : rows_cases) {
        std::vector<__nv_bfloat16> h_x((size_t)rows * H_pad);
        fill_bf16(h_x, 99ull, 1.0f);
        for (int r = 0; r < rows; ++r)               // zero the padding tail
            for (int i = H; i < H_pad; ++i) h_x[(size_t)r * H_pad + i] = f_bf16(0.f);

        uint4* d_x = nullptr; uint4* d_y = nullptr;
        LAB_CHECK(cudaMalloc(&d_x, (size_t)rows * H_pad * sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&d_y, (size_t)rows * H_pad * sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)rows * H_pad * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice));

        auto body = [&](cudaStream_t st) {
            k_rmsnorm<BLOCK><<<rows, BLOCK, shmem, st>>>(d_x, d_gamma, d_y, rows, H, H_vec);
        };
        // Validate once before timing.
        body(nullptr);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_y((size_t)rows * H_pad);
        LAB_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)rows * H_pad * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
        std::vector<float> got((size_t)rows * H), ref((size_t)rows * H);
        for (size_t i = 0; i < (size_t)rows * H; ++i) got[i] = bf16_f(h_y[i]);
        std::vector<__nv_bfloat16> x_contig((size_t)rows * H);   // strip padding
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < H; ++i) x_contig[(size_t)r * H + i] = h_x[(size_t)r * H_pad + i];
        ref_rmsnorm(x_contig, h_gamma, ref, rows, H, kEps);
        ErrStats err = compare_f32(got.data(), ref.data(), (size_t)rows * H,
                                   /*abs_tol*/ 3e-2, /*rel_tol*/ 3e-2);

        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double bytes_moved = (double)rows * H * 2;     // in + out, bf16
        emit_timing(args, "rmsnorm", t, err, bytes_moved, 0.0,
                    ",\"rows\":" + fmt_num(rows) + ",\"hidden\":" + fmt_num(H) +
                    ",\"block\":" + fmt_num(BLOCK) + ",\"dtype\":\"bf16\"");
        std::printf("  rmsnorm rows=%-5d H=%-5d %8.2f us   %6.1f GB/s   correct=%s\n",
                    rows, H, t.median_us, bytes_moved / (t.median_us * 1e-6) / 1e9,
                    err.pass ? "yes" : "NO");
        LAB_CHECK(cudaFree(d_x)); LAB_CHECK(cudaFree(d_y));
    }
    LAB_CHECK(cudaFree(d_gamma));
}
