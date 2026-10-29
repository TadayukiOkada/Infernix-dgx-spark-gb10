// gemv.cu — Layer 3: matrix-vector (decode) and matrix-matrix (prefill) products.
//
// Decode inference is dominated by GEMV: y = W x where W is a layer weight and x
// a single token vector. The op streams W once from DRAM, so its ceiling is
// memory bandwidth, not FLOPs. We implement a straightforward custom GEMV (one
// block per output row, uint4 vectorised inner product, no atomics) and judge it
// against cuBLAS GemmEx at m=1 — expecting cuBLAS to be close to the roofline.
// We then sweep the batch size M through {1,4,128,512} with cuBLAS bf16 GEMM to
// trace the crossover from memory-bound (M~1) to compute/Tensor-Core-bound
// (large M), which is exactly why paged batched decode exists.
#include "gb10_lab/common.h"

namespace {

#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); \
    if(s_!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS error %d @ %s:%d\n",(int)s_,__FILE__,__LINE__); std::exit(1);} } while(0)

// Custom GEMV: y[r] = sum_j W[r,j]*x[j]. One block per output row r; threads
// stride the row in uint4 chunks (coalesced). x is re-read per row but stays
// cached. Accumulate in float, block-reduce via warp shuffles + shared mem.
template <int BLOCK>
__global__ void k_gemv_row(const uint4* __restrict__ W, const uint4* __restrict__ x,
                          float* __restrict__ y, int out_dim, int nv) {
    const int r = blockIdx.x;
    if (r >= out_dim) return;
    const uint4* Wr = W + static_cast<size_t>(r) * nv;
    float local = 0.f;
    for (int g = threadIdx.x; g < nv; g += BLOCK) {
        BF16x8 w = u4_to_bf16(Wr[g]);
        BF16x8 xv = u4_to_bf16(x[g]);
        #pragma unroll
        for (int e = 0; e < 8; ++e) local += bf16_f(w.x[e]) * bf16_f(xv.x[e]);
    }
    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, off);
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

// Host reference: fp32 mat-vec from bf16 inputs.
void ref_gemv(std::vector<float>& y, const std::vector<__nv_bfloat16>& W,
              const std::vector<__nv_bfloat16>& x, int out_dim, int in_dim) {
    y.assign(out_dim, 0.f);
    for (int r = 0; r < out_dim; ++r) {
        double acc = 0.0;
        for (int j = 0; j < in_dim; ++j)
            acc += (double)bf16_f(W[(size_t)r*in_dim+j]) * (double)bf16_f(x[j]);
        y[r] = (float)acc;
    }
}

struct WeightCase { const char* label; int out_dim; int in_dim; };

} // namespace

void run_gemv(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    std::vector<WeightCase> cases = {
        {"lm_head",  p.vocab,   p.hidden},   // [151936, hidden]
        {"mlp_up",   p.inter,   p.hidden},   // [inter, hidden]
        {"attn_o",   p.hidden,  p.hidden},   // [hidden, hidden]
    };
    if (args.quick) cases = {{ "mlp_up", p.inter, p.hidden }};

    constexpr int BLOCK = 256;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float one = 1.f, zero = 0.f;

    for (const auto& c : cases) {
        const int O = c.out_dim, K = c.in_dim;
        const int nv = K / 8;   // K divisible by 8 for these shapes
        std::vector<__nv_bfloat16> hW((size_t)O*K), hx(K);
        fill_bf16(hW, 11ull, 0.5f);
        fill_bf16(hx, 12ull, 0.5f);

        uint4* dW = nullptr; __nv_bfloat16* dx = nullptr; float* dy = nullptr;
        LAB_CHECK(cudaMalloc(&dW, (size_t)O*K*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dx, (size_t)K*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dy, (size_t)O*sizeof(float)));
        LAB_CHECK(cudaMemcpy(dW, hW.data(), (size_t)O*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMemcpy(dx, hx.data(), (size_t)K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        // ---- custom GEMV correctness vs host fp32 reference -----------------
        k_gemv_row<BLOCK><<<O, BLOCK>>>(dW, (uint4*)dx, dy, O, nv);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(O), ref;
        LAB_CHECK(cudaMemcpy(got.data(), dy, (size_t)O*sizeof(float), cudaMemcpyDeviceToHost));
        ref_gemv(ref, hW, hx, O, K);
        ErrStats err = compare_f32(got.data(), ref.data(), (size_t)O, /*abs*/ 2e-2, /*rel*/ 2e-2);

        auto body_custom = [&](cudaStream_t st){
            k_gemv_row<BLOCK><<<O, BLOCK, 0, st>>>(dW, (uint4*)dx, dy, O, nv);
        };
        std::vector<double> us_c = timed_iterations(body_custom, args.warmup, args.iters);
        TimingResult t_c = summarize(us_c);
        double bytes_w = (double)O*K*2 + (double)K*2;   // W stream + small x
        emit_timing(args, "gemv_custom", t_c, err, bytes_w, 2.0*O*K,
                    ",\"label\":\"" + std::string(c.label) + "\"" +
                    ",\"out\":" + fmt_num(O) + ",\"k\":" + fmt_num(K) +
                    ",\"block\":" + fmt_num(BLOCK) + ",\"dtype\":\"bf16\"");
        std::printf("  gemv %-8s out=%-7d k=%-6d custom %8.2f us %6.1f GB/s correct=%s\n",
                    c.label, O, K, t_c.median_us, bytes_w/(t_c.median_us*1e-6)/1e9,
                    err.pass?"yes":"NO");

        // ---- cuBLAS GEMV at m=1 ---------------------------------------------
        // Row-major y[O] = W[O,K] * x[K]. Verified-on-device idiom: interpret the
        // weight buffer (row-major [O,K]) as a col-major [K,O] matrix and transpose
        // it, keep x as-is. Gives C_cm[O,1] = W_cm^T[O,K] * x_cm[K,1].
        auto body_blas = [&](cudaStream_t st){
            CUBLAS_CHECK(cublasSetStream(handle, st));
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      O, 1, K, &one,
                                      dW, CUDA_R_16BF, K,
                                      dx, CUDA_R_16BF, K, &zero,
                                      dy, CUDA_R_32F, O,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        std::vector<double> us_b = timed_iterations(body_blas, args.warmup, args.iters);
        TimingResult t_b = summarize(us_b);
        emit_timing(args, "gemv_cublas", t_b, err, bytes_w, 2.0*O*K,
                    ",\"label\":\"" + std::string(c.label) + "\"" +
                    ",\"out\":" + fmt_num(O) + ",\"k\":" + fmt_num(K) + ",\"m\":1" +
                    ",\"dtype\":\"bf16\"");
        std::printf("  gemv %-8s out=%-7d k=%-6d cublas %8.2f us %6.1f GB/s  (%.2fx vs custom)\n",
                    c.label, O, K, t_b.median_us, bytes_w/(t_b.median_us*1e-6)/1e9,
                    t_c.median_us / std::max(t_b.median_us, 1e-9));

        LAB_CHECK(cudaFree(dW)); LAB_CHECK(cudaFree(dx)); LAB_CHECK(cudaFree(dy));
    }

    // ---- cuBLAS bf16 GEMM batch-size sweep (prefill throughput curve) -------
    const int N = p.inter, K = p.hidden;     // MLP up-proj weight shape
    std::vector<int> M_cases = {1, 4, 128, 512};
    if (args.quick) M_cases = {1, 128};
    const int Mmax = *std::max_element(M_cases.begin(), M_cases.end());
    __nv_bfloat16* dA = nullptr; __nv_bfloat16* dB = nullptr; float* dC = nullptr;
    LAB_CHECK(cudaMalloc(&dA, (size_t)Mmax*K*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dB, (size_t)N*K*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dC, (size_t)Mmax*N*sizeof(float)));
    // B weight matrix [N,K].
    {
        std::vector<__nv_bfloat16> bbuf((size_t)N*K);
        fill_bf16(bbuf, 13ull, 0.5f);
        LAB_CHECK(cudaMemcpy(dB, bbuf.data(), (size_t)N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }
    for (int M : M_cases) {
        std::vector<__nv_bfloat16> abuf((size_t)M*K);
        fill_bf16(abuf, 14ull+M, 0.5f);
        LAB_CHECK(cudaMemcpy(dA, abuf.data(), (size_t)M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        auto body = [&](cudaStream_t st) {
            // Row-major C[M,N] = X_rm[M,K] * W_rm[N,K]^T. Verified idiom: transpose
            // the weight buffer, keep X as-is -> C_cm[N,M] = W_cm^T[N,K] * X_cm[K,M].
            CUBLAS_CHECK(cublasSetStream(handle, st));
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N, M, K, &one,
                                      dB, CUDA_R_16BF, K,
                                      dA, CUDA_R_16BF, K, &zero,
                                      dC, CUDA_R_32F, N,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double flops = 2.0*M*N*K;
        double bytes = ((double)N*K + (double)M*K + (double)M*N)*2;
        emit_timing(args, "gemm_cublas", t, ErrStats{0,0,true}, bytes, flops,
                    ",\"m\":" + fmt_num(M) + ",\"n\":" + fmt_num(N) + ",\"k\":" + fmt_num(K) +
                    ",\"dtype\":\"bf16\"");
        std::printf("  gemm     m=%-4d n=%-6d k=%-6d cublas %8.2f us %7.2f TFLOPs\n",
                    M, N, K, t.median_us, flops/(t.median_us*1e-6)/1e12);
    }
    LAB_CHECK(cudaFree(dA)); LAB_CHECK(cudaFree(dB)); LAB_CHECK(cudaFree(dC));
    CUBLAS_CHECK(cublasDestroy(handle));
}
