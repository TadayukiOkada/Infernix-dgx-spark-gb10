// compute.cu — Layer 2B/C: compute throughput ceilings + launch overhead.
//
// Compute ceilings use cuBLAS as the reference implementation so the numbers
// reflect what the hardware can actually sustain on the relevant precision
// paths (FP32, TF32, BF16). Shapes span decode-scale tiny M through prefill
// scale so the result is not just "peak square GEMM". An FP32 FMA microkernel
// gives the conventional CUDA-core arithmetic rate separately from Tensor Core
// throughput. Launch overhead measures empty-kernel latency, host-side API cost
// and CUDA Graph replay, which dominates small inference operations.
#include "gb10_lab/common.h"

#include <cublas_v2.h>
#include <chrono>

namespace {

#define CUBLAS_CHECK(x)                                                     \
    do {                                                                    \
        cublasStatus_t s_ = (x);                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,      \
                         __LINE__, (int)s_);                               \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---- FP32 FMA ceiling (CUDA cores, no Tensor Cores) -----------------------
// Eight INDEPENDENT accumulators so the FMA units saturate via instruction-level
// parallelism. A single `acc = fmaf(acc,...)` chain is latency-bound and would
// understate throughput; independent chains let the compiler hide FMA latency and
// measure true CUDA-core FMA throughput. Reads stay L1/L2-resident (small buffer),
// so this is a compute-throughput probe, not a bandwidth probe.
__global__ void k_fma(float* __restrict__ x, float a, size_t n_iters) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float c0=x[i], c1=x[i+1], c2=x[i+2], c3=x[i+3],
          c4=x[i+4], c5=x[i+5], c6=x[i+6], c7=x[i+7];
    for (size_t it = 0; it < n_iters; ++it) {
        c0 = fmaf(c0, a, c1); c1 = fmaf(c1, a, c2);
        c2 = fmaf(c2, a, c3); c3 = fmaf(c3, a, c4);
        c4 = fmaf(c4, a, c5); c5 = fmaf(c5, a, c6);
        c6 = fmaf(c6, a, c7); c7 = fmaf(c7, a, c0);
    }
    x[i] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

// ---- GEMM via cuBLAS ------------------------------------------------------
// y = alpha * op(A) op(B) + beta * C with A [K x M] col-major == X^T, etc. We
// build row-major matrices and call column-major cuBLAS with swapped operands
// so the math matches C[M,N] = A[M,K] * B[K,N].
struct GemmResult { double tflops; bool ok; };

GemmResult run_gemm_bf16(int m, int n, int k, int warmup, int iters,
                          cublasComputeType_t ct) {
    __nv_bfloat16* A = nullptr; __nv_bfloat16* B = nullptr; __nv_bfloat16* C = nullptr;
    std::size_t na = (std::size_t)m * k, nb = (std::size_t)k * n, nc = (std::size_t)m * n;
    LAB_CHECK(cudaMalloc(&A, na * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&B, nb * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&C, nc * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMemset(A, 0x11, na * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMemset(B, 0x22, nb * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMemset(C, 0, nc * sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaDeviceSynchronize());

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    // For bf16 we use the 16-bit input / 32-bit accumulate path when available.
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    const float alpha = 1.f, beta = 0.f;

    // Column-major trick: compute C^T = B^T A^T by calling gemm with (n,k,m).
    auto body = [&](cudaStream_t /*stream*/) {
        // cuBLAS operates on the current stream; we call from the default stream.
        // Column-major view: C_cm[n,m] = B_cm[n,k] * A_cm[k,m]; lda must be >= mm=n.
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                     B, CUDA_R_16BF, n, A, CUDA_R_16BF, k, &beta,
                     C, CUDA_R_16BF, n, ct, CUBLAS_GEMM_DEFAULT);
    };

    std::vector<double> us;
    try {
        us = timed_iterations(body, warmup, iters);
    } catch (...) {
        cublasDestroy(h);
        cudaFree(A); cudaFree(B); cudaFree(C);
        return {0.0, false};
    }
    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);

    const double med = percentile(us, 50.0);
    const double flops = 2.0 * m * n * k;
    return {flops / (med * 1e-6) / 1e12, true};
}

GemmResult run_gemm_fp32_tf32(int m, int n, int k, int warmup, int iters,
                              bool allow_tf32) {
    float* A = nullptr; float* B = nullptr; float* C = nullptr;
    std::size_t na = (std::size_t)m * k, nb = (std::size_t)k * n, nc = (std::size_t)m * n;
    LAB_CHECK(cudaMalloc(&A, na * sizeof(float)));
    LAB_CHECK(cudaMalloc(&B, nb * sizeof(float)));
    LAB_CHECK(cudaMalloc(&C, nc * sizeof(float)));
    LAB_CHECK(cudaMemset(A, 0x11, na * sizeof(float)));
    LAB_CHECK(cudaMemset(B, 0x22, nb * sizeof(float)));
    LAB_CHECK(cudaMemset(C, 0, nc * sizeof(float)));
    LAB_CHECK(cudaDeviceSynchronize());

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    if (allow_tf32) {
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    } else {
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    }
    const float alpha = 1.f, beta = 0.f;

    auto body = [&](cudaStream_t /*stream*/) {
        // lda >= mm=n (see run_gemm_bf16); ldb=k, ldc=n.
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    B, n, A, k, &beta, C, n);
    };

    std::vector<double> us;
    try {
        us = timed_iterations(body, warmup, iters);
    } catch (...) {
        cublasDestroy(h);
        cudaFree(A); cudaFree(B); cudaFree(C);
        return {0.0, false};
    }
    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);

    const double med = percentile(us, 50.0);
    const double flops = 2.0 * m * n * k;
    return {flops / (med * 1e-6) / 1e12, true};
}

// ---- Launch overhead ------------------------------------------------------
__global__ void k_empty() {}

struct LaunchStats {
    double empty_kernel_us   = 0.0;   // GPU exec time of an empty kernel
    double host_api_us       = 0.0;   // host-side cost of one launch call
    double graph_replay_us   = 0.0;   // one replayed empty-kernel graph node
    double eager_chain_us    = 0.0;   // N eager launches amortised per launch
    int chain_n               = 0;
};

LaunchStats measure_launch(int warmup, int iters, int chain_n) {
    LaunchStats s;
    const DeviceInfo& d = DeviceInfo::get();
    const int blocks = d.sm_count, threads = 256;

    // 1) Empty-kernel device execution time.
    {
        auto body = [&](cudaStream_t st) { k_empty<<<blocks, threads, 0, st>>>(); };
        std::vector<double> us = timed_iterations(body, warmup, iters);
        s.empty_kernel_us = percentile(us, 50.0);
    }
    // 2) Host-side API cost: enqueue without synchronising, wall-clock the call.
    {
        cudaStream_t st;
        LAB_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        // Warm the submission path.
        for (int i = 0; i < warmup; ++i) k_empty<<<blocks, threads, 0, st>>>();
        LAB_CHECK(cudaStreamSynchronize(st));
        using clock = std::chrono::high_resolution_clock;
        const auto t0 = clock::now();
        for (int i = 0; i < iters; ++i) k_empty<<<blocks, threads, 0, st>>>();
        const auto t1 = clock::now();
        LAB_CHECK(cudaStreamSynchronize(st));
        s.host_api_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                        / 1000.0 / iters;
        LAB_CHECK(cudaStreamDestroy(st));
    }
    // 3) CUDA Graph replay of a single empty kernel.
    {
        TimingResult tr = graph_replay_timing(
            [&](cudaStream_t st) { k_empty<<<blocks, threads, 0, st>>>(); },
            warmup, iters);
        s.graph_replay_us = tr.median_us;
    }
    // 4) Eager chain of many launches — measures per-launch gap when the queue
    //    is kept full (the regime that matters for decode with many small ops).
    s.chain_n = chain_n;
    {
        auto body = [&](cudaStream_t st) {
            for (int i = 0; i < chain_n; ++i) k_empty<<<blocks, threads, 0, st>>>();
        };
        std::vector<double> us = timed_iterations(body, warmup, std::max(3, iters / 4));
        s.eager_chain_us = percentile(us, 50.0) / chain_n;
    }
    return s;
}

} // namespace

void run_compute(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    const int sm = DeviceInfo::get().sm_count;

    // ---- FP32 FMA ceiling (CUDA cores) -------------------------------------
    {
        const size_t n_elem = (size_t)sm * 256 * 64;     // enough to be bandwidth-limited
        float* x = nullptr;
        LAB_CHECK(cudaMalloc(&x, n_elem * sizeof(float)));
        LAB_CHECK(cudaMemset(x, 0x11, n_elem * sizeof(float)));
        LAB_CHECK(cudaDeviceSynchronize());
        const int blocks = sm * 8, threads = 256;
        const size_t n_iters = args.quick ? 20 : 200;
        auto body = [&](cudaStream_t st) {
            k_fma<<<blocks, threads, 0, st>>>(x, 1.000001f, n_iters);
        };
        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        const double med = percentile(us, 50.0);
        const double flops = (double)n_elem * n_iters * 8 /*unroll*/ * 2 /*flop per fma*/;
        const double gflops = flops / (med * 1e-6) / 1e9;
        JsonLine l;
        l.kv("op", "compute").kv("kind", "fp32_fma_cuda_core")
         .append_fragment(",\"n_elements\":" + fmt_num((double)n_elem))
         .append_fragment(",\"iters_per_thread\":" + fmt_num((double)(n_iters * 8)))
         .append_fragment(",\"median_us\":" + fmt_num(med))
         .append_fragment(",\"gflops\":" + fmt_num(gflops));
        emit_record(args, l);
        std::printf("  fp32 FMA (CUDA cores)   %.1f GFLOP/s\n", gflops);
        LAB_CHECK(cudaFree(x));
    }

    // ---- GEMM ceilings across precisions and shapes ------------------------
    struct ShapeCase { const char* label; int m, n, k; };
    // Decode-scale (tiny M), mid, and prefill-scale (large M) for the preset's
    // hidden/intermediate dimensions, plus a large square reference.
    std::vector<ShapeCase> cases = {
        {"decode_m1",   1,   p.hidden, p.inter},
        {"decode_m4",   4,   p.hidden, p.inter},
        {"prefill_m128",128, p.hidden, p.inter},
        {"prefill_m512",512, p.hidden, p.inter},
        {"square_4096", 4096, 4096, 4096},
    };
    if (args.quick) cases = {{ "decode_m1", 1, p.hidden, p.inter },
                             {"prefill_m128", 128, p.hidden, p.inter }};

    for (const auto& c : cases) {
        // BF16 (Tensor Core, 32-bit accumulate)
        GemmResult bf16 = run_gemm_bf16(c.m, c.n, c.k, args.warmup, args.iters,
                                        CUBLAS_COMPUTE_32F);
        // FP32 strict (no TF32)
        GemmResult fp32 = run_gemm_fp32_tf32(c.m, c.n, c.k, args.warmup, args.iters, false);
        // FP32 with TF32 allowed
        GemmResult tf32 = run_gemm_fp32_tf32(c.m, c.n, c.k, args.warmup, args.iters, true);

        JsonLine l;
        l.kv("op", "gemm_ceiling").kv("case", c.label)
         .append_fragment(std::string(",\"shape\":[") + std::to_string(c.m) + "," +
                          std::to_string(c.n) + "," + std::to_string(c.k) + "]")
         .append_fragment(",\"bf16_tflops\":" + fmt_num(bf16.tflops))
         .append_fragment(",\"fp32_tflops\":" + fmt_num(fp32.tflops))
         .append_fragment(",\"tf32_tflops\":" + fmt_num(tf32.tflops));
        emit_record(args, l);
        std::printf("  gemm %-12s M=%-5d N=%-6d K=%-6d | bf16 %7.2f  fp32 %7.2f  tf32 %7.2f TFLOP/s\n",
                    c.label, c.m, c.n, c.k, bf16.tflops, fp32.tflops, tf32.tflops);
    }

    // ---- Launch overhead ---------------------------------------------------
    {
        const int chain_n = args.quick ? 32 : 128;
        LaunchStats s = measure_launch(args.warmup, args.iters, chain_n);
        JsonLine l;
        l.kv("op", "launch_overhead")
         .append_fragment(",\"empty_kernel_us\":" + fmt_num(s.empty_kernel_us))
         .append_fragment(",\"host_api_us\":" + fmt_num(s.host_api_us))
         .append_fragment(",\"graph_replay_us\":" + fmt_num(s.graph_replay_us))
         .append_fragment(",\"eager_chain_us\":" + fmt_num(s.eager_chain_us))
         .append_fragment(",\"chain_n\":" + fmt_num((double)s.chain_n));
        emit_record(args, l);
        std::printf("\nLaunch overhead:\n");
        std::printf("  empty-kernel GPU exec   %8.3f us\n", s.empty_kernel_us);
        std::printf("  host launch API cost    %8.3f us/call\n", s.host_api_us);
        std::printf("  graph replay (1 node)   %8.3f us\n", s.graph_replay_us);
        std::printf("  eager chain amortised   %8.3f us/launch (N=%d)\n", s.eager_chain_us, s.chain_n);
    }
}
