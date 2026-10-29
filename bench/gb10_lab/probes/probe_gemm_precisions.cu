// probe_gemm_precisions.cu — GB10 feature probe: Tensor Core GEMM throughput per dtype.
// Validates and measures the supported matrix-multiply precision paths with real
// TOPS, using cuBLAS / cuBLASLt as the vendor baseline (the mission requires
// optimized-library baselines, not only custom kernels). Reports BF16, FP16,
// TF32 and FP8(E4M3) dense GEMM throughput at a large square shape.
// INT8 GEMM is probed separately (probe_cublaslt_int8 / probe_imma) and is NOT
// exposed by cuBLASLt here.
#include "probe_common.h"
#include <cstdio>
#include <cstdint>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define LTCHECK(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    std::printf("probe_gemm_precisions CUBLAS_FAIL %d @%d\n",(int)s_,__LINE__);} } while(0)

static const int M=2048,N=2048,K=2048;

static double gemm_ex(cublasHandle_t h, cudaDataType_t dt, cublasComputeType_t ct,
                      void* A, void* B, void* C, const char* name) {
    float alpha=1.f, beta=0.f;
    auto body=[&]{
        LTCHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, dt, N, A, dt, K, &beta, C, CUDA_R_32F, N, ct, CUBLAS_GEMM_DEFAULT));
    };
    // verify once
    body(); cudaDeviceSynchronize();
    double ops=2.0*M*N*K;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for(int i=0;i<10;++i) body(); cudaDeviceSynchronize();
    std::vector<double> us(30);
    for(int i=0;i<30;++i){ cudaEventRecord(s); body(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e); us[i]=(double)ms*1000.0; }
    cudaEventDestroy(s); cudaEventDestroy(e);
    double med=probe_percentile(us,50.0);
    double tops=ops/(med*1e-6)/1e12;
    std::printf("%s median_us=%.4f TOPS=%.3f\n", name, med, tops);
    return tops;
}

int main(){
    cublasHandle_t h; LTCHECK(cublasCreate(&h));
    size_t bytes=(size_t)M*K*4;  // room for fp32; smaller dtypes reuse same buffer
    void *A,*B,*C; cudaMalloc(&A,bytes); cudaMalloc(&B,bytes); cudaMalloc(&C,(size_t)M*N*4);
    cudaMemset(A,0x11,bytes); cudaMemset(B,0x22,bytes); cudaMemset(C,0,(size_t)M*N*4);
    std::printf("probe_gemm_precisions OK\n");
    gemm_ex(h, CUDA_R_16BF, CUBLAS_COMPUTE_32F, A,B,C, "gemm_bf16");
    gemm_ex(h, CUDA_R_16F,  CUBLAS_COMPUTE_32F, A,B,C, "gemm_fp16");
    gemm_ex(h, CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_TF32, A,B,C, "gemm_tf32");
    gemm_ex(h, CUDA_R_32F,  CUBLAS_COMPUTE_32F, A,B,C, "gemm_fp32_strict");
    // FP8 E4M3 via cuBLASLt (needs 1-byte operands).
    {
        __nv_fp8_e4m3 *fa=nullptr,*fb=nullptr; int32_t* fc=nullptr;
        cudaMalloc(&fa,(size_t)M*K); cudaMalloc(&fb,(size_t)K*N); cudaMalloc(&fc,(size_t)M*N*4);
        cudaMemset(fa,0x38,(size_t)M*K); cudaMemset(fb,0x38,(size_t)K*N); cudaMemset(fc,0,(size_t)M*N*4);
        cublasLtHandle_t lt; LTCHECK(cublasLtCreate(&lt));
        cublasLtMatmulDesc_t op; LTCHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tn=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof tn);
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof tn);
        cublasLtMatrixLayout_t la,lb,lc;
        LTCHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, M, K, M));
        LTCHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, N, K));
        LTCHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F,     M, N, M));
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        size_t ws=64u<<20; void* dws=nullptr; cudaMalloc(&dws,ws);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws);
        cublasLtMatmulHeuristicResult_t heur[1]; int found=0;
        cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pref,1,heur,&found);
        if(hs==CUBLAS_STATUS_SUCCESS && found>0){
            float alpha=1.f,beta=0.f;
            auto body=[&]{ cublasLtMatmul(lt,op,&alpha,fa,la,fb,lb,&beta,fc,lc,fc,lc,&heur[0].algo,dws,ws,0); };
            body(); cudaDeviceSynchronize();
            cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
            for(int i=0;i<10;++i) body(); cudaDeviceSynchronize();
            std::vector<double> us(30);
            for(int i=0;i<30;++i){ cudaEventRecord(s); body(); cudaEventRecord(e); cudaEventSynchronize(e);
                float ms; cudaEventElapsedTime(&ms,s,e); us[i]=(double)ms*1000.0; }
            cudaEventDestroy(s); cudaEventDestroy(e);
            double med=probe_percentile(us,50.0);
            std::printf("gemm_fp8_e4m3 median_us=%.4f TOPS=%.3f\n", med, 2.0*M*N*K/(med*1e-6)/1e12);
        } else {
            std::printf("gemm_fp8_e4m3 UNSUPPORTED heuristic=%d found=%d\n",(int)hs,found);
        }
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatmulDescDestroy(op);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        cublasLtDestroy(lt); cudaFree(fa); cudaFree(fb); cudaFree(fc); cudaFree(dws);
    }
    cudaFree(A); cudaFree(B); cudaFree(C); cublasDestroy(h);
    return 0;
}
