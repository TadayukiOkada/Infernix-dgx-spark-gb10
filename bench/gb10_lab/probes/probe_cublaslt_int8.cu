// probe_cublaslt_int8.cu — GB10 feature probe: cuBLASLt INT8 GEMM (Tensor Core INT8).
// Validates whether cublasLt supports CUDA_R_8I inputs with CUDA_R_32I accumulate on sm_121.
#include "probe_common.h"
#include <cstdio>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>

#define LT_CHECK(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    std::printf("probe_cublaslt_int8 CUBLASLT_FAIL %d @%d\n",(int)s_,__LINE__); return 4;} } while(0)

int main() {
    const int m=1024,n=1024,k=1024;
    int8_t *A,*B; int32_t *C;
    cudaMalloc(&A,(size_t)m*k); cudaMalloc(&B,(size_t)k*n); cudaMalloc(&C,(size_t)m*n);
    cudaMemset(A,1,(size_t)m*k); cudaMemset(B,1,(size_t)k*n); cudaMemset(C,0,(size_t)m*n*4);
    cublasLtHandle_t lt; LT_CHECK(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t op; LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32F));
    cublasOperation_t tn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof tn);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof tn);
    cublasLtMatrixLayout_t la,lb,lc;
    LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  m, k, m));
    LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  k, n, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, m, n, m));
    float alpha=1.f, beta=0.f;
    // heuristic algo selection
    cublasLtMatmulPreference_t pref; LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws=32*1024*1024; void* dws=nullptr; cudaMalloc(&dws,ws);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    cublasLtMatmulHeuristicResult_t heur[1]; int found=0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, heur, &found);
    if (hs != CUBLAS_STATUS_SUCCESS || found == 0) {
        std::printf("probe_cublaslt_int8 UNSUPPORTED heuristic_status=%d found=%d\n",(int)hs,found);
        return 5;
    }
    cublasStatus_t r = cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc,
                                      &heur[0].algo, dws, ws, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (r != CUBLAS_STATUS_SUCCESS || e != cudaSuccess) {
        std::printf("probe_cublaslt_int8 RUNTIME_FAIL matmul=%d cuda=%s\n",(int)r,cudaGetErrorString(e));
        return 6;
    }
    std::printf("probe_cublaslt_int8 OK (INT8 in / INT32 accum supported)\n");
    // Throughput if (and only if) the API actually supports INT8 on this device:
    // sustained INT8 GEMM TOPS at m=n=k=1024 under the 2MNK convention.
    auto body=[&]{ cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc,
                                  &heur[0].algo, dws, ws, 0); };
    const double ops = 2.0*m*n*k;
    probe_time_and_report("cublaslt_int8_gemm", body, ops);
    return 0;
}
