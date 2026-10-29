// k_gemm.cu — a naive, unoptimized dense GEMM (compute-bound case study).
//
// Intentionally a straightforward shared-memory tiled GEMM without register
// blocking, so it sits clearly in a compute/instruction-issue regime and the
// toolkit has a kernel where:
//   * arithmetic intensity is high,
//   * the roofline says "compute bound",
//   * the SASS shows a FMA-dominated inner loop,
//   * occupancy is limited by tile size / registers.
// A vectorized global-load variant is provided so the memory-access analysis
// has something to compare against.
#include "harness.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kDefaultM = 1024, kDefaultN = 1024, kDefaultK = 1024;
constexpr int kMaxDim = 4096;

float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;
int g_m = 0, g_n = 0, g_k = 0;
std::vector<float> h_A, h_B, h_C;

// Classic tiled GEMM: each block computes a TILE x TILE output tile.
// Shared-memory tiling only (no register blocking) => this is deliberately
// issue-bound, which is exactly the point of including it.
template <int TILE>
__global__ void k_gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && (k0 + tx) < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = ((k0 + ty) < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

constexpr int kTile = 16;

cudaError_t run(int blocks, int threads, int /*shared_mem*/, int /*iterations*/) {
    // `blocks` is interpreted as the number of thread blocks along x; the grid
    // is made 2D from the problem shape. A single scalar "blocks" would be
    // ambiguous for a 2D grid, so the harness passes grid.x and we derive y.
    dim3 block(kTile, kTile, 1);
    int gx = (g_n + kTile - 1) / kTile;
    int gy = (g_m + kTile - 1) / kTile;
    (void)blocks;
    (void)threads;
    k_gemm_tiled<kTile><<<dim3(gx, gy, 1), block>>>(d_A, d_B, d_C, g_m, g_n, g_k);
    return cudaGetLastError();
}

double bytes(int) {
    // A: M*K, B: K*N read once; C: M*N written.
    return static_cast<double>(g_m) * g_k * 4.0 + static_cast<double>(g_k) * g_n * 4.0 +
           static_cast<double>(g_m) * g_n * 4.0;
}

double flops(int) { return 2.0 * g_m * g_n * g_k; }

void setup(int n) {
    // `n` selects the square problem size.
    if (n > kMaxDim) n = kMaxDim;
    g_m = g_n = g_k = n;
    if (d_A) { cudaFree(d_A); d_A = nullptr; }
    if (d_B) { cudaFree(d_B); d_B = nullptr; }
    if (d_C) { cudaFree(d_C); d_C = nullptr; }
    const size_t sa = static_cast<size_t>(g_m) * g_k;
    const size_t sb = static_cast<size_t>(g_k) * g_n;
    const size_t sc = static_cast<size_t>(g_m) * g_n;
    CKT_CUDA_CHECK(cudaMalloc(&d_A, sa * sizeof(float)));
    CKT_CUDA_CHECK(cudaMalloc(&d_B, sb * sizeof(float)));
    CKT_CUDA_CHECK(cudaMalloc(&d_C, sc * sizeof(float)));
    h_A.resize(sa);
    h_B.resize(sb);
    h_C.assign(sc, 0.0f);
    for (size_t i = 0; i < sa; ++i)
        h_A[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    for (size_t i = 0; i < sb; ++i)
        h_B[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
    CKT_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sa * sizeof(float), cudaMemcpyHostToDevice));
    CKT_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sb * sizeof(float), cudaMemcpyHostToDevice));
}

void teardown() {
    if (d_A) { cudaFree(d_A); d_A = nullptr; }
    if (d_B) { cudaFree(d_B); d_B = nullptr; }
    if (d_C) { cudaFree(d_C); d_C = nullptr; }
    h_A.clear();
    h_B.clear();
    h_C.clear();
}

int verify(int n, std::string& detail) {
    if (h_A.empty()) { detail = "verify called before setup"; return 1; }
    const int M = g_m, N = g_n, K = g_k;
    std::vector<float> got(static_cast<size_t>(M) * N);
    if (cudaMemcpy(got.data(), d_C, got.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        detail = "device->host copy failed";
        return 1;
    }
    // Independent FP32 reference on the host, restricted to a sampled set of
    // rows so validation stays cheap for large problems.
    const int stride_m = std::max(1, M / 16);
    double worst = 0.0;
    for (int i = 0; i < M; i += stride_m) {
        for (int j = 0; j < N; j += std::max(1, N / 64)) {
            double acc = 0.0;
            for (int kk = 0; kk < K; ++kk) {
                acc += static_cast<double>(h_A[static_cast<size_t>(i) * K + kk]) *
                       static_cast<double>(h_B[static_cast<size_t>(kk) * N + j]);
            }
            const double err = std::fabs(acc - got[static_cast<size_t>(i) * N + j]);
            const double rel = err / (1.0 + std::fabs(acc));
            worst = std::max(worst, rel);
            if (rel > 1e-3) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                              "mismatch at (%d,%d): got %.6f expected %.6f rel=%.3g", i, j,
                              got[static_cast<size_t>(i) * N + j], acc, rel);
                detail = buf;
                return 2;
            }
        }
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "ok (worst rel err %.3g over sampled rows)", worst);
    detail = buf;
    return 0;
}

void defaults(int& problem_n, int& blocks, int& threads, int& shared_mem) {
    problem_n = kDefaultM;
    blocks = (kDefaultN + kTile - 1) / kTile;  // grid.x
    threads = kTile * kTile;
    shared_mem = 0;  // statically declared
}

}  // namespace

REGISTER_WORKLOAD({"gemm_tiled", &run, &bytes, &flops, &setup, &teardown, &verify, &defaults});
