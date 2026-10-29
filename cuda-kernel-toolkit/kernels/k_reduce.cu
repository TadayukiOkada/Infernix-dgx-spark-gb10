// k_reduce.cu — block reduction where grid size is the tunable parameter.
//
// Included because it is a case where the *number of blocks* (not just the
// block size) is the primary tuning knob, and where a too-small grid leaves
// SMs idle while a too-large grid adds unnecessary global traffic.  It gives
// the launch-parameter search a workload whose optimum is not at an extreme.
#include "harness.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kDefaultN = 1 << 24;  // 16M floats = 64 MiB
constexpr int kMaxN = 1 << 26;
constexpr int kBlock = 256;

float* d_in = nullptr;
float* d_partial = nullptr;
int g_n = 0;
int g_blocks = 0;
std::vector<float> h_in_ref;

// Each block reduces a contiguous chunk into one partial. A grid-stride loop
// over int4 keeps the global read vectorized; the block reduction uses the
// standard warp-shuffle + shared-memory tree.
__global__ void k_reduce_block(const int4* __restrict__ in, float* __restrict__ partial, int n4) {
    __shared__ float warp_sums[32];
    float acc = 0.0f;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        int4 v = in[i];
        acc += static_cast<float>(v.x) + static_cast<float>(v.y) + static_cast<float>(v.z) +
               static_cast<float>(v.w);
    }
    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[blockIdx.x] = v;
    }
}

cudaError_t run(int blocks, int threads, int /*shared_mem*/, int /*iterations*/) {
    g_blocks = blocks;
    const int threads_used = threads > 0 ? (threads & ~31) : kBlock;
    const int n4 = g_n / 4;
    k_reduce_block<<<blocks, threads_used>>>(reinterpret_cast<const int4*>(d_in), d_partial, n4);
    return cudaGetLastError();
}

double bytes(int n) { return static_cast<double>(n) * 4.0; }  // read only
double flops(int n) { return static_cast<double>(n); }        // one add per element

void setup(int n) {
    if (n > kMaxN) n = kMaxN;
    g_n = n - (n % 4);
    if (d_in) { cudaFree(d_in); d_in = nullptr; }
    if (d_partial) { cudaFree(d_partial); d_partial = nullptr; }
    CKT_CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * g_n));
    // Room for the largest grid the search may request.
    CKT_CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * 4096));
    h_in_ref.resize(g_n);
    for (int i = 0; i < g_n; ++i) h_in_ref[i] = static_cast<float>((i % 7) - 3);
    // The kernel reads the buffer as int4, so the device-side values must be
    // the same integers the reference sums.
    std::vector<int> hi(g_n);
    for (int i = 0; i < g_n; ++i) hi[i] = static_cast<int>(h_in_ref[i]);
    CKT_CUDA_CHECK(cudaMemcpy(d_in, hi.data(), sizeof(int) * g_n, cudaMemcpyHostToDevice));
}

void teardown() {
    if (d_in) { cudaFree(d_in); d_in = nullptr; }
    if (d_partial) { cudaFree(d_partial); d_partial = nullptr; }
    h_in_ref.clear();
}

int verify(int n, std::string& detail) {
    if (h_in_ref.empty()) { detail = "verify called before setup"; return 1; }
    const int blocks = g_blocks;
    std::vector<float> part(blocks);
    if (cudaMemcpy(part.data(), d_partial, sizeof(float) * blocks, cudaMemcpyDeviceToHost) != cudaSuccess) {
        detail = "device->host copy failed";
        return 1;
    }
    double sum = 0.0;
    for (float v : part) sum += v;
    double expect = 0.0;
    for (int i = 0; i < g_n; ++i) expect += h_in_ref[i];
    const double tol = 1e-2 * (1.0 + std::fabs(expect));
    if (std::fabs(sum - expect) > tol) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "sum mismatch: got %.4f expected %.4f (tol %.4f)", sum, expect, tol);
        detail = buf;
        return 2;
    }
    char buf[120];
    std::snprintf(buf, sizeof(buf), "ok (sum %.1f over %d blocks)", sum, blocks);
    detail = buf;
    return 0;
}

void defaults(int& problem_n, int& blocks, int& threads, int& shared_mem) {
    problem_n = kDefaultN;
    blocks = 192;  // 48 SMs * 4
    threads = kBlock;
    shared_mem = 0;
}

}  // namespace

REGISTER_WORKLOAD({"reduce_block", &run, &bytes, &flops, &setup, &teardown, &verify, &defaults});
