// k_latency.cu — a deliberately dependency-latency-bound kernel.
//
// Each thread performs a long chain of dependent FMAs with no independent work
// to overlap, so throughput is limited by instruction latency and available
// ILP rather than by any memory or compute ceiling.  This is the reference case
// for the toolkit's "instruction-dependency-bound / latency-bound"
// classification: memory utilisation is low, compute utilisation is low, and
// the stall analysis shows long dependency stalls.
//
// The number of independent chains per thread is controlled by an unroll factor
// so the toolkit can show ILP lifting the kernel off the latency floor.
#include "harness.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kDefaultThreads = 256;
constexpr int kDefaultBlocks = 48 * 2;
constexpr int kIterations = 4096;  // dependent steps per chain

float* d_out = nullptr;
float* d_seed = nullptr;
int g_chains = 1;
std::vector<float> h_seed;

// One thread, `chains` independent accumulator chains of length kIterations.
// With chains=1 there is no ILP and the FMA latency dominates.
template <int CHAINS>
__global__ void k_dependent_fma(const float* __restrict__ seed, float* __restrict__ out, int n_threads,
                                float a) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_threads) return;
    const float s = seed[t & 1023];
    float acc[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) acc[c] = s + static_cast<float>(c);
    #pragma unroll 1
    for (int i = 0; i < kIterations; ++i) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            acc[c] = fmaf(acc[c], a, s);
        }
    }
    float total = 0.0f;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) total += acc[c];
    out[t] = total;
}

cudaError_t run(int blocks, int threads, int /*shared_mem*/, int /*iterations*/) {
    const int n_threads = blocks * threads;
    const float a = 0.999999f;
    switch (g_chains) {
        case 1: k_dependent_fma<1><<<blocks, threads>>>(d_seed, d_out, n_threads, a); break;
        case 2: k_dependent_fma<2><<<blocks, threads>>>(d_seed, d_out, n_threads, a); break;
        case 4: k_dependent_fma<4><<<blocks, threads>>>(d_seed, d_out, n_threads, a); break;
        case 8: k_dependent_fma<8><<<blocks, threads>>>(d_seed, d_out, n_threads, a); break;
        default: k_dependent_fma<1><<<blocks, threads>>>(d_seed, d_out, n_threads, a); break;
    }
    return cudaGetLastError();
}

double bytes(int) { return 0.0; }  // no meaningful data movement

double flops(int) {
    // CHAINS * kIterations FMAs * 2 FLOP, per thread.
    // Threads are supplied at run time, so report per-thread FLOP and let the
    // harness multiply by the launched thread count via `flops_per_thread`.
    return static_cast<double>(g_chains) * kIterations * 2.0;
}

void setup(int n) {
    (void)n;
    // `n` selects the number of chains.
    g_chains = (n >= 8) ? 8 : (n >= 4) ? 4 : (n >= 2) ? 2 : 1;
    if (d_out) { cudaFree(d_out); d_out = nullptr; }
    if (d_seed) { cudaFree(d_seed); d_seed = nullptr; }
    // Sized for the largest plausible thread count.
    CKT_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * (64 * 1024)));
    // The seed buffer is read-only during a launch; keeping it separate from the
    // output removes any read/write race between threads.
    CKT_CUDA_CHECK(cudaMalloc(&d_seed, sizeof(float) * 1024));
    h_seed.assign(1024, 1.0f);
    std::vector<float> seedbuf(1024);
    for (int i = 0; i < 1024; ++i) seedbuf[i] = 1.0f + 1e-6f * i;
    CKT_CUDA_CHECK(cudaMemcpy(d_seed, seedbuf.data(), sizeof(float) * seedbuf.size(), cudaMemcpyHostToDevice));
    h_seed = seedbuf;
    CKT_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * (64 * 1024)));
}

void teardown() {
    if (d_out) { cudaFree(d_out); d_out = nullptr; }
    if (d_seed) { cudaFree(d_seed); d_seed = nullptr; }
    h_seed.clear();
}

int verify(int, std::string& detail) {
    // Validate finiteness and determinism rather than a closed-form value: the
    // point of this kernel is timing, but it must still not be wrong.
    std::vector<float> buf(64 * 1024);
    if (cudaMemcpy(buf.data(), d_out, sizeof(float) * buf.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        detail = "device->host copy failed";
        return 1;
    }
    int nonfinite = 0;
    for (float v : buf) {
        if (!std::isfinite(v)) ++nonfinite;
    }
    if (nonfinite > 0) {
        char b[96];
        std::snprintf(b, sizeof(b), "%d non-finite outputs (a=%g near 1 should stay finite)", nonfinite, 0.999999f);
        detail = b;
        return 2;
    }
    detail = "ok (all outputs finite)";
    return 0;
}

void defaults(int& problem_n, int& blocks, int& threads, int& shared_mem) {
    problem_n = 1;  // one chain
    blocks = kDefaultBlocks;
    threads = kDefaultThreads;
    shared_mem = 0;
}

}  // namespace

REGISTER_WORKLOAD({"dependent_fma", &run, &bytes, &flops, &setup, &teardown, &verify, &defaults});
