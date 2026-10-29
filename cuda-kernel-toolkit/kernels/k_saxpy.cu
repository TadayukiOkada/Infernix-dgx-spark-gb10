// k_saxpy.cu — memory-bandwidth-bound elementwise kernel.
//
// Deliberately the simplest streaming kernel: y = a*x + y over a large array.
// It is included because it is the clearest case of a bandwidth-bound kernel,
// so the toolkit's roofline model and its "memory bandwidth-bound"
// classification can be checked against ground truth.  Two variants (scalar
// 4 B/thread and vectorized 16 B/thread) give the binary inspector a real
// LDG.E vs LDG.E.128 difference to detect: the scalar variant is registered,
// the vectorized one is the same math with float4 access.
#include "harness.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kDefaultN = 1 << 22;  // 4M floats = 16 MiB per array
constexpr float kA = 2.5f;
constexpr int kMaxN = 1 << 26;

float* d_x = nullptr;
float* d_y = nullptr;
int g_n = 0;
// Retained host pre-images so `verify` can compute an independent reference
// instead of trusting anything the kernel produced.
std::vector<float> h_x_ref;
std::vector<float> h_y_pre;

// ---- scalar variant: one float (4 B) per thread per step ----
__global__ void k_saxpy_scalar(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int j = i; j < n; j += stride) {
        y[j] = a * x[j] + y[j];
    }
}

// ---- vectorized variant: float4 (16 B) per thread per step ----
__global__ void k_saxpy_vec4(const float4* __restrict__ x, float4* __restrict__ y, float a, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int j = i; j < n4; j += stride) {
        float4 xv = x[j];
        float4 yv = y[j];
        yv.x = a * xv.x + yv.x;
        yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z;
        yv.w = a * xv.w + yv.w;
        y[j] = yv;
    }
}

cudaError_t run(int blocks, int threads, int /*shared_mem*/, int /*iterations*/) {
    if (g_n % 4 == 0) {
        k_saxpy_vec4<<<blocks, threads>>>(reinterpret_cast<const float4*>(d_x),
                                          reinterpret_cast<float4*>(d_y), kA, g_n / 4);
    } else {
        k_saxpy_scalar<<<blocks, threads>>>(d_x, d_y, kA, g_n);
    }
    return cudaGetLastError();
}

double bytes(int n) {
    // read x (4 B) + read y (4 B) + write y (4 B) per element
    return static_cast<double>(n) * 4.0 * 3.0;
}

double flops(int n) { return static_cast<double>(n) * 2.0; }  // one FMA per element

void setup(int n) {
    if (n > kMaxN) n = kMaxN;
    g_n = n;
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_y) { cudaFree(d_y); d_y = nullptr; }
    CKT_CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * g_n));
    CKT_CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * g_n));
    h_x_ref.resize(g_n);
    h_y_pre.resize(g_n);
    for (int i = 0; i < g_n; ++i) {
        // NOTE: cast to int *before* subtracting, otherwise (i % 97) - 48 on a
        // size_t underflows to a huge unsigned value and silently produces
        // garbage inputs that make every later comparison meaningless.
        h_x_ref[i] = static_cast<float>((i % 97) - 48) * 0.25f;
        h_y_pre[i] = static_cast<float>((i % 31) - 15) * 0.5f;
    }
    CKT_CUDA_CHECK(cudaMemcpy(d_x, h_x_ref.data(), sizeof(float) * g_n, cudaMemcpyHostToDevice));
    CKT_CUDA_CHECK(cudaMemcpy(d_y, h_y_pre.data(), sizeof(float) * g_n, cudaMemcpyHostToDevice));
}

void teardown() {
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_y) { cudaFree(d_y); d_y = nullptr; }
    h_x_ref.clear();
    h_y_pre.clear();
}

int verify(int n, std::string& detail) {
    if (n > kMaxN) n = kMaxN;
    if (static_cast<int>(h_x_ref.size()) != n) {
        detail = "verify called before setup";
        return 1;
    }
    std::vector<float> hout(n);
    if (cudaMemcpy(hout.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        detail = "device->host copy failed";
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        const float expect = kA * h_x_ref[i] + h_y_pre[i];
        if (std::fabs(expect - hout[i]) > 1e-3f * (1.0f + std::fabs(expect))) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "mismatch at %d: got %g expected %g", i, hout[i], expect);
            detail = buf;
            return 2;
        }
    }
    detail = "ok";
    return 0;
}

void defaults(int& problem_n, int& blocks, int& threads, int& shared_mem) {
    problem_n = kDefaultN;
    blocks = 384;  // 48 SMs * 8; the search replaces this
    threads = 256;
    shared_mem = 0;
}

}  // namespace

REGISTER_WORKLOAD({"saxpy", &run, &bytes, &flops, &setup, &teardown, &verify, &defaults});
