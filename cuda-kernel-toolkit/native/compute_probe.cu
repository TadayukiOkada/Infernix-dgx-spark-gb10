// compute_probe — measure achievable arithmetic ceilings on the target GPU.
//
// A roofline model needs a *measured* compute ceiling for the same reason it
// needs a measured memory ceiling: vendor peak numbers describe the best case
// for a kernel that does nothing but arithmetic, and clock/thermal behaviour on
// a real part (especially a power-limited one) can leave it far short.
//
// The probe measures throughput for independent multiply-add chains, which is
// the highest-issue-rate pattern the FMA pipe supports.  Output is emitted as
// `key<TAB>value` records so the Python side can consume it without a JSON
// dependency.
//
// Usage:
//   compute_probe [--reps N] [--warmup N] [--blocks B] [--threads T]
//
// Build:
//   nvcc -arch=sm_121a -O3 -std=c++17 -o compute_probe compute_probe.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                   __FILE__, __LINE__, cudaGetErrorString(err_));               \
      std::exit(2);                                                             \
    }                                                                           \
  } while (0)

// Eight independent accumulator chains per thread.  A single dependent chain
// would measure latency, not throughput; the chains also stop the compiler from
// collapsing the loop into fewer operations.
__global__ void k_fma_throughput(float *__restrict__ out, int iters, float seed) {
  float a0 = seed + threadIdx.x * 1e-3f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
  float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
  const float m = 0.999999f, c = 1e-7f;
  for (int i = 0; i < iters; ++i) {
    a0 = fmaf(a0, m, c);
    a1 = fmaf(a1, m, c);
    a2 = fmaf(a2, m, c);
    a3 = fmaf(a3, m, c);
    a4 = fmaf(a4, m, c);
    a5 = fmaf(a5, m, c);
    a6 = fmaf(a6, m, c);
    a7 = fmaf(a7, m, c);
  }
  float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == 12345.678f) {  // never true; keeps the chains live
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

// FP64 variant.  Measured separately because consumer/SoC parts often have a
// different FP64:FP32 ratio than data-center parts, and assuming parity is a
// common modelling error.
__global__ void k_fma_throughput_f64(double *__restrict__ out, int iters, double seed) {
  double a0 = seed + threadIdx.x * 1e-3, a1 = a0 + 0.1, a2 = a0 + 0.2, a3 = a0 + 0.3;
  const double m = 0.999999, c = 1e-9;
  for (int i = 0; i < iters; ++i) {
    a0 = fma(a0, m, c);
    a1 = fma(a1, m, c);
    a2 = fma(a2, m, c);
    a3 = fma(a3, m, c);
  }
  double s = a0 + a1 + a2 + a3;
  if (s == 12345.6789) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Integer add throughput, for characterising integer-workload kernels.
__global__ void k_iadd_throughput(int *__restrict__ out, int iters, int seed) {
  int a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3;
  for (int i = 0; i < iters; ++i) {
    a0 += i; a1 ^= a0; a2 += a1; a3 ^= a2;
  }
  int s = a0 + a1 + a2 + a3;
  if (s == 0x7fffffff) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// FP16x2 SIMT FMA throughput (__hfma2).
__global__ void k_fma_throughput_f16_simt(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 600
  __half2 a0 = __float2half2_rn(1.0001f + threadIdx.x * 1e-4f);
  __half2 a1 = __float2half2_rn(1.0002f + threadIdx.x * 1e-4f);
  __half2 a2 = __float2half2_rn(1.0003f + threadIdx.x * 1e-4f);
  __half2 a3 = __float2half2_rn(1.0004f + threadIdx.x * 1e-4f);
  const __half2 m = __float2half2_rn(0.9995f);
  const __half2 c = __float2half2_rn(1e-4f);
  for (int i = 0; i < iters; ++i) {
    a0 = __hfma2(a0, m, c);
    a1 = __hfma2(a1, m, c);
    a2 = __hfma2(a2, m, c);
    a3 = __hfma2(a3, m, c);
  }
  __half2 s = __hadd2(__hadd2(a0, a1), __hadd2(a2, a3));
  float sf = __half2float(__low2half(s)) + __half2float(__high2half(s));
  if (sf == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = sf;
#else
  (void)out; (void)iters;
#endif
}

// BF16x2 SIMT FMA throughput (__hfma2 on __nv_bfloat162).
__global__ void k_fma_throughput_bf16_simt(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat162 a0 = __float2bfloat162_rn(1.0001f + threadIdx.x * 1e-4f);
  __nv_bfloat162 a1 = __float2bfloat162_rn(1.0002f + threadIdx.x * 1e-4f);
  __nv_bfloat162 a2 = __float2bfloat162_rn(1.0003f + threadIdx.x * 1e-4f);
  __nv_bfloat162 a3 = __float2bfloat162_rn(1.0004f + threadIdx.x * 1e-4f);
  const __nv_bfloat162 m = __float2bfloat162_rn(0.9995f);
  const __nv_bfloat162 c = __float2bfloat162_rn(1e-4f);
  for (int i = 0; i < iters; ++i) {
    a0 = __hfma2(a0, m, c);
    a1 = __hfma2(a1, m, c);
    a2 = __hfma2(a2, m, c);
    a3 = __hfma2(a3, m, c);
  }
  __nv_bfloat162 s = __hadd2(__hadd2(a0, a1), __hadd2(a2, a3));
  float sf = __bfloat162float(__low2bfloat16(s)) + __bfloat162float(__high2bfloat16(s));
  if (sf == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = sf;
#else
  (void)out; (void)iters;
#endif
}

// TF32 tensor-core throughput (mma.sync m16n8k8 f32.tf32.tf32.f32): 16*8*8*2 = 2048 ops/MMA.
__global__ void k_mma_tf32_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_TF32(D)                                               \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                  \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA_TF32(d0);
    INFERNIX_CKT_MMA_TF32(d1);
    INFERNIX_CKT_MMA_TF32(d2);
    INFERNIX_CKT_MMA_TF32(d3);
#undef INFERNIX_CKT_MMA_TF32
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Int8 tensor-core throughput.  The ggml tile kernels issue
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 exclusively, so an fp32 FMA
// probe is the wrong roofline ceiling for them.  Each warp issues independent
// MMAs (accumulator cycled through 4 live registers so dependent-MMA latency
// does not set the rate); no ldmatrix, so this measures the tensor pipe alone.
__global__ void k_mma_s8_throughput(int *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  int a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  int b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  int d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA(D)                                                    \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "                      \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA(d0);
    INFERNIX_CKT_MMA(d1);
    INFERNIX_CKT_MMA(d2);
    INFERNIX_CKT_MMA(d3);
#undef INFERNIX_CKT_MMA
  }
  int s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 0x7fffffff) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Int4 tensor-core throughput (mma.sync m16n8k64 s32.s4.s4.s32): 16*8*64*2 = 16384 ops/MMA.
__global__ void k_mma_s4_throughput(int *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  int a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  int b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  int d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_S4(D)                                                 \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "                     \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA_S4(d0);
    INFERNIX_CKT_MMA_S4(d1);
    INFERNIX_CKT_MMA_S4(d2);
    INFERNIX_CKT_MMA_S4(d3);
#undef INFERNIX_CKT_MMA_S4
  }
  int s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 0x7fffffff) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// FP16 tensor-core throughput with FP32 accumulation (mma.sync m16n8k16 f32.f16.f16.f32).
__global__ void k_mma_f16_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA16(D)                                                  \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                    \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA16(d0);
    INFERNIX_CKT_MMA16(d1);
    INFERNIX_CKT_MMA16(d2);
    INFERNIX_CKT_MMA16(d3);
#undef INFERNIX_CKT_MMA16
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// FP16 tensor-core throughput with FP16 accumulation (mma.sync m16n8k16 f16.f16.f16.f16).
// On consumer/SoC Blackwell parts, FP16 accumulation runs at 2x the rate of FP32 accumulation.
__global__ void k_mma_f16_f16acc_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  unsigned d0[2] = {0, 0}, d1[2] = {0, 0}, d2[2] = {0, 0}, d3[2] = {0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA16_F16ACC(D)                                           \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "                   \
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                          \
        : "+r"(D[0]), "+r"(D[1])                                               \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA16_F16ACC(d0);
    INFERNIX_CKT_MMA16_F16ACC(d1);
    INFERNIX_CKT_MMA16_F16ACC(d2);
    INFERNIX_CKT_MMA16_F16ACC(d3);
#undef INFERNIX_CKT_MMA16_F16ACC
  }
  unsigned s = d0[0] ^ d1[1] ^ d2[0] ^ d3[1];
  if (s == 0x7fffffffu) out[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<float>(s);
#else
  (void)out; (void)iters;
#endif
}

// BF16 tensor-core throughput with FP32 accumulation (mma.sync m16n8k16 f32.bf16.bf16.f32).
__global__ void k_mma_bf16_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 800
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMABF16(D)                                                \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                 \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMABF16(d0);
    INFERNIX_CKT_MMABF16(d1);
    INFERNIX_CKT_MMABF16(d2);
    INFERNIX_CKT_MMABF16(d3);
#undef INFERNIX_CKT_MMABF16
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Plain FP8 E4M3 tensor-core throughput (mma.sync m16n8k32 f32.e4m3.e4m3.f32): 8192 ops/MMA.
__global__ void k_mma_fp8_e4m3_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 890
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_FP8_E4M3(D)                                           \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                 \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA_FP8_E4M3(d0);
    INFERNIX_CKT_MMA_FP8_E4M3(d1);
    INFERNIX_CKT_MMA_FP8_E4M3(d2);
    INFERNIX_CKT_MMA_FP8_E4M3(d3);
#undef INFERNIX_CKT_MMA_FP8_E4M3
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Plain FP8 E5M2 tensor-core throughput (mma.sync m16n8k32 f32.e5m2.e5m2.f32): 8192 ops/MMA.
__global__ void k_mma_fp8_e5m2_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 890
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_FP8_E5M2(D)                                           \
    asm(                                                                       \
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "                 \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    INFERNIX_CKT_MMA_FP8_E5M2(d0);
    INFERNIX_CKT_MMA_FP8_E5M2(d1);
    INFERNIX_CKT_MMA_FP8_E5M2(d2);
    INFERNIX_CKT_MMA_FP8_E5M2(d3);
#undef INFERNIX_CKT_MMA_FP8_E5M2
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Block-scaled MXFP8 E4M3 tensor-core throughput (kind::mxf8f6f4.block_scale m16n8k32): 8192 ops/MMA.
// On Blackwell (sm_120a/sm_121a), block-scaled FP8 runs at full FP8 rate even with FP32 accumulation.
__global__ void k_mma_mxfp8_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 1000
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  unsigned sfa = 0x7fu, sfb = 0x7fu;
  constexpr unsigned short kBid = 0, kTid = 0;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_MXFP8(D)                                              \
    asm(                                                                       \
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."           \
        "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "                            \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "               \
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"                                \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                \
          "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid))
    INFERNIX_CKT_MMA_MXFP8(d0);
    INFERNIX_CKT_MMA_MXFP8(d1);
    INFERNIX_CKT_MMA_MXFP8(d2);
    INFERNIX_CKT_MMA_MXFP8(d3);
#undef INFERNIX_CKT_MMA_MXFP8
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Block-scaled MXFP6 E3M2 tensor-core throughput (kind::mxf8f6f4.block_scale m16n8k32): 8192 ops/MMA.
__global__ void k_mma_mxfp6_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 1000
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  unsigned sfa = 0x7fu, sfb = 0x7fu;
  constexpr unsigned short kBid = 0, kTid = 0;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_MXFP6(D)                                              \
    asm(                                                                       \
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."           \
        "m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 "                            \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "               \
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"                                \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                \
          "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid))
    INFERNIX_CKT_MMA_MXFP6(d0);
    INFERNIX_CKT_MMA_MXFP6(d1);
    INFERNIX_CKT_MMA_MXFP6(d2);
    INFERNIX_CKT_MMA_MXFP6(d3);
#undef INFERNIX_CKT_MMA_MXFP6
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Block-scaled MXFP4 E2M1 tensor-core throughput (kind::mxf4nvf4.block_scale.scale_vec::2X m16n8k64): 16384 ops/MMA.
__global__ void k_mma_mxfp4_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 1000
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  unsigned sfa = 0x7f7fu, sfb = 0x7f7fu;
  constexpr unsigned short kBid = 0, kTid = 0;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_MXFP4(D)                                              \
    asm(                                                                       \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."           \
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "                            \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "               \
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"                                \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                \
          "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid))
    INFERNIX_CKT_MMA_MXFP4(d0);
    INFERNIX_CKT_MMA_MXFP4(d1);
    INFERNIX_CKT_MMA_MXFP4(d2);
    INFERNIX_CKT_MMA_MXFP4(d3);
#undef INFERNIX_CKT_MMA_MXFP4
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Block-scaled NVFP4 E2M1 tensor-core throughput (kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64): 16384 ops/MMA.
__global__ void k_mma_nvfp4_throughput(float *__restrict__ out, int iters) {
#if __CUDA_ARCH__ >= 1000
  unsigned a0 = threadIdx.x, a1 = threadIdx.x * 3 + 1, a2 = ~threadIdx.x, a3 = threadIdx.x ^ 0x5a;
  unsigned b0 = blockIdx.x, b1 = blockIdx.x * 7 + 3;
  unsigned sfa = 0x38383838u, sfb = 0x38383838u;
  constexpr unsigned short kBid = 0, kTid = 0;
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#define INFERNIX_CKT_MMA_NVFP4(D)                                              \
    asm(                                                                       \
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."           \
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "                            \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "               \
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"                                \
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])                       \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                \
          "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid))
    INFERNIX_CKT_MMA_NVFP4(d0);
    INFERNIX_CKT_MMA_NVFP4(d1);
    INFERNIX_CKT_MMA_NVFP4(d2);
    INFERNIX_CKT_MMA_NVFP4(d3);
#undef INFERNIX_CKT_MMA_NVFP4
  }
  float s = d0[0] + d1[1] + d2[2] + d3[3];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#else
  (void)out; (void)iters;
#endif
}

// Shared-memory bandwidth, for kernels that stage through smem rather than
// saturating HBM.  Distinct from DRAM bandwidth and frequently the real limit.
__global__ void k_smem_throughput(float *__restrict__ out, int iters) {
  extern __shared__ float smem[];
  const int t = threadIdx.x;
  for (int i = t; i < blockDim.x * 4; i += blockDim.x) smem[i] = t * 1e-3f;
  __syncthreads();
  float acc = 0.f;
  const int n = blockDim.x * 4;
  for (int k = 0; k < iters; ++k) {
    acc += smem[(t + k) % n];
    acc += smem[(t + k * 3) % n];
  }
  if (acc == 12345.678f) out[t] = acc;
}

static void emit(const char *key, double value) { std::printf("%s\t%.6f\n", key, value); }
static void emit_str(const char *key, const char *value) { std::printf("%s\t%s\n", key, value); }
static void emit_int(const char *key, long long value) { std::printf("%s\t%lld\n", key, value); }

template <typename K>
static double time_kernel(K launch, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return (double)ms / reps;
}

int main(int argc, char **argv) {
  int reps = 20, warmup = 3, blocks = 0, threads = 256, iters = 20000;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  if (blocks <= 0) {
    // Fill the machine with several waves so the measurement is not launch-limited.
    blocks = p.multiProcessorCount * (p.maxThreadsPerMultiProcessor / threads) * 4;
  }

  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));

  emit_str("probe.schema", "compute_probe/1");
  emit_str("device.name", p.name);
  std::printf("device.compute_capability\t%d.%d\n", p.major, p.minor);
  emit_int("device.sms", p.multiProcessorCount);
  emit_int("device.clock_khz", clock_khz);
  emit_int("config.blocks", blocks);
  emit_int("config.threads", threads);
  emit_int("config.iters", iters);

  float *dout = nullptr;
  double *doutd = nullptr;
  int *douti = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, sizeof(float) * (size_t)blocks * threads));
  CUDA_CHECK(cudaMalloc(&doutd, sizeof(double) * (size_t)blocks * threads));
  CUDA_CHECK(cudaMalloc(&douti, sizeof(int) * (size_t)blocks * threads));

  const double threads_total = (double)blocks * threads;

  // ---- FP32 FMA ---------------------------------------------------------- //
  {
    double ms = time_kernel([&] { k_fma_throughput<<<blocks, threads>>>(dout, iters, 1.0001f); }, warmup, reps);
    // 8 FMA per iteration per thread, each counted as 2 FLOP.
    double flops = threads_total * (double)iters * 8.0 * 2.0;
    double tflops = flops / (ms * 1e-3) / 1e12;
    emit("fp32_fma.ms", ms);
    emit("fp32_fma.tflops", tflops);
    emit("fp32_fma.gflops_per_sm_ghz",
         clock_khz > 0 ? (flops / (ms * 1e-3)) / p.multiProcessorCount / (clock_khz / 1e6) / 1e9 : 0.0);
  }

  // ---- FP64 FMA ---------------------------------------------------------- //
  bool fp64_ok = p.major >= 6;
  if (fp64_ok) {
    double ms = time_kernel([&] { k_fma_throughput_f64<<<blocks, threads>>>(doutd, iters, 1.0001); }, warmup, reps);
    double flops = threads_total * (double)iters * 4.0 * 2.0;
    emit("fp64_fma.ms", ms);
    emit("fp64_fma.tflops", flops / (ms * 1e-3) / 1e12);
  }

  // ---- FP16 SIMT (__hfma2) ----------------------------------------------- //
  if (p.major >= 6) {
    double ms = time_kernel([&] { k_fma_throughput_f16_simt<<<blocks, threads>>>(dout, iters); }, warmup, reps);
    // 4 __hfma2 per iteration per thread, each = 2 halves * 2 FLOP = 4 FLOP -> 16 FLOP/iter.
    double flops = threads_total * (double)iters * 16.0;
    emit("fp16_simt.ms", ms);
    emit("fp16_simt.tflops", flops / (ms * 1e-3) / 1e12);
  }

  // ---- BF16 SIMT (__hfma2) ----------------------------------------------- //
  if (p.major >= 8) {
    double ms = time_kernel([&] { k_fma_throughput_bf16_simt<<<blocks, threads>>>(dout, iters); }, warmup, reps);
    double flops = threads_total * (double)iters * 16.0;
    emit("bf16_simt.ms", ms);
    emit("bf16_simt.tflops", flops / (ms * 1e-3) / 1e12);
  }

  // ---- Integer add ------------------------------------------------------- //
  {
    double ms = time_kernel([&] { k_iadd_throughput<<<blocks, threads>>>(douti, iters, 1); }, warmup, reps);
    // 4 integer operations per iteration per thread.
    double ops = threads_total * (double)iters * 4.0;
    emit("int_add.ms", ms);
    emit("int_add.gopp_per_s", ops / (ms * 1e-3) / 1e9);
  }

  // ---- Shared memory ----------------------------------------------------- //
  {
    size_t smem = (size_t)threads * 4 * sizeof(float);
    int sblocks = blocks;
    int sm_iters = iters / 2;
    double ms = time_kernel(
        [&] { k_smem_throughput<<<sblocks, threads, smem>>>(dout, sm_iters); }, warmup, reps);
    // Two 4-byte loads per iteration per thread.
    double bytes = threads_total * (double)sm_iters * 2.0 * 4.0;
    emit("smem_load.ms", ms);
    emit("smem_load.gbps", bytes / (ms * 1e-3) / 1e9);
  }

  const double warps_total = (double)blocks * (threads / 32);
  const int mma_iters = iters / 8;

  // ---- TF32 tensor core (mma.sync m16n8k8 tf32) -------------------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_tf32_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 2048.0;
    emit("tf32_mma.ms", ms);
    emit("tf32_mma.tflops", ops / (ms * 1e-3) / 1e12);
  }

  // ---- FP16 tensor core (mma.sync m16n8k16 f16, FP32 acc) ---------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_f16_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 4096.0;
    emit("fp16_mma.ms", ms);
    emit("fp16_mma.tflops", ops / (ms * 1e-3) / 1e12);
  }

  // ---- FP16 tensor core (mma.sync m16n8k16 f16, FP16 acc) ---------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_f16_f16acc_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 4096.0;
    emit("fp16_f16acc_mma.ms", ms);
    emit("fp16_f16acc_mma.tflops", ops / (ms * 1e-3) / 1e12);
  }

  // ---- BF16 tensor core (mma.sync m16n8k16 bf16, FP32 acc) --------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_bf16_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 4096.0;
    emit("bf16_mma.ms", ms);
    emit("bf16_mma.tflops", ops / (ms * 1e-3) / 1e12);
  }

  // ---- Int8 tensor core (mma.sync m16n8k32 s8) --------------------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_s8_throughput<<<blocks, threads>>>(douti, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 8192.0;
    emit("int8_mma.ms", ms);
    emit("int8_mma.tops", ops / (ms * 1e-3) / 1e12);
    emit("int8_mma.mma_per_sm_ghz",
         clock_khz > 0 ? warps_total * (double)mma_iters * 4.0 /
                           (ms * 1e-3) / p.multiProcessorCount / (clock_khz / 1e6) : 0.0);
  }

  // ---- Int4 tensor core (mma.sync m16n8k64 s4) --------------------------- //
  if (p.major >= 8) {
    double ms = time_kernel(
        [&] { k_mma_s4_throughput<<<blocks, threads>>>(douti, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 16384.0;
    emit("int4_mma.ms", ms);
    emit("int4_mma.tops", ops / (ms * 1e-3) / 1e12);
  }

  // ---- Plain FP8 tensor core (mma.sync m16n8k32 e4m3 / e5m2, FP32 acc) --- //
  if (p.major >= 9 || (p.major == 8 && p.minor >= 9)) {
    double ms_e4m3 = time_kernel(
        [&] { k_mma_fp8_e4m3_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    double ops = warps_total * (double)mma_iters * 4.0 * 8192.0;
    emit("fp8_e4m3_mma.ms", ms_e4m3);
    emit("fp8_e4m3_mma.tflops", ops / (ms_e4m3 * 1e-3) / 1e12);

    double ms_e5m2 = time_kernel(
        [&] { k_mma_fp8_e5m2_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    emit("fp8_e5m2_mma.ms", ms_e5m2);
    emit("fp8_e5m2_mma.tflops", ops / (ms_e5m2 * 1e-3) / 1e12);
  }

  // ---- Blackwell block-scaled MXFP8 / MXFP6 / MXFP4 / NVFP4 -------------- //
  if (p.major >= 10) {
    double ops_k32 = warps_total * (double)mma_iters * 4.0 * 8192.0;
    double ops_k64 = warps_total * (double)mma_iters * 4.0 * 16384.0;

    double ms_mxfp8 = time_kernel(
        [&] { k_mma_mxfp8_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    emit("mxfp8_mma.ms", ms_mxfp8);
    emit("mxfp8_mma.tflops", ops_k32 / (ms_mxfp8 * 1e-3) / 1e12);

    double ms_mxfp6 = time_kernel(
        [&] { k_mma_mxfp6_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    emit("mxfp6_mma.ms", ms_mxfp6);
    emit("mxfp6_mma.tflops", ops_k32 / (ms_mxfp6 * 1e-3) / 1e12);

    double ms_mxfp4 = time_kernel(
        [&] { k_mma_mxfp4_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    emit("mxfp4_mma.ms", ms_mxfp4);
    emit("mxfp4_mma.tflops", ops_k64 / (ms_mxfp4 * 1e-3) / 1e12);

    double ms_nvfp4 = time_kernel(
        [&] { k_mma_nvfp4_throughput<<<blocks, threads>>>(dout, mma_iters); }, warmup, reps);
    emit("nvfp4_mma.ms", ms_nvfp4);
    emit("nvfp4_mma.tflops", ops_k64 / (ms_nvfp4 * 1e-3) / 1e12);
  }

  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(doutd));
  CUDA_CHECK(cudaFree(douti));
  return 0;
}
