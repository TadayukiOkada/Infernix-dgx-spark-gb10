// probe_async_copy.cu — GB10 feature probe: asynchronous global->shared copy.
// cp.async (LDGSTS) is Ampere+ (sm_80+). Confirmed present on GB10 (sm_121).
// Measures correctness AND sustained copy bandwidth through shared memory.
#include "probe_common.h"
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Grid-stride tile copy: each block moves BLOCK*4 floats global->shared via
// cp.async (16B/thread) then writes them back. Large grid so the launch has
// enough work to measure LDGSTS throughput, not single-block latency.
template <int BLOCK>
__global__ void k_cpasync(const float* src, float* dst, int n) {
    __shared__ float smem[BLOCK*4];
    const int tiles = (n + BLOCK*4 - 1) / (BLOCK*4);
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int base = tile*(BLOCK*4) + threadIdx.x*4;
        if (base + 3 < n) {
            uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[threadIdx.x*4]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(saddr), "l"(src + base));
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (base + 3 < n) {
            for (int j=0;j<4;++j) dst[base+j] = smem[threadIdx.x*4+j];
        }
        __syncthreads();
    }
}

int main() {
    const int nc = 4096;
    float *hc=new float[nc], *ho=new float[nc];
    for (int i=0;i<nc;++i) hc[i]=(float)i;
    float *dc,*doc_; cudaMalloc(&dc,nc*4); cudaMalloc(&doc_,nc*4);
    cudaMemcpy(dc,hc,nc*4,cudaMemcpyHostToDevice);
    k_cpasync<256><<<48*4,256>>>(dc,doc_,nc);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_async_copy RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,doc_,nc*4,cudaMemcpyDeviceToHost);
    bool ok = true; for (int i=0;i<nc;++i) if (ho[i]!=(float)i) ok=false;
    if (!ok){ std::printf("probe_async_copy MISMATCH\n"); return 3; }
    std::printf("probe_async_copy OK\n");
    cudaFree(dc); cudaFree(doc_);
    // Throughput on a 64 MB working set (> L2) => read+write DRAM traffic.
    const int n = 16 << 20;   // 16M floats = 64 MB
    float *da,*do_; cudaMalloc(&da,(size_t)n*4); cudaMalloc(&do_,(size_t)n*4);
    cudaMemset(da,0x3f,(size_t)n*4);
    auto body=[&]{ k_cpasync<256><<<48*8,256>>>(da,do_,n); };
    probe_time_bw("cp.async_perthread", body, 2.0*n*4);   // read + write
    cudaFree(da); cudaFree(do_);
    return 0;
}
