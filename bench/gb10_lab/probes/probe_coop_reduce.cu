// probe_coop_reduce.cu — GB10 feature probe: cooperative-groups block reduction.
#include "probe_common.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>   // cg::reduce / cg::plus live here in CUDA 13
namespace cg = cooperative_groups;

__global__ void k_coop(const float* a, float* out, int n) {
    cg::thread_block block = cg::this_thread_block();
    // CUDA 13 cg::reduce supports tile groups only (not a full thread_block);
    // full-block cg::reduce triggers a static_assert. Warp-tiled reduce is the
    // documented supported form on this target.
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    v = cg::reduce(tile, v, cg::plus<float>());
    if ((threadIdx.x & 31) == 0) out[blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5)] = v;
}

int main() {
    // Working set large enough to exceed L2 (24 MiB): 384 blocks x 1024 threads
    // = 393216 floats ~ 1.5 MB reads but issued across 12288 warps to expose
    // cg::reduce instruction throughput (shuffles), not memory bandwidth.
    const int blocks = 384; const int threads = 1024;
    const int n = blocks * threads; const int warps = blocks * (threads / 32);
    float *ha=new float[n], *ho=new float[warps];
    for (int i=0;i<n;++i) ha[i]=1.f;
    float *da,*do_; cudaMalloc(&da,(size_t)n*4); cudaMalloc(&do_,(size_t)warps*4);
    cudaMemcpy(da,ha,(size_t)n*4,cudaMemcpyHostToDevice);
    k_coop<<<blocks,threads>>>(da,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_coop_reduce RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,(size_t)warps*4,cudaMemcpyDeviceToHost);
    // Each warp reduces 32 lanes of value 1.0 -> 32.0 per warp partial.
    bool ok=true; for(int w=0;w<warps;++w) if (ho[w]!=32.0f) ok=false;
    if(!ok){ std::printf("probe_coop_reduce MISMATCH\n"); return 3; }
    std::printf("probe_coop_reduce OK\n");
    auto body=[&]{ k_coop<<<blocks,threads>>>(da,do_,n); };
    // ops: one add per lane during the warp reduce => n adds.
    probe_time_and_report("coop_groups_reduce", body, (double)n);
    cudaFree(da); cudaFree(do_);
    return 0;
}
