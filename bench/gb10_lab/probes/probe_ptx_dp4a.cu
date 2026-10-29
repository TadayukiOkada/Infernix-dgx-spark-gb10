// probe_ptx_dp4a.cu — GB10 feature probe: inline PTX dp4a (isolated experimental branch).
// Portability risk is explicitly flagged; kept separate from primary benchmarks.
#include "probe_common.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_ptx_dp4a(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = 0;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c) : "r"(a[i]), "r"(b[i]), "r"(c));
    out[i] = c;
}

int main() {
    const int n = 1 << 20;
    int *ha=new int[n], *hb=new int[n], *ho=new int[n];
    for (int i=0;i<n;++i){ ha[i]=0x02020202; hb[i]=0x01010101; }
    int *da,*db,*do_; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    k_ptx_dp4a<<<(n+255)/256,256>>>(da,db,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_ptx_dp4a RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,n*4,cudaMemcpyDeviceToHost);
    long long sum=0; for(int i=0;i<n;++i) sum+=ho[i];
    std::printf("probe_ptx_dp4a OK sum=%lld\n", sum);
    // One inline-PTX dp4a per element = 4 MACs = 8 ops.
    auto body=[&]{ k_ptx_dp4a<<<(n+255)/256,256>>>(da,db,do_,n); };
    probe_time_and_report("ptx_dp4a_inline", body, 8.0*n);
    return 0;
}
