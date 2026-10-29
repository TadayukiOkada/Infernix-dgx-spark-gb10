// probe_dp2a.cu — GB10 feature probe: packed halfword dot-accumulate (__dp2a_lo/_hi).
#include "probe_common.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_dp2a(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lo = __dp2a_lo(a[i], b[i], 0);   // low two 16-bit lanes
    int hi = __dp2a_hi(a[i], b[i], 0);   // high two 16-bit lanes
    out[i] = lo + hi;
}

int main() {
    const int n = 1 << 20;
    int *ha=new int[n], *hb=new int[n], *ho=new int[n];
    for (int i=0;i<n;++i){ ha[i]=0x00030002; hb[i]=0x00040005; } // lanes 2,3 x 5,4
    int *da,*db,*do_; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    k_dp2a<<<(n+255)/256,256>>>(da,db,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_dp2a RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,n*4,cudaMemcpyDeviceToHost);
    long long sum=0; for(int i=0;i<n;++i) sum+=ho[i];
    std::printf("probe_dp2a OK sum=%lld\n", sum);
    // Two __dp2a (lo+hi) per element = 2 halfword-MACs*2 lanes = 4 MACs = 8 ops.
    auto body=[&]{ k_dp2a<<<(n+255)/256,256>>>(da,db,do_,n); };
    probe_time_and_report("dp2a", body, 8.0*n);
    return 0;
}
