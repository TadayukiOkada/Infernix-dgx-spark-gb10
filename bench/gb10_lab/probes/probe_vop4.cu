// probe_vop4.cu — GB10 feature probe: packed byte arithmetic intrinsics.
// These are arithmetic building blocks, not general packed dot products.
#include "probe_common.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_vop(const unsigned* a, const unsigned* b, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned r = 0;
    r += __vadd4(a[i], b[i]);
    r += __vsub4(a[i], b[i]);
    r += __vabsdiffu4(a[i], b[i]);   // CUDA13 spelling (unsigned 4-lane)
    r += __vmaxu4(a[i], b[i]);
    r += __vminu4(a[i], b[i]);
    r += __vavgu4(a[i], b[i]);       // averaging, carries rounding semantics
    out[i] = r;
}

int main() {
    const int n = 1 << 20;
    unsigned *ha=new unsigned[n], *hb=new unsigned[n], *ho=new unsigned[n];
    for (int i=0;i<n;++i){ ha[i]=0x08080808u; hb[i]=0x03030303u; }
    unsigned *da,*db,*do_; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    k_vop<<<(n+255)/256,256>>>(da,db,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_vop4 RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,n*4,cudaMemcpyDeviceToHost);
    unsigned long long sum=0; for(int i=0;i<n;++i) sum+=ho[i];
    std::printf("probe_vop4 OK sum=%llu\n", sum);
    // Six packed-byte ops per element * 4 lanes = 24 lane-ops.
    auto body=[&]{ k_vop<<<(n+255)/256,256>>>(da,db,do_,n); };
    probe_time_and_report("vop4_byte", body, 24.0*n);
    return 0;
}
