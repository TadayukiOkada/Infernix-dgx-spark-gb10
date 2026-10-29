// probe_popc.cu — GB10 feature probe: population count + XNOR binary dot product.
// Binary dot via popcount(~a ^ b) over packed bits. Only valid for binary-valued inputs.
#include "probe_common.h"
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void k_popc(const unsigned* a, const unsigned* b, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned xnor = ~(a[i] ^ b[i]);     // bitwise XNOR
    out[i] = __popc(xnor);              // number of matching bits == binary dot contribution
    // 64-bit popcount variant to confirm intrinsic family on this target.
    unsigned long long w = ((unsigned long long)a[i] << 32) | b[i];
    out[i] += __popcll(w);
}

int main() {
    const int n = 1 << 20;
    unsigned *ha=new unsigned[n], *hb=new unsigned[n], *ho=new unsigned[n];
    for (int i=0;i<n;++i){ ha[i]=0xF0F0F0F0u; hb[i]=0xF0F0F0F0u; }
    unsigned *da,*db,*do_; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    k_popc<<<(n+255)/256,256>>>(da,db,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_popc RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,n*4,cudaMemcpyDeviceToHost);
    unsigned long long sum=0; for(int i=0;i<n;++i) sum+=ho[i];
    std::printf("probe_popc OK sum=%llu\n", sum);
    // Two popcounts per element (32-bit + 64-bit). Count as 2 bit-count ops.
    auto body=[&]{ k_popc<<<(n+255)/256,256>>>(da,db,do_,n); };
    probe_time_and_report("popc_xnor_binary", body, 2.0*n);
    return 0;
}
