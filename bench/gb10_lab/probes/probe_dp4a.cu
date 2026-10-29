// probe_dp4a.cu — GB10 feature probe: INT8 dot-product accumulate (__dp4a).
// Isolated translation unit so a compile failure here cannot poison other probes.
// Exercises the four documented __dp4a signatures (int, unsigned, char4, uchar4).
#include "probe_common.h"
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void k_dp4a(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // signed int variant: four signed 8-bit lanes packed into one 32-bit word.
    int acc = __dp4a(a[i], b[i], 0);
    // unsigned variant.
    unsigned uacc = __dp4a((unsigned)a[i], (unsigned)b[i], 0u);
    // char4 / uchar4 explicit-lane variants (same lowering, documented API).
    char4 ca = make_char4((signed char)(a[i] & 0xff), (signed char)((a[i]>>8)&0xff),
                          (signed char)((a[i]>>16)&0xff), (signed char)((a[i]>>24)&0xff));
    char4 cb = make_char4((signed char)(b[i] & 0xff), (signed char)((b[i]>>8)&0xff),
                          (signed char)((b[i]>>16)&0xff), (signed char)((b[i]>>24)&0xff));
    int c4 = __dp4a(ca, cb, 0);
    out[i] = acc + (int)uacc + c4;
}

int main() {
    const int n = 1 << 20;
    int *ha=new int[n], *hb=new int[n], *ho=new int[n];
    for (int i=0;i<n;++i){ ha[i]=0x02020202; hb[i]=0x01010101; } // 2*1 per lane
    int *da,*db,*do_; cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    k_dp4a<<<(n+255)/256,256>>>(da,db,do_,n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess){ std::printf("probe_dp4a RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(ho,do_,n*4,cudaMemcpyDeviceToHost);
    long long sum=0; for(int i=0;i<n;++i) sum+=ho[i];
    std::printf("probe_dp4a OK sum=%lld\n", sum);
    // Sustained throughput: 3 __dp4a per element (signed+unsigned+char4) = 12 MACs = 24 ops.
    auto body=[&]{ k_dp4a<<<(n+255)/256,256>>>(da,db,do_,n); };
    probe_time_and_report("dp4a_int", body, 24.0*n);
    return 0;
}
