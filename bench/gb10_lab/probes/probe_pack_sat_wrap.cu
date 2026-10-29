// probe_pack_sat_wrap.cu — GB10 feature probe: saturating vs wrapping packed bytes.
// Two DIFFERENT numerical contracts; benchmarked separately, never compared as equal.
#include "probe_common.h"

__global__ void k_wrap(const unsigned* a, const unsigned* b, unsigned* out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned acc=a[i];
    #pragma unroll 8
    for (int r=0;r<8;++r) acc = __vadd4(acc, b[i]);   // wrapping 4-lane byte add
    out[i]=acc;
}
__global__ void k_sat(const unsigned* a, const unsigned* b, unsigned* out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned acc=a[i];
    #pragma unroll 8
    for (int r=0;r<8;++r) acc = __vaddus4(acc, b[i]); // unsigned saturating 4-lane add
    out[i]=acc;
}

int main() {
    const int n=1<<20;
    unsigned *ha=new unsigned[n],*hb=new unsigned[n];
    for(int i=0;i<n;++i){ha[i]=0x01010101u; hb[i]=0x02020202u;}
    unsigned *da,*db,*do_; cudaMalloc(&da,n*4);cudaMalloc(&db,n*4);cudaMalloc(&do_,n*4);
    cudaMemcpy(da,ha,n*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,n*4,cudaMemcpyHostToDevice);
    auto bw=[&]{ k_wrap<<<(n+255)/256,256>>>(da,db,do_,n); };
    auto bs=[&]{ k_sat<<<(n+255)/256,256>>>(da,db,do_,n); };
    bw(); cudaDeviceSynchronize(); bs(); cudaDeviceSynchronize();
    std::printf("probe_pack_sat_wrap OK\n");
    // ops: each lane op counts as 1 add; 4 lanes * 8 iters * n.
    const double ops = 4.0*8*n;
    probe_time_and_report("pack_wrap_vadd4", bw, ops);
    probe_time_and_report("pack_sat_vaddus4", bs, ops);
    cudaFree(da);cudaFree(db);cudaFree(do_);
    return 0;
}
