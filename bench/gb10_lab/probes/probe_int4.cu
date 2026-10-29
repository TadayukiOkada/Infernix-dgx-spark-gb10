// probe_int4.cu — GB10 feature probe: INT4 packed storage + unpack + MAC.
// Two 4-bit signed values per byte. Unpack via shift/mask, multiply into INT32.
// Benchmark the packed-decode path against the same-work INT8 elementwise baseline.
#include "probe_common.h"

__global__ void k_int4(const unsigned* packedA, const unsigned* packedB, int* out, int nwords) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nwords) return;
    unsigned a=packedA[i], b=packedB[i];
    int acc=0;
    // 8 nibbles per word: extract signed 4-bit lanes and multiply pairwise.
    #pragma unroll
    for (int k=0;k<8;++k) {
        int sh=k*4;
        int av=(int)((a>>sh)&0xF); if(av>=8) av-=16;   // sign-extend 4-bit
        int bv=(int)((b>>sh)&0xF); if(bv>=8) bv-=16;
        acc += av*bv;
    }
    out[i]=acc;
}

int main() {
    const int nwords=1<<19;
    unsigned *ha=new unsigned[nwords],*hb=new unsigned[nwords];
    for(int i=0;i<nwords;++i){ha[i]=0x77118877u; hb[i]=0x11223344u;}
    unsigned *da,*db; int* do_; cudaMalloc(&da,nwords*4);cudaMalloc(&db,nwords*4);cudaMalloc(&do_,nwords*4);
    cudaMemcpy(da,ha,nwords*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,nwords*4,cudaMemcpyHostToDevice);
    k_int4<<<(nwords+255)/256,256>>>(da,db,do_,nwords);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){std::printf("probe_int4 RUNTIME_FAIL %s\n",cudaGetErrorString(e));return 2;}
    std::printf("probe_int4 OK\n");
    auto body=[&]{ k_int4<<<(nwords+255)/256,256>>>(da,db,do_,nwords); };
    const double ops = 2.0*8.0*nwords; // 8 MACs per word, 2K convention
    probe_time_and_report("int4_unpack_mac", body, ops);
    cudaFree(da);cudaFree(db);cudaFree(do_);
    return 0;
}
