// probe_lookup.cu — GB10 feature probe: lookup-table / partial-sum dot product.
// For small fixed K and heavily reused operand pairs, precompute a partial-sum
// table over nibble pairs and combine via table lookups instead of MACs.
// Valid only when value range is small (4-bit signed here). Compare vs dp4a.
#include "probe_common.h"

// A 256-entry LUT: lut[(a_nib<<4)|b_nib] = a*b for signed 4-bit a,b in [-8,7].
__constant__ int c_lut[256];

__global__ void k_lookup(const unsigned* A, const unsigned* B, int* out, int rows, int words, int stride) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    const unsigned* Ar=A+(size_t)r*stride; const unsigned* Br=B+(size_t)r*stride;
    int acc=0;
    for(int w=0;w<words;++w){
        unsigned a=Ar[w], b=Br[w];
        #pragma unroll
        for(int k=0;k<8;++k){
            int idx = (int)(((a>>(k*4))&0xF)<<4 | ((b>>(k*4))&0xF));
            acc += c_lut[idx];
        }
    }
    out[r]=acc;
}
__global__ void k_dp4a_s4(const unsigned* A, const unsigned* B, int* out, int rows, int words, int stride) {
    // Same 4-bit data but multiply via shifts/masks (no LUT) for comparison.
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    const unsigned* Ar=A+(size_t)r*stride; const unsigned* Br=B+(size_t)r*stride;
    int acc=0;
    for(int w=0;w<words;++w){
        unsigned a=Ar[w], b=Br[w];
        #pragma unroll
        for(int k=0;k<8;++k){
            int sh=k*4; int av=(int)((a>>sh)&0xF); if(av>=8)av-=16;
            int bv=(int)((b>>sh)&0xF); if(bv>=8)bv-=16; acc+=av*bv;
        }
    }
    out[r]=acc;
}

int main(){
    // build LUT host-side
    int lut[256]; for(int a=0;a<16;++a)for(int b=0;b<16;++b){int av=a>=8?a-16:a,bv=b>=8?b-16:b;lut[a*16+b]=av*bv;}
    cudaMemcpyToSymbol(c_lut,lut,sizeof(lut));
    const int rows=8192,K=64,words=K/8,stride=((words+3)/4)*4;
    unsigned* hA=new unsigned[rows*stride]; unsigned* hB=new unsigned[rows*stride];
    std::uint64_t s=7; for(int i=0;i<rows*stride;++i){s=s*6364136223846793005ull+1442695040888963407ull;hA[i]=(unsigned)(s>>32);} 
    for(int i=0;i<rows*stride;++i){s=s*6364136223846793005ull+1442695040888963407ull;hB[i]=(unsigned)(s>>32);}
    unsigned *dA,*dB; int* dO; cudaMalloc(&dA,(size_t)rows*stride*4);cudaMalloc(&dB,(size_t)rows*stride*4);cudaMalloc(&dO,(size_t)rows*4);
    cudaMemcpy(dA,hA,(size_t)rows*stride*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB,(size_t)rows*stride*4,cudaMemcpyHostToDevice);
    auto bl=[&]{ k_lookup<<<(rows+255)/256,256>>>(dA,dB,dO,rows,words,stride); };
    auto bd=[&]{ k_dp4a_s4<<<(rows+255)/256,256>>>(dA,dB,dO,rows,words,stride); };
    bl(); cudaDeviceSynchronize(); bd(); cudaDeviceSynchronize();
    std::printf("probe_lookup OK\n");
    const double ops=2.0*rows*K;
    probe_time_and_report("lookup_partialsum", bl, ops);
    probe_time_and_report("int4_shiftmul", bd, ops);
    cudaFree(dA);cudaFree(dB);cudaFree(dO);
    return 0;
}
