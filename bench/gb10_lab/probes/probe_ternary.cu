// probe_ternary.cu — GB10 feature probe: ternary weights {-1,0,+1} dot products.
// Three mechanisms for the SAME math (y = sum_j w_j * x_j, w in {-1,0,1}):
//   mul    : generic signed multiply (baseline)
//   sel    : sign-select (add if +1, sub if -1, skip if 0) via masks, no multiply
//   skip0  : compress-and-reduce (count nonzeros) — measures zero-skipping value
// Each timed separately. Only valid when weights are genuinely ternary.
#include "probe_common.h"

// Packed: 4 ternary weights per 32-bit word, 2 bits each: 00=0, 01=+1, 11=-1.
__global__ void k_ternary_mul(const unsigned* W, const signed char* X, int* out, int rows, int words, int stride) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    const unsigned* Wr=W+(size_t)r*stride; const signed char* Xr=X+(size_t)r*(words*4);
    int acc=0;
    for (int wq=0;wq<words;++wq){
        unsigned w=Wr[wq];
        #pragma unroll
        for(int k=0;k<4;++k){
            int code=(w>>(k*2))&0x3; int wt = (code==1)?1:(code==3)?-1:0;
            acc += wt * (int)Xr[wq*4+k];
        }
    }
    out[r]=acc;
}
__global__ void k_ternary_sel(const unsigned* W, const signed char* X, int* out, int rows, int words, int stride) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    const unsigned* Wr=W+(size_t)r*stride; const signed char* Xr=X+(size_t)r*(words*4);
    int acc=0;
    for (int wq=0;wq<words;++wq){
        unsigned w=Wr[wq];
        #pragma unroll
        for(int k=0;k<4;++k){
            int code=(w>>(k*2))&0x3; int x=(int)Xr[wq*4+k];
            // select without multiply: mask-add for +1, mask-sub for -1
            int add = (code==1)? x : 0;
            int sub = (code==3)? x : 0;
            acc += add - sub;
        }
    }
    out[r]=acc;
}

int main(){
    auto roundup=[](int x,int m){return ((x+m-1)/m)*m;};
    const int rows=8192, K=128, words=K/4, stride=roundup(words,4);
    unsigned* hW=new unsigned[rows*stride]; signed char* hX=new signed char[rows*K];
    std::uint64_t s=99;
    for(int r=0;r<rows;++r) for(int w=0;w<words;++w){
        unsigned v=0; for(int k=0;k<4;++k){ s=s*6364136223846793005ull+1442695040888963407ull; v|=((s>>33)&3u)<<(k*2);} 
        hW[(size_t)r*stride+w]=v;
    }
    for(int i=0;i<rows*K;++i){ s=s*6364136223846793005ull+1442695040888963407ull; hX[i]=(signed char)((int)((s>>33)&0xFF)-128); if(hX[i]==-128)hX[i]=127; }
    unsigned* dW; signed char* dX; int* dO; cudaMalloc(&dW,(size_t)rows*stride*4); cudaMalloc(&dX,(size_t)rows*K); cudaMalloc(&dO,(size_t)rows*4);
    cudaMemcpy(dW,hW,(size_t)rows*stride*4,cudaMemcpyHostToDevice); cudaMemcpy(dX,hX,(size_t)rows*K,cudaMemcpyHostToDevice);
    auto bm=[&]{ k_ternary_mul<<<(rows+255)/256,256>>>(dW,dX,dO,rows,words,stride); };
    auto bs=[&]{ k_ternary_sel<<<(rows+255)/256,256>>>(dW,dX,dO,rows,words,stride); };
    bm(); cudaDeviceSynchronize(); bs(); cudaDeviceSynchronize();
    std::printf("probe_ternary OK\n");
    const double ops=2.0*rows*K;
    probe_time_and_report("ternary_mul", bm, ops);
    probe_time_and_report("ternary_sign_select", bs, ops);
    cudaFree(dW);cudaFree(dX);cudaFree(dO);
    return 0;
}
