// probe_imma.cu — GB10 feature probe: INT8 Tensor Core MMA (mma.sync.s8 -> IMMA).
// Critical: cuBLASLt exposes no INT8 GEMM here, but that does NOT prove the
// hardware lacks INT8 MMA. Probe the instruction directly via inline PTX.
// Layout: m16 n8 k32, row.col, .s32.s8.s8.s32. All A,B bytes = 1 => every D
// entry = 32 (sum over k=32 of 1*1), independent of fragment mapping.
#include "probe_common.h"

__global__ void k_imma(int* out) {
    // One warp performs a single INT8 MMA. Fragment registers (packed 4x s8):
    unsigned a0,a1,a2,a3, b0,b1;
    int c0=0,c1=0,c2=0,c3=0;
    a0=a1=a2=a3=0x01010101u;   // 16x32 A, all ones
    b0=b1=0x01010101u;         // 32x8 B, all ones
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    if (blockIdx.x == 0 && threadIdx.x < 32) out[threadIdx.x] = c0+c1+c2+c3;
}

int main() {
    int* dout=nullptr; cudaMalloc(&dout, 32*sizeof(int));
    cudaMemset(dout,0,32*sizeof(int));
    k_imma<<<1,32>>>(dout);
    cudaError_t e=cudaDeviceSynchronize();
    if (e!=cudaSuccess){ std::printf("probe_imma RUNTIME_FAIL %s\n", cudaGetErrorString(e)); return 2; }
    int h[32]; cudaMemcpy(h,dout,32*sizeof(int),cudaMemcpyDeviceToHost);
    bool ok=true; for (int i=0;i<32;++i) if (h[i]!=128) ok=false; // 4 outputs * 32 each
    if (!ok){ std::printf("probe_imma RESULT_MISMATCH h0=%d\n", h[0]); return 3; }
    std::printf("probe_imma OK\n");
    cudaFree(dout);
    // Throughput microbench: 48 SMs x 8 blocks x 1 warp, each warp issues one
    // m16n8k32 INT8 MMA = 16*8*32 MACs = 2*16*8*32 ops. Grid kept resident so
    // this measures sustained Tensor Core issue, not a single launch.
    int* d2=nullptr; cudaMalloc(&d2, 32*sizeof(int));
    auto body2=[&]{ k_imma<<<48*8,32>>>(d2); };
    const double ops = 48.0*8*(16*8*32*2);
    probe_time_and_report("probe_imma_throughput", body2, ops);
    cudaFree(d2);
    return 0;
}
