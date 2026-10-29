// Decompose what gives GB10 its ~90% peak-read efficiency.
// One fixed 512 MiB buffer allocated ONCE, reused across all variants.
// Peak GPU memory footprint: ~512 MiB + negligible overhead.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error L%d: %s\n",__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

constexpr size_t BUF_BYTES = 512ull << 20;  // 512 MiB — fits in DRAM, exceeds L2

// Variant A: scalar uint32 reads, no unroll hint
__global__ void vA_scalar(const unsigned* __restrict__ src, size_t n_words, float* sink){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*(size_t)blockDim.x;
    unsigned acc = 0;
    for(; i<n_words; i+=stride) acc ^= src[i];
    if(acc==0xDEADbeef)*sink=1.f;
}

// Variant B: uint4 vector reads (16 bytes per load), no explicit unroll
__global__ void vB_u4(const uint4* __restrict__ src, size_t n_vec, float* sink){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*(size_t)blockDim.x;
    unsigned acc = 0;
    for(; i<n_vec; i+=stride){ uint4 v=src[i]; acc^=v.x^v.y^v.z^v.w; }
    if(acc==0xDEADbeef)*sink=1.f;
}

// Variant C: uint4 with #pragma unroll 4 to increase MLP
__global__ void vC_u4_unroll4(const uint4* __restrict__ src, size_t n_vec, float* sink){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*(size_t)blockDim.x;
    unsigned acc = 0;
    for(; i+3*stride<n_vec; i+=4*stride){
        uint4 v0=src[i], v1=src[i+stride], v2=src[i+2*stride], v3=src[i+3*stride];
        acc^=v0.x^v0.y^v0.z^v0.w ^ v1.x^v1.y^v1.z^v1.w
           ^ v2.x^v2.y^v2.z^v2.w ^ v3.x^v3.y^v3.z^v3.w;
    }
    for(;i<n_vec;i+=stride){ uint4 v=src[i]; acc^=v.x^v.y^v.z^v.w; }
    if(acc==0xDEADbeef)*sink=1.f;
}

// Variant D: uint4 with ld.nc (L1 non-coherent / evict-first) via inline PTX
__global__ void vD_u4_ldnc(const uint4* __restrict__ src, size_t n_vec, float* sink){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*(size_t)blockDim.x;
    unsigned acc = 0;
    for(; i+3*stride<n_vec; i+=4*stride){
        uint4 a,b,c,d;
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w) : "l"(src+i));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x),"=r"(b.y),"=r"(b.z),"=r"(b.w) : "l"(src+i+stride));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x),"=r"(c.y),"=r"(c.z),"=r"(c.w) : "l"(src+i+2*stride));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.x),"=r"(d.y),"=r"(d.z),"=r"(d.w) : "l"(src+i+3*stride));
        acc^=a.x^a.y^a.z^a.w^b.x^b.y^b.z^b.w^c.x^c.y^c.z^c.w^d.x^d.y^d.z^d.w;
    }
    for(;i<n_vec;i+=stride){ uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(src+i)); acc^=v.x^v.y^v.z^v.w; }
    if(acc==0xDEADbeef)*sink=1.f;
}

// Variant E: uint4 with cp.async.bulk prefetch hint (if supported on sm_121)
// Fallback: same as C but using __builtin_prefetch equivalent via cuda::memcpy_async would be overkill.
// Instead: try L2 persistence hint via cudaStreamSetAttribute — too complex for this probe.
// Skip; use variant F instead.

// Variant F: 2×uint4 interleaved (32 bytes per thread iteration) for maximum MLP without unroll pragma
__global__ void vF_u4x2(const uint4* __restrict__ src, size_t n_vec, float* sink){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*(size_t)blockDim.x;
    unsigned acc = 0;
    for(; i+stride<n_vec; i+=2*stride){
        uint4 v0=src[i], v1=src[i+stride];
        acc^=v0.x^v0.y^v0.z^v0.w ^ v1.x^v1.y^v1.z^v1.w;
    }
    if(i<n_vec){ uint4 v=src[i]; acc^=v.x^v.y^v.z^v.w; }
    if(acc==0xDEADbeef)*sink=1.f;
}

struct R { const char* name; int blocks, threads; double gb_s; };

int main(){
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  SM=%d\n", prop.name, prop.multiProcessorCount);

    // ONE allocation, reused everywhere
    uint4* d_src = nullptr; float* d_sink = nullptr;
    CK(cudaMalloc(&d_src, BUF_BYTES));
    CK(cudaMalloc(&d_sink, sizeof(float)));
    // Initialize with pattern (avoid all-zeros which some HW may optimize)
    std::vector<uint4> h(BUF_BYTES/sizeof(uint4));
    for(size_t i=0;i<h.size();++i) h[i]={(unsigned)i,0,0,0};
    CK(cudaMemcpy(d_src,h.data(),BUF_BYTES,cudaMemcpyHostToDevice));
    h.clear(); h.shrink_to_fit(); // release host copy

    size_t n_words = BUF_BYTES/4;
    size_t n_vec   = BUF_BYTES/16;
    constexpr int REPS = 5;
    std::vector<R> results;

    auto bench = [&](const char* name, int blocks, int threads, auto launch_fn){
        launch_fn(blocks, threads); // warmup
        CK(cudaDeviceSynchronize());
        double total_ms=0;
        for(int r=0;r<REPS;++r){
            cudaEvent_t s_,e_; CK(cudaEventCreate(&s_)); CK(cudaEventCreate(&e_));
            CK(cudaEventRecord(s_)); launch_fn(blocks,threads); CK(cudaEventRecord(e_));
            CK(cudaEventSynchronize(e_));
            float ms=0; CK(cudaEventElapsedTime(&ms,s_,e_));
            total_ms+=ms;
            CK(cudaEventDestroy(s_)); CK(cudaEventDestroy(e_));
        }
        double gbs=(double)BUF_BYTES/(total_ms/REPS*1e-3)/1e9;
        results.push_back({name,blocks,threads,gbs});
    };

    // Best configs from peak_bw probe: ~48-384 blocks, 256-1024 threads
    for(int threads : {256, 512, 1024}){
        int mult = (threads>=1024)?2:(threads>=512)?4:8; // scale down blocks for more threads
        int blocks = prop.multiProcessorCount * mult;
        bench("A_scalar_u32", blocks, threads, [&](int b,int t){ vA_scalar<<<b,t>>>((const unsigned*)d_src,n_words,d_sink); });
        bench("B_u4_plain",   blocks, threads, [&](int b,int t){ vB_u4<<<b,t>>>(d_src,n_vec,d_sink); });
        bench("C_u4_unroll4", blocks, threads, [&](int b,int t){ vC_u4_unroll4<<<b,t>>>(d_src,n_vec,d_sink); });
        bench("D_u4_ldnc",    blocks, threads, [&](int b,int t){ vD_u4_ldnc<<<b,t>>>(d_src,n_vec,d_sink); });
        bench("F_u4x2_interleave", blocks, threads, [&](int b,int t){ vF_u4x2<<<b,t>>>(d_src,n_vec,d_sink); });
    }

    std::sort(results.begin(),results.end(),[](const R&a,const R&b){return a.gb_s>b.gb_s;});
    printf("\n%-24s %6s %6s %8s %7s\n","variant","blocks","threads","GB/s","%peak");
    for(auto&r:results)
        printf("%-24s %6d %6d %8.1f %6.0f%%\n",r.name,r.blocks,r.threads,r.gb_s,r.gb_s/273.0*100);

    CK(cudaFree(d_src)); CK(cudaFree(d_sink));
    return 0;
}
