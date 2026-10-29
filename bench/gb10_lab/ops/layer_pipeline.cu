// layer_pipeline.cu — Layer 5: one decoder layer, one decode token, end to end.
//
// Earlier ops measure each primitive in isolation; this composes them the way a
// real engine does, for a single autoregressive token through every layer, and
// reports the resulting per-token latency and effective weight-streaming GB/s.
// That number is what decode throughput is really bounded by on GB10: streaming
// all the layer weights from DRAM once per token. Projections use cuBLAS bf16
// GemmEx at m=1 (the honest strong baseline); norms, RoPE, GQA attention and the
// MLP non-linearity reuse the same custom-kernel patterns that their dedicated
// ops validate against independent references, so stage-level correctness carries
// over. Here we additionally confirm the composition yields finite outputs.
#include "gb10_lab/common.h"

namespace {

#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); \
    if(s_!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS error %d @ %s:%d\n",(int)s_,__FILE__,__LINE__); std::exit(1);} } while(0)

constexpr int RN_BLOCK   = 256;
constexpr int ATTN_BLOCK = 256;
constexpr float kBase    = 10000.0f;
constexpr float kEps     = 1e-6f;

// ---- RMSNorm, one block per row of width H, bf16 in -> bf16 out ------------
template <int BLOCK>
__global__ void k_rms(const uint4* __restrict__ xr_in, const uint4* __restrict__ g_in,
                      uint4* __restrict__ yr_out, int rows, int H, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int hv   = H / 8;
    const uint4* xr = xr_in + static_cast<size_t>(row) * hv;
    uint4* yr      = yr_out + static_cast<size_t>(row) * hv;
    const uint4* gr = g_in;

    double ssq = 0.0;                                   // pass 1: stream via L1
    for (int c = threadIdx.x; c < hv; c += BLOCK) {
        BF16x8 v = u4_to_bf16(xr[c]);
        #pragma unroll
        for (int e = 0; e < 8; ++e) { float f = bf16_f(v.x[e]); ssq += (double)f * f; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ssq += (double)__shfl_down_sync(0xffffffffu, (float)ssq, off);
    __shared__ float smem[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = (float)ssq;
    __syncthreads();
    // Thread 0 folds the warp partials and stores the INVERSE back into smem[0]
    // so the broadcast below reads rsqrt(mean+eps), not a raw partial sum.
    if (threadIdx.x == 0) {
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < BLOCK/32; ++i) acc += smem[i];
        smem[0] = rsqrtf(acc / H + eps);
    }
    __syncthreads();
    const float inv = smem[0];                            // broadcast inverse
    for (int c = threadIdx.x; c < hv; c += BLOCK) {     // pass 2: re-read from L1
        BF16x8 v = u4_to_bf16(xr[c]);
        BF16x8 g = u4_to_bf16(gr[c]);
        BF16x8 o = u4_to_bf16(make_uint4(0,0,0,0));
        #pragma unroll
        for (int e = 0; e < 8; ++e) o.x[e] = f_bf16(bf16_f(v.x[e]) * inv * bf16_f(g.x[e]));
        yr[c] = bf16_to_u4(o);
    }
}

// ---- RoPE applied in place to `rows` head-vectors of length d --------------
__global__ void k_rope_inplace(uint4* __restrict__ x, int rows, int d, int d_half,
                               const float* __restrict__ inv_freq, int pos) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int d_vec = d / 8;
    uint4* xr = x + static_cast<size_t>(row) * d_vec;
    for (int c = threadIdx.x; c < d_vec; c += blockDim.x) {
        uint4 v = xr[c];
        BF16x8 ob = u4_to_bf16(v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = c * 8 + j;
            const bool first = (i < d_half);
            const int m = first ? i : (i - d_half);
            const float xv = bf16_f(ob.x[j]);
            const int partner_linear = (c * 8 + j) ^ d_half;
            const float xp = bf16_f(u4_to_bf16(xr[partner_linear >> 3]).x[partner_linear & 7]);
            float cs, sn;
            __sincosf((float)pos * inv_freq[m], &sn, &cs);
            ob.x[j] = f_bf16(first ? (xv * cs + xp * sn) : (-xv * cs + xp * sn));
        }
        xr[c] = bf16_to_u4(ob);
    }
}

// ---- GQA single-token attention over a KV cache ----------------------------
template <int BLOCK>
__global__ void k_attn_gqa(const __nv_bfloat16* __restrict__ Q,     // [heads, d]
                           const __nv_bfloat16* __restrict__ K,     // [seq, kv_heads, d]
                           const __nv_bfloat16* __restrict__ V,     // [seq, kv_heads, d]
                           float scale, int seq, int d, int kv_heads,
                           float* __restrict__ O) {                  // [heads, d]
    const int h    = blockIdx.x;
    const int kvh  = h % kv_heads;
    const int tid  = threadIdx.x;
    extern __shared__ float smem[];      // [seq] scores + max at smem[seq]
    float* s = smem;
    const __nv_bfloat16* Kh = K + static_cast<size_t>(kvh) * d;
    const __nv_bfloat16* Vh = V + static_cast<size_t>(kvh) * d;
    const __nv_bfloat16* Qh = Q + static_cast<size_t>(h) * d;

    for (int i = tid; i < seq; i += BLOCK) {
        float acc = 0.f;
        #pragma unroll 4
        for (int j = 0; j < d; ++j)
            acc += bf16_f(Qh[j]) * bf16_f(Kh[static_cast<size_t>(i)*d + j]);
        s[i] = acc * scale;
    }
    __syncthreads();
    float m_local = -INFINITY;
    for (int i = tid; i < seq; i += BLOCK) m_local = fmaxf(m_local, s[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m_local = fmaxf(m_local, __shfl_down_sync(0xffffffffu, m_local, off));
    __shared__ float warp_max[BLOCK/32];
    if ((tid & 31) == 0) warp_max[tid >> 5] = m_local;
    __syncthreads();
    if (tid == 0) {
        float mm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < BLOCK/32; ++w) mm = fmaxf(mm, warp_max[w]);
        smem[seq] = mm;
    }
    __syncthreads();
    const float m_row = smem[seq];
    float l_local = 0.f;
    for (int i = tid; i < seq; i += BLOCK) l_local += __expf(s[i] - m_row);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        l_local += __shfl_down_sync(0xffffffffu, l_local, off);
    __shared__ float warp_sum[BLOCK/32];
    if ((tid & 31) == 0) warp_sum[tid >> 5] = l_local;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float ss = 0.f;
        #pragma unroll
        for (int w = 0; w < BLOCK/32; ++w) ss += warp_sum[w];
        s_inv = 1.f / ss;
    }
    __syncthreads();
    const float inv_l = s_inv;
    if (tid < d) {
        float acc = 0.f;
        for (int i = 0; i < seq; ++i)
            acc += __expf(s[i] - m_row) * inv_l * bf16_f(Vh[static_cast<size_t>(i)*d + tid]);
        O[h * d + tid] = acc;
    }
}

// ---- SiLU(gate) * up: halves of a [1, 2*inter] buffer -> [1, inter] --------
__global__ void k_silu_mul(const uint4* __restrict__ gu_in, uint4* __restrict__ out,
                           int inter_vec) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= inter_vec) return;
    BF16x8 gate = u4_to_bf16(gu_in[c]);
    BF16x8 up   = u4_to_bf16(gu_in[c + inter_vec]);
    BF16x8 o = u4_to_bf16(make_uint4(0,0,0,0));
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        float g = bf16_f(gate.x[e]);
        o.x[e] = f_bf16((g / (1.f + __expf(-g))) * bf16_f(up.x[e]));
    }
    out[c] = bf16_to_u4(o);
}

// ---- in-place residual add x += h over rows of width H ---------------------
template <int BLOCK>
__global__ void k_add(uint4* __restrict__ x, const uint4* __restrict__ h, int rows, int H) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int hv = H / 8;
    uint4* xr = x + static_cast<size_t>(row) * hv;
    const uint4* hr = h + static_cast<size_t>(row) * hv;
    for (int c = threadIdx.x; c < hv; c += BLOCK) {
        BF16x8 xv = u4_to_bf16(xr[c]);
        BF16x8 hvv = u4_to_bf16(hr[c]);
        #pragma unroll
        for (int e = 0; e < 8; ++e) xv.x[e] = f_bf16(bf16_f(xv.x[e]) + bf16_f(hvv.x[e]));
        xr[c] = bf16_to_u4(xv);
    }
}

double layer_weight_bytes(const ModelPreset& p) {
    const int heads = p.n_heads, kvh_n = p.n_kv_heads, d = p.head_dim,
              H = p.hidden, inter = p.inter;
    double elems = (double)(heads + 2*kvh_n)*d * H     // qkv proj
                 + (double)H * heads*d                 // o proj
                 + (double)2*inter * H                 // gate_up proj
                 + (double)H * inter;                  // down proj
    return elems * 2.0;                                // bf16
}

} // namespace

void run_layer_pipeline(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    const int H = p.hidden, heads = p.n_heads, kvh_n = p.n_kv_heads,
              d = p.head_dim, inter = p.inter;
    const int qkv_out = (heads + 2*kvh_n) * d;
    const float scale = 1.0f / std::sqrt((double)d);
    const int layers = p.layers;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float one = 1.f, zero = 0.f;

    std::vector<float> h_inv(d/2);
    for (int i = 0; i < d/2; ++i) h_inv[i] = std::pow(kBase, -2.0f*i/d);
    float* d_inv = nullptr;
    LAB_CHECK(cudaMalloc(&d_inv, (d/2)*sizeof(float)));
    LAB_CHECK(cudaMemcpy(d_inv, h_inv.data(), (d/2)*sizeof(float), cudaMemcpyHostToDevice));

    // Allocate the shared weight and activation buffers once.
    __nv_bfloat16 *Wqkv=nullptr,*Wo=nullptr,*Wgu=nullptr,*Wdown=nullptr,*gA=nullptr,*gM=nullptr;
    __nv_bfloat16 *dx=nullptr,*dan=nullptr,*dm=nullptr,*dqkv=nullptr,*doAttnB=nullptr,
                  *dproj=nullptr,*dguOut=nullptr,*dact=nullptr,*dKc=nullptr,*dVc=nullptr;
    float *dOattn=nullptr;
    // Xavier-style weight init: scale = 1/sqrt(fan_in) so each projection output
    // stays O(1). With raw ±0.5 weights, a fan_in of thousands makes every
    // projection O(sqrt(fan_in)) and the residual stream compounds past bf16
    // range over 64 layers (the NOT-FINITE probe). Scaling keeps activations O(1)
    // without changing byte counts or measured latency.
    auto up=[&](unsigned long long seed,int n,int fan_in,__nv_bfloat16**pp){
        const float wscale = 1.0f / std::sqrt((double)fan_in);
        std::vector<__nv_bfloat16>b(n); fill_bf16(b,seed,wscale);
        void* tmp=nullptr;
        LAB_CHECK(cudaMalloc(&tmp,(size_t)n*sizeof(__nv_bfloat16))); *pp=(__nv_bfloat16*)tmp;
        LAB_CHECK(cudaMemcpy(*pp,b.data(),(size_t)n*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice)); };
    up(0x77ull, qkv_out*H,     H,        &Wqkv); up(0x6Full, H*(heads*d),   heads*d, &Wo);
    up(0x75ull, 2*inter*H,     H,        &Wgu);  up(0x64ull, H*inter,       inter,   &Wdown);
    up(0x61ull, H,             1,        &gA);   up(0x6Dull, H,             1,       &gM);
    { std::vector<__nv_bfloat16>b(H); fill_bf16(b,7ull,0.5f);
      LAB_CHECK(cudaMalloc(&dx,(size_t)H*sizeof(__nv_bfloat16)));
      LAB_CHECK(cudaMemcpy(dx,b.data(),(size_t)H*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice)); }
    LAB_CHECK(cudaMalloc(&dan,(size_t)H*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dm ,(size_t)H*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dqkv,(size_t)qkv_out*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&doAttnB,(size_t)heads*d*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dproj,(size_t)H*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dguOut,(size_t)2*inter*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dact,(size_t)inter*sizeof(__nv_bfloat16)));
    LAB_CHECK(cudaMalloc(&dOattn,(size_t)heads*d*sizeof(float)));

    // y = W x via cuBLAS at m=1: column-major out(out,1)=op(W) op(x).
    auto gemm_mv=[&](int out,int k,const __nv_bfloat16*W,const __nv_bfloat16*x,
                     __nv_bfloat16*outp,cudaStream_t st){
        CUBLAS_CHECK(cublasSetStream(handle,st));
        CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,1,k,&one,
                                  W,CUDA_R_16BF,k, x,CUDA_R_16BF,k, &zero,
                                  outp,CUDA_R_16BF,out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); };
    // copy a device buffer (bf16) to another on-device.
    auto devcopy=[&](const void*src,void*dst,size_t bytes,cudaStream_t st){
        LAB_CHECK(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,st)); };
    // convert fp32 attn output [heads*d] to bf16 for the o-projection.
    auto f2b=[&](const float*srcf,__nv_bfloat16*dstb,int n,cudaStream_t st){
        std::vector<float>h(n); LAB_CHECK(cudaMemcpyAsync(h.data(),srcf,n*sizeof(float),cudaMemcpyDeviceToHost,st));
        cudaStreamSynchronize(st);
        std::vector<__nv_bfloat16>b(h.size()); for(size_t i=0;i<h.size();++i)b[i]=f_bf16(h[i]);
        LAB_CHECK(cudaMemcpyAsync(dstb,b.data(),n*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice,st)); };

    size_t shmem_max = sizeof(float)*(args.quick?1024+1:8192+1);
    LAB_CHECK(cudaFuncSetAttribute(k_attn_gqa<ATTN_BLOCK>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem_max));

    std::vector<int> ctx_cases = args.quick ? std::vector<int>{256,512} : std::vector<int>{256,512,1024,2048};
    for (int ctx : ctx_cases) {
        const int tot = ctx + 1;   // attend over cached keys plus the new token
        LAB_CHECK(cudaMalloc(&dKc,(size_t)tot*kvh_n*d*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dVc,(size_t)tot*kvh_n*d*sizeof(__nv_bfloat16)));
        { std::vector<__nv_bfloat16>k((size_t)tot*kvh_n*d),v((size_t)tot*kvh_n*d);
          fill_bf16(k,201ull,0.5f); fill_bf16(v,202ull,0.5f);
          LAB_CHECK(cudaMemcpy(dKc,k.data(),k.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
          LAB_CHECK(cudaMemcpy(dVc,v.data(),v.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice)); }
        // Reset hidden state to the initial value before each ctx run.
        { std::vector<__nv_bfloat16>b(H); fill_bf16(b,7ull,0.5f);
          LAB_CHECK(cudaMemcpy(dx,b.data(),(size_t)H*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice)); }
        const int pos = ctx;   // append the new token's K,V here so length becomes tot

        // One full layer forward pass for the single token.
        auto one_layer = [&](cudaStream_t st){
            // Optional per-stage magnitude trace (only when GB10_PIPE_STAGE=1).
            auto stage_max = [&](const char* tag, const void* buf, int n, bool is_f32){
                if (!std::getenv("GB10_PIPE_STAGE")) return;
                std::vector<float> tmp(n);
                if (is_f32) LAB_CHECK(cudaMemcpy(tmp.data(),buf,n*sizeof(float),cudaMemcpyDeviceToHost));
                else { std::vector<__nv_bfloat16> b(n); LAB_CHECK(cudaMemcpy(b.data(),buf,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
                       for(int i=0;i<n;++i) tmp[i]=bf16_f(b[i]); }
                double mx=0; bool f=true; for(float v:tmp){double a=std::fabs((double)v); mx=std::max(mx,a); if(!std::isfinite(v))f=false;}
                std::printf("      [%s] max=%.3e finite=%s\n", tag, mx, f?"yes":"NO");
            };
            // --- attention block --------------------------------------------
            k_rms<RN_BLOCK><<<1,RN_BLOCK,0,st>>>((uint4*)dx,(uint4*)gA,(uint4*)dan,1,H,kEps);
            stage_max("dan(norm)", dan, H, false);
            gemm_mv(qkv_out,H,Wqkv,dan,dqkv,st);                       // Q,Knew,Vnew
            stage_max("qkv", dqkv, qkv_out, false);
            k_rope_inplace<<<heads+kvh_n,64,0,st>>>(reinterpret_cast<uint4*>(dqkv),
                                                    heads+kvh_n,d,d/2,d_inv,pos);
            stage_max("qkv+rope", dqkv, qkv_out, false);
            // append Knew,Vnew into the cache row `pos` (in-place, once).
            devcopy(dqkv + (size_t)heads*d,      dKc + (size_t)pos*kvh_n*d, (size_t)kvh_n*d*sizeof(__nv_bfloat16), st);
            devcopy(dqkv + (size_t)(heads+kvh_n)*d, dVc + (size_t)pos*kvh_n*d, (size_t)kvh_n*d*sizeof(__nv_bfloat16), st);
            k_attn_gqa<ATTN_BLOCK><<<heads,ATTN_BLOCK,sizeof(float)*(tot+1),st>>>(
                dqkv, dKc, dVc, scale, tot, d, kvh_n, dOattn);
            stage_max("attn_out(f32)", dOattn, heads*d, true);
            f2b(dOattn, doAttnB, heads*d, st);
            gemm_mv(H, heads*d, Wo, doAttnB, dproj, st);               // o proj
            stage_max("o_proj", dproj, H, false);
            k_add<RN_BLOCK><<<1,RN_BLOCK,0,st>>>((uint4*)dx,(uint4*)dproj,1,H);
            stage_max("dx_after_attn", dx, H, false);
            // --- MLP block --------------------------------------------------
            k_rms<RN_BLOCK><<<1,RN_BLOCK,0,st>>>((uint4*)dx,(uint4*)gM,(uint4*)dm,1,H,kEps);
            stage_max("dm(norm)", dm, H, false);
            gemm_mv(2*inter,H,Wgu,dm,dguOut,st);                       // gate_up
            stage_max("gate_up", dguOut, 2*inter, false);
            k_silu_mul<<<(inter/8+255)/256,256,0,st>>>((uint4*)dguOut,(uint4*)dact,inter/8);
            stage_max("act(silu)", dact, inter, false);
            gemm_mv(H,inter,Wdown,dact,dproj,st);                      // down
            stage_max("down_proj", dproj, H, false);
            k_add<RN_BLOCK><<<1,RN_BLOCK,0,st>>>((uint4*)dx,(uint4*)dproj,1,H);
            stage_max("dx_after_mlp", dx, H, false);
        };

        // Full-model per-token pass = every layer in sequence on one stream.
        auto body = [&](cudaStream_t st){
            for (int L = 0; L < layers; ++L) one_layer(st);
        };
        // Finiteness sanity check on the running hidden state after one pass.
        {
          // Diagnostic mode (GB10_PIPE_DEBUG=1): run one layer at a time and
          // report the residual magnitude after each, to localize divergence.
          if (std::getenv("GB10_PIPE_DEBUG")) {
            const int dbg_layers = std::min(layers, 12);
            for (int L = 0; L < dbg_layers; ++L) {
                one_layer(nullptr); LAB_CHECK(cudaDeviceSynchronize());
                std::vector<__nv_bfloat16>hx(H);
                LAB_CHECK(cudaMemcpy(hx.data(),dx,(size_t)H*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
                double mx=0; bool f=true;
                for(int i=0;i<H;++i){ double v=std::fabs((double)bf16_f(hx[i])); mx=std::max(mx,v); if(!std::isfinite(v)) f=false; }
                std::printf("    [dbg] L=%2d |dx|max=%.3e finite=%s\n", L, mx, f?"yes":"NO");
                if(!f) break;
            }
          }
          body(nullptr); LAB_CHECK(cudaDeviceSynchronize());
          std::vector<__nv_bfloat16>hx(H); LAB_CHECK(cudaMemcpy(hx.data(),dx,(size_t)H*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
          bool finite=true; for(int i=0;i<H;++i) if(!std::isfinite((double)bf16_f(hx[i]))) finite=false;
          emit_record(args, JsonLine().kv("op","layer_pipeline_check")
                        .append_fragment(std::string(",\"ctx\":")+fmt_num(ctx) + ",\"finite\":")
                        .append_fragment(finite?"true":"false"));
          std::printf("  pipeline ctx=%-5d all-layers output %s\n", ctx, finite?"finite":"NOT-FINITE");
        }

        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double bytes_moved = (double)layers * layer_weight_bytes(p);    // weights streamed/token
        emit_timing(args, "layer_pipeline", t, ErrStats{0,0,true}, bytes_moved, 0.0,
                    ",\"ctx\":" + fmt_num(ctx) + ",\"layers\":" + fmt_num(layers) +
                    ",\"weights_gb_per_token\":" + fmt_num(bytes_moved/1e9) +
                    ",\"dtype\":\"bf16\"");
        std::printf("  pipeline ctx=%-5d layers=%-3d %10.2f us/token   %6.1f GB/s weight-stream   (%.3f ms)\n",
                    ctx, layers, t.median_us, bytes_moved/(t.median_us*1e-6)/1e9, t.median_us/1e6);
        LAB_CHECK(cudaFree(dKc)); LAB_CHECK(cudaFree(dVc));
    }

    #define FREE(p) if(p){cudaFree(p);p=nullptr;}
    FREE(Wqkv);FREE(Wo);FREE(Wgu);FREE(Wdown);FREE(gA);FREE(gM);FREE(dx);FREE(dan);FREE(dm);
    FREE(dqkv);FREE(doAttnB);FREE(dproj);FREE(dguOut);FREE(dact);FREE(dOattn);FREE(d_inv);
    CUBLAS_CHECK(cublasDestroy(handle));
}
