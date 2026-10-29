// attention_decode.cu — Layer 3: single-token causal self-attention over a KV cache.
//
// At decode time each new token attends over every cached key/value, so one
// forward step costs O(seq_len * head_dim) of streamed KV data per head. The
// operation is bandwidth-bound: the cost is reading K and V from DRAM, not the
// arithmetic. We implement a two-pass kernel (one block per head): pass 1 reads
// K once to build the score row and reduce its max/denominator; pass 2 reads V
// once and accumulates the weighted sum. Both passes are coalesced along the
// head-dim axis. We sweep the KV length to trace how decode latency grows with
// context, and compare the measured GB/s against the memory roofline.
#include "gb10_lab/common.h"

namespace {

template <int BLOCK>
__global__ void k_attn_decode(const __nv_bfloat16* __restrict__ Q,   // [heads, d]
                             const __nv_bfloat16* __restrict__ K,    // [seq, heads, d]
                             const __nv_bfloat16* __restrict__ V,    // [seq, heads, d]
                             float scale, int seq, int heads, int d,
                             float* __restrict__ O) {                 // [heads, d]
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const __nv_bfloat16* Kh = K + static_cast<size_t>(h) * d;
    const __nv_bfloat16* Vh = V + static_cast<size_t>(h) * d;
    const __nv_bfloat16* Qh = Q + static_cast<size_t>(h) * d;

    extern __shared__ float smem[];   // [seq] scores, then smem[seq] holds the row max
    float* s = smem;

    // Pass 1: s[i] = scale * dot(Q_h, K[i,h,:]).
    for (int i = tid; i < seq; i += nthr) {
        float acc = 0.f;
        #pragma unroll 4
        for (int j = 0; j < d; ++j)
            acc += bf16_f(Qh[j]) * bf16_f(Kh[static_cast<size_t>(i)*heads*d + j]);
        s[i] = acc * scale;
    }
    __syncthreads();

    // Reduce row max across the block into smem[seq].
    float m_local = -INFINITY;
    for (int i = tid; i < seq; i += nthr) m_local = fmaxf(m_local, s[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m_local = fmaxf(m_local, __shfl_down_sync(0xffffffffu, m_local, off));
    __shared__ float warp_max[BLOCK / 32];
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

    // Denominator: sum of exp(s[i]-m), reduced across the block and broadcast.
    float l_local = 0.f;
    for (int i = tid; i < seq; i += nthr) l_local += __expf(s[i] - m_row);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        l_local += __shfl_down_sync(0xffffffffu, l_local, off);
    __shared__ float warp_sum[BLOCK / 32];
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

    // Pass 2: o[j] = sum_i exp(s[i]-m)*inv_l * V[i,j]. Thread owns element j.
    if (tid < d) {
        float acc = 0.f;
        for (int i = 0; i < seq; ++i)
            acc += __expf(s[i] - m_row) * inv_l * bf16_f(Vh[static_cast<size_t>(i)*heads*d + tid]);
        O[h * d + tid] = acc;
    }
}

void ref_attn(std::vector<float>& out, const std::vector<__nv_bfloat16>& Q,
              const std::vector<__nv_bfloat16>& K, const std::vector<__nv_bfloat16>& V,
              int heads, int seq, int d) {
    const float scale = 1.0f / std::sqrt((double)d);
    out.assign((size_t)heads * d, 0.f);
    for (int h = 0; h < heads; ++h) {
        double mx = -1e30;
        for (int i = 0; i < seq; ++i) {
            double sc = 0.0;
            for (int j = 0; j < d; ++j)
                sc += (double)bf16_f(Q[(size_t)h*d+j]) * (double)bf16_f(K[((size_t)i*heads+h)*d+j]);
            sc *= scale; mx = std::max(mx, sc);
        }
        double denom = 0.0;
        for (int i = 0; i < seq; ++i) {
            double sc = 0.0;
            for (int j = 0; j < d; ++j)
                sc += (double)bf16_f(Q[(size_t)h*d+j]) * (double)bf16_f(K[((size_t)i*heads+h)*d+j]);
            denom += std::exp(sc*scale - mx);
        }
        for (int j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int i = 0; i < seq; ++i) {
                double sc = 0.0;
                for (int k = 0; k < d; ++k)
                    sc += (double)bf16_f(Q[(size_t)h*d+k]) * (double)bf16_f(K[((size_t)i*heads+h)*d+k]);
                acc += std::exp(sc*scale - mx) * (double)bf16_f(V[((size_t)i*heads+h)*d+j]);
            }
            out[(size_t)h*d + j] = (float)(acc / denom);
        }
    }
}

} // namespace

void run_attention_decode(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    const int heads = p.n_heads, d = p.head_dim;
    std::vector<int> seq_cases = {256, 1024, 4096, 8192};
    if (args.quick) seq_cases = {256, 1024};

    constexpr int BLOCK = 256;
    size_t shmem_max = (sizeof(float) * (8192 + 1));   // scores + max slot
    LAB_CHECK(cudaFuncSetAttribute(k_attn_decode<BLOCK>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shmem_max));
    const float scale = 1.0f / std::sqrt((double)d);

    for (int seq : seq_cases) {
        std::vector<__nv_bfloat16> hQ((size_t)heads*d),
                                  hK((size_t)seq*heads*d),
                                  hV((size_t)seq*heads*d);
        fill_bf16(hQ, 21ull, 0.7f);
        fill_bf16(hK, 22ull, 0.7f);
        fill_bf16(hV, 23ull, 0.7f);
        __nv_bfloat16* dQ=nullptr; __nv_bfloat16* dK=nullptr; __nv_bfloat16* dV=nullptr;
        float* dO=nullptr;
        LAB_CHECK(cudaMalloc(&dQ,(size_t)heads*d*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dK,(size_t)seq*heads*d*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dV,(size_t)seq*heads*d*sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&dO,(size_t)heads*d*sizeof(float)));
        LAB_CHECK(cudaMemcpy(dQ,hQ.data(),(size_t)heads*d*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMemcpy(dK,hK.data(),(size_t)seq*heads*d*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
        LAB_CHECK(cudaMemcpy(dV,hV.data(),(size_t)seq*heads*d*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));

        auto body = [&](cudaStream_t st){
            k_attn_decode<BLOCK><<<heads, BLOCK, sizeof(float)*(seq+1), st>>>(
                dQ, dK, dV, scale, seq, heads, d, dO);
        };
        body(nullptr);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)heads*d), ref;
        LAB_CHECK(cudaMemcpy(got.data(),dO,(size_t)heads*d*sizeof(float),cudaMemcpyDeviceToHost));
        ref_attn(ref, hQ, hK, hV, heads, seq, d);
        ErrStats err = compare_f32(got.data(), ref.data(), (size_t)heads*d,
                                   /*abs_tol*/ 3e-2, /*rel_tol*/ 3e-2);

        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double bytes_moved = (double)seq*heads*d*2*2;   // K + V streams
        emit_timing(args, "attention_decode", t, err, bytes_moved, 2.0*(double)seq*heads*d*2,
                    ",\"seq\":" + fmt_num(seq) + ",\"heads\":" + fmt_num(heads) +
                    ",\"head_dim\":" + fmt_num(d) + ",\"block\":" + fmt_num(BLOCK) +
                    ",\"dtype\":\"bf16\"");
        std::printf("  attn_dec seq=%-6d heads=%-4d d=%-4d %9.2f us   %6.1f GB/s   correct=%s\n",
                    seq, heads, d, t.median_us, bytes_moved/(t.median_us*1e-6)/1e9,
                    err.pass?"yes":"NO");
        LAB_CHECK(cudaFree(dQ));LAB_CHECK(cudaFree(dK));LAB_CHECK(cudaFree(dV));LAB_CHECK(cudaFree(dO));
    }
}
