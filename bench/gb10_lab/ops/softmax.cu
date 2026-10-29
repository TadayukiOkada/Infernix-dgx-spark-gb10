// softmax.cu — Layer 3: online (one-pass) softmax over attention score rows.
//
// Standard two-pass softmax needs the row max before it can normalise, forcing
// a second read of the scores. The online formulation carries a running max m_i
// and denominator l_i and rescales partial results whenever the max grows, so a
// single streaming pass suffices. That matters on GB10 where re-reading large
// score matrices from DRAM is expensive. One warp per row reduces the running
// (m, l) pairs with __shfl_down; tails shorter than a warp are masked. Used for
// both prefill (long rows) and decode (short rows over the KV length).
#include "gb10_lab/common.h"

namespace {

template <int BLOCK>
__global__ void k_softmax_online(const float* __restrict__ in,  // [rows, width]
                                 float* __restrict__ out,        // [rows, width]
                                 int rows, int width) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = in  + static_cast<size_t>(row) * width;
    float* yr      = out + static_cast<size_t>(row) * width;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;

    // Each thread streams a strided subset of the row, keeping a private (m, l).
    float m_i = -INFINITY, l_i = 0.f;
    for (int i = tid; i < width; i += nthr) {
        const float v = xr[i];
        const float m_new = fmaxf(m_i, v);
        l_i = l_i * __expf(m_i - m_new) + __expf(v - m_new);
        m_i = m_new;
    }

    // Reduce (m, l) across the block. Pairwise combine via shared memory.
    extern __shared__ float smem[];     // 2 * BLOCK floats: m and l arrays
    float* s_m = smem;
    float* s_l = smem + BLOCK;
    s_m[tid] = m_i; s_l[tid] = l_i;
    __syncthreads();
    #pragma unroll
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float m_a = s_m[tid], m_b = s_m[tid + offset];
            const float m_new = fmaxf(m_a, m_b);
            s_l[tid] = s_l[tid] * __expf(m_a - m_new) + s_l[tid + offset] * __expf(m_b - m_new);
            s_m[tid] = m_new;
        }
        __syncthreads();
    }
    const float m_row = s_m[0];
    const float inv_l = 1.f / s_l[0];

    // Second read of the row (L1/L2 resident for moderate widths) to emit.
    for (int i = tid; i < width; i += nthr)
        yr[i] = __expf(xr[i] - m_row) * inv_l;
}

void ref_softmax(std::vector<float>& out, const std::vector<float>& in,
                 int rows, int width) {
    out.assign((size_t)rows * width, 0.f);
    for (int r = 0; r < rows; ++r) {
        double mx = -1e30;
        for (int i = 0; i < width; ++i) mx = std::max(mx, (double)in[(size_t)r*width+i]);
        double denom = 0.0;
        for (int i = 0; i < width; ++i) denom += std::exp(in[(size_t)r*width+i] - mx);
        for (int i = 0; i < width; ++i)
            out[(size_t)r*width+i] = (float)(std::exp(in[(size_t)r*width+i] - mx) / denom);
    }
}

} // namespace

void run_softmax(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    // Row width models the attention score length: head_dim for QK^T scale or
    // sequence length for decode over the KV cache. Sweep both regimes.
    struct Case { const char* label; int rows; int width; };
    std::vector<Case> cases = {
        {"decode_kv256",   p.n_heads, 256},
        {"decode_kv4096",  p.n_heads, 4096},
        {"prefill_s512",   p.n_heads * 8, 512},
        {"prefill_s2048",  p.n_heads * 8, 2048},
    };
    if (args.quick) cases = {{ "decode_kv256", p.n_heads, 256 },
                             {"prefill_s512",  p.n_heads * 8, 512 }};

    constexpr int BLOCK = 256;
    size_t shmem = 2 * sizeof(float) * BLOCK;
    LAB_CHECK(cudaFuncSetAttribute(k_softmax_online<BLOCK>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shmem));

    for (const auto& c : cases) {
        std::vector<float> h_in((size_t)c.rows * c.width);
        fill_f32(h_in, 7ull, 2.0f);   // moderate magnitudes exercise rescaling
        float* d_in = nullptr; float* d_out = nullptr;
        LAB_CHECK(cudaMalloc(&d_in, (size_t)c.rows * c.width * sizeof(float)));
        LAB_CHECK(cudaMalloc(&d_out, (size_t)c.rows * c.width * sizeof(float)));
        LAB_CHECK(cudaMemcpy(d_in, h_in.data(),
                             (size_t)c.rows * c.width * sizeof(float), cudaMemcpyHostToDevice));

        auto body = [&](cudaStream_t st) {
            k_softmax_online<BLOCK><<<c.rows, BLOCK, shmem, st>>>(d_in, d_out, c.rows, c.width);
        };
        body(nullptr);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)c.rows * c.width), ref;
        LAB_CHECK(cudaMemcpy(got.data(), d_out,
                             (size_t)c.rows * c.width * sizeof(float), cudaMemcpyDeviceToHost));
        ref_softmax(ref, h_in, c.rows, c.width);
        ErrStats err = compare_f32(got.data(), ref.data(), (size_t)c.rows * c.width,
                                   /*abs_tol*/ 1e-4, /*rel_tol*/ 1e-4);

        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double bytes_moved = (double)c.rows * c.width * sizeof(float) * 2;  // in + out
        emit_timing(args, "softmax", t, err, bytes_moved, 0.0,
                    ",\"label\":\"" + std::string(c.label) + "\"" +
                    ",\"rows\":" + fmt_num(c.rows) + ",\"width\":" + fmt_num(c.width) +
                    ",\"block\":" + fmt_num(BLOCK) + ",\"dtype\":\"fp32\"");
        std::printf("  softmax %-14s rows=%-6d width=%-5d %8.2f us   %6.1f GB/s   correct=%s\n",
                    c.label, c.rows, c.width, t.median_us,
                    bytes_moved / (t.median_us * 1e-6) / 1e9, err.pass ? "yes" : "NO");
        LAB_CHECK(cudaFree(d_in)); LAB_CHECK(cudaFree(d_out));
    }
}
