// rope.cu — Layer 3: rotary position embeddings over attention heads.
//
// rotate-half scheme. For element i of a head vector of length d at position
// pos, define m = min(i, i - d/2) and sign = +1 if i < d/2 else -1:
//     out_i = x_i * cos(pos*theta_m) * sign + x_partner * sin(pos*theta_m)
// where theta_m = base^{-2m/d} and x_partner is the element paired with i under
// the half-rotation. Applied to Q and K before attention. One block handles one
// (token, head) row; each thread owns 8 consecutive bf16 elements (one uint4)
// and applies the rotation with per-element __sincosf. Every lane stays active
// because the rotation is expressed per-element rather than by pairing two
// vectors, and neighbour threads touch neighbouring addresses (coalesced).
#include "gb10_lab/common.h"

namespace {

constexpr float kBase = 10000.0f;

template <int BLOCK>
__global__ void k_rope(const uint4* __restrict__ x_in,   // [tokens*heads*(d_pad/8)]
                       uint4* __restrict__ y_out,         // same layout
                       int rows,                          // tokens * heads
                       int d, int d_half, int d_vec,
                       const float* __restrict__ inv_freq, // [d/2]: base^{-2m/d}
                       int n_heads, int start_pos) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tok    = row / n_heads;
    const int pos    = start_pos + tok;
    const uint4* xr  = x_in  + static_cast<size_t>(row) * d_vec;
    uint4* yr       = y_out + static_cast<size_t>(row) * d_vec;
    const int tid   = threadIdx.x;
    const int nthr  = blockDim.x;

    for (int c = tid; c < d_vec; c += nthr) {
        uint4 v   = xr[c];
        BF16x8 ob = u4_to_bf16(v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = c * 8 + j;
            if (i >= d) { ob.x[j] = f_bf16(0.f); continue; }
            const bool first = (i < d_half);
            const int  m     = first ? i : (i - d_half);
            const float xv   = bf16_f(ob.x[j]);
            // Partner lives in another thread's chunk; reload it (L1 hit).
            const int partner_linear = (c * 8 + j) ^ d_half;
            const float xp   = bf16_f(u4_to_bf16(xr[partner_linear >> 3]).x[partner_linear & 7]);
            float cs, sn;
            __sincosf(static_cast<float>(pos) * inv_freq[m], &sn, &cs);
            ob.x[j] = f_bf16(first ? (xv * cs + xp * sn) : (-xv * cs + xp * sn));
        }
        yr[c] = bf16_to_u4(ob);
    }
}

// Reference implementation for correctness checking.
void ref_rope(std::vector<float>& out, const std::vector<__nv_bfloat16>& x,
              int num_tokens, int num_heads, int d, int start_pos) {
    out.assign((size_t)num_tokens * num_heads * d, 0.f);
    for (int t = 0; t < num_tokens; ++t)
      for (int h = 0; h < num_heads; ++h) {
          size_t base = ((size_t)t * num_heads + h) * d;
          const int pos = start_pos + t;
          for (int i = 0; i < d; ++i) {
              const bool first = (i < d / 2);
              const int  m     = first ? i : (i - d / 2);
              const float xi = bf16_f(x[base + i]);
              const float xp = bf16_f(x[base + (first ? i + d/2 : i - d/2)]);
              const float ang = (float)pos * (float)std::pow(kBase, -2.0 * m / d);
              const float cs = std::cos(ang), sn = std::sin(ang);
              out[base + i] = first ? (xi * cs + xp * sn) : (-xi * cs + xp * sn);
          }
      }
}

} // namespace

void run_rope(const LabArgs& args) {
    const ModelPreset& p = args.preset;
    const int d = p.head_dim;
    const int d_pad = ((d + 7) / 8) * 8;
    const int d_vec = d_pad / 8;
    const int inv_len = d / 2;

    std::vector<int> token_cases = {1, 32, 256, 2048};
    if (args.quick) token_cases = {1, 256};

    // inv_freq[m] = base^{-2m/d}.
    std::vector<float> h_inv(inv_len);
    for (int i = 0; i < inv_len; ++i) h_inv[i] = std::pow(kBase, -2.0f * i / d);
    float* d_inv = nullptr;
    LAB_CHECK(cudaMalloc(&d_inv, inv_len * sizeof(float)));
    LAB_CHECK(cudaMemcpy(d_inv, h_inv.data(), inv_len * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 128;
    const int block = std::min(BLOCK, d_vec);

    for (int tokens : token_cases) {
        const int heads = p.n_heads;
        const int rows  = tokens * heads;
        std::vector<__nv_bfloat16> h_x((size_t)rows * d_pad);
        fill_bf16(h_x, 42ull, 1.0f);
        for (size_t r = 0; r < (size_t)rows; ++r)
            for (int i = d; i < d_pad; ++i) h_x[r * d_pad + i] = f_bf16(0.f);

        uint4* d_x = nullptr; uint4* d_y = nullptr;
        LAB_CHECK(cudaMalloc(&d_x, (size_t)rows * d_pad * sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMalloc(&d_y, (size_t)rows * d_pad * sizeof(__nv_bfloat16)));
        LAB_CHECK(cudaMemcpy(d_x, h_x.data(),
                             (size_t)rows * d_pad * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice));

        auto body = [&](cudaStream_t st) {
            k_rope<BLOCK><<<rows, block, 0, st>>>(d_x, d_y, rows, d, d / 2, d_vec, d_inv,
                                                  heads, 0);
        };
        body(nullptr);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_y((size_t)rows * d_pad);
        LAB_CHECK(cudaMemcpy(h_y.data(), d_y,
                             (size_t)rows * d_pad * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
        std::vector<float> got((size_t)rows * d), ref;
        for (size_t i = 0; i < (size_t)rows * d; ++i) got[i] = bf16_f(h_y[i]);
        std::vector<__nv_bfloat16> x_contig((size_t)rows * d);
        for (size_t r = 0; r < (size_t)rows; ++r)
            for (int i = 0; i < d; ++i) x_contig[r * d + i] = h_x[r * d_pad + i];
        ref_rope(ref, x_contig, tokens, heads, d, 0);
        ErrStats err = compare_f32(got.data(), ref.data(), (size_t)rows * d,
                                   /*abs_tol*/ 3e-2, /*rel_tol*/ 3e-2);

        std::vector<double> us = timed_iterations(body, args.warmup, args.iters);
        TimingResult t = summarize(us);
        double bytes_moved = (double)rows * d * 2;   // in + out bf16
        emit_timing(args, "rope", t, err, bytes_moved, 0.0,
                    ",\"tokens\":" + fmt_num(tokens) + ",\"heads\":" + fmt_num(heads) +
                    ",\"head_dim\":" + fmt_num(d) + ",\"block\":" + fmt_num(block) +
                    ",\"dtype\":\"bf16\"");
        std::printf("  rope     tok=%-5d heads=%-3d d=%-4d %8.2f us   %6.1f GB/s   correct=%s\n",
                    tokens, heads, d, t.median_us,
                    bytes_moved / (t.median_us * 1e-6) / 1e9, err.pass ? "yes" : "NO");
        LAB_CHECK(cudaFree(d_x)); LAB_CHECK(cudaFree(d_y));
    }
    LAB_CHECK(cudaFree(d_inv));
}
