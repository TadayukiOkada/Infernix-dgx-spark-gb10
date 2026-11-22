// Native sm_121a NVFP4 OMMA narrow/medium W4A4 MoE kernels (1 <= count <= moe::kMaxColumns,
// or split-input experts via count-8 chunking), included by moe_layer.cu inside its anonymous namespace.
//
// Matrix A = 16 weight rows x 64 K elements (`rg16_a_fragments_seq`, sequential units u0..u3);
// Matrix B = up to 8 token/activation columns x 64 K elements (`OmmaXChunk` in SMEM, zero bank conflicts);
// Output C = FP32 16 x 8 (`c0`=(g, 2t), `c1`=(g, 2t+1), `c2`=(g+8, 2t), `c3`=(g+8, 2t+1)).
// Both A and B use `plane_block` [0, 2, 1, 3] ordering within each 4-block K64 chunk for `scale_vec::4X`.
// Intermediate activations `h` are stored in `h_blocks` as `OmmaHRow` (368 B <= kHRowStride 480 B)
// so down_omma reads h directly with zero SMEM bank conflicts and zero cross-warp reduction.

struct OmmaXChunk {
    std::uint32_t codes[16][8][2]; // [pos = 4*k64 + plane_block(b & 3)][col 0..7][lo=0 / hi=1]
    std::uint32_t scales4[4][8];   // [k64 0..3][col 0..7] -> bytes [b0, b2, b1, b3] E4M3FN scales
};

template <class G>
struct OmmaShape {
    static constexpr int kThreads        = 128; // 4 warps
    static constexpr int kWarps          = 4;
    static constexpr int kGateUpSlices   = G::kGateUpRowGroups / 2; // 40 (16 intermediates = 1 h block each)
    static constexpr int kGateUpChunks   = G::kGateUpBlocks / 16;   // 10 chunks x 16 units
    static constexpr int kGateUpStageB   = 2 * 16 * static_cast<int>(moe::kUnitBytes); // 4,608 B
    static constexpr int kDownTiles      = G::kDownRowGroups / 4;   // 40 (4 row groups per tile)
    static constexpr int kDownChunks     = G::kDownBlocks / 8;      // 5 stages x 8 units
    static constexpr int kDownWarpStageB = 8 * static_cast<int>(moe::kUnitBytes);      // 1,152 B
    static constexpr int kDownRgBytes    = G::kDownBlocks * static_cast<int>(moe::kUnitBytes); // 5,760 B
    static constexpr int kHRowBytes      = 368; // 320 code B + 40 scale B, odd x 16 B -> 0 bank conflicts
    static_assert(G::kGateUpBlocks % 16 == 0 && G::kDownBlocks % 8 == 0 && G::kDownRowGroups % 4 == 0);
    static_assert(moe::wide::kHRowStride >= kHRowBytes && moe::wide::kHCodeBytes == 320);
};

__device__ __forceinline__ std::uint32_t omma_lds32(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ void rg16_a_fragments_seq(const std::uint8_t* units, int group, int t, std::uint32_t& a0,
                                                      std::uint32_t& a1, std::uint32_t& a2, std::uint32_t& a3,
                                                      std::uint32_t& sfa) {
    const int quad            = 64 * (t & 1) + 4 * group;
    const std::uint8_t* u02   = units + static_cast<std::size_t>(2 * (t >> 1)) * moe::kUnitBytes + quad;
    const std::uint8_t* u13   = units + static_cast<std::size_t>(1 + 2 * (t >> 1)) * moe::kUnitBytes + quad;
    const std::uint32_t w0    = omma_lds32(u02);
    const std::uint32_t w1    = omma_lds32(u02 + 32);
    const std::uint32_t v0    = omma_lds32(u13);
    const std::uint32_t v1    = omma_lds32(u13 + 32);
    a0                        = (w0 & 0x0F0F0F0FU) | ((w1 & 0x0F0F0F0FU) << 4);
    a1                        = ((w0 >> 4) & 0x0F0F0F0FU) | (w1 & 0xF0F0F0F0U);
    a2                        = (v0 & 0x0F0F0F0FU) | ((v1 & 0x0F0F0F0FU) << 4);
    a3                        = ((v0 >> 4) & 0x0F0F0F0FU) | (v1 & 0xF0F0F0F0U);
    sfa                       = 0U;
    if ((t & 2) == 0) {
        const int sw            = 128 + ((t & 1) << 3) + 4 * (group >> 2);
        const unsigned sel      = static_cast<unsigned>(group & 3) | (static_cast<unsigned>(4 + (group & 3)) << 4);
        const std::uint32_t s0  = omma_lds32(units + sw);
        const std::uint32_t s1  = omma_lds32(units + moe::kUnitBytes + sw);
        const std::uint32_t s2  = omma_lds32(units + 2 * moe::kUnitBytes + sw);
        const std::uint32_t s3  = omma_lds32(units + 3 * moe::kUnitBytes + sw);
        sfa                     = __byte_perm(__byte_perm(s0, s2, sel), __byte_perm(s1, s3, sel), 0x5410);
    }
}

template <class G>
__device__ __forceinline__ void omma_load_x_f32_amax(const bf16* x, int token_col, int chunk, int b_local,
                                                     float (&v)[16], float& amax) {
    const auto* src = reinterpret_cast<const uint4*>(
        reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(token_col) * G::kHidden +
        static_cast<std::size_t>(chunk * 16 + b_local) * 16);
    const uint4 s0 = src[0], s1 = src[1];
    const unsigned words[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float f0 = canon::bf16_to_f32(static_cast<std::uint16_t>(words[i] & 0xFFFFU));
        const float f1 = canon::bf16_to_f32(static_cast<std::uint16_t>(words[i] >> 16));
        v[2 * i]       = f0;
        v[2 * i + 1]   = f1;
        const float a0 = fabsf(f0);
        const float a1 = fabsf(f1);
        amax           = a0 > amax || a0 != a0 ? a0 : amax;
        amax           = a1 > amax || a1 != a1 ? a1 : amax;
    }
}

__device__ __forceinline__ std::uint32_t omma_pack_scales4(std::uint8_t sw, int lane) {
    const unsigned base_lane = static_cast<unsigned>(lane & ~3);
    const unsigned s0        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 0U);
    const unsigned s1        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 2U);
    const unsigned s2        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 1U);
    const unsigned s3        = __shfl_sync(0xFFFFFFFFU, static_cast<unsigned>(sw), base_lane + 3U);
    return s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
}

template <class G>
__device__ __forceinline__ void omma_quantize_x_chunk(const bf16* x, int token_col, bool live_col, int chunk,
                                                      float input_scale, OmmaXChunk& dst) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & 31;
    const int col     = tid >> 4;   // 0..7
    const int b_local = tid & 15;   // 0..15 blocks within the 256-elem chunk
    const int k64     = b_local >> 2;
    std::uint8_t sw   = 0U;
    if (live_col) {
        float v[16];
        float amax = 0.0F;
        omma_load_x_f32_amax<G>(x, token_col, chunk, b_local, v, amax);
        unsigned low = 0U, high = 0U;
        canon::quantize_a4_plane_from_f32(v, amax, input_scale, low, high, sw);
        const int pos = 4 * k64 + moe::wide::plane_block(b_local & 3);
        *reinterpret_cast<uint2*>(dst.codes[pos][col]) = make_uint2(low, high);
    }
    const std::uint32_t packed = omma_pack_scales4(sw, lane);
    if (live_col && (b_local & 3) == 0) { dst.scales4[k64][col] = packed; }
}

template <class G>
__device__ __forceinline__ void omma_quantize_x_chunk_split(const bf16* x, int token_col, bool live_col, int chunk,
                                                            float scale_gate, float scale_up, OmmaXChunk& dst_gate,
                                                            OmmaXChunk& dst_up) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & 31;
    const int col     = tid >> 4;
    const int b_local = tid & 15;
    const int k64     = b_local >> 2;
    std::uint8_t sw_g = 0U, sw_u = 0U;
    if (live_col) {
        float v[16];
        float amax = 0.0F;
        omma_load_x_f32_amax<G>(x, token_col, chunk, b_local, v, amax);
        const int pos = 4 * k64 + moe::wide::plane_block(b_local & 3);
        unsigned low = 0U, high = 0U;
        canon::quantize_a4_plane_from_f32(v, amax, scale_gate, low, high, sw_g);
        *reinterpret_cast<uint2*>(dst_gate.codes[pos][col]) = make_uint2(low, high);
        canon::quantize_a4_plane_from_f32(v, amax, scale_up, low, high, sw_u);
        *reinterpret_cast<uint2*>(dst_up.codes[pos][col]) = make_uint2(low, high);
    }
    const std::uint32_t pg = omma_pack_scales4(sw_g, lane);
    const std::uint32_t pu = omma_pack_scales4(sw_u, lane);
    if (live_col && (b_local & 3) == 0) {
        dst_gate.scales4[k64][col] = pg;
        dst_up.scales4[k64][col]   = pu;
    }
}

template <class G>
union GateUpOmmaShared {
    struct {
        alignas(16) std::uint8_t w[2][2][16 * moe::kUnitBytes]; // 9,216 B
        OmmaXChunk x[2];                                         // 2,304 B
    } pipe;
    struct {
        float partial[2][16][8]; // 1,024 B
        std::uint16_t h[8][16];  //   256 B
    } epi;
};

template <class G>
struct DownOmmaShared {
    alignas(16) std::uint8_t w[2][OmmaShape<G>::kWarps][OmmaShape<G>::kDownWarpStageB]; // 9,216 B
    alignas(16) std::uint8_t h[8][OmmaShape<G>::kHRowBytes];                              // 2,944 B
};

template <class G>
__device__ __forceinline__ void gate_up_omma_slice(GateUpOmmaShared<G>& sm, const bf16* x,
                                                     const MoeDispatch& dispatch, int top_k,
                                                     const std::uint8_t* record, const moe::ExpertScales& scales,
                                                     int first, int count, int slice, std::uint8_t* h_base) {
    using S              = OmmaShape<G>;
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int group      = lane >> 2;
    const int t          = lane & 3;
    const int rg_local   = warp & 1; // 0: intermediates 0..7, 1: intermediates 8..15
    const int ks         = warp >> 1; // 0: K64 steps 0,2 in chunk; 1: K64 steps 1,3 in chunk
    const bool split_in  = scales.input_gate != scales.input_up;
    const std::uint8_t* units_base =
        record + static_cast<std::size_t>(2 * slice) * G::kGateUpBlocks * moe::kUnitBytes;

    const auto issue_w = [&](int chunk, int buf) {
        const std::size_t k_off = static_cast<std::size_t>(chunk) * 16 * moe::kUnitBytes;
        constexpr int kRgBytes  = 16 * static_cast<int>(moe::kUnitBytes);
        constexpr int kVecs     = kRgBytes / 16; // 144
        for (int i = tid; i < 2 * kVecs; i += S::kThreads) {
            const int rg = i / kVecs;
            const int k  = i % kVecs;
            const std::uint8_t* src =
                units_base + static_cast<std::size_t>(rg) * G::kGateUpBlocks * moe::kUnitBytes + k_off + 16 * k;
            ggml::tile::cp_async16(sm.pipe.w[buf][rg] + 16 * k, src, 16);
        }
    };

    for (int c_base = 0; c_base < count; c_base += 8) {
        const int n_cols    = min(8, count - c_base);
        const int col       = tid >> 4;
        const bool live_col = col < n_cols;
        const int tok =
            live_col ? moe::wide::entry_column(dispatch.entries[first + c_base + col], top_k, dispatch.shared_first)
                     : 0;

        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        if (!split_in) {
            issue_w(0, 0);
            ggml::tile::cp_async_commit();
            omma_quantize_x_chunk<G>(x, tok, live_col, 0, scales.input_gate, sm.pipe.x[0]);
            issue_w(1, 1);
            ggml::tile::cp_async_commit();
            ggml::tile::cp_async_wait<1>();
            __syncthreads();

            for (int ch = 0; ch < S::kGateUpChunks; ++ch) {
                const OmmaXChunk& xc = sm.pipe.x[ch & 1];
                const int b_col      = min(group, n_cols - 1);
                const std::uint8_t* wu = sm.pipe.w[ch & 1][rg_local];
#pragma unroll
                for (int k_step = 0; k_step < 2; ++k_step) {
                    const int w_idx = ks + 2 * k_step; // {0, 2} for ks=0, {1, 3} for ks=1
                    std::uint32_t a0, a1, a2, a3, sfa;
                    rg16_a_fragments_seq(wu + w_idx * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                    const std::uint32_t b0  = xc.codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t b1  = xc.codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t sfb = xc.scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
                if (ch + 1 < S::kGateUpChunks) {
                    omma_quantize_x_chunk<G>(x, tok, live_col, ch + 1, scales.input_gate, sm.pipe.x[(ch + 1) & 1]);
                }
                __syncthreads();
                if (ch + 2 < S::kGateUpChunks) {
                    issue_w(ch + 2, ch & 1);
                    ggml::tile::cp_async_commit();
                }
                if (ch + 1 < S::kGateUpChunks) {
                    if (ch + 2 < S::kGateUpChunks) {
                        ggml::tile::cp_async_wait<1>();
                    } else {
                        ggml::tile::cp_async_wait<0>();
                    }
                    __syncthreads();
                }
            }
        } else {
            float acc_g[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float acc_u[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            issue_w(0, 0);
            ggml::tile::cp_async_commit();
            omma_quantize_x_chunk_split<G>(x, tok, live_col, 0, scales.input_gate, scales.input_up, sm.pipe.x[0],
                                           sm.pipe.x[1]);
            issue_w(1, 1);
            ggml::tile::cp_async_commit();
            ggml::tile::cp_async_wait<1>();
            __syncthreads();

            for (int ch = 0; ch < S::kGateUpChunks; ++ch) {
                const int b_col      = min(group, n_cols - 1);
                const std::uint8_t* wu = sm.pipe.w[ch & 1][rg_local];
#pragma unroll
                for (int k_step = 0; k_step < 2; ++k_step) {
                    const int w_idx = ks + 2 * k_step;
                    std::uint32_t a0, a1, a2, a3, sfa;
                    rg16_a_fragments_seq(wu + w_idx * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                    const std::uint32_t bg0 = sm.pipe.x[0].codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t bg1 = sm.pipe.x[0].codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t sg  = sm.pipe.x[0].scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc_g[0], acc_g[1], acc_g[2], acc_g[3], a0, a1, a2, a3, bg0, bg1, sfa, sg);
                    const std::uint32_t bu0 = sm.pipe.x[1].codes[4 * w_idx + (t >> 1)][b_col][t & 1];
                    const std::uint32_t bu1 = sm.pipe.x[1].codes[4 * w_idx + 2 + (t >> 1)][b_col][t & 1];
                    const std::uint32_t su  = sm.pipe.x[1].scales4[w_idx][b_col];
                    mma_nvfp4_e4m3(acc_u[0], acc_u[1], acc_u[2], acc_u[3], a0, a1, a2, a3, bu0, bu1, sfa, su);
                }
                __syncthreads();
                if (ch + 2 < S::kGateUpChunks) {
                    issue_w(ch + 2, ch & 1);
                    ggml::tile::cp_async_commit();
                }
                if (ch + 1 < S::kGateUpChunks) {
                    omma_quantize_x_chunk_split<G>(x, tok, live_col, ch + 1, scales.input_gate, scales.input_up,
                                                   sm.pipe.x[0], sm.pipe.x[1]);
                    if (ch + 2 < S::kGateUpChunks) {
                        ggml::tile::cp_async_wait<1>();
                    } else {
                        ggml::tile::cp_async_wait<0>();
                    }
                    __syncthreads();
                }
            }
            const bool is_gate = (group & 1) == 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[i] = is_gate ? acc_g[i] : acc_u[i]; }
        }

        if (ks == 1) {
            sm.epi.partial[rg_local][group][2 * t]         = acc[0];
            sm.epi.partial[rg_local][group][2 * t + 1]     = acc[1];
            sm.epi.partial[rg_local][group + 8][2 * t]     = acc[2];
            sm.epi.partial[rg_local][group + 8][2 * t + 1] = acc[3];
        }
        __syncthreads();
        if (ks == 0) {
            const float s0     = acc[0] + sm.epi.partial[rg_local][group][2 * t];
            const float s1     = acc[1] + sm.epi.partial[rg_local][group][2 * t + 1];
            const float s2     = acc[2] + sm.epi.partial[rg_local][group + 8][2 * t];
            const float s3     = acc[3] + sm.epi.partial[rg_local][group + 8][2 * t + 1];
            const bool is_gate = (group & 1) == 0;
            const float alpha  = is_gate ? scales.alpha_gate : scales.alpha_up;
            const std::uint16_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(s0, alpha));
            const std::uint16_t y1 = canon::f32_to_bf16_rn(canon::mul_rn(s1, alpha));
            const std::uint16_t y2 = canon::f32_to_bf16_rn(canon::mul_rn(s2, alpha));
            const std::uint16_t y3 = canon::f32_to_bf16_rn(canon::mul_rn(s3, alpha));
            const auto u0          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y0, 4));
            const auto u1          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y1, 4));
            const auto u2          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y2, 4));
            const auto u3          = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y3, 4));
            if (is_gate) {
                const int i_lo = 8 * rg_local + (group >> 1);
                const int i_hi = i_lo + 4;
                const int c0 = 2 * t, c1 = 2 * t + 1;
                if (c0 < n_cols) {
                    sm.epi.h[c0][i_lo] = canon::swiglu_bf16(y0, u0);
                    sm.epi.h[c0][i_hi] = canon::swiglu_bf16(y2, u2);
                }
                if (c1 < n_cols) {
                    sm.epi.h[c1][i_lo] = canon::swiglu_bf16(y1, u1);
                    sm.epi.h[c1][i_hi] = canon::swiglu_bf16(y3, u3);
                }
            }
        }
        __syncthreads();
        if (tid < n_cols) {
            float hv[16];
            float amax = 0.0F;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float f = canon::bf16_to_f32(sm.epi.h[tid][j]);
                hv[j]         = f;
                const float a = fabsf(f);
                amax          = a > amax || a != a ? a : amax;
            }
            unsigned low = 0U, high = 0U;
            std::uint8_t sw = 0U;
            canon::quantize_a4_plane_from_f32(hv, amax, scales.input_down, low, high, sw);
            const int pos       = (slice & ~3) + moe::wide::plane_block(slice & 3);
            std::uint8_t* h_row = h_base + static_cast<std::size_t>(first + c_base + tid) * moe::wide::kHRowStride;
            *reinterpret_cast<uint2*>(h_row + 8 * pos) = make_uint2(low, high);
            h_row[moe::wide::kHCodeBytes + pos]        = sw;
        }
        if (c_base + 8 < count) { __syncthreads(); }
    }
}

template <class G>
__device__ __forceinline__ void down_omma_tile_compute(DownOmmaShared<G>& sm, const MoeDispatch& dispatch,
                                                       const std::uint8_t* record, const moe::ExpertScales& scales,
                                                       int first, int count, int tile, const std::uint8_t* h_base,
                                                       bf16* outputs, bool preloaded_stage0) {
    using S        = OmmaShape<G>;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int t     = lane & 3;
    const int rg    = 4 * tile + warp;
    const std::uint8_t* rg_base =
        record + G::kGateUpBytes + static_cast<std::size_t>(4 * tile) * S::kDownRgBytes;

    const auto stage_w = [&](int st, int buf) {
        constexpr int kVecs = S::kDownWarpStageB / 16; // 72
        const std::uint8_t* src =
            rg_base + static_cast<std::size_t>(warp) * S::kDownRgBytes + static_cast<std::size_t>(st) * S::kDownWarpStageB;
        for (int i = lane; i < kVecs; i += 32) { ggml::tile::cp_async16(sm.w[buf][warp] + 16 * i, src + 16 * i, 16); }
    };

    for (int c_base = 0; c_base < count; c_base += 8) {
        const int n_cols = min(8, count - c_base);
        if (!preloaded_stage0 || c_base > 0) {
            stage_w(0, 0);
            ggml::tile::cp_async_commit();
        }
        constexpr int kHVecs = S::kHRowBytes / 16; // 23
        for (int i = tid; i < n_cols * kHVecs; i += S::kThreads) {
            const int c = i / kHVecs;
            const int k = i % kHVecs;
            ggml::tile::cp_async16(
                sm.h[c] + 16 * k,
                h_base + static_cast<std::size_t>(first + c_base + c) * moe::wide::kHRowStride + 16 * k, 16);
        }
        ggml::tile::cp_async_commit();
        stage_w(1, 1);
        ggml::tile::cp_async_commit();
        ggml::tile::cp_async_wait<1>();
        __syncthreads();

        float acc[4]              = {0.0F, 0.0F, 0.0F, 0.0F};
        const std::uint8_t* h_row = sm.h[min(group, n_cols - 1)];
        for (int st = 0; st < S::kDownChunks; ++st) {
            const std::uint8_t* wu = sm.w[st & 1][warp];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int k64 = 2 * st + half;
                std::uint32_t a0, a1, a2, a3, sfa;
                rg16_a_fragments_seq(wu + half * 4 * moe::kUnitBytes, group, t, a0, a1, a2, a3, sfa);
                const std::uint32_t b0  = omma_lds32(h_row + 32 * k64 + 4 * t);
                const std::uint32_t b1  = omma_lds32(h_row + 32 * k64 + 16 + 4 * t);
                const std::uint32_t sfb = omma_lds32(h_row + moe::wide::kHCodeBytes + 4 * k64);
                mma_nvfp4_e4m3(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
            __syncthreads();
            if (st + 2 < S::kDownChunks) {
                stage_w(st + 2, st & 1);
                ggml::tile::cp_async_commit();
            }
            if (st + 1 < S::kDownChunks) {
                if (st + 2 < S::kDownChunks) {
                    ggml::tile::cp_async_wait<1>();
                } else {
                    ggml::tile::cp_async_wait<0>();
                }
                __syncthreads();
            }
        }

        const float alpha = scales.alpha_down;
        const int row0    = 16 * rg + group;
        const int row1    = row0 + 8;
        auto* out_u16     = reinterpret_cast<std::uint16_t*>(outputs);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int col_local = 2 * t + i;
            if (col_local < n_cols) {
                const int out_col = dispatch.entries[first + c_base + col_local];
                std::uint16_t* dst = out_u16 + static_cast<std::size_t>(out_col) * G::kHidden;
                dst[row0]          = canon::f32_to_bf16_rn(canon::mul_rn(acc[i], alpha));
                dst[row1]          = canon::f32_to_bf16_rn(canon::mul_rn(acc[2 + i], alpha));
            }
        }
        if (c_base + 8 < count) { __syncthreads(); }
    }
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 3)
    gate_up_omma_kernel(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                        const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ flags,
                        int base, std::uint8_t* __restrict__ h_base, PassPhase phase) {
    __shared__ GateUpOmmaShared<G> sm;
    pdl::wait_for_dependencies();
    const int job   = base + static_cast<int>(blockIdx.y);
    const int slice = static_cast<int>(blockIdx.x);
    if (job < *dispatch.job_count && (flags == nullptr || flags[job] == 0)) {
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
        if (record != nullptr) {
            const moe::ExpertScales scales = source.scales[expert];
            const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
            if (count > 0 && !moe::wide_route(count, scales)) {
                gate_up_omma_slice<G>(sm, x, dispatch, top_k, record, scales, first, count, slice, h_base);
            }
        }
    }
    pdl::trigger_dependents();
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 3)
    down_omma_kernel(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* __restrict__ job_records,
                     const std::int32_t* __restrict__ flags, int base, const std::uint8_t* __restrict__ h_base,
                     bf16* __restrict__ outputs, PassPhase phase) {
    using S = OmmaShape<G>;
    __shared__ DownOmmaShared<G> sm;
    const int job  = base + static_cast<int>(blockIdx.y);
    const int tile = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    moe::ExpertScales scales{};
    bool preloaded = false;
    // dispatch.* is read-only across gate/up -> down, so prefetching immutable down weights stage 0
    // before the PDL wait hides latency without depending on h_base.
    if (job < *dispatch.job_count && (flags == nullptr || flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
        if (record != nullptr) {
            scales = source.scales[expert];
            first  = dispatch.offsets[expert];
            count  = dispatch.offsets[expert + 1] - first;
            if (count <= 0 || moe::wide_route(count, scales)) {
                record = nullptr;
            } else {
                const std::uint8_t* rg_base =
                    record + G::kGateUpBytes + static_cast<std::size_t>(4 * tile) * S::kDownRgBytes;
                const std::uint8_t* src = rg_base + static_cast<std::size_t>(warp) * S::kDownRgBytes;
                for (int i = lane; i < S::kDownWarpStageB / 16; i += 32) {
                    ggml::tile::cp_async16(sm.w[0][warp] + 16 * i, src + 16 * i, 16);
                }
                ggml::tile::cp_async_commit();
                preloaded = true;
            }
        }
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    down_omma_tile_compute<G>(sm, dispatch, record, scales, first, count, tile, h_base, outputs, preloaded);
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 3)
    gate_up_omma_persistent_kernel(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                                   const std::uint8_t* const* __restrict__ job_records,
                                   const std::int32_t* __restrict__ narrow_jobs,
                                   const std::int32_t* __restrict__ pass_narrow, int pass_index,
                                   std::uint8_t* __restrict__ h_base) {
    using S = OmmaShape<G>;
    __shared__ GateUpOmmaShared<G> sm;
    pdl::wait_for_dependencies();
    const int n_begin     = pass_narrow[pass_index];
    const int n_count     = pass_narrow[pass_index + 1] - n_begin;
    const int total_tasks = n_count * S::kGateUpSlices;
    for (int task = static_cast<int>(blockIdx.x); task < total_tasks; task += static_cast<int>(gridDim.x)) {
        const int job              = narrow_jobs[n_begin + task / S::kGateUpSlices];
        const int slice            = task % S::kGateUpSlices;
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(PassPhase::All, source, expert, job_records, job);
        if (record == nullptr) { continue; }
        const moe::ExpertScales scales = source.scales[expert];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        if (count <= 0 || moe::wide_route(count, scales)) { continue; }
        gate_up_omma_slice<G>(sm, x, dispatch, top_k, record, scales, first, count, slice, h_base);
        __syncthreads();
    }
    pdl::trigger_dependents();
}

template <class G>
__global__ void __launch_bounds__(OmmaShape<G>::kThreads, 3)
    down_omma_persistent_kernel(MoeDispatch dispatch, MoeExpertSource source,
                                const std::uint8_t* const* __restrict__ job_records,
                                const std::int32_t* __restrict__ narrow_jobs,
                                const std::int32_t* __restrict__ pass_narrow, int pass_index,
                                const std::uint8_t* __restrict__ h_base, bf16* __restrict__ outputs) {
    using S = OmmaShape<G>;
    __shared__ DownOmmaShared<G> sm;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int n_begin     = pass_narrow[pass_index];
    const int n_count     = pass_narrow[pass_index + 1] - n_begin;
    const int total_tasks = n_count * S::kDownTiles;
    const int task0       = static_cast<int>(blockIdx.x);

    bool preloaded0            = false;
    const std::uint8_t* rec0   = nullptr;
    int first0 = 0, count0 = 0, tile0 = 0;
    moe::ExpertScales scales0{};
    if (task0 < total_tasks) {
        const int job = narrow_jobs[n_begin + task0 / S::kDownTiles];
        tile0         = task0 % S::kDownTiles;
        const int exp = dispatch.jobs[job];
        rec0          = pass_record(PassPhase::All, source, exp, job_records, job);
        if (rec0 != nullptr) {
            scales0 = source.scales[exp];
            first0  = dispatch.offsets[exp];
            count0  = dispatch.offsets[exp + 1] - first0;
            if (count0 <= 0 || moe::wide_route(count0, scales0)) {
                rec0 = nullptr;
            } else {
                const std::uint8_t* rg_base =
                    rec0 + G::kGateUpBytes + static_cast<std::size_t>(4 * tile0) * S::kDownRgBytes;
                const std::uint8_t* src = rg_base + static_cast<std::size_t>(warp) * S::kDownRgBytes;
                for (int i = lane; i < S::kDownWarpStageB / 16; i += 32) {
                    ggml::tile::cp_async16(sm.w[0][warp] + 16 * i, src + 16 * i, 16);
                }
                ggml::tile::cp_async_commit();
                preloaded0 = true;
            }
        }
    }
    pdl::wait_for_dependencies();
    if (task0 < total_tasks && rec0 != nullptr) {
        down_omma_tile_compute<G>(sm, dispatch, rec0, scales0, first0, count0, tile0, h_base, outputs, preloaded0);
        __syncthreads();
    }
    for (int task = task0 + static_cast<int>(gridDim.x); task < total_tasks; task += static_cast<int>(gridDim.x)) {
        const int job              = narrow_jobs[n_begin + task / S::kDownTiles];
        const int tile             = task % S::kDownTiles;
        const int expert           = dispatch.jobs[job];
        const std::uint8_t* record = pass_record(PassPhase::All, source, expert, job_records, job);
        if (record == nullptr) { continue; }
        const moe::ExpertScales scales = source.scales[expert];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        if (count <= 0 || moe::wide_route(count, scales)) { continue; }
        down_omma_tile_compute<G>(sm, dispatch, record, scales, first, count, tile, h_base, outputs, false);
        __syncthreads();
    }
}

void launch_expert_pass_a4_omma(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                canon::A4Block* h_blocks, const moe::wide::ExpertsWorkspace& layout, int pass_index,
                                bool use_xplane, bf16* outputs, cudaStream_t stream, PassPhase phase, PassPart part) {
    using G                    = moe::FlashNextExperts;
    using S                    = OmmaShape<G>;
    auto* h_base               = reinterpret_cast<std::uint8_t*>(h_blocks);
    const bool unstaged_direct = source.staging_slots == 0 && source.fetch.request == nullptr && phase == PassPhase::All;
    if (use_xplane) {
        const int persistent = persistent_narrow_ctas();
        if (part != PassPart::Down) {
            CUDA_CHECK(pdl::launch_with(unstaged_direct ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized,
                                        {dim3(persistent), dim3(S::kThreads), 0, stream},
                                        gate_up_omma_persistent_kernel<G>, x, dispatch, source, top_k, job_records,
                                        layout.narrow_jobs, layout.pass_narrow, pass_index, h_base));
            check_launch("A4 OMMA gate/up xplane");
        }
        if (part == PassPart::GateUp) { return; }
        const auto dep = part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic;
        CUDA_CHECK(pdl::launch_with(dep, {dim3(persistent), dim3(S::kThreads), 0, stream},
                                    down_omma_persistent_kernel<G>, dispatch, source, job_records, layout.narrow_jobs,
                                    layout.pass_narrow, pass_index, h_base, outputs));
        check_launch("A4 OMMA down xplane");
        return;
    }
    if (part != PassPart::Down) {
        const auto gu_dep = unstaged_direct ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized;
        CUDA_CHECK(pdl::launch_with(gu_dep, {dim3(S::kGateUpSlices, jobs), dim3(S::kThreads), 0, stream},
                                    gate_up_omma_kernel<G>, x, dispatch, source, top_k, job_records, flags, base,
                                    h_base, phase));
        check_launch("A4 OMMA gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    const auto dep = part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic;
    CUDA_CHECK(pdl::launch_with(dep, {dim3(S::kDownTiles, jobs), dim3(S::kThreads), 0, stream}, down_omma_kernel<G>,
                                dispatch, source, job_records, flags, base, h_base, outputs, phase));
    check_launch("A4 OMMA down");
}
