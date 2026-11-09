#pragma once

#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

__launch_bounds__(256) __global__
void mtp_pack_fc_input_vec8_kernel(const uint4* __restrict__ embedding_norm,
                                   const uint4* __restrict__ hidden_norm,
                                   uint4* __restrict__ out,
                                   std::int32_t rows_vec8) {
    pdl::enter();
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= rows_vec8) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows_vec8 + col;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows_vec8);
    out[out_base + col]             = embedding_norm[in_idx];
    out[out_base + rows_vec8 + col] = hidden_norm[in_idx];
}

__launch_bounds__(256) __global__
void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                              const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                              std::int32_t rows) {
    pdl::enter();
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]        = embedding_norm[in_idx];
    out[out_base + rows + row] = hidden_norm[in_idx];
}

__launch_bounds__(256) __global__
void mtp_split_attn_in_vec8_kernel(const uint4* __restrict__ attn_in,
                                   uint4* __restrict__ q,
                                   uint4* __restrict__ k,
                                   uint4* __restrict__ gate,
                                   uint4* __restrict__ v,
                                   std::int32_t tokens) {
    pdl::enter();
    constexpr int kVecAttnRows = kMtpAttnRows / 8; // 1792
    constexpr int kVecQRows    = kMtpQRows / 8;    // 768
    constexpr int kVecKvRows   = kMtpKvRows / 8;   // 128

    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kVecAttnRows) * tokens;
    if (idx >= n) { return; }

    const int row         = static_cast<int>(idx % kVecAttnRows);
    const int token       = static_cast<int>(idx / kVecAttnRows);
    const uint4 value     = attn_in[idx];

    if (row < kVecQRows) {
        q[static_cast<std::int64_t>(token) * kVecQRows + row] = value;
        return;
    }
    if (row < kVecQRows + kVecKvRows) {
        const int local = row - kVecQRows;
        k[static_cast<std::int64_t>(token) * kVecKvRows + local] = value;
        return;
    }
    if (row < kVecQRows + kVecKvRows + kVecQRows) {
        const int local = row - kVecQRows - kVecKvRows;
        gate[static_cast<std::int64_t>(token) * kVecQRows + local] = value;
        return;
    }

    const int local = row - kVecQRows - kVecKvRows - kVecQRows;
    v[static_cast<std::int64_t>(token) * kVecKvRows + local] = value;
}

__launch_bounds__(256) __global__
void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                              __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                              std::int32_t tokens) {
    pdl::enter();
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kMtpAttnRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % kMtpAttnRows);
    const int token           = static_cast<int>(idx / kMtpAttnRows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < kMtpQRows) {
        q[static_cast<std::int64_t>(token) * kMtpQRows + row] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows) {
        const int local                                          = row - kMtpQRows;
        k[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows + kMtpQRows) {
        const int local                                            = row - kMtpQRows - kMtpKvRows;
        gate[static_cast<std::int64_t>(token) * kMtpQRows + local] = value;
        return;
    }

    const int local = row - kMtpQRows - kMtpKvRows - kMtpQRows;
    v[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
}

} // namespace infernix::ops
