#include "ops/offloaded_sparse_moe/cuda/narrow_expert.h"
#include "core/pdl.cuh"

#include <stdexcept>
#include <string>

namespace infernix::ops::offloaded_moe {
namespace {

using canon::A4Block;

constexpr int kThreads = 256;

struct Workspace {
    A4Block* x_gate; // [job][kMaxColumns][kGateUpBlocks]
    A4Block* x_up;   // [job][kMaxColumns][kGateUpBlocks]
    A4Block* h;      // [job][kMaxColumns][kHBlocks]
};

__host__ __device__ inline Workspace carve(void* base, int jobs) {
    auto* p           = static_cast<A4Block*>(base);
    const std::size_t x = static_cast<std::size_t>(jobs) * kMaxColumns * kGateUpBlocks;
    return {p, p + x, p + 2 * x};
}

// Four doubled E2M1 codes of row r (0..15) at k = 4q..4q+3 of a unit staged in SMEM, packed as signed bytes.
__device__ __forceinline__ int smem_row_quad(const std::uint32_t* u32, int r, int q) {
    const std::uint32_t word = u32[8 * q + (r & 7)];
    const int shift          = r < 8 ? 0 : 4;
    return static_cast<int>(canon::e2m1_x2_quad((word >> shift) & 0x0F0F0F0FU));
}

__device__ __forceinline__ int act_quad(const A4Block& a, int q) {
    return *reinterpret_cast<const int*>(&a.c2[4 * q]);
}

__global__ void __launch_bounds__(kThreads) quantize_x_kernel(const NarrowJob* jobs, Workspace ws) {
    const NarrowJob job = jobs[blockIdx.x];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kMaxColumns * kGateUpBlocks;
    for (int i = threadIdx.x; i < job.ncols * kGateUpBlocks; i += blockDim.x) {
        const int c = i / kGateUpBlocks, b = i % kGateUpBlocks;
        const std::uint16_t* v = job.x + static_cast<std::size_t>(c) * kHidden + 16 * b;
        ws.x_gate[base + i] = canon::quantize_a4_block(v, job.scales.input_gate);
        ws.x_up[base + i]   = job.scales.input_up == job.scales.input_gate
                                  ? ws.x_gate[base + i]
                                  : canon::quantize_a4_block(v, job.scales.input_up);
    }
    pdl::trigger_dependents();
}

// One CTA per (16-intermediate unit u, job): row groups 2u and 2u+1, gate rows even, up rows odd.
// All 8 warps split K (b = warp, warp+8, ...) and reuse loaded weights across all ncols.
__global__ void __launch_bounds__(kThreads, 2) gate_up_kernel(const NarrowJob* jobs, Workspace ws) {
    const int u         = blockIdx.x;
    const int j         = blockIdx.y;
    const NarrowJob job = jobs[j];
    const int ncols     = job.ncols;
    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int half      = lane >> 4;
    const int rg        = 2 * u + half;
    const int r         = lane & 15;
    const bool gate     = (lane & 1) == 0;
    const std::uint8_t* rg_base = job.record + static_cast<std::size_t>(rg) * kGateUpBlocks * kUnitBytes;

    // 10 uint4s per half-warp (9 used = 144 bytes, 10th pads stride to 160B for 0 bank conflicts).
    __shared__ uint4 smem_units[8][2][10];
    auto load_unit_smem = [&](int b) {
        if (r < 9) {
            smem_units[warp][half][r] =
                reinterpret_cast<const uint4*>(rg_base + b * kUnitBytes)[r];
        }
        __syncwarp();
    };

    // Preload first unit (immutable weights) before waiting on quantize_x_kernel.
    load_unit_smem(warp);
    pdl::wait_for_dependencies();

    const A4Block* acts_base =
        (gate ? ws.x_gate : ws.x_up) + static_cast<std::size_t>(j) * kMaxColumns * kGateUpBlocks;
    const std::uint32_t* u32 = reinterpret_cast<const std::uint32_t*>(&smem_units[warp][half][0]);
    const std::uint8_t* u8s  = reinterpret_cast<const std::uint8_t*>(&smem_units[warp][half][8]);

    std::int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    const auto accum_col = [&](int c, std::int64_t& sc, int b, int rq0, int rq1, int rq2, int rq3, int sw) {
        if (c < ncols) {
            const A4Block& a = acts_base[static_cast<std::size_t>(c) * kGateUpBlocks + b];
            int p            = 0;
            p                = __dp4a(rq0, act_quad(a, 0), p);
            p                = __dp4a(rq1, act_quad(a, 1), p);
            p                = __dp4a(rq2, act_quad(a, 2), p);
            p                = __dp4a(rq3, act_quad(a, 3), p);
            sc += static_cast<std::int64_t>(p * sw) * a.scale_scaled;
        }
    };
    if (ncols == 1) {
        for (int b = warp; b < kGateUpBlocks; b += 8) {
            if (b != warp) { load_unit_smem(b); }
            const int rq0 = smem_row_quad(u32, r, 0);
            const int rq1 = smem_row_quad(u32, r, 1);
            const int rq2 = smem_row_quad(u32, r, 2);
            const int rq3 = smem_row_quad(u32, r, 3);
            const int sw  = canon::e4m3_scaled(u8s[r]);
            accum_col(0, s0, b, rq0, rq1, rq2, rq3, sw);
        }
    } else {
        for (int b = warp; b < kGateUpBlocks; b += 8) {
            if (b != warp) { load_unit_smem(b); }
            const int rq0 = smem_row_quad(u32, r, 0);
            const int rq1 = smem_row_quad(u32, r, 1);
            const int rq2 = smem_row_quad(u32, r, 2);
            const int rq3 = smem_row_quad(u32, r, 3);
            const int sw  = canon::e4m3_scaled(u8s[r]);
            accum_col(0, s0, b, rq0, rq1, rq2, rq3, sw);
            accum_col(1, s1, b, rq0, rq1, rq2, rq3, sw);
            accum_col(2, s2, b, rq0, rq1, rq2, rq3, sw);
            accum_col(3, s3, b, rq0, rq1, rq2, rq3, sw);
            accum_col(4, s4, b, rq0, rq1, rq2, rq3, sw);
            accum_col(5, s5, b, rq0, rq1, rq2, rq3, sw);
            accum_col(6, s6, b, rq0, rq1, rq2, rq3, sw);
            accum_col(7, s7, b, rq0, rq1, rq2, rq3, sw);
        }
    }

    __shared__ std::int64_t smem_sum[8][32];
    __shared__ std::uint16_t y[kMaxColumns][32];
    __shared__ std::uint16_t h[kMaxColumns][16];
    const float alpha = gate ? job.scales.alpha_gate : job.scales.alpha_up;
    const auto reduce_col = [&](int c, std::int64_t sc) {
        if (c < ncols) {
            smem_sum[warp][lane] = sc;
            __syncthreads();
            if (warp == 0) {
                std::int64_t total = 0;
#pragma unroll
                for (int w = 0; w < 8; ++w) { total += smem_sum[w][lane]; }
                y[c][lane] = canon::a4_row_output(total, alpha);
            }
            __syncthreads();
        }
    };
    reduce_col(0, s0);
    if (ncols > 1) {
        reduce_col(1, s1);
        reduce_col(2, s2);
        reduce_col(3, s3);
        reduce_col(4, s4);
        reduce_col(5, s5);
        reduce_col(6, s6);
        reduce_col(7, s7);
    }
    for (int i = threadIdx.x; i < 16 * ncols; i += blockDim.x) {
        const int k = i % 16, c = i / 16;
        h[c][k] = canon::swiglu_bf16(y[c][2 * k], y[c][2 * k + 1]);
    }
    __syncthreads();
    if (threadIdx.x < ncols) {
        const int c = threadIdx.x;
        ws.h[(static_cast<std::size_t>(j) * kMaxColumns + c) * kHBlocks + u] =
            canon::quantize_a4_block(h[c], job.scales.input_down);
    }
    pdl::trigger_dependents();
}

// One CTA per (4 down row groups, job): 64 output rows for every column.
// 4 K-splits x 2 row-halves (all 8 warps active even at ncols == 1).
__global__ void __launch_bounds__(kThreads, 2) down_kernel(const NarrowJob* jobs, Workspace ws) {
    const int j         = blockIdx.y;
    const NarrowJob job = jobs[j];
    const int ncols     = job.ncols;
    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int half      = lane >> 4;
    const int k_split   = warp >> 1;
    const int row       = ((warp & 1) << 5) + lane;
    const int rg        = 4 * blockIdx.x + (row >> 4);
    const int r         = row & 15;
    const std::uint8_t* down    = job.record + kGateUpBytes;
    const std::uint8_t* rg_base = down + static_cast<std::size_t>(rg) * kDownBlocks * kUnitBytes;

    __shared__ uint4 smem_units[8][2][10];
    auto load_unit_smem = [&](int b) {
        if (r < 9) {
            smem_units[warp][half][r] =
                reinterpret_cast<const uint4*>(rg_base + b * kUnitBytes)[r];
        }
        __syncwarp();
    };

    // Preload first unit (immutable weights) before waiting on gate_up_kernel.
    load_unit_smem(k_split);
    pdl::wait_for_dependencies();

    const A4Block* acts_base    = ws.h + static_cast<std::size_t>(j) * kMaxColumns * kHBlocks;
    const std::uint32_t* u32    = reinterpret_cast<const std::uint32_t*>(&smem_units[warp][half][0]);
    const std::uint8_t* u8s     = reinterpret_cast<const std::uint8_t*>(&smem_units[warp][half][8]);

    std::int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    const auto accum_col = [&](int c, std::int64_t& sc, int b, int rq0, int rq1, int rq2, int rq3, int sw) {
        if (c < ncols) {
            const A4Block& a = acts_base[static_cast<std::size_t>(c) * kHBlocks + b];
            int p            = 0;
            p                = __dp4a(rq0, act_quad(a, 0), p);
            p                = __dp4a(rq1, act_quad(a, 1), p);
            p                = __dp4a(rq2, act_quad(a, 2), p);
            p                = __dp4a(rq3, act_quad(a, 3), p);
            sc += static_cast<std::int64_t>(p * sw) * a.scale_scaled;
        }
    };
    if (ncols == 1) {
        for (int b = k_split; b < kDownBlocks; b += 4) {
            if (b != k_split) { load_unit_smem(b); }
            const int rq0 = smem_row_quad(u32, r, 0);
            const int rq1 = smem_row_quad(u32, r, 1);
            const int rq2 = smem_row_quad(u32, r, 2);
            const int rq3 = smem_row_quad(u32, r, 3);
            const int sw  = canon::e4m3_scaled(u8s[r]);
            accum_col(0, s0, b, rq0, rq1, rq2, rq3, sw);
        }
    } else {
        for (int b = k_split; b < kDownBlocks; b += 4) {
            if (b != k_split) { load_unit_smem(b); }
            const int rq0 = smem_row_quad(u32, r, 0);
            const int rq1 = smem_row_quad(u32, r, 1);
            const int rq2 = smem_row_quad(u32, r, 2);
            const int rq3 = smem_row_quad(u32, r, 3);
            const int sw  = canon::e4m3_scaled(u8s[r]);
            accum_col(0, s0, b, rq0, rq1, rq2, rq3, sw);
            accum_col(1, s1, b, rq0, rq1, rq2, rq3, sw);
            accum_col(2, s2, b, rq0, rq1, rq2, rq3, sw);
            accum_col(3, s3, b, rq0, rq1, rq2, rq3, sw);
            accum_col(4, s4, b, rq0, rq1, rq2, rq3, sw);
            accum_col(5, s5, b, rq0, rq1, rq2, rq3, sw);
            accum_col(6, s6, b, rq0, rq1, rq2, rq3, sw);
            accum_col(7, s7, b, rq0, rq1, rq2, rq3, sw);
        }
    }

    __shared__ std::int64_t smem_sum[4][64];
    const auto reduce_col = [&](int c, std::int64_t sc) {
        if (c < ncols) {
            smem_sum[k_split][row] = sc;
            __syncthreads();
            if (k_split == 0) {
                std::int64_t total = 0;
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) { total += smem_sum[ks][row]; }
                job.y[static_cast<std::size_t>(c) * kHidden + 16 * rg + r] =
                    canon::a4_row_output(total, job.scales.alpha_down);
            }
            __syncthreads();
        }
    };
    reduce_col(0, s0);
    if (ncols > 1) {
        reduce_col(1, s1);
        reduce_col(2, s2);
        reduce_col(3, s3);
        reduce_col(4, s4);
        reduce_col(5, s5);
        reduce_col(6, s6);
        reduce_col(7, s7);
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string("offloaded_moe narrow route: ") + what + ": " + cudaGetErrorString(e)); }
}

} // namespace

std::size_t narrow_workspace_bytes(int jobs) {
    return static_cast<std::size_t>(jobs) * kMaxColumns * (2 * kGateUpBlocks + kHBlocks) * sizeof(canon::A4Block);
}

void launch_narrow_experts(const NarrowJob* jobs, int job_count, void* workspace, cudaStream_t stream) {
    if (job_count <= 0) { return; }
    if (job_count > 65535) { throw std::invalid_argument("offloaded_moe narrow route: too many jobs"); }
    const Workspace ws = carve(workspace, job_count);
    quantize_x_kernel<<<job_count, kThreads, 0, stream>>>(jobs, ws);
    check(cudaGetLastError(), "quantize");
    check(pdl::launch_consumer({dim3(kHBlocks, job_count), kThreads, 0, stream}, gate_up_kernel, jobs, ws),
          "gate/up");
    check(pdl::launch_consumer({dim3(kDownRowGroups / 4, job_count), kThreads, 0, stream}, down_kernel, jobs, ws),
          "down");
}

} // namespace infernix::ops::offloaded_moe
