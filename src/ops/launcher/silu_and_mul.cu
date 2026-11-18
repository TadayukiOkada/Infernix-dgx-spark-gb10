// infernix::ops — silu_mul launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/silu_and_mul.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/silu_and_mul.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

bool can_use_dim0_split_fast_path(const Tensor& gate, const Tensor& up, const Tensor& out) {
    if (gate.ne[2] != 1 || gate.ne[3] != 1) { return false; }
    if ((gate.ne[0] & 1) != 0) { return false; }
    if (gate.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        up.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16))) {
        return false;
    }
    if ((gate.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0 ||
        (up.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0) {
        return false;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    return ((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) == 0;
}

} // namespace

void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 128;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split_fast_path(gate, up, out)) {
            const std::int64_t row_pairs          = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
            const int grid_x =
                static_cast<int>(std::max<std::int64_t>(1, div_up(row_pairs, kPairsPerBlock)));
            for_each_token_slice(gate.ne[1], 1, [&](int token_offset, int token_count) {
                const Tensor gate_slice = gate.slice(1, token_offset, token_count);
                const Tensor up_slice   = up.slice(1, token_offset, token_count);
                Tensor out_slice        = out.slice(1, token_offset, token_count);
                const dim3 grid(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(token_count));
                silu_and_mul_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(gate_slice.data),
                    static_cast<const __nv_bfloat16*>(up_slice.data),
                    static_cast<__nv_bfloat16*>(out_slice.data), gate.ne[0],
                    gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                    up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
                CUDA_CHECK(cudaGetLastError());
            });
            return;
        }
        const int scalar_grid = static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_and_mul_strided_input_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n,
            gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3],
            up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);

    // Fast 128-bit vectorized path for 16-byte aligned tensors (LDG.128 / STG.128)
    constexpr std::int64_t kElemsPerVector = 8;
    if (((gate_addr | up_addr | out_addr) & (alignof(uint4) - 1)) == 0 && n >= kElemsPerVector) {
        const auto* gate_bf = static_cast<const __nv_bfloat16*>(gate.data);
        const auto* up_bf   = static_cast<const __nv_bfloat16*>(up.data);
        auto* out_bf        = static_cast<__nv_bfloat16*>(out.data);
        if (n == 6144) {
            silu_and_mul_exact_kernel<16><<<48, 16, 0, stream>>>(gate_bf, up_bf, out_bf);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (n == 98304) {
            silu_and_mul_exact_kernel<128><<<96, 128, 0, stream>>>(gate_bf, up_bf, out_bf);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (n == 786432) {
            silu_and_mul_prefill_staged_kernel<<<768, 128, 0, stream>>>(gate_bf, up_bf, out_bf);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const std::int64_t num_vectors = n / kElemsPerVector;
        constexpr int kBlock256 = 256;
        const auto min_blocks = static_cast<unsigned int>(div_up(num_vectors, static_cast<std::int64_t>(kBlock256)));
        const unsigned int grid = std::min<unsigned int>(min_blocks, 384u);

        silu_and_mul_uint4_kernel<<<grid, kBlock256, 0, stream>>>(
            reinterpret_cast<const uint4*>(gate.data),
            reinterpret_cast<const uint4*>(up.data),
            reinterpret_cast<uint4*>(out.data),
            num_vectors);
        CUDA_CHECK(cudaGetLastError());

        const std::int64_t remainder = n % kElemsPerVector;
        if (remainder > 0) {
            const std::int64_t tail_start = num_vectors * kElemsPerVector;
            silu_and_mul_scalar_kernel<<<1, static_cast<int>(remainder), 0, stream>>>(
                static_cast<const __nv_bfloat16*>(gate.data) + tail_start,
                static_cast<const __nv_bfloat16*>(up.data) + tail_start,
                static_cast<__nv_bfloat16*>(out.data) + tail_start,
                remainder);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    if (((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_and_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
    const int grid = static_cast<int>(std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock)));

    silu_and_mul_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                                     static_cast<const __nv_bfloat16*>(up.data),
                                                     static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
