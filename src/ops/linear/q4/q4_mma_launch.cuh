#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_a16_mma.cuh"

namespace infernix::ops::detail {

template <int Bytes, auto Kernel>
int q4_prepare_mma_shared() {
    if constexpr (Bytes > 48 * 1024) {
        static const cudaError_t attribute =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(attribute);
        return Bytes;
    } else {
        return 0;
    }
}

template <int BlockRows>
struct Q4MmaIdentityRows {
    static constexpr int kOutputRowsPerCta = BlockRows;
    static int output_rows(int rows) { return rows; }
    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return row0 + local_row;
    }
};

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue,
          class RowPolicy = Q4MmaIdentityRows<Schedule::kBlockRows>>
void launch_q4_a16_mma(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q4_operands(operands);
    if (operands.padded_k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("Q4 MMA padded K must contain complete K tiles");
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(
            static_cast<unsigned>(
                div_up(row_policy.output_rows(operands.rows), RowPolicy::kOutputRowsPerCta)),
            static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x          = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full_rows_k = operands.rows % Schedule::kBlockRows == 0 &&
                                 operands.k == operands.padded_k &&
                                 operands.k % Schedule::kBlockK == 0;
        const bool full_tokens    = count % Schedule::kBlockTokens == 0;
        constexpr int kSharedBytes = q4_mma_shared_bytes<Schedule, Epilogue>;
        const auto launch          = [&]<bool FullRowsK, bool FullTokens>() {
            constexpr auto kernel =
                q4_a16_mma_kernel<Schedule, FullRowsK, Output, Epilogue, RowPolicy, FullTokens>;
            const int shared = q4_prepare_mma_shared<kSharedBytes, kernel>();
            kernel<<<grid, Schedule::kThreads, shared, stream>>>(
                x, operands.codes, operands.scales, output, epilogue, row_policy, operands.rows,
                operands.k, count, operands.padded_k, offset);
        };
        constexpr bool independent_tokens = [] {
            if constexpr (requires { Schedule::kIndependentTokenTail; })
                return Schedule::kIndependentTokenTail;
            else
                return false;
        }();
        if constexpr (independent_tokens) {
            // Keep one token-tail variant across full and partial tiles. Row/K bounds
            // remain independent, without changing register policy at exact token tiles.
            if (full_rows_k)
                launch.template operator()<true, false>();
            else
                launch.template operator()<false, false>();
        } else {
            if (full_rows_k && full_tokens)
                launch.template operator()<true, true>();
            else
                launch.template operator()<false, false>();
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
