#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_mma.cuh"

namespace infernix::ops::detail {

template <int Bytes, auto Kernel>
int q6_prepare_mma_shared() {
    if constexpr (Bytes > 48 * 1024) {
        static const cudaError_t attribute =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(attribute);
        return Bytes;
    } else {
        return 0;
    }
}

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_mma(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    validate_q6_operands(operands);
    if (operands.padded_k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("Q6 MMA padded K must contain complete K tiles");
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x   = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
                          operands.k % Schedule::kBlockK == 0;
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = q6_a16_mma_kernel<Schedule, Full, Output, Epilogue>;
            const int shared      = q6_prepare_mma_shared<Schedule::kSharedBytes, kernel>();
            kernel<<<grid, Schedule::kThreads, shared, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        };
        if (full) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
