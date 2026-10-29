#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_stream.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <stdexcept>

namespace infernix::ops::detail {

// One launch per grid.y-sized token slice; each CTA covers up to kMaxCols columns.
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_stream(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.k != Schedule::kK)
        throw std::invalid_argument("Q8 register-streamed route requires its static K");
    for_each_token_slice(operands.tokens, Schedule::kMaxCols, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows, Schedule::kBlockRows), div_up(count, Schedule::kMaxCols));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const auto launch = [&]<bool FullRows, class S = Schedule>() {
            constexpr auto kernel = q8_a16_stream_kernel<S, FullRows, Output, Epilogue>;
            const int shared      = q8_prepare_shared<S::kSharedBytes, kernel>();
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(S::kThreads), static_cast<std::size_t>(shared), stream},
                                            kernel, x, operands.codes, operands.scales, output, epilogue,
                                            operands.rows, operands.padded_k, count, offset));
        };
        const auto dispatch_rows = [&]<class S = Schedule>() {
            if (operands.rows % S::kBlockRows == 0)
                launch.template operator()<true, S>();
            else
                launch.template operator()<false, S>();
        };
        if constexpr (Schedule::kPasses == 1) {
            switch (count) {
                case 1:
                    if constexpr (Schedule::kColsPerPass >= 1)
                        return dispatch_rows.template operator()<typename Schedule::template with_cols<1>>();
                    break;
                case 2:
                    if constexpr (Schedule::kColsPerPass >= 2)
                        return dispatch_rows.template operator()<typename Schedule::template with_cols<2>>();
                    break;
                case 5:
                    if constexpr (Schedule::kColsPerPass >= 5 && Schedule::kChunks <= 5)
                        return dispatch_rows.template operator()<typename Schedule::template with_cols<5>>();
                    break;
                default:
                    break;
            }
        }
        dispatch_rows.template operator()<Schedule>();
    });
}

template <class Schedule>
void launch_q8_a16_stream_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_stream<Schedule>(q8_linear_operands(x, weight),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                   LinearIdentityEpilogue{}, stream);
}

} // namespace infernix::ops::detail
