#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t fp8_linear_add_partial_capacity_bytes(std::int32_t input_rows,
                                                                std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                  std::int32_t input_rows,
                                                                  LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
void fp8_linear_add_matrix_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
// A16ColumnInvariant: one sliced-K schedule (8 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_linear_add_invariant_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream);

// Qwen3.5-122B-A10B's static-scale fp8_e4m3fn_row_f32 [3072,8192] (attention and GDN output).
[[nodiscard]] std::size_t fp8_linear_add_n3072_partial_capacity_bytes(std::int32_t max_tokens);
void fp8_linear_add_n3072_invariant_launch(const Tensor& x, const Weight& weight,
                                           Tensor& residual, cudaStream_t stream);
void fp8_linear_add_n3072_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);
void fp8_linear_add_n3072_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    WorkspaceArena& workspace, cudaStream_t stream);

// A8 GEMM schedules reachable for the 122B [3072,8192] shape, exposed as opaque ids so the bench
// can drive every legal schedule directly (candidate sweep) instead of the production route.
enum class Fp8LinearAddN3072ScheduleId {
    A8Tma32x64,
    A8Small,
    A8Mid,
    A8Wide,
    A8Tma128x128G8,
    A8Tma192x128G8,
    A8Bulk,
    A8BulkG8,
};

const char* fp8_linear_add_n3072_schedule_name(Fp8LinearAddN3072ScheduleId schedule) noexcept;
bool fp8_linear_add_n3072_candidate_is_legal(Fp8LinearAddN3072ScheduleId schedule,
                                            std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t cols) noexcept;
void fp8_linear_add_n3072_execute_candidate(Fp8LinearAddN3072ScheduleId schedule, const Tensor& x,
                                            const Weight& weight, Tensor& residual,
                                            WorkspaceArena& workspace, cudaStream_t stream);
// Largest split-K partial plane any candidate schedule may need, so a bench arena sized with this
// can force any split-K schedule at any token count.
[[nodiscard]] std::size_t fp8_linear_add_n3072_candidate_partial_capacity_bytes();

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
