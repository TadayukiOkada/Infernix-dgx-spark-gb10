#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// Two registered parents: fp8_e4m3fn_row_bf16 [16384,5120] (q/k/v 10240, z 6144) and the
// Qwen3.5-122B-A10B static-scale fp8_e4m3fn_row_f32 [20480,3072] (q/k/v 12288, z 8192), told apart
// by K. The latter's A8 route encodes with the parent's input scale and has no residual profile.
enum class Fp8GdnInputScheduleId {
    A8T32R32K128,
    A8T64R128K256,
    A8Tma64x128,
    A8Tma192x128,
    A8MidBulk,
    A8Bulk,
};

struct Fp8GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

const char* fp8_gdn_input_schedule_name(Fp8GdnInputScheduleId schedule) noexcept;
bool fp8_gdn_input_candidate_is_legal(Fp8GdnInputScheduleId schedule,
                                      const Fp8GdnInputProblem& problem) noexcept;
void fp8_gdn_input_execute_candidate(Fp8GdnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& qkv, Tensor& z,
                                     WorkspaceArena& workspace, cudaStream_t stream);
// Largest split-K partial plane any candidate schedule may need, so a bench arena sized with this
// can force MidBulk/Bulk at any token count without tripping the aligned-partials check.
[[nodiscard]] std::size_t fp8_gdn_input_candidate_partial_capacity_bytes();

[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens);
[[nodiscard]] std::size_t fp8_gdn_input_n20480_partial_capacity_bytes(std::int32_t max_tokens);
// The A8 partials of the parent with this K.
[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t input_rows,
                                                               std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

void fp8_gdn_input_n16384_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream);
void fp8_gdn_input_n20480_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream);
void fp8_gdn_input_n20480_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream);
void fp8_gdn_input_n20480_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);
// AllowA8Residual: the workspace's two E4M3 planes into two accumulators (MMA tiles).
void fp8_gdn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream);

// Exact contraction mechanisms shared by G1/G2/G3. Semantic Ops own their route frontier and
// call one of these launchers after resolving their complete-form plan.
// A16ColumnInvariant: one sliced-K schedule (4 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream);
void fp8_gdn_input_a8_residual_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                        Tensor& z, WorkspaceArena& workspace,
                                        cudaStream_t stream);

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

// A8 GEMM schedules reachable for the 122B [20480,3072] shape, exposed as opaque ids so the bench
// can drive every legal schedule directly (candidate sweep) instead of the production route. The
// Wave48 split-K ids carry the GB10-resident-grid wave model (48 SMs) instead of the [170]-SM one.
enum class Fp8GdnInputN20480ScheduleId {
    A8T32R32K128,
    A8T64R128K256,
    A8Tma64x128,
    A8Tma64x128G8,
    A8Tma192x128,
    A8Tma192x128G8,
    A8Tma64x256G8,
    A8Tma128x128G8,
    A8MidBulk,
    A8Bulk,
    A8BulkG8,
    A8BulkG8Wave48,
};

const char* fp8_gdn_input_n20480_schedule_name(Fp8GdnInputN20480ScheduleId schedule) noexcept;
bool fp8_gdn_input_n20480_candidate_is_legal(Fp8GdnInputN20480ScheduleId schedule,
                                             const Fp8GdnInputProblem& problem) noexcept;
void fp8_gdn_input_n20480_execute_candidate(Fp8GdnInputN20480ScheduleId schedule, const Tensor& x,
                                            const Weight& weight, Tensor& qkv, Tensor& z,
                                            WorkspaceArena& workspace, cudaStream_t stream);
// Largest split-K partial plane any candidate schedule may need, so a bench arena sized with this
// can force any split-K schedule at any token count.
[[nodiscard]] std::size_t fp8_gdn_input_n20480_candidate_partial_capacity_bytes();

} // namespace infernix::ops::detail
