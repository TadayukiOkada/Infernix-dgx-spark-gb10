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

// A8 GEMM schedules reachable for the 27B attention input projection shape, exposed as opaque ids so
// the bench can drive every legal schedule directly (candidate sweep) instead of the production route.
enum class Fp8AttnInputScheduleId {
    A8T32R32K128,
    A8T32R128K128,
    A8Tma64x128,
    A8Tma64x256,
    A8Tma96x256,
    A8Bulk,
};

struct Fp8AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

const char* fp8_attn_input_schedule_name(Fp8AttnInputScheduleId schedule) noexcept;
bool fp8_attn_input_candidate_is_legal(Fp8AttnInputScheduleId schedule,
                                       const Fp8AttnInputProblem& problem) noexcept;
void fp8_attn_input_execute_candidate(Fp8AttnInputScheduleId schedule, const Tensor& x,
                                      const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                      Tensor& v, WorkspaceArena& workspace, cudaStream_t stream);
// Largest split-K partial plane any candidate schedule may need, so a bench arena sized with this can
// force Bulk at any token count without tripping the aligned-partials check.
[[nodiscard]] std::size_t fp8_attn_input_candidate_partial_capacity_bytes();

[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens);

inline constexpr int kFp8AttnInputLastSmallMmaT = 32;

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// A16ColumnInvariant: one sliced-K schedule (4 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_attn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);
// AllowA8Residual: the workspace's two E4M3 planes into two accumulators (MMA tiles).
void fp8_attn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v,
                                       Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

// Qwen3.5-122B-A10B: one fp8_e4m3fn_row_f32 parent [17408,3072] (query 8192, key 512, gate 8192,
// value 512). AllowA8 encodes with the parent's static input scale from T = 17; a static scale has
// no A8 residual profile.
[[nodiscard]] std::size_t fp8_attn_input_n17408_partial_capacity_bytes(std::int32_t max_tokens);
[[nodiscard]] std::size_t fp8_attn_input_n17408_workspace_capacity_bytes(LinearPolicy policy,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);
void fp8_attn_input_n17408_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                            Tensor& gate, Tensor& k, Tensor& v,
                                            cudaStream_t stream);
void fp8_attn_input_n17408_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_n17408_a8_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                     Tensor& gate, Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                     cudaStream_t stream);
void fp8_attn_input_n17408_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                    Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                    WorkspaceArena* workspace, cudaStream_t stream);

// A8 GEMM schedules reachable for the 122B [17408,3072] shape, exposed as opaque ids so the bench
// can drive every legal schedule directly (candidate sweep) instead of the production route.
enum class Fp8AttnInputN17408ScheduleId {
    A8T32R32K128,
    A8T32R128K128,
    A8Tma64x128,
    A8Tma64x128G8,
    A8Tma64x256,
    A8Tma64x256G8,
    A8Tma96x256,
    A8Tma96x256G8,
    A8Tma128x128G8,
    A8Bulk,
    A8BulkG8,
};

const char* fp8_attn_input_n17408_schedule_name(Fp8AttnInputN17408ScheduleId schedule) noexcept;
bool fp8_attn_input_n17408_candidate_is_legal(Fp8AttnInputN17408ScheduleId schedule,
                                              const Fp8AttnInputProblem& problem) noexcept;
void fp8_attn_input_n17408_execute_candidate(Fp8AttnInputN17408ScheduleId schedule,
                                             const Tensor& x, const Weight& weight, Tensor& q,
                                             Tensor& gate, Tensor& k, Tensor& v,
                                             WorkspaceArena& workspace, cudaStream_t stream);
// Largest split-K partial plane any candidate schedule may need, so a bench arena sized with this
// can force any split-K schedule at any token count.
[[nodiscard]] std::size_t fp8_attn_input_n17408_candidate_partial_capacity_bytes();

} // namespace infernix::ops::detail
