#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

enum class Q8AttnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    SimtR8C4,
    MmaR32C64,
    MmaR32C128,
    MmaR64C64,
    MmaR64C96,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
    DFlash2SmallT,
    DFlash2MmaR16C64K128,
    DFlash2MmaR32C32K128,
    DFlash2MmaR32C64K128,
    DFlash2MmaR32C64,
    DFlash2MmaR64C128,
    // Request-invariant exact route of the four-output parents (35B-A3B, 122B-A10B): one sliced-K
    // reduction at every width.
    ColumnInvariant,
};

struct Q8AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8AttnInputPlan {
    Q8AttnInputScheduleId schedule;
};

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept;
bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept;
// column_invariant is registered for the four-output parents ([9216,2048] and [17408,3072]) only.
Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem,
                                           bool column_invariant = false);

// Candidate-sweep surface: force a specific schedule for a given shape so the bench can measure every
// legal schedule directly instead of the production route. candidate_is_legal checks the schedule
// belongs to the shape's allowed set (target / companion / dflash2).
bool q8_attn_input_candidate_is_legal(Q8AttnInputScheduleId schedule,
                                      const Q8AttnInputProblem& problem) noexcept;
void q8_attn_input_execute_candidate(Q8AttnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                     Tensor& v, cudaStream_t stream);
void q8_attn_input_execute_candidate(Q8AttnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                     cudaStream_t stream);

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream,
                                bool column_invariant = false);
void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream,
                            bool column_invariant = false);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream);

} // namespace infernix::ops::detail
