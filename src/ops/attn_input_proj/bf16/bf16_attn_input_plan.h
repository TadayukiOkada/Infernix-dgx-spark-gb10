#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// The Attention epilogue owns its measured crossover independently of pure Linear.
inline constexpr std::int32_t kBf16AttnInputSmallTMinTokens   = 2;
inline constexpr std::int32_t kBf16AttnInputSmallTMaxTokens   = 4;
inline constexpr std::int32_t kBf16AttnInputSmallTDispatchEnd = 4;

void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream);

void bf16_attn_input_column_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                             Tensor& gate, Tensor& k, Tensor& v,
                                             cudaStream_t stream);

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream,
                              bool column_invariant = false);

// Internal GEMM schedules that the projection selects per token band. Exposed so the bench can
// sweep them directly and retune the band edges for the current device (GB10, 48 SMs). The HEAD
// edges inherited 5090 tuning.
enum class Bf16AttnInputScheduleId {
    Decode,
    SimtSmall,
    SlicedR32T16W4,
    MmaR32T32K128S3,
    TmaR64T64K64S3,
    MmaR64T32K64S3,
    TmaR64T128K64S2,
};

struct Bf16AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

const char* bf16_attn_input_schedule_name(Bf16AttnInputScheduleId schedule) noexcept;
bool bf16_attn_input_candidate_is_legal(Bf16AttnInputScheduleId schedule,
                                        const Bf16AttnInputProblem& problem) noexcept;
void bf16_attn_input_execute_candidate(Bf16AttnInputScheduleId schedule, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream);

} // namespace infernix::ops::detail
