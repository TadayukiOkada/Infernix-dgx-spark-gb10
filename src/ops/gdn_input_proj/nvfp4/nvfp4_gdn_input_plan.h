#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// Internal A4 GEMM schedules that the projection selects per token band. Exposed so the bench can
// sweep them directly and retune the band edges for the current device.
enum class Nvfp4GdnInputA4ScheduleId {
    M32N64,
    M32N128,
    M128N128Pipelined,
    M64N128,
    M128N128Resident,
    Tma,
};

struct Nvfp4GdnInputA4Problem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// A16ColumnInvariant: one sliced-K schedule family (8 K-warps) for every T, so a column's
// arithmetic does not depend on the call's other columns.
void nvfp4_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

const char* nvfp4_gdn_input_a4_schedule_name(Nvfp4GdnInputA4ScheduleId schedule) noexcept;
bool nvfp4_gdn_input_a4_admits(const Nvfp4GdnInputA4Problem& problem) noexcept;
bool nvfp4_gdn_input_a4_candidate_is_legal(Nvfp4GdnInputA4ScheduleId schedule,
                                           const Nvfp4GdnInputA4Problem& problem) noexcept;
void nvfp4_gdn_input_a4_execute_candidate(Nvfp4GdnInputA4ScheduleId schedule, const Tensor& x,
                                          const Weight& weight, Tensor& qkv, Tensor& z,
                                          WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
