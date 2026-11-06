// linear_topk over Qwen3.5-122B-A10B's head (nvfp4_rg16_v1 [248320, 3072]): the head's canonical
// W4A16 projection (rg16_linear, exact sums, so the scores are the target head's own FP32 logits)
// into the call's workspace, the head read once for every column; then each 128-row producer group's
// top sixteen per column, published as the other heads' producers publish theirs for the merge.
#include "core/weight.h"
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "infernix/ops/rg16_linear.h"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_merge_sort.cuh>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kThreads = kLinearTopKGroupedRows; // one row per thread
using GroupSort        = cub::BlockMergeSort<std::uint64_t, kThreads, 1>;

// One CTA per (producer group, column) tile, the merge kernel's grid shape: its 128 rows' keys
// (rows past valid_rows the sentinel) sorted, the first sixteen published. The column dimension
// is parallel across the SMs, so no per-CTA serial column loop and no barrier between columns.
// Each warp stages its 32 rows in eight warp-local LDG.128 loads and shuffles them back to the
// sort's one-row-per-thread layout; a tile the vector loads cannot cover falls back to one
// coalesced scalar load per thread.
__global__ __launch_bounds__(kThreads, 8) void nvfp4_logits_topk_kernel(const float* __restrict__ logits,
                                                                        std::int32_t rows, std::int32_t valid_rows,
                                                                        std::uint64_t* __restrict__ partial_keys,
                                                                        std::int32_t producer_groups) {
    pdl::enter();
    __shared__ typename GroupSort::TempStorage storage;
    const int group  = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int row    = group * kThreads + tid;
    const std::int64_t base = static_cast<std::int64_t>(column) * rows + group * kThreads;
    float score;
    if (rows % 4 == 0 && group * kThreads + kThreads <= rows) {
        // The warp's 32 rows are eight float4s: lanes 0..7 load them and every lane picks its
        // own row, component (lane & 3) of lane (lane >> 2), from the loaders.
        constexpr int kFloat4sPerWarp = 32 / 4;
        const int lane                = tid & 31;
        const float4 loaded           = lane < kFloat4sPerWarp
                                            ? reinterpret_cast<const float4*>(logits + base + (tid & ~31))[lane]
                                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float components[4]     = {loaded.x, loaded.y, loaded.z, loaded.w};
        score                         = 0.0f;
        #pragma unroll
        for (int component = 0; component < 4; ++component) {
            const float shuffled = __shfl_sync(0xffffffffU, components[component], lane >> 2);
            if ((lane & 3) == component) score = shuffled;
        }
    } else { // tail tile: only the rows below `rows` exist
        score = row < rows ? logits[base + tid] : 0.0f;
    }
    std::uint64_t key[1] = {row < valid_rows ? score_id_order_key(score, row) : 0};
    GroupSort(storage).Sort(key, ScoreIdOrderGreater{});
    if (tid < kLinearTopK) {
        partial_keys[(static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + tid] = key[0];
    }
}

} // namespace

void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows, const LinearTopKWorkspace& workspace,
                               cudaStream_t stream) {
    if (workspace.rows_per_producer != kThreads || logits.dtype != DType::FP32 ||
        logits.ne[1] != workspace.columns) {
        throw std::logic_error("linear_topk logits: workspace does not match the call");
    }
    CUDA_CHECK(pdl::launch_with(
        pdl::Dependency::Programmatic,
        pdl::LaunchConfig{dim3(static_cast<unsigned>(workspace.producer_groups),
                               static_cast<unsigned>(workspace.columns)), kThreads, 0, stream},
        nvfp4_logits_topk_kernel, static_cast<const float*>(logits.data), logits.ne[0], valid_rows,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups));
    CUDA_CHECK(cudaGetLastError());
}

void linear_topk_nvfp4_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                              const LinearTopKWorkspace& workspace, Tensor& logits, WorkspaceArena& arena,
                              cudaStream_t stream) {
    if (logits.ne[0] != head.n || logits.ne[1] != hidden.ne[1]) {
        throw std::logic_error("linear_topk NVFP4: workspace does not match the call");
    }
    rg16_linear(hidden, head, logits, arena, stream);
    linear_topk_logits_launch(logits, valid_rows, workspace, stream);
}

} // namespace infernix::ops::detail
