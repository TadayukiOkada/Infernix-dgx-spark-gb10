#include "core/weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/layout.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// The D4 epilogue warms at most 1/12 of the device's L2 cache (up to 8 MiB on 96 MiB L2, 2 MiB on
// GB10's 24 MiB L2) of the next consumer's weights, so the warmed block cannot evict what it was
// meant to help.
constexpr std::size_t kNextWeightPrefetchLimit = std::size_t{8} << 20;

std::size_t device_prefetch_limit() {
    thread_local int cached_device          = -1;
    thread_local std::size_t cached_limit   = kNextWeightPrefetchLimit;
    int current                             = 0;
    if (cudaGetDevice(&current) == cudaSuccess) {
        if (current != cached_device) {
            int l2 = 0;
            if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, current) == cudaSuccess && l2 > 0) {
                cached_limit = std::clamp<std::size_t>(static_cast<std::size_t>(l2) / 12,
                                                       std::size_t{2} << 20, kNextWeightPrefetchLimit);
            } else {
                cached_limit = kNextWeightPrefetchLimit;
            }
            cached_device = current;
        }
    }
    return cached_limit;
}
} // namespace

std::size_t sparse_moe_decode_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_decode_workspace(layout);
    return layout.peak_bytes(1);
}

SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up, QType routed_down,
                                                   const SparseMoeHints& hints) {
    const bool main_profile =
        routed_gate_up == QType::Q4_G64_FP16 &&
        (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16);
    const bool mtp_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    if (!main_profile && !mtp_profile) {
        throw std::invalid_argument("sparse_moe: unsupported routed codec profile");
    }

    SparseMoeDecodePlan plan;
    plan.workspace_bytes = sparse_moe_decode_workspace_bytes();
    if (hints.next_weight_prefetch != nullptr) {
        const std::size_t limit  = device_prefetch_limit();
        plan.next_weight_prefetch = hints.next_weight_prefetch;
        plan.next_weight_prefetch_bytes =
            hints.next_weight_prefetch_bytes < limit
                ? hints.next_weight_prefetch_bytes
                : limit;
    }
    return plan;
}

} // namespace infernix::ops::detail
