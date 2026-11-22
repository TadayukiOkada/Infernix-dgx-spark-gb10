#pragma once
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace infernix::models::qwen3_5::execution {

[[nodiscard]] inline ScopedL2AccessPolicyWindow make_decode_l2_persist_window(const DeviceContext& device,
                                                                              const WorkspaceArena& work) {
    const std::size_t persist_bytes = device.persisting_l2_cache_bytes();
    const std::size_t max_window    = device.access_policy_max_window_bytes();
    if (persist_bytes == 0 || max_window == 0 || work.base() == nullptr) { return {}; }
    enable_persisting_l2_cache(device);
    const std::size_t bytes = std::min({work.capacity(), max_window, persist_bytes});
    return ScopedL2AccessPolicyWindow(device.stream, work.base(), bytes, max_window, 1.0F,
                                      cudaAccessPropertyPersisting, cudaAccessPropertyNormal);
}

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        enable_persisting_l2_cache(state.execution.device);
        executable->launch(state.execution.device.stream);
    } else {
        nvtx::ScopedRange eager_range(nvtx::Name::DecodeEager, nvtx::Category::Decode);
        const auto l2_window = make_decode_l2_persist_window(state.execution.device, state.execution.work);
        body();
    }
}

template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    const auto l2_window = make_decode_l2_persist_window(state.execution.device, state.execution.work);
    definition.capture(state.execution.device.stream, body);
}

} // namespace infernix::models::qwen3_5::execution
