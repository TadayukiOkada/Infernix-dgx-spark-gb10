#pragma once

#include "models/qwen4_exp/model.h"

#include <string>

namespace infernix::models::qwen4_exp {

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
//
// Each architecture derives its own key: the digest's bits are the contract between a measured
// ContextPrefillCost entry and the configurations it prices, so re-deriving one architecture's
// key to share code with another would silently invalidate every cost table already bound to it.
[[nodiscard]] std::string prefill_signature(const Model& model);

} // namespace infernix::models::qwen4_exp
