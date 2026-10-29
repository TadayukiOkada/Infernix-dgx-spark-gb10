#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"

namespace infernix::ops::detail {

void int8_kv_fast_prompt_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
