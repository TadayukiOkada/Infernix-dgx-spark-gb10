#pragma once

#include "core/device.h"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"

namespace infernix::ops::detail {

void int8_kv_tiled_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                             cudaStream_t stream);

} // namespace infernix::ops::detail
