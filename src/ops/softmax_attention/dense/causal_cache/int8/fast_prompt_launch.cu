#include "ops/softmax_attention/dense/causal_cache/int8/fast_prompt_launch.h"

#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_launch.cuh"

namespace infernix::ops::detail {

void int8_kv_fast_prompt_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    with_causal_geometry(p.query_heads, [&]<class G>() {
        launch_int8_kv_fast_tiled_mma<G>(p, cache, envelope, workspace, stream);
    });
}

} // namespace infernix::ops::detail
