#include "models/qwen4_exp/measurement.h"

#include "models/qwen3_5/frontend/digest.h"
#include "models/registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <vector>

namespace infernix::models::qwen4_exp {

std::string prefill_signature(const Model& model) {
    using Json         = nlohmann::json;
    const auto& config = model.config();
    const auto& t      = config.text;
    Json facts{{"implementation", "qwen4_exp-prefill-1"},
               {"architecture", architecture_name(Architecture::Qwen4Exp)},
               {"hidden_size", t.hidden_size},
               {"vocab_size", t.vocab_size},
               {"public_token_count", model.resources().public_token_count},
               {"num_hidden_layers", t.num_hidden_layers},
               {"max_position_embeddings", t.max_position_embeddings},
               {"rms_norm_eps", t.rms_norm_eps},
               // The mixer of every layer, in order: an Attention layer costs per pair, a Gdn layer
               // per token, so the interleaving is the first-order shape of a prefill.
               {"layer_types", t.layer_types},
               {"attention", {t.attention.heads, t.attention.kv_heads, t.attention.head_dim}},
               {"rope", {t.rope.theta, t.rope.rotary_dim, t.rope.pair_axes}},
               {"gdn",
                {t.gdn.key_heads, t.gdn.key_head_dim, t.gdn.value_heads, t.gdn.value_head_dim,
                 t.gdn.conv_kernel, t.gdn.output_gate, t.gdn.output_ggml_tiled, t.gdn.a_direct}},
               {"moe",
                {t.moe.experts, t.moe.top_k, t.moe.intermediate, t.moe.shared_intermediate,
                 t.moe.a16_experts, t.moe.a4_optional}},
               {"hc", {t.hc.streams, t.hc.rank}},
               {"qsa", {t.qsa.index_heads, t.qsa.index_head_dim, t.qsa.budget, t.qsa.compress_ratio}},
               // The n-gram injection reads one row per token per head from its volume: its geometry
               // and row format are prefill traffic, not just the model's identity.
               {"ple",
                {t.ple.layer, t.ple.embed_dim, t.ple.conv_kernel, t.ple.conv_dilation,
                 t.ple.ngram.ngram_size, t.ple.ngram.heads_per_ngram, t.ple.split_parts,
                 t.ple.table.format, t.ple.table.rows, t.ple.table.row_bytes,
                 t.ple.table.rows_per_block, t.ple.table.block_bytes}},
               // A GGUF import reads ggml blocks; the lossless route rounds every operation. Both
               // change the cost of the same token count.
               {"ggml_weights", t.ggml_weights},
               {"exact_experts", t.exact_experts},
               {"lossless", t.lossless},
               {"design_reference", t.design_reference}};
    if (config.vision) {
        const auto& v   = *config.vision;
        facts["vision"] = {
            v.depth,      v.hidden_size,         v.intermediate_size,  v.num_heads,
            v.patch_size, v.temporal_patch_size, v.spatial_merge_size, v.num_position_embeddings};
    }
    std::vector<const BoundWeight*> weights;
    // The context-cost model prices primary Text/Vision reconstruction. The coefficients were
    // measured for that work; the optional MTP/DFlash2/proposal weights a draft call reads do not
    // change the baseline's applicability.
    for (const auto& weight : model.weight_data()) {
        if (weight.name.starts_with("text/") || weight.name.starts_with("vision/")) {
            weights.push_back(&weight);
        }
    }
    std::sort(weights.begin(), weights.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });
    std::map<const WeightParent*, std::size_t> parents;
    auto& inventory = facts["weights"] = Json::array();
    for (const auto* weight : weights) {
        Json item{{"role", weight->name},
                  {"shape", weight->view.shape},
                  {"policy", weight->policy},
                  {"parts", Json::array()}};
        for (const auto& part : weight->view.parts) {
            const auto [it, inserted] = parents.emplace(part.parent, parents.size());
            const auto& geometry      = part.parent->geometry;
            item["parts"].push_back({it->second, part.begin, part.end, geometry.format,
                                     geometry.layout, geometry.shape, geometry.padded_columns,
                                     geometry.group_size});
        }
        inventory.push_back(std::move(item));
    }
    // The digest is the Qwen3.5 frontend's (the frontend itself is shared by both architectures).
    return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(facts.dump()));
}

} // namespace infernix::models::qwen4_exp
