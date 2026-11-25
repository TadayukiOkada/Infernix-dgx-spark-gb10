from tools.convert.methods import grouped_mse, nvfp4_mse, nvfp4_rht16_mse


def configure(model, recipe, sources):
    hc_targets = (
        "/attn_hc/down",
        "/attn_hc/inject",
        "/attn_hc/up",
        "/mlp_hc/down",
        "/mlp_hc/inject",
        "/mlp_hc/up",
        "/final_mixer/down",
        "/final_mixer/up",
    )
    shared_targets = (
        "/moe/shared/gate",
        "/moe/shared/up",
        "/moe/shared/down",
    )
    gdn_qkvz_targets = (
        "/gdn/query",
        "/gdn/key",
        "/gdn/value",
        "/gdn/z",
    )
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name == "text/output_head" or name.endswith(hc_targets):
            recipe.assign(
                name,
                format="q8_g32_fp16",
                method=grouped_mse,
                source=parameter.source,
                activation_policy="A16Only",
            )
        elif not name.startswith("text/layers/"):
            continue
        elif name.endswith(shared_targets) or name.endswith("/gdn/output"):
            recipe.assign(
                name,
                format="nvfp4",
                method=nvfp4_mse,
                source=parameter.source,
                activation_policy="A16Only",
            )
        elif name.endswith(gdn_qkvz_targets):
            parts = name.split("/")
            layer_idx = int(parts[2])
            if 14 <= layer_idx <= 33:
                recipe.assign(
                    name,
                    format="nvfp4",
                    method=nvfp4_mse,
                    source=parameter.source,
                    activation_policy="A16Only",
                )
            else:
                recipe.assign(
                    name,
                    format="nvfp4",
                    layout="block_scale_k16_m128x4_rht16_v1",
                    method=nvfp4_rht16_mse,
                    source=parameter.source,
                    activation_policy="A16Only",
                )
