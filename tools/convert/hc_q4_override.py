from tools.convert.methods import grouped_mse


def configure(model, recipe, sources):
    targets = (
        "/attn_hc/down",
        "/attn_hc/inject",
        "/attn_hc/up",
        "/mlp_hc/down",
        "/mlp_hc/inject",
        "/mlp_hc/up",
        "/final_mixer/down",
        "/final_mixer/up",
        "/moe/shared/gate",
        "/moe/shared/up",
        "/moe/shared/down",
    )
    for name in model.parameters:
        if name.startswith("text/") and name.endswith(targets):
            recipe.assign(name, format="q4_g64_fp16", method=grouped_mse)
