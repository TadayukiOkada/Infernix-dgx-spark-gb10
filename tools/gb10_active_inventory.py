#!/usr/bin/python3
"""Map the weight formats and (N, K) shapes actually bound by active .infernix artifacts.

Reads each artifact's .conversion.json report and prints, per artifact, the format ->
(N, K) multiset so a kernel sweep can be limited to routes the model really executes.
"""
import json
import glob
import os
import sys
from collections import Counter, defaultdict

REPORTS = {
    "dense8": "/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/"
              "Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix.conversion.json",
    "nvfp4_w4dense": "/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/"
                     "Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix.conversion.json",
    "iq3s": "/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/"
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix.conversion.json",
    "mtp_head": "/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/"
                "Qwen3.8-Flash-Next-DFlash2-Infernix.infernix.conversion.json",
}


def role_of(param_names):
    """Classify a bound parameter by its logical name, matching the roadmap's columns."""
    joined = " ".join(param_names)
    if "moe.experts" in joined or "experts" in joined:
        return "routed_experts"
    if "hyper" in joined or "hc" in joined:
        return "hyper_connection"
    if "shared_expert" in joined:
        return "shared_experts"
    if "lm_head" in joined or "embed" in joined:
        return "lm_head"
    for key, label in (
        ("self_attn", "trunk_attn"),
        ("attn", "trunk_attn"),
        ("linear_attn", "trunk_gdn"),
        ("gdn", "trunk_gdn"),
        ("mlp", "trunk_mlp"),
        ("norm", "norm"),
    ):
        if key in joined:
            return label
    return "other"


def main():
    wanted = sys.argv[1:] or list(REPORTS)
    for tag in wanted:
        path = REPORTS.get(tag, tag)
        if not os.path.exists(path):
            print(f"!! missing report for {tag}: {path}")
            continue
        d = json.load(open(path))
        recipe = (d.get("provenance") or {}).get("recipe")
        cfg = ((d.get("components") or {}).get("text") or {}).get("config") or {}
        print(f"=== {tag}  recipe={recipe}  hidden={cfg.get('hidden_size')} "
              f"layers={cfg.get('num_hidden_layers')} fmt={d.get('formats')}")
        by_role = defaultdict(Counter)
        for m in d.get("methods", []):
            shape = m.get("shape") or []
            if len(shape) < 2:
                continue
            r = role_of(m.get("parameters") or [])
            n, k = shape[0], shape[-1] if len(shape) > 1 else shape[0]
            by_role[r][(m.get("format"), m.get("layout"), n, k,
                        "x".join(str(s) for s in shape[1:-1]))] += 1
        for r in sorted(by_role):
            print(f"  [{r}]")
            for (fmt, layout, n, k, mid), cnt in sorted(
                    by_role[r].items(), key=lambda x: (-x[1], x[0][2] * x[0][3])):
                extra = f" mid={mid}" if mid else ""
                print(f"    {cnt:5d}x  {fmt:<16} {layout:<28} N={n:<8} K={k:<7}{extra}")


if __name__ == "__main__":
    main()
