"""Structural and per-parameter format audit of a Qwen3.8-Flash-Next artifact.

Answers, for a produced artifact:

1. Does the artifact open, and which components / config does it declare?
2. Does every logical binding of the recipe's model resolve, and what format,
   layout and activation policy does each stored object carry?
3. Does the realized format assignment match the recipe *exactly* — i.e. is the
   W4 dense tier actually 4-bit, and did anything unintended change width?
4. Do the NVFP4 dense shapes the artifact now contains all have a registered
   dispatch kernel in `src/ops/linear/nvfp4/nvfp4_shapes.h`?
5. Byte totals per format, so the expected per-token weight traffic can be
   compared with the plan.

Usage:
    .venv-convert/bin/python -m tools.flash_next.audit_artifact \
        --artifact OUT.infernix [--recipe RECIPE] [--model CHECKPOINT] [--json R]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject

# Dispatch shapes compiled into `src/ops/linear/nvfp4/shapes/`. Kept in sync with
# nvfp4_shapes.h; the audit reports any artifact shape missing here.
_NVFP4_DISPATCH = {
    (14336, 5120), (16384, 5120), (34816, 5120), (5120, 6144), (5120, 17408),
    (16384, 2560), (13952, 2560), (12800, 2560), (2560, 6144),
    (2048, 2560), (6144, 2560), (512, 2560), (128, 2560), (10240, 2560),
    (2560, 2560),
}

_BYTES_PER_WEIGHT = {
    "nvfp4": 0.5625,      # E2M1 4-bit + E4M3 per 16
    "nvfp4_mul": 0.5625,
    "q8_g32_fp16": 1.0625,  # int8 + fp16 per 32
    "q4_g64_fp16": 0.5625,
    "bf16": 2.0,
    "fp32": 4.0,
}


def _shape_of(obj: TensorObject) -> tuple[int, ...]:
    return tuple(int(d) for d in obj.shape)


def _binding_objects(artifact: Artifact, binding: dict) -> list[dict]:
    return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]


def audit(artifact_path: Path, recipe: str | None, model_path: Path | None) -> dict:
    report: dict = {"artifact": str(artifact_path), "problems": [], "notes": []}

    with Artifact(artifact_path) as artifact:
        directory = artifact.directory
        objects = {o.id: o for o in artifact.objects if isinstance(o, TensorObject)}
        bindings = directory.bindings

        report["components"] = sorted(directory.components)
        report["bindings"] = len(bindings)
        report["objects"] = len(objects)
        report["volume_id"] = (directory.components.get("text", {})
                               .get("config", {})
                               .get("ngram_table", {})
                               .get("volume_id"))

        # ---- 1. per-binding format / layout / activation policy -------------
        per_binding: dict[str, list[str]] = {}
        per_binding_policy: dict[str, str] = {}
        logical_shapes: dict[str, tuple[int, ...]] = {}
        missing_objects = 0

        # Object-level accounting: a packed object is referenced by several
        # bindings, so bytes are counted once per object, not per binding part.
        object_users: dict[str, set[str]] = collections.defaultdict(set)

        for name, binding in bindings.items():
            formats: list[str] = []
            for part in _binding_objects(artifact, binding):
                object_users[part["object"]].add(name)
                obj = objects.get(part["object"])
                if obj is None:
                    missing_objects += 1
                    formats.append("<missing>")
                    continue
                formats.append(obj.format)
            per_binding[name] = formats
            if "activation_policy" in binding:
                per_binding_policy[name] = binding["activation_policy"]

        format_bytes: collections.Counter = collections.Counter()
        format_count: collections.Counter = collections.Counter()
        for oid, obj in objects.items():
            format_bytes[obj.format] += obj.bytes
            format_count[obj.format] += 1

        if missing_objects:
            report["problems"].append(
                f"{missing_objects} binding parts reference an absent object")

        # collapse packed parts to one entry per logical name
        logical_format: dict[str, str] = {}
        for name, formats in per_binding.items():
            uniq = sorted(set(formats))
            logical_format[name] = "+".join(uniq) if len(uniq) > 1 else uniq[0]

        report["format_histogram"] = dict(sorted(format_count.items()))
        report["format_bytes_mib"] = {
            k: round(v / 2**20, 2) for k, v in sorted(format_bytes.items())}

        # ---- 2. compare against the recipe's intent -------------------------
        if recipe and model_path is not None:
            from tools.convert import qwen4_exp
            from tools.convert.official_recipes import RECIPES
            from tools.convert.recipe import Recipe
            from tools.convert.sources.safetensors import SafetensorsSource

            with SafetensorsSource(model_path) as store:
                model = qwen4_exp.build_model(
                    store, components=tuple(directory.components))
                expected = Recipe(model)
                RECIPES[recipe](model, expected, {"base": store})

                mismatches: list[str] = []
                for name in model.parameters:
                    want = tuple(sorted({s.format for s in expected.selections[name]}))
                    got = tuple(sorted(set(per_binding.get(name, ["<absent>"]))))
                    if not want:
                        continue
                    if want != got:
                        mismatches.append(f"{name}: want {want} got {got}")

                # Logical (n, k) per parameter. The runtime keys NVFP4 dispatch on
                # these, NOT on packed-object shapes: Binder::binding() validates
                # that the object's element count equals the logical shape and
                # Weight.n/k come from the parameter, so a packed 16384x2560 object
                # backs four 2048x2560 / 6144x2560 bindings.
                logical_shapes.update({
                    name: tuple(int(d) for d in p.shape)
                    for name, p in model.parameters.items()})

                report["recipe"] = recipe
                report["recipe_parameters"] = len(model.parameters)
                report["recipe_mismatches"] = mismatches[:50]
                report["recipe_mismatch_total"] = len(mismatches)
                if mismatches:
                    report["problems"].append(
                        f"{len(mismatches)} parameters differ from recipe {recipe!r}")

                # activation policy parity
                policy_mismatch = []
                for (name, _input), policy in expected.policies.items():
                    stored = per_binding_policy.get(name)
                    if stored is not None and stored != policy:
                        policy_mismatch.append(
                            f"{name}: want {policy} got {stored}")
                report["policy_mismatches"] = policy_mismatch[:50]
                if policy_mismatch:
                    report["problems"].append(
                        f"{len(policy_mismatch)} activation policies differ")

        # ---- 3. NVFP4 dense dispatch coverage -------------------------------
        # Prefer logical parameter shapes: that is what Nvfp4Geometry receives at
        # runtime. Without a model, fall back to packed object shapes (which are
        # only an upper bound on what dispatch actually sees).
        nvfp4_shapes: set[tuple[int, int]] = set()
        nvfp4_shape_source = "object (packed)"
        for name, formats in per_binding.items():
            if "nvfp4" not in formats:
                continue
            if name in logical_shapes:
                nvfp4_shape_source = "logical (parameter)"
                shape = logical_shapes[name]
                if len(shape) >= 2:
                    nvfp4_shapes.add((shape[0], shape[1]))
                continue
            for part in _binding_objects(artifact, bindings[name]):
                obj = objects.get(part["object"])
                if obj is None or obj.format not in ("nvfp4", "nvfp4_mul"):
                    continue
                shape = _shape_of(obj)
                if len(shape) >= 2:
                    nvfp4_shapes.add((shape[0], shape[1]))

        uncovered = sorted(s for s in nvfp4_shapes if s not in _NVFP4_DISPATCH)
        report["nvfp4_shape_source"] = nvfp4_shape_source
        report["nvfp4_shapes"] = [f"{n}x{k}" for n, k in sorted(nvfp4_shapes)]
        report["nvfp4_shapes_without_dispatch"] = [
            f"{n}x{k}" for n, k in uncovered]
        if uncovered:
            report["problems"].append(
                "NVFP4 shapes with no registered dispatch: "
                + ", ".join(f"{n}x{k}" for n, k in uncovered))

        # ---- 4. dense (non-expert) per-token traffic ------------------------
        expert_re = re.compile(r"/moe/experts$")
        dense_bytes = 0
        expert_bytes = 0
        for oid, obj in objects.items():
            users = object_users.get(oid, set())
            if not users:
                continue  # unreferenced object: not on any path
            # An object shared by an expert bank and a dense binding does not
            # occur in practice; classify by the first binding that owns it.
            if any(expert_re.search(u) for u in users):
                expert_bytes += obj.bytes
            else:
                dense_bytes += obj.bytes
        report["dense_mib"] = round(dense_bytes / 2**20, 1)
        report["expert_mib"] = round(expert_bytes / 2**20, 1)

        # dense classes that are still 8-bit or wider, by suffix
        tiers: dict[str, set[str]] = collections.defaultdict(set)
        class_shapes: dict[str, set[str]] = collections.defaultdict(set)
        class_counts: collections.Counter = collections.Counter()
        for name, fmt in logical_format.items():
            if expert_re.search(name):
                continue
            if not name.startswith("text/layers/"):
                continue
            suffix = re.sub(r"^text/layers/\d+/", "", name)
            tiers[suffix].add(fmt)
            class_counts[suffix] += 1
            shape = logical_shapes.get(name)
            if shape is not None and len(shape) >= 2:
                class_shapes[suffix].add(f"{shape[0]}x{shape[1]}")
        report["dense_classes"] = {
            k: sorted(v) for k, v in sorted(tiers.items())}
        report["dense_class_shapes"] = {
            k: sorted(v) for k, v in sorted(class_shapes.items())}
        report["dense_class_counts"] = dict(sorted(class_counts.items()))

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--recipe", help="compare against this official recipe")
    parser.add_argument("--model", type=Path,
                        help="source checkpoint (required with --recipe)")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--show-classes", action="store_true",
                        help="print every dense class and its realized format")
    args = parser.parse_args(argv)

    if args.recipe and args.model is None:
        parser.error("--recipe requires --model")

    report = audit(args.artifact, args.recipe, args.model)

    print(f"artifact        : {report['artifact']}")
    print(f"components      : {', '.join(report['components'])}")
    print(f"bindings        : {report['bindings']}")
    print(f"objects         : {report['objects']}")
    print(f"volume id       : {report['volume_id']}")
    print(f"dense MiB       : {report['dense_mib']}")
    print(f"expert MiB      : {report['expert_mib']}")
    print()
    print("objects per format:")
    for fmt, n in report["format_histogram"].items():
        print(f"  {fmt:<16} {n:>7} objects  {report['format_bytes_mib'][fmt]:>12.2f} MiB")
    print()
    print(f"NVFP4 dense shapes ({len(report['nvfp4_shapes'])}, "
          f"{report['nvfp4_shape_source']}):")
    for s in report["nvfp4_shapes"]:
        n, k = s.split("x")
        ok = (int(n), int(k)) in _NVFP4_DISPATCH
        print(f"  {s:<14} {'dispatch OK' if ok else '** NO DISPATCH **'}")

    if "recipe" in report:
        print()
        print(f"recipe {report['recipe']!r}: "
              f"{report['recipe_parameters']} parameters, "
              f"{report.get('recipe_mismatch_total', 0)} format mismatches")
        for m in report.get("recipe_mismatches", []):
            print(f"  MISMATCH {m}")
        for m in report.get("policy_mismatches", []):
            print(f"  POLICY   {m}")

    if args.show_classes:
        print()
        print("dense layer classes: format, logical shape, count, dispatch")
        for suffix, fmts in report["dense_classes"].items():
            shapes = ",".join(report["dense_class_shapes"].get(suffix, []))
            count = report["dense_class_counts"].get(suffix, 0)
            state = ""
            if fmts == ["nvfp4"]:
                bad = [s for s in report["dense_class_shapes"].get(suffix, [])
                       if tuple(int(x) for x in s.split("x"))
                       not in _NVFP4_DISPATCH]
                state = "** NO DISPATCH **" if bad else "dispatch OK"
            print(f"  {suffix:<34} {', '.join(fmts):<12} {shapes:<12} "
                  f"x{count:<4} {state}")

    print()
    if report["problems"]:
        print("PROBLEMS:")
        for p in report["problems"]:
            print(f"  - {p}")
    else:
        print("OK: structure, formats and dispatch coverage consistent")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {args.json}")

    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
