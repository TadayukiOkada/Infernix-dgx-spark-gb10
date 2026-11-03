"""``ckt kt`` - qualify and explore the product kernels with Kernel Tuner."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from .bridge import REPO_ROOT, SHIM_DIR, SRC_ROOT, config_key, run_tuning
from .problems import PROBLEMS

# Positional shorthand for --baseline, in the order the schedule structs list their parameters.
BASELINE_KEYS = {
    "nvfp4_a16_mma": ("KT_BM", "KT_BN", "KT_BK", "KT_WM", "KT_WN", "KT_STAGES", "KT_MINB"),
    "nvfp4_a16_sliced_k": ("KT_KW", "KT_TT", "KT_ROWTILES", "KT_STAGES", "KT_MINB"),
}


# --------------------------------------------------------------------------- #
# Shim drift check
# --------------------------------------------------------------------------- #
def _code_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        if "KT-SUBST" in raw:  # deliberate NVRTC substitution, documented on the line itself
            continue
        line = re.sub(r"//.*", "", raw).strip()
        if line and not line.startswith("#"):
            out.append(re.sub(r"\s+", " ", line))
    return out


def check_shim_drift() -> list[str]:
    """Every code line of a ``MIRROR-OF`` shim file must still exist verbatim in its source file."""
    problems = []
    for shim in sorted(SHIM_DIR.rglob("*")):
        if not shim.is_file():
            continue
        m = re.search(r"MIRROR-OF:\s*(\S+)", shim.read_text())
        if not m:
            continue
        source = REPO_ROOT / m.group(1)
        have = set(_code_lines(source.read_text()))
        for line in _code_lines(shim.read_text()):
            if line not in have and line not in {"}", "} // namespace infernix::pdl"}:
                problems.append(f"{shim.relative_to(SHIM_DIR)}: line not in {m.group(1)}: {line}")
    return problems


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def sensitivity(rows: list[dict], params: list[str]) -> list[tuple[str, list[tuple[str, float]]]]:
    """Best verified median time per value of each tunable (a one-factor 'what matters' view)."""
    out = []
    for p in params:
        best: dict[str, float] = {}
        for r in rows:
            if r.get("verified") is not True:
                continue
            key = str(r[p])
            best[key] = min(best.get(key, float("inf")), r["median_us"])
        if len(best) > 1:
            out.append((p, sorted(best.items(), key=lambda kv: kv[1])))
    return out


def render(problem, rows, baseline_key, top, counts) -> str:
    ok = [r for r in rows if r.get("verified") is True]
    bad = [r for r in rows if r.get("verified") is False]
    base = next((r for r in rows if config_key({k: r[k] for k in problem.tune_params}) == baseline_key), None)
    lines = [f"### {problem.name}", ""]
    lines.append(
        f"{counts['space']} valid configurations, {len(rows)} timed, {len(ok)} verified, "
        f"{len(bad)} failed the oracle, {counts['compile_fail']} rejected at compile time."
    )
    if base is not None:
        lines.append(f"Baseline `{base['label']}`: {base['median_us']:.2f} us, {base['tflops']:.2f} TFLOP/s, "
                     f"{base['pct_pipe']:.1f}% pipe, {base['pct_mem']:.0f}% mem, verified={base['verified']}.")
    elif baseline_key:
        lines.append("Baseline configuration was not in the timed set (compile/launch failure).")
    lines += ["", "| # | schedule | us | TFLOP/s | %pipe | GB/s | %mem | regs | local B | vs base |",
              "|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for i, r in enumerate(ok[:top], 1):
        delta = f"{100 * (r['median_us'] / base['median_us'] - 1):+.1f}%" if base else "-"
        lines.append(
            f"| {i} | `{r['label']}` | {r['median_us']:.2f} | {r['tflops']:.2f} | {r['pct_pipe']:.1f}% | "
            f"{r['gbps']:.0f} | {r['pct_mem']:.0f}% | {r['num_regs']} | {r['local_bytes']} | {delta} |"
        )
    if bad:
        lines += ["", "Failed the oracle (excluded from the ranking; fastest 5 by time, which would otherwise "
                  "look like wins):", "", "| schedule | us | max err / tol |", "|---|--:|--:|"]
        for r in bad[:5]:
            lines.append(f"| `{r['label']}` | {r['median_us']:.2f} | {r.get('v_max_err_over_tol', float('nan')):.3g} |")
    sens = sensitivity(rows, [p for p in problem.tune_params if p != "block_size_x"])
    if sens:
        lines += ["", "Best verified time (us) per parameter value:", ""]
        for p, vals in sens:
            lines.append(f"- `{p}`: " + ", ".join(f"{v}={t:.1f}" for v, t in vals))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(args: Any) -> int:
    for name in PROBLEMS:
        print(f"{name}  baseline keys: {', '.join(BASELINE_KEYS[name])}")
    return 0


def cmd_doctor(args: Any) -> int:
    drift = check_shim_drift()
    print(f"shim dir:   {SHIM_DIR}")
    print(f"source dir: {SRC_ROOT}")
    for d in drift:
        print("DRIFT", d)
    print("shim mirrors in sync" if not drift else f"{len(drift)} drifted line(s): update the shim")
    return 1 if drift else 0


def cmd_tune(args: Any) -> int:
    drift = check_shim_drift()
    if drift:
        print("shim drift detected (run `ckt kt doctor`):", *drift, sep="\n  ", file=sys.stderr)
        return 3
    problem = PROBLEMS[args.problem](args.n, args.k, args.tokens, wide=args.wide)
    baseline_key = None
    if args.baseline:
        keys = BASELINE_KEYS[args.problem]
        vals = [int(v) for v in args.baseline.split(",")]
        if len(vals) != len(keys):
            raise SystemExit(f"--baseline needs {len(keys)} values: {', '.join(keys)}")
        pinned = dict(zip(keys, vals))
        for key, val in pinned.items():
            if val not in problem.tune_params[key]:
                problem.tune_params[key] = sorted(problem.tune_params[key] + [val])
        # block size is a function of the schedule; the problem's restriction enforces it.
        candidate = None
        for threads in problem.tune_params["block_size_x"]:
            probe = {k: v[0] for k, v in problem.tune_params.items()}
            probe.update(pinned)
            probe["block_size_x"] = threads
            if "KT_FULL" in probe:
                probe["KT_FULL"] = "true" if (args.tokens % pinned.get("KT_BN", 1) == 0) else "false"
            if problem.restrictions[0](probe):
                candidate = probe
        if candidate is None:
            raise SystemExit("baseline configuration is invalid for this problem (see restriction)")
        baseline_key = config_key(candidate)
    rows, _ = run_tuning(
        problem,
        strategy=args.strategy,
        strategy_options={"max_fevals": args.max_fevals} if args.max_fevals else {},
        iterations=args.iterations,
        flush_mib=args.flush_mib,
        compiler=args.compiler,
    )
    from .bridge import COMPILE_LOG

    # count the space for the report (restrictions applied by KT; recount through the same predicate)
    import itertools

    names = list(problem.tune_params)
    space = sum(
        1 for combo in itertools.product(*problem.tune_params.values())
        if problem.restrictions[0](dict(zip(names, combo)))
    )
    counts = {"space": space, "compile_fail": len(COMPILE_LOG)}
    text = render(problem, rows, baseline_key, args.top, counts)
    print(text)
    if COMPILE_LOG and args.show_compile_errors:
        for k, msg in list(COMPILE_LOG.items())[:5]:
            print("\ncompile failure", k, "\n", msg[:600])
    if args.json:
        Path(args.json).write_text(json.dumps({"problem": problem.name, "rows": rows,
                                              "compile_failures": {str(k): v for k, v in COMPILE_LOG.items()}},
                                             default=str, indent=1))
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 0


def register_kt_subparsers(subparsers: Any, add_common: Any) -> None:
    kt = subparsers.add_parser("kt", help="tune/qualify production kernels in-process with Kernel Tuner (NVRTC)")
    add_common(kt)
    sub = kt.add_subparsers(dest="kt_action", required=True)

    ls = sub.add_parser("list", help="list tunable kernel problems")
    ls.set_defaults(func=cmd_list)

    dr = sub.add_parser("doctor", help="check that the NVRTC host-header shim still mirrors src/")
    dr.set_defaults(func=cmd_doctor)

    tu = sub.add_parser("tune", help="sweep a kernel's template space on one workload and verify every candidate")
    tu.add_argument("problem", choices=sorted(PROBLEMS))
    tu.add_argument("--n", type=int, required=True, help="output rows (weight N)")
    tu.add_argument("--k", type=int, required=True, help="reduction length (weight K)")
    tu.add_argument("--tokens", "-t", type=int, required=True, help="token count T")
    tu.add_argument("--wide", action="store_true", help="also sweep cp.async cache hints / staging modes")
    tu.add_argument("--baseline", help="current production schedule, positional (see `ckt kt list`)")
    tu.add_argument("--strategy", default="brute_force", help="Kernel Tuner strategy (brute_force, random_sample, "
                    "genetic_algorithm, ...)")
    tu.add_argument("--max-fevals", type=int, help="evaluation budget for non-exhaustive strategies")
    tu.add_argument("--iterations", type=int, default=12, help="timed launches per candidate (default 12)")
    tu.add_argument("--flush-mib", type=int, default=48, help="cold-L2 eviction read size; 0 disables (default 48)")
    tu.add_argument("--compiler", choices=["nvrtc", "nvcc"], default="nvrtc",
                    help="nvrtc: fast JIT-PTX sweep; nvcc: product flags and SASS (slower, exact codegen)")
    tu.add_argument("--top", type=int, default=10)
    tu.add_argument("--json", help="write raw rows to a JSON file")
    tu.add_argument("--md", help="write the report to a markdown file")
    tu.add_argument("--show-compile-errors", action="store_true")
    tu.set_defaults(func=cmd_tune)
