"""ab_runner — first-class interleaved A/B benchmark & shape-CSV comparator.

Eliminates ad-hoc shell/Python A/B scripts and manual statistical hand math:
1. ``run_interleaved_ab()``: executes ``A -> B -> A -> B ...`` rounds on the GPU
   (canceling thermal/clock drift on unified-memory GB10), extracts timing
   metrics from ``--ckt`` harnesses, ``infernix_linear_bench``, ``infernix_bench``,
   or a custom regex, and runs :func:`ckt.kernel_math.calculate_ab_stats`.
2. ``compare_shape_csvs()``: compares two ``infernix_linear_bench --csv-out``
   (or ``ckt shapes bench``) files point-by-point across ``(qtype, policy, N, K, T)``,
   computing exact per-point deltas, geometric mean speedup, total latency delta,
   win/tie/loss counts, best improvement, and worst regression.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any

from .kernel_math import calculate_ab_stats
from .util import ToolError, run


_RESULT_RE = re.compile(r"\b(?:median_us|mean_us|latency_us)=([0-9]+(?:\.[0-9]+)?)")
_LINEAR_ROW_RE = re.compile(
    r"^\S+\s+(?:Q4|Q5|Q6|Q8|BF16|NVFP4|FP8)\s+\S+\s+\d+\s+\d+\s+\d+\s+([0-9]+\.[0-9]+)",
    re.M,
)
_TOKS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s", re.I)


def extract_metric_from_stdout(text: str, metric_regex: str | None = None) -> tuple[float, str]:
    """Extract a numeric timing/throughput metric and its unit from command stdout."""
    if metric_regex:
        m = re.search(metric_regex, text, re.M)
        if not m:
            raise ToolError(f"custom --metric-regex '{metric_regex}' did not match stdout:\n{text[:400]}")
        return float(m.group(1)), "custom"

    # 1. CKT harness `RESULT ... median_us=<val>`
    m = _RESULT_RE.search(text)
    if m:
        return float(m.group(1)), "us"

    # 2. `infernix_linear_bench` stdout table (`median_us` column)
    matches = _LINEAR_ROW_RE.findall(text)
    if matches:
        vals = [float(x) for x in matches]
        # Sum across rows if multiple points were run in one invocation, or single point
        return (vals[0] if len(vals) == 1 else sum(vals)), "us"

    # 3. `infernix_bench` tok/s output (convert tok/s to us/tok so lower is better, or report both)
    m_tok = _TOKS_RE.search(text)
    if m_tok:
        tok_s = float(m_tok.group(1))
        if tok_s > 0:
            return 1e6 / tok_s, "us_per_tok"

    # 4. JSON output containing `median_us`
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "median_us" in obj:
                return float(obj["median_us"]), "us"
            if "points" in obj and isinstance(obj["points"], list) and obj["points"]:
                return float(obj["points"][0]["median_us"]), "us"
    except Exception:
        pass

    raise ToolError(
        "could not auto-detect timing metric (`median_us=`, `infernix_linear_bench` row, or `tok/s`) "
        f"from stdout:\n{text[:400]}"
    )


def _parse_env_kv(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        for part in item.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def run_interleaved_ab(
    *,
    cmd_a: str | list[str],
    cmd_b: str | list[str],
    env_a: dict[str, str] | None = None,
    env_b: dict[str, str] | None = None,
    label_a: str = "baseline_A",
    label_b: str = "candidate_B",
    rounds: int = 6,
    warmup_rounds: int = 1,
    metric_regex: str | None = None,
    kernel_share_of_total: float | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run interleaved ``A -> B -> A -> B ...`` rounds on the GPU and compute zero-hand-math A/B stats."""
    argv_a = shlex.split(cmd_a) if isinstance(cmd_a, str) else list(cmd_a)
    argv_b = shlex.split(cmd_b) if isinstance(cmd_b, str) else list(cmd_b)

    merged_env_a = {**os.environ, **(env_a or {})}
    merged_env_b = {**os.environ, **(env_b or {})}

    samples_a: list[float] = []
    samples_b: list[float] = []
    round_log: list[dict[str, Any]] = []
    unit = "us"

    total_rounds = warmup_rounds + rounds
    for r_idx in range(total_rounds):
        is_warmup = r_idx < warmup_rounds
        # Alternate starting order on odd/even rounds (`A, B` then `B, A`) to eliminate directional drift!
        order = [("A", argv_a, merged_env_a), ("B", argv_b, merged_env_b)]
        if r_idx % 2 == 1:
            order.reverse()

        pair_vals: dict[str, float] = {}
        for tag, argv, env_map in order:
            res = run(argv, env=env_map, timeout=timeout, check=False, log=False)
            if res.returncode != 0:
                raise ToolError(f"A/B variant {tag} exited with code {res.returncode}: {res.stderr.strip()[:300]}")
            val, detected_unit = extract_metric_from_stdout(res.stdout + "\n" + res.stderr, metric_regex)
            unit = detected_unit
            pair_vals[tag] = val

        if not is_warmup:
            samples_a.append(pair_vals["A"])
            samples_b.append(pair_vals["B"])
            round_log.append(
                {
                    "round": r_idx - warmup_rounds + 1,
                    "order": f"{order[0][0]}->{order[1][0]}",
                    "a_us": round(pair_vals["A"], 4),
                    "b_us": round(pair_vals["B"], 4),
                    "delta_pct": round((pair_vals["B"] - pair_vals["A"]) / pair_vals["A"] * 100.0, 2)
                    if pair_vals["A"] > 0
                    else 0.0,
                }
            )

    stats = calculate_ab_stats(
        samples_a,
        samples_b,
    )
    if kernel_share_of_total and 0.0 < kernel_share_of_total <= 1.0 and stats.get("speedup", 0) > 0:
        sp = float(stats["speedup"])
        amdahl = 1.0 / ((1.0 - kernel_share_of_total) + (kernel_share_of_total / sp))
        stats["amdahl_e2e_speedup"] = round(amdahl, 4)
        stats["amdahl_e2e_pct_gain"] = round((amdahl - 1.0) * 100.0, 2)
    return {
        "mode": "interleaved_command_ab",
        "label_a": label_a,
        "label_b": label_b,
        "cmd_a": " ".join(argv_a),
        "cmd_b": " ".join(argv_b),
        "env_a": env_a or {},
        "env_b": env_b or {},
        "warmup_rounds": warmup_rounds,
        "measured_rounds": rounds,
        "metric_unit": unit,
        "round_log": round_log,
        "stats": stats,
    }


def _load_shape_rows(path: Path) -> list[dict[str, Any]]:
    from .shapes import extract_route_signature

    if not path.exists():
        raise ToolError(f"A/B input file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "points" in data:
            qtype = str(data.get("qtype", "")).upper()
            policy = str(data.get("policy", "")).upper()
            n = int(data.get("N", 0))
            k = int(data.get("K", 0))
            out = []
            for p in data["points"]:
                route = str(p.get("active_route", ""))
                sig_info = extract_route_signature(route, policy=policy.lower()) if route else {}
                out.append(
                    {
                        "qtype": qtype,
                        "policy": policy,
                        "N": n,
                        "K": k,
                        "T": int(p["T"]),
                        "median_us": float(p["median_us"]),
                        "min_us": float(p.get("min_us", p["median_us"])),
                        "p95_us": float(p.get("p95_us", p["median_us"])),
                        "effective_gbs": float(p.get("effective_gbs", 0.0)),
                        "useful_tflops": float(p.get("useful_tflops", 0.0)),
                        "active_route": route,
                        "reduction_signature": str(
                            sig_info.get("reduction_signature") or p.get("reduction_signature", "")
                        ),
                    }
                )
            return out
        raise ToolError(f"unsupported JSON structure in {path}; expected `points` list")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            route = str(r.get("active_route", ""))
            pol = str(r.get("policy", "a16")).upper()
            sig_info = extract_route_signature(route, policy=pol.lower()) if route else {}
            rows.append(
                {
                    "qtype": str(r.get("qtype", "")).upper(),
                    "policy": pol,
                    "N": int(r.get("N", 0) or 0),
                    "K": int(r.get("K", 0) or 0),
                    "T": int(r.get("T", 0) or 0),
                    "median_us": float(r.get("median_us", 0.0) or 0.0),
                    "min_us": float(r.get("min_us", 0.0) or 0.0),
                    "p95_us": float(r.get("p95_us", 0.0) or 0.0),
                    "effective_gbs": float(r.get("effective_gbs", 0.0) or 0.0),
                    "useful_tflops": float(r.get("useful_tflops", 0.0) or 0.0),
                    "active_route": route,
                    "reduction_signature": str(
                        sig_info.get("reduction_signature") or r.get("reduction_signature", "")
                    ),
                }
            )
        return rows


def compare_shape_csvs(
    path_a: str | Path,
    path_b: str | Path,
    *,
    noise_threshold_pct: float = 2.0,
    require_bit_exact: bool = True,
) -> dict[str, Any]:
    """Compare two shape-benchmark CSV/JSON files point-by-point across ``(qtype, policy, N, K, T)``."""
    rows_a = _load_shape_rows(Path(path_a))
    rows_b = _load_shape_rows(Path(path_b))

    map_a = {(r["qtype"], r["policy"], r["N"], r["K"], r["T"]): r for r in rows_a}
    map_b = {(r["qtype"], r["policy"], r["N"], r["K"], r["T"]): r for r in rows_b}

    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_keys:
        raise ToolError(f"no matching (qtype, policy, N, K, T) points between {path_a} and {path_b}")

    comparisons: list[dict[str, Any]] = []
    log_speedups: list[float] = []
    speedups: list[float] = []
    sum_a_us = 0.0
    sum_b_us = 0.0
    wins = 0
    ties = 0
    losses = 0
    bit_exact_count = 0
    reduction_changed_count = 0

    for key in common_keys:
        ra = map_a[key]
        rb = map_b[key]
        ua = float(ra["median_us"])
        ub = float(rb["median_us"])
        sum_a_us += ua
        sum_b_us += ub
        delta_us = ub - ua
        delta_pct = (delta_us / ua * 100.0) if ua > 0 else 0.0
        sp = (ua / ub) if ub > 0 else 1.0
        speedups.append(sp)
        if sp > 0:
            log_speedups.append(math.log(sp))

        if delta_pct <= -noise_threshold_pct:
            status = "WIN"
            wins += 1
        elif delta_pct >= noise_threshold_pct:
            status = "REGRESSION"
            losses += 1
        else:
            status = "TIE"
            ties += 1

        sig_a = ra.get("reduction_signature", "")
        sig_b = rb.get("reduction_signature", "")
        if sig_a and sig_b:
            is_bit_exact = sig_a == sig_b
            bit_exact_status = "BIT_EXACT" if is_bit_exact else f"REDUCTION_CHANGED ({sig_a} -> {sig_b})"
            if is_bit_exact:
                bit_exact_count += 1
            else:
                reduction_changed_count += 1
        else:
            is_bit_exact = None
            bit_exact_status = "UNKNOWN (no route metadata)"

        comparisons.append(
            {
                "qtype": key[0],
                "policy": key[1],
                "N": key[2],
                "K": key[3],
                "T": key[4],
                "base_us": round(ua, 3),
                "cand_us": round(ub, 3),
                "delta_us": round(delta_us, 3),
                "delta_pct": round(delta_pct, 2),
                "speedup_x": round(sp, 4),
                "base_gbs": round(float(ra["effective_gbs"]), 2),
                "cand_gbs": round(float(rb["effective_gbs"]), 2),
                "status": status,
                "route_a": ra.get("active_route", ""),
                "route_b": rb.get("active_route", ""),
                "reduction_sig_a": sig_a,
                "reduction_sig_b": sig_b,
                "bit_exact": is_bit_exact,
                "bit_exact_status": bit_exact_status,
            }
        )

    geomean_speedup = math.exp(sum(log_speedups) / len(log_speedups)) if log_speedups else 1.0
    arith_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    total_delta_pct = ((sum_b_us - sum_a_us) / sum_a_us * 100.0) if sum_a_us > 0 else 0.0

    best_point = min(comparisons, key=lambda c: c["delta_pct"])
    worst_point = max(comparisons, key=lambda c: c["delta_pct"])

    if require_bit_exact and reduction_changed_count > 0:
        verdict = "REJECT_BIT_EXACTNESS_VIOLATION"
    elif losses == 0 and wins > 0:
        verdict = "PASS_STRICT_IMPROVEMENT"
    elif geomean_speedup > 1.01 and worst_point["delta_pct"] < 5.0:
        verdict = "PASS_NET_IMPROVEMENT"
    elif losses > 0 and geomean_speedup < 1.0:
        verdict = "REJECT_REGRESSION"
    else:
        verdict = "NEUTRAL_OR_MIXED"

    return {
        "mode": "shape_csv_ab",
        "file_a": str(path_a),
        "file_b": str(path_b),
        "matched_points": len(comparisons),
        "noise_threshold_pct": noise_threshold_pct,
        "geomean_speedup_x": round(geomean_speedup, 4),
        "arith_mean_speedup_x": round(arith_speedup, 4),
        "sum_base_us": round(sum_a_us, 3),
        "sum_cand_us": round(sum_b_us, 3),
        "sum_delta_us": round(sum_b_us - sum_a_us, 3),
        "sum_delta_pct": round(total_delta_pct, 2),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bit_exact_points": bit_exact_count,
        "reduction_changed_points": reduction_changed_count,
        "all_bit_exact": reduction_changed_count == 0 and bit_exact_count > 0,
        "best_gain": best_point,
        "worst_change": worst_point,
        "verdict": verdict,
        "points": comparisons,
    }
