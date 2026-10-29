"""Benchmark and correctness execution harnesses for autotuning.

Executes candidate workloads with honest cold-L2 caching, CUDA event timing,
sample capture, and strict gating against the repository's existing correctness tests,
including byte-for-byte output identity verification.
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .database import CorrectnessTestSummary, TuningRecord
from .profile import HardwareProfile, SoftwareProfile
from .registry import CandidateSpec
from .workload import WorkloadSpec
from ..shapes import find_linear_bench, optimal_flush_mib
from ..util import ToolError, run, to_plain


BF16_SHAPES: dict[tuple[int, int], str] = {
    (13952, 2560): "qkvg",
    (96, 2560): "gdnab",
    (16384, 2560): "gdnqkvz",
    (2560, 6144): "out6144",
    (12800, 2560): "plekv",
    (1280, 2560): "shgu",
    (2560, 640): "shdown",
    (324, 10240): "hcdown",
    (10240, 320): "hcup",
    (320, 10240): "fmdown",
    (248320, 2560): "head",
    (131072, 2560): "prophead",
}

FP8_SHAPES: dict[tuple[int, int], str] = {
    (16384, 2560): "gdnqkvz",
    (13312, 2560): "qkvg",
    (2560, 6144): "out6144",
    (1280, 2560): "shgu",
    (2560, 640): "shdown",
    (14336, 5120): "ffn14336",
    (16384, 5120): "ffn16384",
    (34816, 5120): "ffn34816",
    (5120, 6144): "down5120",
    (5120, 17408): "down17408",
}


def normalize_route_name(name_or_launch: str) -> str:
    s = name_or_launch.strip()
    s = s.replace("return ", "").replace(";", "")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"<[^>]*>", "", s)
    for pfx in ("launch_q8_a16_", "launch_q4_a16_", "launch_q5_a16_", "launch_q6_a16_", "launch_bf16_", "fp8_linear_a16_"):
        if s.startswith(pfx):
            s = s[len(pfx):]
    if s.endswith("_a1"):
        s = s[:-3]
    return s.strip()


@dataclass
class BenchmarkMeasurement:
    """Raw and aggregated timing metrics from a benchmark execution."""

    median_us: float
    min_us: float
    p95_us: float
    max_us: float
    mad_us: float
    samples_us: list[float]
    effective_gbs: float
    useful_tflops: float
    warmup: int
    repeat: int
    flush_mib: int
    is_noisy: bool = False
    noise_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


class CorrectnessValidator:
    """Validates candidate correctness against existing repository tests."""

    def __init__(self, repo_root: str | Path = "."):
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / "build-native").exists() and (self.repo_root.parent / "build-native").exists():
            self.repo_root = self.repo_root.parent
        self.tests_dir = self.repo_root / "build-native" / "tests"
        self._test_cache: dict[str, tuple[int, str]] = {}

    def validate(
        self,
        candidate: CandidateSpec,
        workload: WorkloadSpec,
        required_tests: Sequence[str] | None = None,
    ) -> CorrectnessTestSummary:
        """Run existing correctness tests applicable to this candidate and workload."""
        summary = CorrectnessTestSummary()

        # 1. Determine applicable tests
        qtype = workload.quantization_family.lower()
        tests_to_run: list[str] = list(required_tests) if required_tests else []

        if not tests_to_run:
            # Map by operation and policy
            if workload.operation == "linear":
                if workload.policy in ("a16", "invariant"):
                    tests_to_run.append(f"infernix_linear_{qtype}_a16_test")
                elif workload.policy == "a4":
                    tests_to_run.append(f"infernix_linear_{qtype}_a4_test")
                elif workload.policy == "a8":
                    tests_to_run.append(f"infernix_linear_{qtype}_a8_test")

                # Column-invariance / byte-identity test
                if workload.requires_byte_identity:
                    tests_to_run.append("infernix_column_invariant_projections_test")
            elif workload.operation == "linear_add":
                tests_to_run.append(f"infernix_linear_add_{qtype}_a16_test")
            elif workload.operation == "linear_swiglu":
                tests_to_run.append(f"infernix_linear_swiglu_{qtype}_test")

        summary.required_tests = tests_to_run
        summary.byte_identity_enforced = workload.requires_byte_identity

        # 2. Execute each test
        for test_name in tests_to_run:
            test_exe = self.tests_dir / test_name
            if not test_exe.exists() or not os.access(test_exe, os.X_OK):
                summary.skipped.append(test_name)
                summary.details[test_name] = "executable_not_found"
                continue

            try:
                t_cache_key = f"{test_name}:{workload.n}:{workload.k}"
                if t_cache_key in self._test_cache:
                    returncode, output = self._test_cache[t_cache_key]
                else:
                    cmd = [str(test_exe)]
                    if test_name.startswith("infernix_linear_") and "column_invariant" not in test_name and workload.n > 0 and workload.k > 0:
                        cmd.extend(["--shape", str(workload.n), str(workload.k)])
                    res = subprocess.run(
                        cmd,
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    returncode = res.returncode
                    output = res.stdout + "\n" + res.stderr
                    self._test_cache[t_cache_key] = (returncode, output)

                if returncode == 0:
                    summary.passed.append(test_name)
                    summary.details[test_name] = "pass"
                else:
                    summary.failed.append(test_name)
                    summary.details[test_name] = f"failed (code {returncode}): {output[:300]}"

                    # If this is a byte-identity test, record strict failure
                    if "column_invariant" in test_name or "exact" in output.lower():
                        summary.byte_identity_passed = False
            except subprocess.TimeoutExpired:
                summary.failed.append(test_name)
                summary.details[test_name] = "timeout"
            except Exception as e:
                summary.failed.append(test_name)
                summary.details[test_name] = str(e)

        return summary


class BenchmarkRunner:
    """Executes candidates on specified workloads with cold-L2 timing."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        hw_profile: HardwareProfile | None = None,
        bench_exe: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        if not (self.repo_root / "build-native").exists() and (self.repo_root.parent / "build-native").exists():
            self.repo_root = self.repo_root.parent

        self.hw_profile = hw_profile
        self.bench_exe = Path(bench_exe).resolve() if bench_exe else find_linear_bench(self.repo_root)
        self._meas_cache: dict[tuple[Any, ...], BenchmarkMeasurement] = {}
        self._sweep_data_cache: dict[tuple[str, int, int], dict[int, dict[str, dict[str, Any]]]] = {}

        # Right-size L2 flush: 2x target L2 cache (or 32-48 MiB on GB10)
        l2_bytes = hw_profile.l2_cache_size if hw_profile else (24 * 1024 * 1024)
        l2_mib = max(16, math.ceil(l2_bytes / (1024 * 1024)))
        self.default_flush_mib = max(16, int(round(l2_mib * 1.5)))

    def _find_sweep_binary(self, qtype: str, n: int, k: int) -> tuple[Path | None, str | None]:
        bench_dir = self.repo_root / "build-native" / "bench"
        q = qtype.lower()
        if q == "q8":
            bin_path = bench_dir / "infernix_q8_route_sweep"
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, None
        elif q == "q4":
            bin_path = bench_dir / "infernix_q4_route_sweep"
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, None
        elif q == "q5":
            bin_path = bench_dir / "infernix_q5_route_sweep"
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, None
        elif q == "q6":
            bin_path = bench_dir / "infernix_q6_route_sweep"
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, None
        elif q == "nvfp4":
            bin_path = bench_dir / "infernix_nvfp4_route_sweep"
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, None
        elif q == "bf16":
            bin_path = bench_dir / "infernix_bf16_flash_next_sweep"
            if (n, k) in BF16_SHAPES and bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, BF16_SHAPES[(n, k)]
        elif q == "fp8":
            bin_path = bench_dir / "infernix_fp8_flash_next_sweep"
            if (n, k) in FP8_SHAPES and bin_path.exists() and os.access(bin_path, os.X_OK):
                return bin_path, FP8_SHAPES[(n, k)]
        return None, None

    def preload_sweep(
        self,
        qtype: str,
        n: int,
        k: int,
        tokens_list: Sequence[int],
        warmup: int = 3,
        repeat: int = 15,
        flush_mib: int | None = None,
    ) -> None:
        """Preload all route sweep measurements for a set of token widths."""
        sweep_bin, shape_label = self._find_sweep_binary(qtype, n, k)
        if not sweep_bin:
            return

        cache_key = (qtype.lower(), n, k)
        if cache_key not in self._sweep_data_cache:
            self._sweep_data_cache[cache_key] = {}

        existing_tokens = self._sweep_data_cache[cache_key].keys()
        needed_tokens = [t for t in tokens_list if t not in existing_tokens]
        if not needed_tokens:
            return

        eff_flush = flush_mib if flush_mib is not None else self.default_flush_mib
        q = qtype.lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sweep_out.csv"
            if q in ("q4", "q5", "q6", "q8", "nvfp4"):
                cmd = [
                    str(sweep_bin),
                    "--n", str(n),
                    "--k", str(k),
                    "--tokens", ",".join(str(t) for t in sorted(needed_tokens)),
                    "--warmup", str(warmup),
                    "--repeat", str(repeat),
                    "--flush-mib", str(eff_flush),
                    "-o", str(csv_path),
                ]
            else:  # bf16 or fp8
                prefill_tokens = [t for t in needed_tokens if t >= 9]
                cmd = [
                    str(sweep_bin),
                    "--shape", shape_label,
                    "--t-list", ",".join(str(t) for t in sorted(prefill_tokens)) if prefill_tokens else "9",
                    "--warmup", str(warmup),
                    "--repeat", str(repeat),
                    "--flush-mib", str(eff_flush),
                    "--csv", str(csv_path),
                ]

            try:
                res = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, timeout=300)
                if res.returncode != 0 or not csv_path.exists():
                    return

                with csv_path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t = int(row.get("T", row.get("t", 1)))
                        route = row.get("route", "").strip()
                        if not route:
                            continue
                        if t not in self._sweep_data_cache[cache_key]:
                            self._sweep_data_cache[cache_key][t] = {}

                        bit_exact = bool(int(row.get("bit_exact", 1)))
                        t_invariant = bool(int(row.get("t_invariant", 1)))
                        med_us = float(row.get("median_us", 0.0))
                        min_us = float(row.get("min_us", med_us))
                        p95_us = float(row.get("p95_us", med_us))
                        gbs = float(row.get("gbs", row.get("effective_gbs", 0.0)) or 0.0)
                        tflops = float(row.get("tflops", row.get("useful_tflops", 0.0)) or 0.0)

                        self._sweep_data_cache[cache_key][t][route] = {
                            "median_us": med_us,
                            "min_us": min_us,
                            "p95_us": p95_us,
                            "gbs": gbs,
                            "tflops": tflops,
                            "bit_exact": bit_exact,
                            "t_invariant": t_invariant,
                        }
            except Exception:
                pass

    def measure(
        self,
        candidate: CandidateSpec,
        workload: WorkloadSpec,
        *,
        warmup: int = 3,
        repeat: int = 20,
        flush_mib: int | None = None,
        execution_mode: str | None = None,
    ) -> BenchmarkMeasurement:
        """Measure cold-cache latency for candidate on workload."""
        eff_flush = flush_mib if flush_mib is not None else self.default_flush_mib
        eff_exec = execution_mode or workload.execution_mode
        qtype = workload.quantization_family.lower()

        cache_key = (
            candidate.id,
            candidate.launch_expr,
            qtype,
            workload.policy.lower(),
            workload.n,
            workload.k,
            workload.tokens,
            eff_exec,
            workload.routes,
            warmup,
            repeat,
            eff_flush,
        )
        if cache_key in self._meas_cache:
            return self._meas_cache[cache_key]

        # 1. Check if a route sweep binary is available for this family and geometry
        sweep_bin, shape_label = self._find_sweep_binary(qtype, workload.n, workload.k)
        if sweep_bin:
            geom_key = (qtype, workload.n, workload.k)
            if geom_key not in self._sweep_data_cache or workload.tokens not in self._sweep_data_cache[geom_key]:
                self.preload_sweep(qtype, workload.n, workload.k, [workload.tokens], warmup, repeat, eff_flush)

            t_data = self._sweep_data_cache.get(geom_key, {}).get(workload.tokens, {})
            if t_data:
                # Map candidate to route entry in sweep
                matched_entry = None
                norm_cand = normalize_route_name(candidate.launch_expr)

                is_baseline = candidate.kernel_family in ("production", "registered", "baseline") or candidate.id.endswith("_production")

                if is_baseline:
                    matched_entry = t_data.get("production") or t_data.get("registered") or t_data.get("decode_registered")
                else:
                    # Match by exact name or normalized name
                    for rname, rinfo in t_data.items():
                        norm_r = normalize_route_name(rname)
                        if norm_cand == norm_r or norm_cand in norm_r or norm_r in norm_cand:
                            matched_entry = rinfo
                            break
                        if candidate.template_name and candidate.template_name.lower() in rname.lower():
                            matched_entry = rinfo
                            break

                if matched_entry is not None:
                    # Gating: for q4, bit_exact encodes numeric_ok; for q8/others, enforce bit_exact when byte-identity is required
                    if (qtype in ("q4", "q5", "q6") or workload.requires_byte_identity) and not matched_entry.get("bit_exact", True):
                        meas = BenchmarkMeasurement(
                            median_us=float("inf"),
                            min_us=float("inf"),
                            p95_us=float("inf"),
                            max_us=float("inf"),
                            mad_us=0.0,
                            samples_us=[],
                            effective_gbs=0.0,
                            useful_tflops=0.0,
                            warmup=warmup,
                            repeat=repeat,
                            flush_mib=eff_flush,
                            is_noisy=True,
                            noise_ratio=1.0,
                        )
                        self._meas_cache[cache_key] = meas
                        return meas

                    med_us = matched_entry["median_us"]
                    min_us = matched_entry["min_us"]
                    p95_us = matched_entry["p95_us"]
                    gbs = matched_entry["gbs"]
                    tflops = matched_entry["tflops"]
                    mad = abs(med_us - min_us) / 1.4826 if med_us > 0 else 0.0
                    noise_ratio = (p95_us - min_us) / med_us if med_us > 0 else 0.0

                    meas = BenchmarkMeasurement(
                        median_us=med_us,
                        min_us=min_us,
                        p95_us=p95_us,
                        max_us=p95_us,
                        mad_us=mad,
                        samples_us=[min_us, med_us, p95_us],
                        effective_gbs=gbs,
                        useful_tflops=tflops,
                        warmup=warmup,
                        repeat=repeat,
                        flush_mib=eff_flush,
                        is_noisy=noise_ratio > 0.15,
                        noise_ratio=noise_ratio,
                    )
                    self._meas_cache[cache_key] = meas
                    return meas
                else:
                    # Candidate not in sweep: if baseline, fall back to production in sweep;
                    # if synthetic candidate not in sweep, it cannot be run, so return inf
                    if not is_baseline:
                        meas = BenchmarkMeasurement(
                            median_us=float("inf"),
                            min_us=float("inf"),
                            p95_us=float("inf"),
                            max_us=float("inf"),
                            mad_us=0.0,
                            samples_us=[],
                            effective_gbs=0.0,
                            useful_tflops=0.0,
                            warmup=warmup,
                            repeat=repeat,
                            flush_mib=eff_flush,
                            is_noisy=False,
                            noise_ratio=0.0,
                        )
                        self._meas_cache[cache_key] = meas
                        return meas

        # 2. Fallback to infernix_linear_bench (for compiled production route)
        fallback_key = (
            "__compiled_production__",
            qtype,
            workload.policy.lower(),
            workload.n,
            workload.k,
            workload.tokens,
            eff_exec,
            workload.routes,
            warmup,
            repeat,
            eff_flush,
        )
        if fallback_key in self._meas_cache:
            meas = self._meas_cache[fallback_key]
            self._meas_cache[fallback_key] = meas
            self._meas_cache[cache_key] = meas
            return meas

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bench_out.csv"

            cmd = [
                str(self.bench_exe),
                "--qtype", workload.quantization_family,
                "--policy", workload.policy,
                "--n", str(workload.n),
                "--k", str(workload.k),
                "--t", str(workload.tokens),
                "--execution", eff_exec,
                "--warmup", str(warmup),
                "--repeat", str(repeat),
                "--flush-mib", str(eff_flush),
                "--routes", workload.routes,
                "--csv-out", str(csv_path),
            ]

            res = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

            if res.returncode != 0:
                raise ToolError(
                    f"Benchmark executable failed ({res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
                )

            if not csv_path.exists():
                raise ToolError(f"Benchmark executable did not produce CSV at {csv_path}")

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                raise ToolError("Benchmark CSV was empty")

            row = rows[0]
            med_us = float(row.get("median_us", 0.0))
            min_us = float(row.get("min_us", med_us))
            p95_us = float(row.get("p95_us", med_us))
            gbs = float(row.get("effective_gbs", 0.0) or 0.0)
            tflops = float(row.get("useful_tflops", 0.0) or 0.0)

            samples = [min_us, med_us, p95_us]
            max_us = p95_us
            mad = abs(med_us - min_us) / 1.4826 if med_us > 0 else 0.0
            noise_ratio = (p95_us - min_us) / med_us if med_us > 0 else 0.0
            is_noisy = noise_ratio > 0.15

            meas = BenchmarkMeasurement(
                median_us=med_us,
                min_us=min_us,
                p95_us=p95_us,
                max_us=max_us,
                mad_us=mad,
                samples_us=samples,
                effective_gbs=gbs,
                useful_tflops=tflops,
                warmup=warmup,
                repeat=repeat,
                flush_mib=eff_flush,
                is_noisy=is_noisy,
                noise_ratio=noise_ratio,
            )
            self._meas_cache[fallback_key] = meas
            self._meas_cache[cache_key] = meas
            return meas
