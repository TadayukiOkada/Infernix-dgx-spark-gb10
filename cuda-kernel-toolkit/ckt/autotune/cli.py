"""CLI handler for ckt autotune.

Implements the user-facing CLI commands:
  ckt autotune discover
  ckt autotune benchmark
  ckt autotune tune
  ckt autotune generate
  ckt autotune validate-policy
  ckt autotune report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .database import TuningDatabase, TuningRecord
from .diagnostics import CustomKernelDiagnostician
from .discovery import RepositoryDiscovery
from .generator import DispatchGenerator, PolicyCompressor
from .profile import HardwareProfile, SoftwareProfile, TuningProfile
from .registry import CandidateRegistry, CandidateSpec
from .reporting import AutotuneReporter
from .runner import BenchmarkRunner, CorrectnessValidator
from .search import AdaptiveRefiner, NoiseAwareSelector, TuningSearch, WinnerSelection
from .validator import PolicyValidator
from .workload import WorkloadSpec, WorkloadSweep
from ..environment import Environment
from ..shapes import discover_shapes


def register_autotune_subparsers(subparsers: Any, add_common: Any) -> None:
    """Register 'autotune' and its sub-subcommands."""
    at_p = subparsers.add_parser(
        "autotune",
        help="offline, benchmark-driven CUDA autotuning and dispatch generation",
    )
    add_common(at_p)
    at_sub = at_p.add_subparsers(dest="autotune_action", required=True)

    # 1. ckt autotune discover
    disc_p = at_sub.add_parser("discover", help="discover all supported quantization families, operations, and test suites")
    add_common(disc_p)
    disc_p.add_argument("--repo-root", default=".", help="repository root (default: current directory)")
    disc_p.add_argument("-o", "--output", help="save Markdown report to path")
    disc_p.add_argument("--json", action="store_true", help="output discovery metadata as JSON")
    disc_p.set_defaults(func=cmd_autotune)

    # 2. ckt autotune benchmark
    bench_p = at_sub.add_parser("benchmark", help="benchmark candidates on specified workloads with cold-L2 timing")
    add_common(bench_p)
    bench_p.add_argument("--operation", default="linear", help="target operation (default: linear)")
    bench_p.add_argument("--family", "--qtype", dest="family", required=True, help="quantization family (e.g. nvfp4, fp8, q4, q8, bf16)")
    bench_p.add_argument("-n", "--n", type=int, required=True, help="output dimension N")
    bench_p.add_argument("-k", "--k", type=int, required=True, help="reduction dimension K")
    bench_p.add_argument("-t", "--tokens", default="1", help="token count or sweep, e.g. '1' or '1,8,16,32' or '1:32:1'")
    bench_p.add_argument("--policy", default="a16", choices=["a16", "a8", "a4", "invariant"], help="activation compute policy")
    bench_p.add_argument("--execution", default="eager", choices=["eager", "graph"], help="execution mode")
    bench_p.add_argument("--routes", default="exact", choices=["exact", "fast"], help="routes set")
    bench_p.add_argument("--warmup", type=int, default=3, help="warmup calls (default: 3)")
    bench_p.add_argument("--repeat", type=int, default=20, help="repeat sample count (default: 20)")
    bench_p.add_argument("--flush-mib", type=int, help="override cold-L2 flush buffer size in MiB")
    bench_p.add_argument("--repo-root", default=".")
    bench_p.add_argument("--bench-exe", help="explicit path to benchmark binary")
    bench_p.add_argument("--results", help="path to persist results JSON/JSONL")
    bench_p.add_argument("--json", action="store_true")
    bench_p.set_defaults(func=cmd_autotune)

    # 3. ckt autotune tune
    tune_p = at_sub.add_parser("tune", help="run complete autotuning campaign: candidates, sweep, refinement, selection, and generation")
    add_common(tune_p)
    tune_p.add_argument("--operation", default="linear", help="target operation (default: linear)")
    tune_p.add_argument("--family", "--qtype", dest="family", required=True, help="quantization family (e.g. nvfp4, fp8, q4, q8, bf16)")
    tune_p.add_argument("-n", "--n", type=int, required=True, help="output dimension N")
    tune_p.add_argument("-k", "--k", type=int, required=True, help="reduction dimension K")
    tune_p.add_argument("-t", "--tokens", help="optional custom token list or sweep spec (default: representative sweep)")
    tune_p.add_argument("--policy", default="a16", choices=["a16", "a8", "a4", "invariant"])
    tune_p.add_argument("--execution", default="eager", choices=["eager", "graph"])
    tune_p.add_argument("--warmup", type=int, default=3)
    tune_p.add_argument("--repeat", type=int, default=20)
    tune_p.add_argument("--flush-mib", type=int)
    tune_p.add_argument("--min-improvement-pct", type=float, default=2.0, help="minimum percentage gain to switch candidate (default: 2.0%%)")
    tune_p.add_argument("--objective", default="median_us", choices=["median_us", "p95_us"])
    tune_p.add_argument("--no-adaptive-refinement", dest="adaptive_refinement", action="store_false", default=True, help="disable boundary refinement")
    tune_p.add_argument("--results", help="path to persist results database (JSON or JSONL)")
    tune_p.add_argument("--generate", help="output path for generated C++ dispatch file")
    tune_p.add_argument("--report", help="output path for Markdown tuning report")
    tune_p.add_argument("--validate", action="store_true", default=True, help="run policy validation gate after generation")
    tune_p.add_argument("--repo-root", default=".")
    tune_p.add_argument("--json", action="store_true")
    tune_p.set_defaults(func=cmd_autotune)

    # 4. ckt autotune generate
    gen_p = at_sub.add_parser("generate", help="compress winners from results database and emit deterministic C++ dispatch code")
    add_common(gen_p)
    gen_p.add_argument("--results", required=True, help="path to results JSON/JSONL database")
    gen_p.add_argument("-o", "--output", help="path to write generated C++ file (stdout by default)")
    gen_p.add_argument("--operation", default="linear")
    gen_p.add_argument("--family", "--qtype", dest="family", required=True)
    gen_p.add_argument("-n", "--n", type=int, required=True)
    gen_p.add_argument("-k", "--k", type=int, required=True)
    gen_p.add_argument("--token-step", type=int, default=1)
    gen_p.add_argument("--repo-root", default=".")
    gen_p.set_defaults(func=cmd_autotune)

    # 5. ckt autotune validate-policy
    val_p = at_sub.add_parser("validate-policy", help="validate generated policy source against compilation, test suites, and performance")
    add_common(val_p)
    val_p.add_argument("--policy", required=True, help="path to generated or existing C++ dispatch file")
    val_p.add_argument("--family", "--qtype", dest="family", required=True)
    val_p.add_argument("-n", "--n", type=int, required=True)
    val_p.add_argument("-k", "--k", type=int, required=True)
    val_p.add_argument("--operation", default="linear")
    val_p.add_argument("--results", help="optional results JSON/JSONL for performance regression check")
    val_p.add_argument("--max-regression-pct", type=float, default=1.0)
    val_p.add_argument("--repo-root", default=".")
    val_p.add_argument("--json", action="store_true")
    val_p.set_defaults(func=cmd_autotune)

    # 6. ckt autotune report
    rep_p = at_sub.add_parser("report", help="render human-readable tuning report from results database")
    add_common(rep_p)
    rep_p.add_argument("--results", required=True, help="path to results JSON/JSONL database")
    rep_p.add_argument("-o", "--output", help="save report to Markdown file")
    rep_p.add_argument("--json", action="store_true")
    rep_p.set_defaults(func=cmd_autotune)


def cmd_autotune(args: argparse.Namespace) -> int:
    """Entry point dispatch for ckt autotune."""
    action = getattr(args, "autotune_action", "discover")
    repo_root = Path(getattr(args, "repo_root", ".")).resolve()

    # Load environment and hardware profile
    cache_dir = Path(getattr(args, "cache_dir", ".ckt_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = Environment.discover(cache_dir=cache_dir)
    dev = env.device
    profile = TuningProfile.create(dev, repo_root=repo_root)

    if action == "discover":
        disco = RepositoryDiscovery(repo_root=repo_root)
        report = disco.discover()
        if getattr(args, "json", False):
            print(json.dumps(report.to_dict(), indent=2))
        else:
            rendered = report.render_markdown()
            if getattr(args, "output", None):
                out_p = Path(args.output)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(rendered)
                print(f"Discovery report written to {out_p}")
            else:
                print(rendered)
        return 0

    if action == "benchmark":
        runner = BenchmarkRunner(
            repo_root=repo_root,
            hw_profile=profile.hardware,
            bench_exe=getattr(args, "bench_exe", None),
        )
        validator = CorrectnessValidator(repo_root=repo_root)
        db = TuningDatabase()
        if getattr(args, "results", None) and Path(args.results).exists():
            db.load(args.results)

        tokens_spec = getattr(args, "tokens", "1")
        tokens_list = WorkloadSweep.from_string(tokens_spec)

        qtype = args.family.lower()
        cand = CandidateSpec(
            id=f"{args.operation}_{qtype}_n{args.n}_k{args.k}_explicit",
            operation=args.operation,
            quantization_family=qtype,
            kernel_family="active_route",
            n=args.n,
            k=args.k,
            template_name="ActiveRoute",
            launch_expr="explicit",
        )

        # Correctness check
        first_wl = WorkloadSpec(
            operation=args.operation,
            quantization_family=qtype,
            n=args.n,
            k=args.k,
            tokens=tokens_list[0],
            policy=args.policy,
            execution_mode=args.execution,
            routes=args.routes,
        )
        c_summary = validator.validate(cand, first_wl)

        results = []
        for t in tokens_list:
            wl = WorkloadSpec(
                operation=args.operation,
                quantization_family=qtype,
                n=args.n,
                k=args.k,
                tokens=t,
                policy=args.policy,
                execution_mode=args.execution,
                routes=args.routes,
            )
            meas = runner.measure(
                cand,
                wl,
                warmup=args.warmup,
                repeat=args.repeat,
                flush_mib=getattr(args, "flush_mib", None),
            )
            rec = TuningRecord(
                schema_version=1,
                hardware_profile_id=profile.profile_id,
                software_profile=profile.software.to_dict(),
                workload=wl.to_dict(),
                candidate=cand.to_dict(),
                compile_status="success",
                correctness_status="pass" if len(c_summary.failed) == 0 else "fail",
                correctness_tests=c_summary,
                latency_us=meas.median_us,
                latency_p95_us=meas.p95_us,
                latency_min_us=meas.min_us,
                latency_max_us=meas.max_us,
                latency_mad_us=meas.mad_us,
                samples_us=meas.samples_us,
                effective_gbs=meas.effective_gbs,
                useful_tflops=meas.useful_tflops,
                warmup=args.warmup,
                repeat=args.repeat,
                flush_mib=meas.flush_mib,
            )
            db.add(rec)
            results.append(rec)

        if getattr(args, "results", None):
            res_path = Path(args.results)
            if res_path.suffix == ".jsonl":
                db.save_jsonl(res_path)
            else:
                db.save_json(res_path)
            print(f"Persisted {len(results)} records to {res_path}")

        if getattr(args, "json", False):
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"Benchmark results for {args.operation} {qtype.upper()} [{args.n}, {args.k}] on {profile.hardware.gpu_name}:")
            print(f"  {'T':>6}  {'median_us':>11}  {'p95_us':>11}  {'effective_GB/s':>16}  {'useful_TFLOP/s':>16}  correctness")
            for r in results:
                t_val = r.workload.get("tokens", 1)
                corr_s = "pass" if r.correctness_status == "pass" else "FAIL"
                print(f"  {t_val:>6}  {r.latency_us:>11.3f}  {r.latency_p95_us:>11.3f}  {r.effective_gbs:>16.1f}  {r.useful_tflops:>16.2f}  {corr_s}")

        return 0

    if action == "tune":
        disco = RepositoryDiscovery(repo_root=repo_root)
        disco_report = disco.discover()

        qtype = args.family.lower()
        shape_specs = discover_shapes(repo_root, qtype=qtype, n=args.n, k=args.k, dev=dev)
        existing_spec = shape_specs[0] if shape_specs else None

        runner = BenchmarkRunner(repo_root=repo_root, hw_profile=profile.hardware)
        validator = CorrectnessValidator(repo_root=repo_root)
        db = TuningDatabase()

        target_policy = getattr(args, "policy", "a16")
        existing_bands = [
            b.to_dict() for b in existing_spec.bands
            if getattr(b, "policy", "a16") == target_policy
        ] if existing_spec else []
        aliases = existing_spec.aliases if existing_spec else {}
        disabled_routes = []
        if existing_spec and "uses_a4" in Path(existing_spec.file_path).read_text(errors="replace"):
            disabled_routes.append("a4_route_disabled")

        registry = CandidateRegistry(hw_profile=profile.hardware)
        candidates = registry.generate_candidates_for_shape(
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            existing_bands=existing_bands,
            aliases=aliases,
            policy=target_policy,
        )

        tokens_spec = getattr(args, "tokens", None)
        tokens_list = (
            WorkloadSweep.from_string(tokens_spec)
            if tokens_spec
            else WorkloadSweep.standard_sweep()
        )
        token_step = existing_spec.token_step if existing_spec else 1
        if token_step > 1:
            tokens_list = [t for t in tokens_list if t >= token_step and t % token_step == 0]

        searcher = TuningSearch(
            runner=runner,
            validator=validator,
            database=db,
            profile=profile,
            min_improvement_pct=args.min_improvement_pct,
        )

        winners = searcher.tune_shape(
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            candidates=candidates,
            tokens_list=tokens_list,
            adaptive_refinement=args.adaptive_refinement,
            policy=args.policy,
            execution_mode=args.execution,
            warmup=args.warmup,
            repeat=args.repeat,
            flush_mib=getattr(args, "flush_mib", None),
            existing_bands=existing_bands,
        )

        # Check if any winning candidate genuinely improved over the existing baseline
        def baseline_route_at(t: int) -> str:
            if existing_bands:
                for b in existing_bands:
                    min_t = b.get("min_tokens", 1)
                    max_t = b.get("max_tokens")
                    if min_t <= t and (max_t is None or t <= max_t):
                        return b.get("route_expr", "")
            return ""

        improvements: list[str] = []
        for t, w in sorted(winners.items()):
            base_route = baseline_route_at(t)
            if base_route and w.winning_candidate.launch_expr != base_route and w.margin_pct >= args.min_improvement_pct:
                ru_lat = w.runner_up_latency_us or 0.0
                improvements.append(
                    f"T={t:>4d}: {base_route} ({ru_lat:.2f} us) -> {w.winning_candidate.launch_expr} ({w.winning_latency_us:.2f} us, {w.margin_pct:+.1f}%)"
                )

        if not improvements and existing_spec is not None:
            print(f"\n[OK] Shape {qtype.upper()} [{args.n}, {args.k}] is ALREADY OPTIMAL.")
            print(f"No candidate beat the incumbent dispatch by >= {args.min_improvement_pct}%.")
            print("Preserving existing shape file without modification.")
            return 0

        print(f"\n[PERF] Shape {qtype.upper()} [{args.n}, {args.k}] improved over baseline at {len(improvements)} token widths:")
        for imp in improvements:
            print(f"  {imp}")

        # Compress winners into intervals
        token_step = existing_spec.token_step if existing_spec else 1
        latencies = getattr(searcher, "latencies_per_token", None)
        intervals = PolicyCompressor.compress(
            winners,
            token_step=token_step,
            latencies_per_token=latencies,
            tolerance_pct=args.min_improvement_pct,
        )

        # Diagnostics for any high latency workloads
        diagnostician = CustomKernelDiagnostician(hw_profile=profile.hardware)
        diagnostics = []
        for t, w in sorted(winners.items()):
            diag = diagnostician.diagnose(w.record)
            diagnostics.append(diag)

        # Generate C++ code
        generator = DispatchGenerator(profile=profile)
        res_hash = hashlib.sha256(
            json.dumps([w.to_dict() for w in winners.values()], sort_keys=True).encode()
        ).hexdigest()[:16]

        gen_target = getattr(args, "generate", None)
        existing_code = None
        if gen_target and Path(gen_target).exists():
            existing_code = Path(gen_target).read_text(errors="replace")
        elif existing_spec and Path(existing_spec.file_path).exists():
            existing_code = Path(existing_spec.file_path).read_text(errors="replace")

        cpp_code = generator.generate_cpp(
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            intervals=intervals,
            token_step=token_step,
            disabled_routes=disabled_routes,
            results_hash=res_hash,
            existing_code=existing_code,
        )

        # Optional policy validation
        val_report = None
        if getattr(args, "validate", True):
            policy_val = PolicyValidator(repo_root=repo_root, runner=runner, validator=validator)
            target_f = existing_spec.file_path if existing_spec else f"src/ops/linear/{qtype}/shapes/n{args.n}_k{args.k}.cu"
            val_report = policy_val.validate_policy_source(
                cpp_code=cpp_code,
                operation=args.operation,
                qtype=qtype,
                n=args.n,
                k=args.k,
                target_shape_file=target_f,
                expected_winners=winners,
            )

        # Persist results
        if getattr(args, "results", None):
            res_p = Path(args.results)
            if res_p.suffix == ".jsonl":
                db.save_jsonl(res_p)
            else:
                db.save_json(res_p)
            print(f"Tuning database saved to {res_p}")

        # Persist generated C++
        if getattr(args, "generate", None):
            gen_p = Path(args.generate)
            gen_p.parent.mkdir(parents=True, exist_ok=True)
            gen_p.write_text(cpp_code)
            print(f"Generated C++ dispatch code written to {gen_p}")

        # Generate and save report
        rendered_rep = AutotuneReporter.render_tuning_campaign_report(
            profile=profile,
            discovery=disco_report,
            shape_key=f"{args.operation}_{qtype}_n{args.n}_k{args.k}",
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            winners=winners,
            intervals=intervals,
            diagnostics=diagnostics,
            validation_report=val_report,
            generated_cpp_code=cpp_code,
        )

        if getattr(args, "report", None):
            rep_p = Path(args.report)
            rep_p.parent.mkdir(parents=True, exist_ok=True)
            rep_p.write_text(rendered_rep)
            print(f"Tuning report written to {rep_p}")
        elif not getattr(args, "json", False):
            print(rendered_rep)

        if getattr(args, "json", False):
            summary = {
                "profile": profile.to_dict(),
                "shape": {"operation": args.operation, "qtype": qtype, "n": args.n, "k": args.k},
                "winners": {str(t): w.to_dict() for t, w in winners.items()},
                "intervals": [i.to_dict() for i in intervals],
                "validation": val_report.to_dict() if val_report else None,
                "diagnostics": [d.to_dict() for d in diagnostics],
            }
            print(json.dumps(summary, indent=2))

        return 0

    if action == "generate":
        db = TuningDatabase()
        db.load(args.results)
        qtype = args.family.lower()
        winners_raw = db.get_valid_winners_per_token(
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            profile_id=profile.profile_id,
        )
        if not winners_raw:
            raise SystemExit(f"No valid records found for {args.operation} {qtype} [{args.n}, {args.k}] in {args.results}")

        # Reconstruct WinnerSelection dict
        winners: dict[int, WinnerSelection] = {}
        for t, rec in sorted(winners_raw.items()):
            cand = CandidateSpec.from_dict(rec.candidate)
            winners[t] = WinnerSelection(
                tokens=t,
                winning_candidate=cand,
                runner_up=None,
                winning_latency_us=rec.latency_us,
                runner_up_latency_us=None,
                margin_pct=0.0,
                is_tie=False,
                tie_breaker_applied=False,
                record=rec,
            )

        intervals = PolicyCompressor.compress(winners, token_step=args.token_step)
        generator = DispatchGenerator(profile=profile)
        gen_out = getattr(args, "output", None)
        existing_code = None
        if gen_out and Path(gen_out).exists():
            existing_code = Path(gen_out).read_text(errors="replace")
        cpp = generator.generate_cpp(
            operation=args.operation,
            qtype=qtype,
            n=args.n,
            k=args.k,
            intervals=intervals,
            token_step=args.token_step,
            existing_code=existing_code,
        )
        if getattr(args, "output", None):
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(cpp)
            print(f"Generated C++ dispatch written to {out_p}")
        else:
            print(cpp)
        return 0

    if action == "validate-policy":
        policy_path = Path(args.policy).resolve()
        if not policy_path.exists():
            raise SystemExit(f"Policy file not found: {policy_path}")
        cpp_code = policy_path.read_text(errors="replace")

        runner = BenchmarkRunner(repo_root=repo_root, hw_profile=profile.hardware)
        validator = CorrectnessValidator(repo_root=repo_root)
        policy_val = PolicyValidator(repo_root=repo_root, runner=runner, validator=validator)

        # Expected winners from results if provided
        expected_winners: dict[int, WinnerSelection] = {}
        if getattr(args, "results", None) and Path(args.results).exists():
            db = TuningDatabase()
            db.load(args.results)
            qtype = args.family.lower()
            winners_raw = db.get_valid_winners_per_token(
                operation=args.operation,
                qtype=qtype,
                n=args.n,
                k=args.k,
                profile_id=profile.profile_id,
            )
            for t, rec in winners_raw.items():
                expected_winners[t] = WinnerSelection(
                    tokens=t,
                    winning_candidate=CandidateSpec.from_dict(rec.candidate),
                    runner_up=None,
                    winning_latency_us=rec.latency_us,
                    runner_up_latency_us=None,
                    margin_pct=0.0,
                    is_tie=False,
                    tie_breaker_applied=False,
                    record=rec,
                )

        rep = policy_val.validate_policy_source(
            cpp_code=cpp_code,
            operation=args.operation,
            qtype=args.family.lower(),
            n=args.n,
            k=args.k,
            target_shape_file=policy_path,
            expected_winners=expected_winners,
            max_allowed_regression_pct=args.max_regression_pct,
        )

        if getattr(args, "json", False):
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            status = "PASSED (Ready for production)" if rep.production_ready else "FAILED"
            print(f"=== Policy Validation Verdict: {status} ===")
            print(f"  Compilation check:       {'PASS' if rep.compilation_passed else 'FAIL'}")
            print(f"  Existing test suite:     {'PASS' if rep.correctness_passed else 'FAIL'}")
            print(f"  Strict byte identity:    {'PASS (0 ULP mismatch)' if rep.byte_identity_passed else 'FAIL'}")
            print(f"  Performance regression:  {'PASS (within threshold)' if rep.benchmark_verified else 'FAIL'}")
            if rep.regressions:
                for reg in rep.regressions:
                    print(f"    Regression at T={reg.get('tokens')}: {reg.get('regression_pct')}%")
        return 0 if rep.production_ready else 1

    if action == "report":
        db = TuningDatabase()
        db.load(args.results)
        print(f"Loaded {db.count()} tuning records from {args.results}")
        # Group by shape
        records = db.records
        if not records:
            print("No records in database.")
            return 0
        first_r = records[0]
        wl = first_r.workload
        qtype = wl.get("quantization_family", "unknown")
        n = wl.get("n", 0)
        k = wl.get("k", 0)
        op = wl.get("operation", "linear")

        winners_raw = db.get_valid_winners_per_token(
            operation=op,
            qtype=qtype,
            n=n,
            k=k,
            profile_id=profile.profile_id,
        )
        winners: dict[int, WinnerSelection] = {}
        for t, rec in sorted(winners_raw.items()):
            winners[t] = WinnerSelection(
                tokens=t,
                winning_candidate=CandidateSpec.from_dict(rec.candidate),
                runner_up=None,
                winning_latency_us=rec.latency_us,
                runner_up_latency_us=None,
                margin_pct=0.0,
                is_tie=False,
                tie_breaker_applied=False,
                record=rec,
            )
        intervals = PolicyCompressor.compress(winners)
        rendered = AutotuneReporter.render_tuning_campaign_report(
            profile=profile,
            discovery=None,
            shape_key=f"{op}_{qtype}_n{n}_k{k}",
            operation=op,
            qtype=qtype,
            n=n,
            k=k,
            winners=winners,
            intervals=intervals,
        )
        if getattr(args, "output", None):
            Path(args.output).write_text(rendered)
            print(f"Report written to {args.output}")
        else:
            print(rendered)
        return 0

    return 0
