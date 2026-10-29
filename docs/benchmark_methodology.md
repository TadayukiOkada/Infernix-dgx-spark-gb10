# GB10 Benchmark Methodology

Protocol for every measurement in the GB10 retune program. Core rule:
**Measure first → hypothesis → change one variable → benchmark again → validate correctness →
keep or revert on evidence.** No GPU clock or system-level GPU setting is ever changed.

## 0. Fixed conditions (identical for baseline and candidate)

- Device locked: GB10 sm_121a, 48 SMs, build gate `CMAKE_CUDA_ARCHITECTURES = 121a` (hard-error otherwise).
- Clocks untouched; rely on default DVFS. Record `nvidia-smi` temp/power/util before and after each run
  and discard runs where thermal state drifted between baseline and candidate.
- Same build tree (`build-native/`), same binary, only the one variable under test differs.
- Weights fixed at `/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/`.
- Every timed quantity reports **median + variability (min, p95)**, never a single sample.

## 1. Timing layers — measure separately, never conflate

Each number is attributed to exactly one layer so a regression is localizable:

1. **Kernel-only** — device launch time of a single schedule, L2 cold.
2. **Host launch / dispatch** — `resolve_route`/plan resolution + launch overhead (CPU-side).
3. **Allocation** — workspace/arena sizing and map/unmap cost (`WorkspaceArena`, VMM).
4. **Scheduler** — round admission / graph replay scheduling.
5. **Full pipeline** — one op executed through the production wrapper+plan path.
6. **End-to-end** — `bench/inference/infernix_bench` prefill/decode tokens-per-second.

## 2. Cold-launch discipline (how the existing benches time)

The authoritative per-kernel method already in the harnesses:
- Flush L2 (≥256 MB scratch) **before each warmup and before each timed sample**.
- `measure_cold_launch_prepared(restore, launch, flush, stream, warmup, repeat)` when the op needs a
  prepared operand state restored per sample; `measure_cold_launch(launch, …)` otherwise.
- For nvfp4, quantize (A4) happens **inside** the timed launch, so an A16↔A4 crossover is apples-to-apples
  (quantize cost is part of the route).
- CUDA-graph ops: `resolve_plan` runs once at capture; time the captured replay, not re-planning.

## 3. Candidate enumeration (bypass policy gates)

Policy-gated benches hide the true crossover because the production route is chosen *before* timing.
To fit a band table, use `--variants`, which times **every internal schedule candidate independently per
token count** and prints a CSV. This is how the nvfp4 add/swiglu tables were derived.

Commands (exact flags verified against `nvfp4_linear_add_bench.cu`):

```bash
cmake --build build-native --target infernix_nvfp4_linear_add_bench
build-native/bench/infernix_nvfp4_linear_add_bench \
  --n 5120 --k 6144 --variants \
  --t-sweep 1,2,4,8,16,20,32,48,64,96,128,160,256,288,512,1024 \
  --warmup 5 --repeat 15 --csv-out /tmp/gb10_more/nvfp4add_cand_k6144.csv
```

CSV header: `path,kind,K,T,median_us,min_us,p95_us` (swiglu: `path,kind,K,T,...`). Production-path check uses
`--policy a16|a4` instead of `--variants`.

Ops that already have a `--variants` harness: nvfp4 `linear_add`, nvfp4 `linear_swiglu`, fp8 `linear_add`,
fp8 `linear_swiglu`. **To extend** (Phase 2 work): `linear_pair` (q8), `q8_linear_swiglu`, `q4_linear_swiglu`,
`sparse_moe`, `gdn_gating_proj` (bf16 SplitK), and the 61 `linear/*/shapes/*.cu` `select_*` families.

## 4. Regression thresholds (noise floor)

Before declaring any change a win/revert, establish the noise floor per bench:
1. Run the **same** binary/config 5× back-to-back.
2. Threshold = max observed median spread across those 5 runs (per token count).
3. A candidate "wins" only if its median improves by **more than the noise floor** AND the improvement is
   stable across ≥2 independent runs. Improvements inside the noise floor are **not wins**.
4. Record the measured noise floor alongside results so later comparisons reuse it.

Observed so far: run-to-run noise is small but nonzero; the nvfp4 production dispatch matched measured winners
within noise. Formal per-bench floors to be recorded in `docs/experiments/` as they are measured.

## 5. Correctness hooks (mandatory before keeping a change)

A faster route that changes accumulation order is not accepted on speed alone.

- **Bit-exact class**: routes that keep the same K-order MMA tiles must remain bit-identical — assert against a
  reference output (`bf16_flash_next_sweep.cu`-style route-vs-reference bit invariance).
- **Order-changing class**: split-K swaps or route crossovers alter accumulation order → must pass
  `tests/test_perplexity_evaluation.cpp` + the relevant `eval/run_qwen3_8_27b_nvfp4_*` script with quality within
  tolerance. CPU-vs-GPU W4A4 expert parity asserted for MoE routes.
- Any accepted change ships with a unit test or bench asserting the new route is selected for the intended band
  (`routes_are_closed` static_asserts already guard band coverage).

## 6. Experiment record template (`docs/experiments/`)

Each experiment is one file recording exactly:

```
Hypothesis:      <what we expect and why>
Baseline:        <command> -> median/p95, correctness status
Changed variable: <the single thing altered>
Candidate:       <command> -> median/p95
Correctness:     <bit-exact assertion or perplexity result>
Decision:        KEEP | REVERT  (+ reason vs noise floor)
Artifacts:       <CSV paths, nsys/ncu profiles>
```

Representative set required (Phase 4): 1 bandwidth-sensitive, 1 compute-intensive, 1 launch-sensitive short
kernel, 1 fused/multi-stage, 1 dispatch path, 1 allocation/memory path, 1 end-to-end workload.

## 7. End-to-end acceptance baseline

Final comparison against `bench/gb10/matrix_table.txt` baseline (pp2048 ≈ 1183 t/s; pp4096+tg128 prefill ≈
1172.83 / decode 44.33; pp2048+tg128 1158.23 / 49.96; tg128 39.23; pp512 940.32; pp4096 1157.80). Acceptance
requires: improvement is GB10-measured, reproducible across ≥2 runs beyond the noise floor, and correctness holds.
