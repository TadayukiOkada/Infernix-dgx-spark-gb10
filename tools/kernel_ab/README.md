# Kernel A/B tooling

Low-overhead harness for the kernel optimization loop in
[`docs/performance/gb10-kernel-optimization-results.md`](../../docs/performance/gb10-kernel-optimization-results.md):
profile, change one kernel, gate on the exact oracles, then measure the pair end to end.

All scripts run from the repository root and default to the local Qwen3.8-Flash-Next
artifact. Override the inputs with `INFERNIX_WEIGHTS`, `INFERNIX_NGRAM`, and
`INFERNIX_CORPUS`.

## Default decode workload

Decode measurements use MTP speculative decoding with the optimized proposal head, the
n-gram table resident in RAM, and `k8v4` KV cache:

```
--spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype k8v4 --ngram-residency ram
```

`--draft-tokens` and `--lm-head-draft` require an explicit `--spec`; without it the bench
rejects them. Three draft tokens is the measured optimum on this host (see the sweep
below). Plain decode (`--kv-dtype int8`, no `--spec`) is a different workload and gives
materially different results — record which one a number came from.

## Scripts

| Script | What it does |
|---|---|
| `decode_ab.sh BASE CAND [ROUNDS] [DRAFT]` | Paired interleaved A/B of two `infernix_bench` builds; reports `pp4096` and `tg128` medians, paired delta, win count, and speculation acceptance. |
| `draft_sweep.sh BINARY [COUNTS] [ROUNDS]` | Sweeps `--draft-tokens` and reports `pp4096`/`tg128` medians and acceptance per count. |
| `profile_decode.sh [decode\|prefill] [BINARY] [SAMPLES]` | Captures an NSYS trace with `--cuda-graph-trace=node` and prints the ranked kernel table. |
| `rank_kernels.py TRACE.sqlite` | Ranks kernels from an NSYS SQLite export, splits phases at large gaps, and reports per-round GPU busy fraction and per-call medians. |

```bash
# Rank decode kernels
./tools/kernel_ab/profile_decode.sh decode build-native/bench/infernix_bench 16 3

# Rank prefill kernels
./tools/kernel_ab/profile_decode.sh prefill build-native/bench/infernix_bench 4096

# Compare two builds, four paired rounds
./tools/kernel_ab/decode_ab.sh /tmp/ex/base /tmp/ex/cand 4 3

# Find the draft-token optimum
./tools/kernel_ab/draft_sweep.sh build-native/bench/infernix_bench "2 3 4" 3

# Re-rank an existing trace
python3 tools/kernel_ab/rank_kernels.py profiles/nsys/kernel_ab_decode.sqlite
```

## Two measurement rules this tooling exists to enforce

**Interleave both orders.** On this host the binary that runs *first* in a round measures
about 0.2 t/s higher regardless of which binary it is, while an `A/A` null control
(`base` vs `base`) has a ~0.6 % spread. A single ordering can therefore invent or hide a
0.3 % change. `decode_ab.sh` alternates the order every round; never accept or reject a
candidate from one order.

**Trace CUDA graphs as nodes.** CUDA-graph decode rounds are invisible to a plain
`nsys profile --trace=cuda`: the graph collapses into one activity and the report shows
2–12 % GPU-busy with a ~27 ms "gap" per token, which is a trace artifact rather than
engine behavior. `profile_decode.sh` always passes `--cuda-graph-trace=node`. Separately,
whole-capture busy fractions are dominated by the one-time weight upload (tens of GB of
`CUPTI_ACTIVITY_KIND_MEMCPY`), so `rank_kernels.py` splits the timeline into phases
before reporting occupancy.

## Measured reference values (GB10, Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S)

Draft-token sweep on the default decode config, three rounds each:

| `--draft-tokens` | `tg128` median | spec acceptance | rounds |
|---|---|---|---|
| 2 | 53.36 t/s | 0.541 | 122 |
| **3** | **58.84 t/s** | 0.531 | 98 |
| 4 | 52.81 t/s | 0.400 | 98 |

Plain decode (`int8` KV, no `--spec`) runs about 36.7–36.9 t/s on the same artifact.
