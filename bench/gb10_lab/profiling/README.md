# Profiling toolkit (works on any CUDA app)

Ready-to-run wrappers around **Nsight Compute** (`ncu`, per-kernel roofline) and
**Nsight Systems** (`nsys`, CPU/GPU timeline + launch gaps), each paired with a
parser that turns profiler output into a compact table plus machine-readable JSONL.

## Files

| File | Purpose |
|---|---|
| `ncu_profile.sh` | Run `ncu` over any app; emits `*.csv` + browsable `*.ncu-rep`. Metric presets: `roofline`, `tensor`, `bw`, `launch`, `full`. |
| `parse_ncu.py` | Parse the `ncu --csv` dump → per-kernel SOL/Tensor/occupancy table + JSONL, with an automatic roofline class. |
| `nsys_profile.sh` | Run `nsys profile`; emits `*.nsys-rep` + `*.sqlite` + built-in stats CSVs. |
| `parse_nsys.py` | Parse the sqlite export → GPU busy vs idle-gap %, launch count, per-kernel totals, host API time. |

## Quick start

```bash
# Roofline on one kernel family:
./ncu_profile.sh -k 'regex:k_rmsnorm' -c 1 -- ./build/gb10_bench --op rmsnorm
python3 parse_ncu.py ncu_profile.csv --jsonl ncu.jsonl

# Timeline / launch-gap analysis on a multi-kernel op:
./nsys_profile.sh -- ./build/gb10_bench --op layer_pipeline
python3 parse_nsys.py nsys_profile.sqlite --jsonl nsys.jsonl
```

## Which tool answers which question

- **"Is this kernel near peak?"** → `ncu_profile.sh` + `parse_ncu.py`. Read `SM%`,
  `MEM%`, `TENSOR%`, `waves/SM`. A `TENSOR-UNDERUSED` class means cuBLAS/CUTLASS
  picked a legacy tile instead of a Blackwell-native one.
- **"Why is power/utilization low across a run?"** → `nsys_profile.sh` +
  `parse_nsys.py`. High `gap_pct` with tiny per-kernel durations = the GPU is
  idle between launches (launch/serialization bound), which is exactly the
  microbenchmark low-wattage signature.

## GB10 / unified-memory caveats (already handled by the presets)

- `dram__*` counters **do not exist** on GB10 — bandwidth is anchored by
  `lts__throughput` / `l1tex__throughput` SOL % and `*_t_bytes.sum`, not DRAM
  counters. The `bw` preset uses only valid metrics.
- `sm__pipe_tensor_*` **is** available and is the ground truth for Tensor-Core
  saturation.
- The target app's own stdout interleaves with `ncu --csv`; `parse_ncu.py` locates
  the header row and discards non-matching lines, so it stays robust.

## Useful knobs

- `METRICS="..." ./ncu_profile.sh -- app` — override the metric list entirely.
- `-p full` — add warp-latency and stall-reason metrics (long_scoreboard = waiting
  on global loads; barrier = `__syncthreads` contention).
- `NSYS_EXTRA="--backtrace=full"` — richer host-side attribution.
- `nsys_profile.sh -t cuda,nvtx,osrt -- app` — correlate NVTX ranges you add in code.
