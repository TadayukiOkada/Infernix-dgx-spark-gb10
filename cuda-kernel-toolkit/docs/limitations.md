# Known limitations and platform caveats

These constraints were discovered while validating the toolkit on the target
hardware and materially affect interpretation of its reports. They are recorded
here so they are not rediscovered, re-derived incorrectly, or silently papered
over.

## GB10 unified memory: `dram__*` counters unavailable

On GB10 (`integrated_uma = true`) the HBM is system memory behind the C2C link,
and NCU does not expose `dram__bytes.sum`-family counters for this topology.
Consequences:

- Hardware-measured DRAM bandwidth cannot be read directly from NCU. The
  toolkit therefore derives effective bandwidth from harness-reported byte
  counts divided by Stage-A wall-clock latency, and uses
  `gpu__compute_memory_throughput.*` plus `lts__t_sector_hit_rate.pct` for the
  on-chip picture. Reports label these as **derived**, not hardware-counted.
- A bus-width × clock formula overstates achievable bandwidth roughly 2.2× on
  this part (measured read ceiling ≈ 246–250 GB/s). All bandwidth claims must
  come from measurement; the theoretical number appears only as context.

## NCU `--kernel-name` matching semantics

NCU matches the **exact undecorated function base name** — not a substring, not
a regex, not a demangled template specialization:

- `--kernel-name k_saxpy_vec4` works; `saxpy`, `k_saxpy`, `saxpy_vec4`, and any
  mangled symbol silently match nothing.
- Template instantiations collapse to their base name: `k_gemm_tiled<(int)16>`
  is listed simply as `k_gemm_tiled`.
- On a filter miss NCU prints `==WARNING== No kernels were profiled.` followed
  by an `Available Kernels:` numbered list. The profiler parses that list and
  retries each candidate before failing, so mis-named kernels self-correct at
  the cost of one extra collection pass.
- This build's `cuobjdump` has no `regex:` function filter either; `-fun` takes
  exact stored symbols, and file-static kernels carry their
  `__nv_static_<hash>_...` prefix. Short names return exit code 0 with zero
  instructions — instruction count must be checked before drawing conclusions.

## NCU must never run on real model weights

Hard policy inherited from the project: NCU replays kernels many times and
buffers large state; it may only instrument synthetic harnesses with shapes,
dtypes, layouts, alignment, and scale metadata matching production workloads.
NSYS remains permitted against real invocations for timeline analysis. The CLI
does not enforce this — the operator chooses what executable to point at.

## `sm__throughput.avg.pct_of_peak_sustained_elapsed` aliases L1TEX

This metric is the MAX over sub-units and was observed to equal
`l1tex__throughput.avg.pct_of_peak_sustained_elapsed` exactly (73.08% for both
on `gemm_tiled`). It must not be interpreted as "compute utilization". The
classifier therefore reads per-pipe active-cycle metrics
(`sm__pipe_{fma,alu,fp64,xu,tensor}_cycles_active.*`,
`l1tex__data_pipe_lsu_wavefronts_mem_shared.*`, `l1tex__lsu_writeback_active.*`)
for arbitration signals.

## FP64 penalty

Measured FP64:FP32 throughput ratio is approximately **67:1** (~0.42 vs ~27.5
TFLOP/s). Any kernel emitting `DFMA`/`FMUL.D` in the inner loop pays this
explicitly; the SASS instruction mix should be inspected for double-precision
ops whenever FP64 is unexpectedly slow.

## Occupancy prediction assumes full co-residency

`predict_occupancy` computes blocks/SM under unlimited grid assumptions. With
small grids (fewer CTAs than SM capacity) measured `warps_active` can be far
below the predicted value (e.g., dependent_fma: predicted 100%, measured 33%).
Reports present both numbers side by side; classification prefers the measured
one when available.

## Unexplained anomaly: `sectors_per_request` for grid-stride vectorized loads

`reduce_block` performs int4-grid-stride global loads yet reports
`global_load_requests.sectors_per_request = 16.0` consistently across sizes,
which contradicts the expected 4 sectors/request for fully-coalesced 128-bit
accesses. Not chased further during validation; flagged here so future runs do
not mistake it for evidence of bad coalescing without cross-checking against
SASS width classification first.

## NCU-replayed timings are not wall-clock

Replay inflates launch overhead and serializes concurrent work. Latency figures
in reports always come from Stage-A baseline timing of the actual process; NCU
contributes utilisation counters, never µs.
