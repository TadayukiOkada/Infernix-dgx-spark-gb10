# `cuda-kernel-toolkit` (`ckt`) Architecture

`cuda-kernel-toolkit` turns benchmark executables, `/shapes/` dispatch files,
NSYS traces, and CUDA kernel sources into provenance-tagged, zero-hand-math
optimization diagnostics and recommendations. It is built for and validated on
NVIDIA GB10 (`sm_121a`, DGX Spark unified memory).

## Design Principles

1. **Zero hand math.** Every quantitative figure needed to tune a kernel
   (occupancy, register step tables, shared-memory step tables, bank-conflict
   degrees, wave quantization on 48 SMs, GGML/Infernix quantized byte footprints,
   roofline bounds across all 21 measured SIMT/Tensor Core pipes, Little's Law
   MLP, and A/B statistics) is computed deterministically by `ckt`. Agents must
   never estimate or calculate numbers mentally.
2. **Every number carries provenance.** Every reported figure is tagged as
   `measured` (from GPU probes, Stage-A timing, or NCU counters), `compiler`
   (from `cuobjdump -res-usage`), `static` (from source or SASS), or `modelled`
   (deterministic formula on measured/compiler inputs).
3. **Machine code over source intent.** SASS instruction mix (`IMMA`, `HMMA`,
   `QMMA`, `OMMA`, `LDG.E.128`, `LDS.128`, `LDGSTS`, `UTMALDG`, `SHFL`, `MUFU`,
   `DEPBAR`, `DFMA`) and register/spill counts come from per-symbol `cuobjdump`
   inspection — unfiltered whole-binary dumps are forbidden.
4. **Correctness gates performance.** No candidate parameter or schedule is
   accepted unless verification passes.
5. **Stdlib-only Python core.** All analysis, SQLite parsing, CSV/JSON/Markdown/HTML
   reporting, and config loading work out-of-the-box on Python 3.11+ without
   external package dependencies.

## Module Map

| Module | Responsibility |
| --- | --- |
| `environment.py` + `native/*.cu` | Compiles and runs `device_probe.cu`, `bandwidth_probe.cu`, and `compute_probe.cu` (with `-gencode=arch=compute_121a,code=sm_121a`), measuring sustained DRAM bandwidth (`242.7 GB/s`), shared-memory bandwidth (`3,987.7 GB/s`), and arithmetic ceilings across all supported pipes (`fp32`, `fp64`, `fp16_simt`, `bf16_simt`, `tf32`, `fp16`, `fp16_f16acc`, `bf16`, `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`, `mxfp8`, `mxfp6`, `mxfp4`, `nvfp4`, `int32`). |
| `kernel_math.py` | Zero-hand-math calculators for LLM agents (`calculate_occupancy_tables`, `calculate_wave_tables`, `calculate_roofline_math`, `calculate_workload_math` for 34 dense + GGML + Infernix quantized formats, `calculate_smem_layout`, `calculate_mlp`, `calculate_ab_stats`). |
| `shapes.py` | Discovers and parses Infernix `src/ops/linear/*/shapes/n<N>_k<K>.{cu,cpp}` files (`ckt shapes list`, `ckt shapes inspect`) and benchmarks shapes (`ckt shapes bench`) using the fastest GB10 cold-L2 method (in-process weight reuse, GB10-sized 32–48 MiB L2 flush, and route-boundary probing), automatically flagging transition cliffs. |
| `ab_runner.py` | First-class A/B testing engine (`ckt ab`): supports interleaved GPU command execution (`B->A`, `A->B`), same-executable env-var A/B, and point-by-point shape benchmark CSV/JSON comparison. |
| `hotspots.py` | Parses NSYS `.sqlite` exports (`ckt hotspots`), ranking kernels by functional block, demangled symbol, and launch geometry while automatically computing GB10 occupancy and wave efficiency for every launch. |
| `binary_inspector.py` & `static_analyzer.py` | Per-function `cuobjdump -res-usage` and `cuobjdump -sass -fun <exact_symbol>` inspection (`ckt static --sass`, `ckt compare`), classifying load/store widths, Tensor Core pipes (`IMMA`, `HMMA`, `QMMA`, `OMMA`), `cp.async`, TMA, shuffles, uniform ops, `MUFU`, `DEPBAR`, and FP64 ops. |
| `performance_model.py` & `recommendations.py` | Occupancy step-function model, auto-dtype roofline model with L2 working-set residency (`l2-resident`, `l2-borderline`, `dram-streaming`), wave quantization, 9-category bottleneck classifier (calibrated for GB10 unified-memory counter semantics), and evidence-linked recommendations. |
| `profiler.py`, `search.py`, `experiments.py` | Staged NCU metric collection (`Stage B` core + `Stage C` targeted focus), correctness-gated launch-parameter search (`ckt search`) and environment sweeps (`ckt sweep`), and persistent experiment tracking (`ckt experiments`). |
| `reporting.py` & `cli.py` | Renders deterministic JSON, Markdown, and standalone HTML reports; exposes the unified `python3 -m ckt` CLI. |
