# `cuda-kernel-toolkit` (`ckt`) — Step-by-Step User & LLM Agent Guide

All commands run from `/opt/etc/Infernix` using the repository's Python 3.11+
virtualenv (`/opt/etc/Infernix/.venv-convert/bin/python3`):

```bash
PYTHONPATH=/opt/etc/Infernix/cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt <subcommand> [options]
```

> **Golden Rule for LLM Agents:**
> 1. **Never perform mental or hand math** for occupancy, register thresholds,
>    shared-memory allocation, bank-conflict padding, wave quantization,
>    quantized weight bytes, roofline ceilings, Little's Law MLP, or A/B
>    statistics. Always invoke `ckt math`, `ckt static`, `ckt shapes`, or
>    `ckt ab`.
> 2. **Never write ad-hoc Python or shell scripts** to parse NSYS `.sqlite`
>    files, run `cuobjdump`, sweep parameters, benchmark `/shapes/`, or compute
>    A/B statistics. Use the built-in `ckt` subcommands below.

---

## Step 1: Hardware & Precision Ceiling Discovery (`ckt env` & `ckt doctor`)

Before analyzing or tuning any kernel, query the cached hardware & precision
snapshot (or rebuild/refresh it if running on a new GPU):

```bash
# Read cached environment snapshot (.ckt_cache/env.json)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt env

# Force native probe rebuild & re-measurement of bandwidth + all SIMT/Tensor Core pipes
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt env --refresh --rebuild --tools

# Verify NCU, NSYS, cuobjdump, nvcc, and metric catalog readiness
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt doctor
```

`ckt env` measures and caches:
- Device topology (`48 SMs`, `1536 max threads/SM = 48 warps/SM`, `65,536 regs/SM`, `102,400 B smem/SM`, `1,024 B reserved smem/block`, `24 MiB L2 cache`).
- Sustained global memory bandwidth (`242.7 GB/s` on GB10 LPDDR5X unified memory) and shared memory bandwidth (`3,987.7 GB/s`).
- Measured arithmetic ceilings across every supported precision (`fp32`, `fp64`, `fp16_simt`, `bf16_simt`, `tf32`, `fp16`, `fp16_f32acc`, `fp16_f16acc`, `bf16`, `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`, `mxfp8`, `fp8`, `mxfp6`, `fp6`, `mxfp4`, `nvfp4`, `fp4`, `int32`).

---

## Step 2: Whole-Graph / Decode Hotspot Attribution (`ckt hotspots` & `ckt nsys`)

### Unlocking Hardware Performance Counters & Kernel Call Stacks
To profile full CPU host activity, PMU hardware counters, and resolve kernel symbol backtraces without sudo permissions, configure:
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
```
Inspect current permissions at any time via `ckt env` or `ckt nsys`.

### Generating or Running the Optimal NSYS Profile Command (`ckt nsys`)
```bash
# Check recommended flags based on current Linux permissions
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt nsys \
    ./build-native/bench/infernix_bench --weights model.infernix

# Or execute profiling directly with full PMU & DWARF backtraces:
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt nsys \
    --full --run -o profiles/nsys/trace \
    ./build-native/bench/infernix_bench --weights model.infernix
```

### Analyzing the NSYS SQLite Trace (`ckt hotspots`)
When starting from an end-to-end model or benchmark trace (`nsys profile -c cudaProfilerApi --cuda-graph-trace=node --stats=true -o trace ...`), rank kernels and functional blocks directly from the exported `.sqlite` file:

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt hotspots \
    profiles/nsys/iq3s_prefill.sqlite --top 15 --reps 1
```

What `ckt hotspots` computes automatically:
- **Timeline & Host Starvation**: Computes wall span, total GPU busy time, GPU duty cycle %, and **host dispatch idle bubbles** (time the GPU was stalled waiting for the CPU host).
- **Host CUDA Runtime Overhead**: Top host-side CUDA runtime API calls (`cudaLaunchKernel`, `cudaMalloc`, `cudaStreamSynchronize`, `cuLibraryLoadData`).
- **Synchronization Breakdown**: Distinguishes Stream sync vs Event sync vs Context sync wait latencies.
- **Host CPU Hotspots & Kernel Symbols**: Ranks CPU-side function samples from PMU hardware counters and flags masked kernel symbols.
- **Functional Block Budget**: Groups kernels by functional family (`moe_expert`, `dense_ggml`, `attention`, `hyper_conn`, `delta_net`, `bf16_gemm`, `norm_route`).
- **Exact Launch Geometry**: Runs `occupancy_model()` and `wave_analysis()` on every launch geometry showing `active_blocks_per_sm`, `occupancy_pct`, `limiting_factor`, `waves`, `tail_efficiency_pct`, and compiler stack spills next to its `% share of GPU time`.

---

## Step 3: Static SASS & Resource Inspection (`ckt static` & `ckt compare`)

Never run unfiltered `cuobjdump -sass` on a large Infernix binary. Use `ckt static` (which queries `cuobjdump -res-usage` in ~1.1 s and disassembles only the requested symbol via `cuobjdump -sass -fun <exact_symbol>` in ~1.5 s):

```bash
# Inspect register/smem/spill usage, GB10 occupancy, and SASS instruction mix for matching kernels
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static \
    --exe ./build-native/tests/infernix_ggml_linear_test \
    --kernel mma_rows_kernel --threads 256 --sass
```

What `ckt static --sass` reports:
- Compiler `REG`, `STACK` (spills), `SHARED`, and `LOCAL` bytes.
- Exact GB10 occupancy (`blocks/SM`, `warps/SM`, `occupancy %`, `limiting_factor`, and `max_regs_to_gain_one_block`).
- SASS instruction mix: global/shared load & store width histograms (`4 B`, `8 B`, `16 B`), `IMMA` (INT8 Tensor Core), `HMMA` (FP16/BF16/TF32 Tensor Core), `QMMA` (FP8/FP6 Tensor Core), `OMMA` (NVFP4/MXFP4 Tensor Core), `LDGSTS` (`cp.async`), `UTMALDG` (TMA), `SHFL`, `MUFU`, `DEPBAR`, and `FP64` instructions.

### Comparing Two Symbols or Binaries (`ckt compare`)

After editing a kernel or comparing two template schedules, verify the exact machine-code and register delta before or alongside benchmarking:

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt compare \
    --exe-a ./build-native/tests/infernix_ggml_linear_test \
    --symbol-a "mma_rows_kernel<false, (infernix::QType)23" \
    --symbol-b "rows_kernel<(infernix::QType)23" \
    --threads 256
```

---

## Step 4: Zero-Hand-Math Analytical Calculators (`ckt math`)

Never estimate tuning numbers by hand. Run `ckt math <topic>`:

### 4.1 Occupancy, Register Step Table & Block-Size Sweep (`ckt math occupancy`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math occupancy \
    --threads 256 --regs 128 --smem 38912
```
Returns:
- Current active blocks/SM, warps/SM, occupancy %, limiting factor (`registers`, `shared memory`, or `threads/warps/block-count`).
- `free_reg_headroom_per_thread` (how many registers the compiler can add before dropping a block) and `max_regs_per_thread_to_gain_one_block` (exact target for `__launch_bounds__`).
- Complete **register step table** (`1..N blocks/SM` -> exact max `regs/thread` and `smem/block` allowed on GB10, including the `1,024 B` CUDA-reserved shared memory per block).
- Complete **block-size sweep** (`32..1024 threads/block`).

### 4.2 Grid Wave Quantization (`ckt math waves`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math waves \
    --ctas 256 --threads 256 --regs 128 --smem 38912 --items 4096
```
Returns full waves, tail blocks, tail efficiency %, max theoretical speedup from eliminating the partial tail wave, and a table of wave-aligned grid sizes (`1x..8x` waves on 48 SMs) with `items_per_cta`.

### 4.3 Shared-Memory Bank Conflict & Padding Calculator (`ckt math smem`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math smem \
    --rows 128 --cols 64 --elem-bytes 2 --stages 2 --threads 256 --regs 96
```
Returns exact per-stage and total shared-memory bytes (with GB10's `1,024 B` reserved block header and `128 B` rounding), column-stride bank conflict degree (`gcd(stride_words, 32)`), and optimal `+elems` row padding for conflict-free `4 B` scalar and `16 B` (`float4`/`ldmatrix`) accesses.

### 4.4 Quantized & Dense Workload Byte / FLOP / Roofline Calculator (`ckt math workload` & `ckt math roofline`)
Supports 34 formats including Infernix native (`q4`, `q5`, `q6`, `q8`, `fp8_e4m3`, `nvfp4`, `mxfp4`, `bf16`, `fp16`, `fp32`) and all GGML block-quantized formats (`q4_0`..`q8_1`, `q2_k`..`q8_k`, `iq1_s`..`iq4_xs`):

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math workload \
    --m 4 --n 4096 --k 14336 --weight-dtype iq3_s --act-dtype q8_1 \
    --out-dtype fp32 --compute-dtype int8 --latency-us 492
```
Returns exact weight/activation/output bytes, traffic saved by `M>1` weight reuse, L2 cache residency vs GB10's 24 MiB L2 (`l2-resident`, `l2-borderline`, `dram-streaming`), effective GB/s, effective TFLOP/s, % of GB10 memory and compute ceilings, and headroom in microseconds.

### 4.5 Little's Law Memory-Level Parallelism (`ckt math mlp`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math mlp \
    --warps-per-sm 8 --load-width-bytes 16
```
Computes bytes in flight required per SM to saturate `242.7 GB/s` at ~250 ns memory latency and how many independent loads per thread (`#pragma unroll`) are required at `4 B`, `8 B`, and `16 B` load widths.

---

## Step 5: Benchmarking & Retuning Infernix `/shapes/` (`ckt shapes`)

Infernix stores per-shape token dispatch ladders under `src/ops/linear/{q4,q5,q6,q8,fp8,nvfp4,bf16}/shapes/n<N>_k<K>.{cu,cpp}`. Many shape files were originally fitted on RTX 5090 (`170 SMs`, `96 MiB L2`) and exhibit sharp latency cliffs or wave quantization underfill on GB10 (`48 SMs`, `24 MiB L2`).

### What is the fastest method to benchmark `/shapes/` on GB10?
`ckt shapes bench` implements the fastest cold-L2 benchmark method on GB10 by combining three techniques:
1. **In-process grouped execution via `infernix_linear_bench`**: allocates and quantizes the synthetic weight matrix **once per `(qtype, N, K)` shape** in memory instead of spawning a fresh process per token count `T`.
2. **GB10-sized L2 eviction (`--flush-mib 32` or `48` instead of `256`)**: `infernix_linear_bench` defaults to a `256 MiB` L2 flush buffer (sized for RTX 5090's 96 MiB L2). On GB10 (`24 MiB` L2, `242.7 GB/s` LPDDR5X), flushing `256 MiB` before every single repeat wastes `~1.05 ms/repeat` (`~21 ms` per `T` point). `ckt shapes bench` automatically sizes `--flush-mib` to GB10's 24 MiB L2 (`32 MiB` by default, or `48 MiB` with `--full-flush`), cutting benchmark wall-clock time by **5x–8x** while still completely evicting the 24 MiB L2 cache!
3. **Boundary-aware token probing (`--mode boundaries`)**: instead of blindly sweeping all `T=1..512`, `ckt shapes bench` parses `src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cu,cpp}` to extract the exact `if (tokens <= B)` cutoff points and benchmarks `T=1` plus `[B, B+1]` at every route transition.

### Step-by-Step `/shapes/` Workflow

```bash
# 1. List all /shapes/ files (or filter by --qtype q4|q5|q6|q8|fp8|nvfp4|bf16)
#    Shows weight MB, GB10 L2 residency regime, T=1 memory floor (us), band count, and tuning origin (RTX 5090 vs GB10)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype q4

# 2. Inspect a shape's C++ dispatch ladder before running GPU benchmarks
#    Shows every token band, parsed tile (Rows x Tokens), CTA count on 48 SMs, wave efficiency %, and roofline floor (us)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes inspect \
    --qtype q4 --n 4096 --k 5120

# 3. Benchmark the shape on GB10 at its route transition boundaries (fastest mode, ~3-10 seconds)
#    Maps each T row back to its exact C++ line in /shapes/, reports GB/s & TFLOP/s, and flags route-transition cliffs!
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --policy a16 --mode boundaries \
    --csv-out cuda-kernel-toolkit/results/q4_n4096_k5120_base.csv \
    --output cuda-kernel-toolkit/results/q4_n4096_k5120_base.json

# 4. Optional: Run a continuous token sweep across a specific T band (e.g. T=1..64)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --mode sweep --sweep 1:64:1
```

### Cross-Family Cliff Report (`ckt cliffs`)
Scans all 23 Infernix kernel families (`linear:*`, `linear_add`, `linear_swiglu`, `attn_input_proj`, `gdn_input_proj`, `linear_pair`, `ggml_linear`, `causal_softmax_attention`, `gated_delta_net`, `sparse_moe`, `hyper_connection`, `qsa_attention`, `rmsnorm_rope`, `offloaded_sparse_moe`, `candidate_selector`, `dynamic_grouped_conv`) with cold-L2 (`32 MiB` on GB10) boundary probing and emits one consolidated JSON and Markdown report:

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt cliffs \
    --families all --max-tokens 512 --warmup 2 --repeat 8 \
    --output cuda-kernel-toolkit/results/cliffs_report.json \
    --markdown-out cuda-kernel-toolkit/results/cliffs_report.md

# Or incrementally re-scan selected families and merge into an existing JSON report:
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt cliffs \
    --families linear:q8,qsa_attention --merge-report cuda-kernel-toolkit/results/cliffs_report.json \
    --output cuda-kernel-toolkit/results/cliffs_report.json --markdown-out cuda-kernel-toolkit/results/cliffs_report.md
```

---

## Step 6: A/B Testing Inside `cuda-kernel-toolkit` (`ckt ab`)

Every kernel or `/shapes/` change must be validated with `ckt ab`. `ckt ab` supports three built-in modes and computes all statistical metrics (median, mean, stdev, CV%, IQR, min/max, speedup ratio, SNR vs noise floor, geomean, and `PASS / REGRESSION / WITHIN_NOISE` verdict) without hand math:

### Mode A: Point-by-Point Shape CSV / JSON Comparison (`--csv-a` vs `--csv-b`)
Use this when tuning a `/shapes/n<N>_k<K>.{cu,cpp}` file or a linear kernel across multiple token counts `T`:

```bash
# 1. Capture baseline CSV before editing
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --csv-out cuda-kernel-toolkit/results/base.csv

# 2. Edit src/ops/linear/q4/shapes/n4096_k5120.cu and rebuild infernix_linear_bench:
cmake --build build-native --parallel "$(nproc)" --target infernix_linear_bench

# 3. Capture candidate CSV after editing
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --csv-out cuda-kernel-toolkit/results/cand.csv

# 4. Run point-by-point A/B comparison across all T points
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --csv-a cuda-kernel-toolkit/results/base.csv \
    --csv-b cuda-kernel-toolkit/results/cand.csv \
    --output cuda-kernel-toolkit/results/ab_report.json
```

### Mode B: Interleaved GPU Command / Binary A/B (`--cmd-a` vs `--cmd-b`)
Executes alternating rounds (`B->A`, `A->B`, `B->A`, `A->B`...) on the GPU so thermal and LPDDR5X clock drift on unified-memory GB10 cancel out:

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --cmd-a "./build-native/bench/infernix_linear_bench --qtype q4 --n 4096 --k 5120 --t 128 --flush-mib 32" \
    --cmd-b "./build-native/bench/infernix_linear_bench --qtype q4 --n 4096 --k 5120 --t 129 --flush-mib 32" \
    --label-a "sliced_r32_t32_w2_s2" \
    --label-b "mma_r32_t32_k64" \
    --rounds 6 --warmup-rounds 1 \
    --output cuda-kernel-toolkit/results/ab_interleaved.json
```

### Mode C: Same-Executable Environment Toggle A/B (`--exe` + `--env-a` / `--env-b`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --exe "./build-native/tests/infernix_ggml_linear_test --ckt --workload mma_rows_kernel --n 4096" \
    --env-a INFERNIX_CKT_COLUMNS=1 \
    --env-b INFERNIX_CKT_COLUMNS=4 \
    --rounds 6
```

---

## Step 7: Deep NCU Counter Profiling, Parameter Sweeps & Full Reports (`ckt profile`, `ckt sweep`, `ckt search`, `ckt analyze`)

> **Important (`AGENTS.md` rule):** Never run `ncu` (`ckt profile` or `ckt analyze` with NCU enabled) on a process that loads real model weights. Always run NCU on synthetic test/benchmark harnesses (`infernix_*_test --ckt` or `infernix_linear_bench --profile`).

```bash
# 1. Correctness-gated environment/parameter sweep on an Infernix --ckt test harness
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt sweep \
    --exe ./build-native/tests/infernix_ggml_linear_test --ckt \
    --workload mma_rows_kernel --env INFERNIX_CKT_COLUMNS=1,2,4,8 \
    --output cuda-kernel-toolkit/results/sweep_ggml_mma.json

# 2. Staged NCU profiling on a synthetic harness
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt profile \
    --exe ./build-native/tests/infernix_ggml_linear_test \
    --workload mma_rows_kernel --problem-size 4096 --stage B

# 3. Full end-to-end analysis pipeline (timing + verification + SASS + occupancy + NCU + roofline + recommendations)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt analyze \
    --config cuda-kernel-toolkit/configs/infernix_ggml_linear.yaml \
    --output cuda-kernel-toolkit/results/infernix-ggml-linear
```

---

## Step 8: Offline CUDA Autotuning & Dispatch Generation (`ckt autotune`)

The `ckt autotune` suite provides quantization-family agnostic discovery, candidate compilation & legality verification, byte-identical test contract gating, cold-L2 benchmark sweeps with adaptive crossover refinement, noise-aware selection, and deterministic C++ dispatch code generation.

On GB10 (`sm_121a`), specify `--flush-mib 32` (1.33x the 24 MiB L2 cache) to guarantee complete cold-cache eviction while accelerating sweeps by 5x–8x. For the step-by-step maintainer runbook and verified GB10 results, see [docs/maintainer/autotune-workflow.md](../../docs/maintainer/autotune-workflow.md).

### 1. Repository Discovery (`ckt autotune discover`)
Discovers supported quantization families (`bf16`, `fp8`, `nvfp4`, `q4`, `q4_q5`, `q5`, `q6`, `q8`), registered geometries (`N x K`), kernel families (`gemv`, `simt`, `sliced_k`, `mma`), coupled/disabled routes, and test associations without manual configuration:
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune discover
```

### 2. Standalone Benchmark Runner (`ckt autotune benchmark`)
Benchmarks candidates across token counts with cold-L2 buffer flushing (GB10-sized 32 MiB flush):
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune benchmark \
    --family nvfp4 -n 128 -k 2560 -t 1,2,4,8,16,32,64,128 --flush-mib 32
```

### 3. End-to-End Autotuning Campaign (`ckt autotune tune`)
Runs the full pipeline:
- Legal candidate instantiation and compile checks
- Byte-identity correctness gate against existing test suites (`infernix_column_invariant_projections_test`)
- Workload sweep across token counts
- Adaptive crossover refinement near transition boundaries
- Noise-aware winner selection with minimum improvement margin (`--min-improvement-pct 2.0`)
- Deterministic C++ dispatch ladder generation with provenance header
- Post-generation policy validation gate
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune tune \
    --family nvfp4 -n 128 -k 2560 \
    --results results/autotune_nvfp4_n128_k2560.json \
    --generate results/n128_k2560_tuned.cu \
    --report results/autotune_nvfp4_n128_k2560.md
```

### 4. Deterministic Dispatch Generation (`ckt autotune generate`)
Compresses winner intervals from an existing database and emits production-ready C++ dispatch code:
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune generate \
    --results results/autotune_nvfp4_n128_k2560.json \
    --family nvfp4 -n 128 -k 2560 \
    -o results/n128_k2560_tuned.cu
```

### 5. Policy Validation Gate (`ckt autotune validate-policy`)
Validates a generated dispatch file against:
- C++/CUDA compilation syntax and template completeness
- Existing test suites (asserting bit-exact identity and zero ULP drift)
- Performance regression check against measured winner latency
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune validate-policy \
    --policy results/n128_k2560_tuned.cu \
    --family nvfp4 -n 128 -k 2560 \
    --results results/autotune_nvfp4_n128_k2560.json
```

### 6. Tuning Report Generation (`ckt autotune report`)
Generates comprehensive Markdown reports with hardware profile, discovery inventory, interval ladders, winner tables, and custom kernel roofline diagnostics:
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt autotune report \
    --results results/autotune_nvfp4_n128_k2560.json \
    -o results/autotune_nvfp4_n128_k2560.md
```

---

## Step 9: In-Process Kernel Tuner Qualification (`ckt kt`)

Sweeps the *template space of a production kernel* with [Kernel Tuner](https://kerneltuner.github.io/kernel_tuner/)
(NVRTC + driver API, no product rebuild) and verifies every candidate against a float64 decode of the stored NVFP4
operands. Requires `kernel_tuner` in `.venv-convert`.

```bash
PYTHONPATH=cuda-kernel-toolkit .venv-convert/bin/python3 -m ckt kt list
PYTHONPATH=cuda-kernel-toolkit .venv-convert/bin/python3 -m ckt kt doctor     # shim still mirrors src/?
PYTHONPATH=cuda-kernel-toolkit .venv-convert/bin/python3 -m ckt kt tune nvfp4_a16_mma \
    --n 2560 --k 2560 --tokens 1024 --baseline 64,128,64,64,16,2,2 --md profiles/kt/out.md
# decode: nvfp4_a16_sliced_k --baseline KW,TT,ROWTILES,STAGES,MINB ; --wide adds cache hints/staging modes;
# --strategy genetic_algorithm --max-fevals 300 for budgeted search; --compiler nvcc for product SASS.
```

What to know before trusting a row:
- **Timing protocol.** KT's default loop (sync, record, launch) adds ~7 us of host launch latency to a ~20 us kernel.
  The bridge queues the cold-L2 flush *ahead of* the start event instead (the `bench::measure_cold_launch` order);
  with it the harness reproduced `infernix_linear_bench` (20.27 vs 20.29 us, [2048,2560] T=24).
- **Always pass `--baseline`**, and keep the GPU otherwise idle; re-time any claimed win with a repeat run.
- **Not modelled:** PDL overlap inside the decode graph, eager vs graph launch gaps, the A4/TMA routes.
- NVRTC compiles PTX that the driver JIT-assembles (`--compiler nvcc` gives the product's SASS; register counts differ,
  timing matched in the one case checked). `ckt/kt/shim/` replaces only host-only headers; `ckt kt doctor` guards drift.
- A sweep ranks schedules; it does not change the product. Moving a winner into `shapes/*.cu` needs the normal
  `ckt shapes bench` / test gates, and a reproducible-route bit check where the reduction order could change.
