# cuda-kernel-toolkit (`ckt`)

A stdlib-only Python toolkit for automated CUDA kernel profiling, zero-hand-math
analytical calculations, `/shapes/` dispatch benchmarking, bit-exactness
verification, and interleaved A/B validation on NVIDIA GB10 (`sm_121a`, DGX Spark
unified memory).

Designed for both **automated LLM agents** and **human GPU engineers** so that:
1. **Agents never guess or perform mental/hand math.** `ckt math` and `ckt shapes`
   compute exact register step tables, full `32..1024` thread-range block sweeps,
   shared-memory step tables, bank-conflict degrees, wave quantization on 48 SMs,
   GGML/Infernix quantized byte footprints, roofline bounds across all 21 hardware
   pipes (`fp32`, `fp64`, `fp16_simt`, `bf16_simt`, `tf32`, `fp16`, `bf16`, `int8`,
   `int4`, `fp8`, `mxfp8`, `mxfp6`, `mxfp4`, `nvfp4`), Little's Law MLP requirements,
   and outlier-resistant A/B statistical verdicts.
2. **Agents never write ad-hoc profiling, parsing, or A/B scripts.** `ckt`
   provides built-in subcommands for NSYS SQLite hotspot ranking (`ckt hotspots`),
   per-symbol SASS + resource inspection with automatic C++ template thread-count
   inference (`ckt static` / `ckt compare --threads auto`), Infernix `/shapes/`
   discovery and fast cold-L2 benchmarking (`ckt shapes`), bit-exactness verification
   (`--require-bit-exact`), and interleaved GPU A/B testing (`ckt ab`).

See [`docs/usage.md`](docs/usage.md) for the CLI reference,
[`docs/architecture.md`](docs/architecture.md) for module design, and
[`docs/limitations.md`](docs/limitations.md) for GB10 unified-memory profiling
notes.

---

## 1. Measured NVIDIA GB10 (`sm_121a`) Hardware Baseline

Running `python3 -m ckt env --refresh --rebuild` compiles and executes native CUDA
probes (`device_probe`, `bandwidth_probe`, `compute_probe`) on the active GPU and
records every supported SIMT and Tensor Core ceiling in `.ckt_cache/env.json`:

| Property / Pipe | Measured / Queried on GB10 (`sm_121a`) | Notes |
| --- | --- | --- |
| **Compute Capability / SMs** | `12.1` (`sm_121a`), **48 SMs** | Max **1,536 threads/SM** (**48 warps/SM**, not 64!), **32 blocks/SM** |
| **Register File / SM** | **65,536** 32-bit regs | 256-reg warp allocation unit (**step of 8 regs/thread**), max 255/thread |
| **Shared Memory / SM** | **102,400 B** (100 KiB) | 49,152 B default/block, 101,376 B opt-in, **1,024 B reserved/block**, 128 B alloc unit |
| **L2 Cache / Memory** | **24 MiB** (`25,165,824 B`), Unified LPDDR5X | **242.7 GB/s** measured sustained DRAM read; **3,987.7 GB/s** measured SMEM |
| **FP32 / FP64 SIMT FMA** | **27.76 TFLOP/s** FP32 / **0.411 TFLOP/s** FP64 | **67.6:1 FP64 penalty** — accidental `double` literals destroy performance |
| **FP16 / BF16 SIMT (`__hfma2`)** | **30.17 TFLOP/s** | Scalar/vector SIMT half/bfloat16 pipe |
| **TF32 Tensor Core (`HMMA`)** | **61.00 TFLOP/s** | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` |
| **FP16 / BF16 Tensor Core (`HMMA`)** | **123.76 TFLOP/s** | Full rate with **both FP32 and FP16 accumulation** (1:1 parity on GB10) |
| **INT8 Tensor Core (`IMMA`)** | **247.61 TOP/s** | `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (used by GGML tiles) |
| **FP8 / MXFP8 / MXFP6 (`QMMA`)** | **244.30 TFLOP/s** | `e4m3` / `e5m2` / `e3m2` (`m16n8k32`, full rate with FP32 accumulate) |
| **MXFP4 / NVFP4 (`OMMA`)** | **488.40 – 494.56 TFLOP/s** | `kind::mxf4nvf4.block_scale` `m16n8k64` `e2m1` (native hardware pipe) |
| **Legacy INT4 (`s4` `IMMA`)** | **34.95 TOP/s** | **7.1x slower than INT8 and 14.1x slower than NVFP4** — avoid `s4` on GB10! |

---

## 2. GB10 Tunable Parameter Ranges (`min`, `max`, `step`, & Sweet Spots)

Every kernel tuning parameter on NVIDIA GB10 (`sm_121a`) is bounded by hardware
resource quantization. Never guess parameter limits — inspect them directly with
`ckt math occupancy --threads all --regs <R> --smem <S>`.

### 2.1 Complete GB10 Parameter Range Reference Table

| Parameter | Hardware `min` | Hardware `max` | Allocation `step` / Granularity | GB10 Sweet-Spot Variants & Constraints |
| --- | --- | --- | --- | --- |
| **`threads` (Threads / CTA)** | `1` (`32` for warp-aligned kernels) | `1,024` | `32` (`1` warp) | **100% Occupancy Divisors of 1,536**: `64, 128, 192, 256, 384, 512, 768`. Non-divisors (`96, 160, 224, 320, 448, 640, 896, 1024`) leave partial warp slots idle (`66.7%..93.8%` max occupancy). At `threads=1024`, only `1` block/SM can ever fit (`66.7%` max occupancy) and **`REG > 64` fails to launch (`0` blocks/SM)**! |
| **`warps_per_block` (Warps / CTA)** | `1` (`32` threads) | `32` (`1,024` threads) | `1` warp (`32` threads) | Infernix linear/MMA kernels use `2, 4, 6, 8, 10, 12, 14, 16` warps/CTA (`64..512` threads). |
| **`warps_per_sm` (Active Warps / SM)** | `1` | `48` (`1,536` threads/SM) | `1` warp | Note: GB10 has **48 warps/SM**, not 64 (Hopper/Ada). Occupancy `% = active_warps_per_sm / 48`. |
| **`blocks_per_sm` (Active CTAs / SM)** | `1` | `32` | `1` CTA | Capped by `min(32, floor(48 / warps_per_block), floor(65536 / regs_per_block), floor(102400 / smem_per_block_alloc))`. |
| **`regs_per_thread` (`REG`)** | `1` | `255` | **`8` regs/thread** (`256` regs/warp allocation unit) | **Zero-spill sweet spots for 100% occupancy (`48` warps/SM)**: `<= 40` regs (`40` warps = `83.3%`), `<= 32` regs (`48` warps = `100%`). At `24` warps/SM (`50%` occupancy): `<= 80` regs/thread. At `16` warps/SM (`33.3%` occupancy): `<= 128` regs/thread. |
| **`smem_bytes` (Dynamic + Static SMEM / CTA)** | `0 B` | `49,152 B` (default) / `101,376 B` (opt-in) | **`128 B`** + **`1,024 B` CUDA driver reserved per CTA** | Total SM capacity is `102,400 B` (`100 KiB`). Effective per-CTA allocation is `ceil(smem_bytes / 128) * 128 + 1024`. Max SMEM for `N` blocks/SM: `1 blk = 101,376 B`, `2 blk = 50,176 B`, `3 blk = 33,024 B`, `4 blk = 24,576 B`, `6 blk = 16,000 B`, `8 blk = 11,776 B`. |
| **`MinBlocksPerSm` (`__launch_bounds__`)** | `1` | `32` | `1` CTA | **Critical Rule**: Never set `MinBlocksPerSm > floor(102400 / (aligned_smem + 1024))`! Over-constraining `MinBlocksPerSm` forces `ptxas` to spill registers to local memory (`STACK > 0`) for an occupancy target that shared memory physically prevents. |
| **`stages` (`cp.async` Pipeline Stages)** | `1` (synchronous / double-buffered) | `4` (practical limit on 100 KiB SMEM) | `1` stage | `stages = 1..2` for small tiles (`T <= 64`) to preserve multi-CTA residency (`3..6` blk/SM); `stages = 2..3` for compute-bound tiles (`64x96`, `64x128`). |
| **`k_slices` (Split-K / Sliced-K Warps)** | `1` (serial K loop) | `8` | Power of 2 (`1, 2, 4, 8`) | `k_slices = 8` for `T=1` GEMV (`256` threads); `k_slices = 4` (`128` threads) or `2` (`64` threads) for `T=2..128` sliced-K MMA; `k_slices = 1` for full-tile MMA (`T >= 129`). **Changing `k_slices` alters FP32 reduction order (`0..1 ULP`)!** |
| **`vector_bytes` (Global/Shared Load Width)** | `1 B` | `16 B` (`uint4` / `float4` / `LDG.E.128` / `LDS.128`) | Power of 2 (`1, 2, 4, 8, 16 B`) | Always target `16 B` (`128-bit`) per thread with `128 B` base alignment (`32 threads * 4 B = 128 B` cache line). |
| **`flush_mib` (L2 Cache Flush Size)** | `0 MiB` (warm L2) | `256 MiB` (RTX 5090 default) | `1 MiB` | On GB10 (`24 MiB` L2), **`32 MiB` (`1.35x` L2, default in `ckt shapes bench`)** or **`48 MiB` (`2.0x` L2, `--full-flush`)** achieves cold-L2 timing 5x–8x faster than `256 MiB`. |

### 2.2 Why We Previously Passed `--threads 256` and How `ckt` Supports the Full `32..1024` Range

In `cuobjdump -res-usage`, the ELF resource table reports `REG`, `STACK`, `SHARED`, and
`LOCAL` per kernel symbol, **not** `blockDim.x` (unless encoded in the C++ template
arguments). Previously, when auditing `Q4A16SlicedKMmaSchedule<16, 24, 8, ...>` (where
`KWarps = 8` -> `8 * 32 = 256` threads), we passed `--threads 256`.

However, Infernix kernels span the full range of thread counts (`64, 128, 192, 256, 320, 384, 448, 512`):
- **Sliced-K MMA (`Q4A16SlicedKMmaSchedule<BlockRows, BlockTokens, KWarps, ...>`)**:
  - `KWarps = 2` -> **`64` threads** (`2` warps, e.g. `sliced_r32_t32_w2_s2`)
  - `KWarps = 4` -> **`128` threads** (`4` warps, e.g. `sliced_r32_t16_w4_s2`)
  - `KWarps = 8` -> **`256` threads** (`8` warps, e.g. `sliced_k5120_t24`)
- **Full-Tile MMA (`Q4A16MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, ...>`)**:
  - `threads = (BlockRows / WarpRows) * (BlockTokens / WarpTokens) * 32`:
  - `r64_t128` (`WarpRows=32, WarpTokens=64`) -> **`128` threads** (`4` warps)
  - `r64_t72` (`WarpRows=32, WarpTokens=24`) -> **`192` threads** (`6` warps)
  - `r64_t64` / `r64_t80` -> **`256` threads** (`8` warps)
  - `r64_t120` -> **`320` threads** (`10` warps)
  - `r64_t48` / `r64_t96` -> **`384` threads** (`12` warps)
  - `r64_t112` -> **`448` threads** (`14` warps)

**How `ckt` handles all thread counts automatically**:
1. **`ckt static` and `ckt compare` default to `--threads auto`**:
   `ckt` parses the demangled C++ template signature of every symbol (`*SlicedKMmaSchedule`,
   `*MmaSchedule`, `*GemvSchedule`, `*SimtSchedule`, etc.) and automatically computes each
   symbol's exact compiled `threads_per_block` (`64..512`) and `MinBlocksPerSm`! You can
   also pass an explicit `--threads <32..1024>` override at any time.
2. **`ckt math occupancy --threads all --regs <R> --smem <S>`**:
   Sweeps all 16 warp-multiple block sizes across GB10's entire `32..1024` hardware range
   (`32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024`) and
   prints `warps/CTA`, `max_blocks/SM`, `active_warps/SM`, `occupancy_%`, and `limiter`
   for every block size in a single table:

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math occupancy \
    --threads all --regs 72 --smem 29952
```

---

## 3. Bit-Exactness (`0 ULP`) vs. Numerical Oracle Qualification

Per `AGENTS.md`:
> *The exact route (`--disable-fast-paths`) stays bit-exact from release to release and,
> where the artifact's formats allow, request-invariant; a change that alters it for speed
> ships as a fast path, qualified under the numerical acceptance policy.*

### How `ckt` Tracks and Enforces Bit-Exactness

Why do two valid CUDA linear kernels for the same `(QType, N, K, T)` produce either
**100% bit-identical (`0 ULP`)** outputs or **`0..1 ULP`** differences?
Because IEEE-754 floating-point addition `(a + b) + c != a + (b + c)` is non-associative:
- Within a single warp's Tensor Core MMA instruction (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`),
  the `K=16` inner dot product and sequential `K`-tile accumulation per `k_slice` are
  deterministic for a given `k_slices` count (`K / k_slices` elements accumulated serially
  per slice).
- **Same `reduction_signature` = 100% Bit-Exact (`0 ULP`)**:
  When two routes share the same `compute_policy` (`a16` vs `q8_1`), same `mma_family`
  (`mma`, `sliced_mma`, `gemv`), same **`k_slices`** (`1, 2, 4, 8`), and same inter-slice
  `reduction_order` (`linear`, `pairwise`, `serial`), every output element `(n, t)`
  undergoes the **exact same sequence of FP32 instructions** regardless of tile dimensions
  `BlockRows x BlockTokens`, pipeline stages `s1/s2/s3`, or `__launch_bounds__`!
  - Example (`BIT_EXACT`): Changing `launch_q4_a16_sliced_r16_t8_w4_s2` (`k_slices=4`) ->
    `launch_q4_a16_sliced_r32_t8_w4_s2` (`k_slices=4`), or changing
    `launch_q4_a16_mma_r32_t32_k64_...` (`k_slices=1`) -> `launch_q4_a16_mma_r64_t96` (`k_slices=1`),
    or clamping `kMinBlocksPerSm` in `Q4A16SlicedKMmaSchedule` is **100% bit-exact (`0 ULP`)**.
- **Changed `reduction_signature` = Fast-Path Candidate (`REDUCTION_CHANGED`)**:
  Changing `k_slices` (for example, replacing `sliced_r32_t32_w4_s1` (`k_slices=4`) or
  `sliced_r32_t32_w2_s2` (`k_slices=2`) with `mma_r64_t64` (`k_slices=1`) at `T=65..128`)
  changes the final cross-slice FP32 sum from a 4-way/2-way slice reduction to a 1-way
  serial accumulation. Even though both pass the FP32/FP64 mathematical oracle in `ctest`
  within `< 1e-5`, they are not bit-identical (`0..1 ULP` BF16 rounding difference) and
  must only be enabled behind a `LinearRoutes` fast-path check (`routes.allow_*`) rather
  than unconditionally mutating the default `LinearRoutes::Exact` dispatch table.

**Built-in `ckt` Bit-Exactness Commands (Enabled by Default)**:
1. **`ckt shapes inspect --qtype <q> --n <N> --k <K>`**:
   Displays `thr` (`threads_per_block`), `k_sl` (`k_slices`), and `reduction_sig`
   (e.g., `a16:sliced_mma:k_slices=4:linear`, `a16:mma:k_slices=1:serial`) for every line
   of a `/shapes/` file before you edit it.
2. **`ckt ab --csv-a <before.json> --csv-b <after.json>` (`--require-bit-exact` is `True` by default)**:
   Compares the baseline and candidate `reduction_signature` at every token count `T`,
   prints a `bit_exact` column (`BIT_EXACT` vs `REDUCTION_CHANGED(...)`), and **by default**
   returns verdict `REJECT_BIT_EXACTNESS_VIOLATION` (exit code `1`) if any token point
   changed its reduction tree. To evaluate a fast-path candidate that intentionally changes
   `k_slices`, pass `--allow-reduction-change` (or `--no-require-bit-exact`).

---

## 4. Quick Start Command Cheat Sheet (Exact Session Commands)

```bash
# 1. Hardware & 21-pipe precision ceilings (cached in .ckt_cache/env.json)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt env

# 2. Zero-hand-math calculators (--threads=auto is default and prints the full 32..1024 block-size sweep!)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math occupancy --regs 72 --smem 29952
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math waves --ctas 256 --threads 128 --regs 72 --smem 21760 --items 4096
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math smem --rows 128 --cols 64 --elem-bytes 2 --stages 2 --threads 256 --regs 96
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math workload --m 4 --n 4096 --k 14336 --weight-dtype iq3_s --act-dtype q8_1 --compute-dtype int8 --latency-us 492
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math mlp --warps-per-sm 24 --load-width-bytes 16 --mem-latency-ns 250

# 3. Per-symbol static SASS & resource inspection (--threads=auto is default and infers 64..512 threads from C++ templates!)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static \
    build-native/bench/infernix_linear_bench --kernel q4_a16_sliced_k_mma_kernel --spills-only
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt compare \
    --exe-a /tmp/infernix_linear_bench_base \
    --exe-b build-native/bench/infernix_linear_bench \
    --kernel "Q4A16SlicedKMmaScheduleILi16ELi24ELi8ELi1ELNS0_5CacheE1ELS4_0ELi6ELi5120ELi24E" --sass

# 4. Inspect & benchmark Infernix /shapes/ with bit-exact reduction signatures & 32 MiB cold-L2 flush
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype q4
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes inspect --qtype q4 --n 4096 --k 5120
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --execution graph --mode boundaries \
    --csv-out cuda-kernel-toolkit/results/q4_n4096_k5120_before.csv \
    --output cuda-kernel-toolkit/results/q4_n4096_k5120_before.json

# 5. Interleaved A/B testing & bit-exact /shapes/ comparison (--require-bit-exact is default!)
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --csv-a cuda-kernel-toolkit/results/q4_n4096_k5120_before.json \
    --csv-b cuda-kernel-toolkit/results/q4_n4096_k5120_after_bitexact.json \
    --output cuda-kernel-toolkit/results/q4_n4096_k5120_ab_bitexact.json
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --cmd-a "/tmp/infernix_linear_bench_base --qtype q4 --policy a16 --n 6144 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --cmd-b "./build-native/bench/infernix_linear_bench --qtype q4 --policy a16 --n 6144 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --rounds 8 --warmup-rounds 2
```

---

## 5. Step-by-Step Playbooks for All Common Optimization Tasks

### Task 1: Family `/shapes/` Dispatch Table & Schedule Optimization (Real Case Study: Infernix `Q4` Family on GB10)

**When to use**: Optimizing `src/ops/linear/<qtype>/shapes/n<N>_k<K>.{cu,cpp}` and family
schedule headers (`src/ops/linear/<qtype>/*_schedule.cuh`) for GB10 (`sm_121a`).

#### Step 1.1: Audit `/shapes/` Routes, Thread Counts & Bit-Exact Reduction Signatures (`ckt shapes inspect`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes list --qtype q4
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes inspect --qtype q4 --n 4096 --k 5120
```
- **What to check**: Look at `thr` (`64..384`), `k_sl` (`k_slices = 8, 4, 2, 1`), `CTAs(min..max)`,
  `wave_eff%` (how cleanly `CTAs` divide GB10's `48` SMs), and `reduction_sig`.
- **Bit-exactness rule**: To keep the default `LinearRoutes::Exact` schedule 100% bit-exact
  (`0 ULP`), only swap schedules at a given token `T` that have the **exact same
  `reduction_sig`** (same `k_slices` and reduction order).

#### Step 1.2: Audit Family Template Instantiations Across All Thread Counts (`ckt static --spills-only`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static \
    build-native/bench/infernix_linear_bench --kernel q4_a16_sliced_k_mma_kernel --spills-only
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math occupancy \
    --regs 40 --smem 29952
```
- Notice `--threads=auto` is the default: `ckt static` automatically infers `64` threads
  for `KWarps=2`, `128` threads for `KWarps=4`, and `256` threads for `KWarps=8`, plus
  `MinBlocksPerSm` from the 7th template parameter!
- **Discovery on `Q4`**: `ckt static` flagged **34 spilling instantiations** (`32` with
  `[OVER-CONSTRAINED __launch_bounds__?]`) where `Q4A16SlicedKMmaSchedule` requested
  `MinBlocksPerSm = 6` (`256` threads) even though `SHARED = 21,760..38,144 B` physically
  caps GB10's `102,400 B` SM shared memory at `2..4` blocks/SM.

#### Step 1.3: Benchmark Baseline Route Boundaries (`ckt shapes bench`)
```bash
cp build-native/bench/infernix_linear_bench /tmp/infernix_linear_bench_base
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --execution graph --mode boundaries \
    --csv-out cuda-kernel-toolkit/results/q4_n4096_k5120_before.csv \
    --output  cuda-kernel-toolkit/results/q4_n4096_k5120_before.json
```
- **How to interpret `ckt shapes bench` output**:
  1. **`[UPWARD CLIFF]` (`step_% >= +15%` at `T = B -> B+1`)**: Moving from `T=128` to `T=129`
     jumped **`+76.2%` (`149.28 us -> 263.01 us`)** when switching to
     `launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2`.
  2. **`[INVERTED CLIFF / DOMINATED BAND]` (`step_% <= -5%` at `T = B -> B+1`)**: Moving from
     `T=320` (`450.27 us`) to `T=321` (`263.90 us`, `launch_q4_a16_mma_r64_t128`) **dropped
     latency by `-41.4%`**, proving that the `T=129..320` route is strictly dominated on GB10!
  3. **Low `wave_%` at small `T`**: At `T=2..8`, `launch_q4_a16_sliced_r16_t8_w4_s2` launched
     `256` CTAs (`61.25 us` at `T=8`), slower than `T=16` (`57.02 us`, `128` CTAs).

#### Step 1.4: Fix `__launch_bounds__` & Verify via `ckt compare` and `ckt ab`
In `src/ops/linear/q4/q4_schedule.cuh`, we clamped `kMinBlocksPerSm` to
`kMaxBlocksBySmem = 102400 / (((kSharedBytes + 127) / 128 * 128) + 1024)`, rebuilt, and ran:
```bash
cmake --build build-native --parallel "$(nproc)" --target infernix_linear_bench
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static \
    build-native/bench/infernix_linear_bench --kernel q4_a16_sliced_k_mma_kernel --spills-only
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt compare \
    --exe-a /tmp/infernix_linear_bench_base \
    --exe-b build-native/bench/infernix_linear_bench \
    --kernel "Q4A16SlicedKMmaScheduleILi16ELi24ELi8ELi1ELNS0_5CacheE1ELS4_0ELi6ELi5120ELi24E" --sass
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --cmd-a "/tmp/infernix_linear_bench_base --qtype q4 --policy a16 --n 6144 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --cmd-b "./build-native/bench/infernix_linear_bench --qtype q4 --policy a16 --n 6144 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --rounds 8 --warmup-rounds 2
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --cmd-a "/tmp/infernix_linear_bench_base --qtype q4 --policy a16 --n 7168 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --cmd-b "./build-native/bench/infernix_linear_bench --qtype q4 --policy a16 --n 7168 --k 5120 --t 24 --execution graph --warmup 3 --repeat 12 --flush-mib 32" \
    --rounds 6 --warmup-rounds 2
```
- Eliminated spills in **30 of 34 instantiations** (`STACK: 16..96 B -> 0 B`,
  `LDL/STL: -100%`, total SASS instructions `-15.47%` to `-19.51%`).
- Delivered **`+10.46%` (`1.1046x`, `99.22 -> 89.82 us`, `SNR = 4.34`)** on
  `Q4 N=6144 K=5120 T=24` and **`+8.76%` (`1.0876x`, `115.44 -> 106.14 us`, `SNR = 4.45`)**
  on `Q4 N=7168 K=5120 T=24` with **100% bit-exactness (`0 ULP`)**.

#### Step 1.5: Compare 100% Bit-Exact Schedule vs. Fast-Path Schedule (`ckt ab` with default `--require-bit-exact`)
```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt shapes bench \
    --qtype q4 --n 4096 --k 5120 --execution graph \
    --tokens 1,2,4,5,8,9,16,17,64,65,96,97,128,129,288,289,320,321 \
    --csv-out cuda-kernel-toolkit/results/q4_n4096_k5120_after_bitexact.csv \
    --output cuda-kernel-toolkit/results/q4_n4096_k5120_after_bitexact.json
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt ab \
    --csv-a cuda-kernel-toolkit/results/q4_n4096_k5120_before.json \
    --csv-b cuda-kernel-toolkit/results/q4_n4096_k5120_after_bitexact.json \
    --output cuda-kernel-toolkit/results/q4_n4096_k5120_ab_bitexact.json
```
In `src/ops/linear/q4/shapes/n4096_k5120.cu`, we tested two candidate schedules against
baseline (`q4_n4096_k5120_before.json`):
1. **100% Bit-Exact Schedule (`14/14 BIT_EXACT`, committed in `n4096_k5120.cu`)**:
   - Preserves `k_slices` at every single `T`:
     - `T <= 4`: `sliced_r16_t8_capacity4` (`128` threads, `k_slices=4`)
     - `T <= 8`: `sliced_r32_t8_w4_s2` (`128` threads, `k_slices=4`) -> **`-9.40%` (`1.104x`) at `T=8`**
     - `T <= 16`: `sliced_r32_t16_w4_s2` (`128` threads, `k_slices=4`)
     - `T <= 64`: `sliced_r32_t32_w4_s2` (`128` threads, `k_slices=4`)
     - `T <= 96`: `sliced_r32_t32_w4_s1` (`128` threads, `k_slices=4`)
     - `T <= 128`: `sliced_r32_t32_w2_s2` (`64` threads, `k_slices=2`)
     - `T <= 288`: `mma_r64_t96` (`384` threads, `k_slices=1`) -> **`-29.26%` (`1.414x`) at `T=129`**
     - `T > 288`: `mma_r64_t128` (`128` threads, `k_slices=1`) -> **`-40.42%` (`1.678x`) at `T=320`**
   - Passes `ckt ab` (with default `--require-bit-exact`) with **`14/14 BIT_EXACT`** and
     **`3W / 11T / 0L` (`PASS_STRICT_IMPROVEMENT`)**, achieving **`1.0729x` geomean speedup
     (`1.0875x` arith mean)** and reducing total boundary latency sum from
     **`1970.43 us -> 1704.70 us (-13.49%)`**.
2. **Fast-Path Candidate (`T=65..128` routed to `mma_r64_t64_k64_wr32_wt16_s2_a2_b2`)**:
   - Further reduces `T=97..128` latency by **`-16.5% to -17.6%` (`151.26 -> 124.64 us`)**,
     reaching **`1.1090x` geomean (`-16.70%` total latency sum)** when evaluated with
     `ckt ab --allow-reduction-change`.
   - With default `--require-bit-exact`, `ckt ab` flags `T=65, 96, 97, 128` as
     `REDUCTION_CHANGED(a16:sliced_mma:k_slices=4/2:linear->a16:mma:k_slices=1:serial)`
     and returns `REJECT_BIT_EXACTNESS_VIOLATION`, preventing accidental reduction-order
     mutations in an exact route!

---

### Task 2: Compiler Resource, Register Spill & `__launch_bounds__` Repair Across Thread Ranges (`32..1024`)

**When to use**: A kernel has `STACK > 0` (`STL`/`LDL` local memory spills in SASS) or
suboptimal occupancy on GB10 (`sm_121a`).

1. **Scan the binary for spills with automatic template thread-count extraction**:
   ```bash
   ckt static build-native/bench/infernix_linear_bench --kernel <substring> --spills-only
   ```
2. **Check the exact register and shared-memory step table across all block sizes (`32..1024`)**:
   ```bash
   ckt math occupancy --threads all --regs <REG> --smem <SHARED>
   ```
3. **Branch on the `ckt static` & `ckt math` diagnosis**:
   - **Case A (`[OVER-CONSTRAINED __launch_bounds__?]` — `blocks_by_shared < MinBlocksPerSm`)**:
     Lower `MinBlocksPerSm` in `__launch_bounds__(MaxThreads, MinBlocksPerSm)` to match
     `blocks_by_shared = 102400 / (aligned_smem + 1024)`. This gives `ptxas` more registers
     per thread at **zero occupancy cost**.
   - **Case B (Register-limited occupancy cliff — e.g. `REG = 66` when `64` unlocks the next CTA/SM)**:
     Check `ckt math occupancy`'s `register_step_table`: if `REG` is within `1..6` registers
     above a step boundary (`32, 40, 48, 56, 64, 72, 80, 96, 128`), test `__launch_bounds__`
     or narrow live accumulator temporaries, then run `ckt compare` to confirm `STACK == 0`
     before benchmarking on GPU.
   - **Case C (`threads > 512` causing register starvation)**:
     At `threads = 768` (`24` warps/CTA) or `1024` (`32` warps/CTA), a single CTA consumes
     half or two-thirds of the SM's `65,536` registers. Use `ckt math occupancy --threads all`
     to pick a smaller divisor of `1,536` (`128, 192, 256, 384`).

---

### Task 3: Single-Kernel Block-Size (`32..1024`), Occupancy & Unroll/MLP Tuning

**When to use**: Tuning block size (`blockDim`), thread coarsening (`ITEMS_PER_THREAD`),
or loop unroll factor (`#pragma unroll`) for a custom CUDA kernel.

1. **Sweep valid block sizes (`32..1024`) analytically first**:
   ```bash
   ckt math occupancy --threads all --regs 64 --smem 16384
   ```
2. **Calculate Little's Law MLP (Memory-Level Parallelism) requirement**:
   ```bash
   ckt math mlp --latency-cycles 450 --throughput-bytes-per-cycle 104 \
       --bytes-per-inst 16 --warps-per-sm 24
   ```
   - `ckt math mlp` computes exact `in_flight_bytes_required_per_sm` (`46,800 B` on GB10)
     and `loads_per_thread_needed` so you know whether `1`, `2`, or `4` independent `128-bit`
     (`uint4`) loads per thread are needed to saturate GB10's `242.7 GB/s` LPDDR5X bus.
3. **Run a correctness-gated parameter sweep on GPU (`ckt sweep`)**:
   ```bash
   ckt sweep build-native/tests/infernix_ggml_linear_test --args "--ckt" \
       --workload mma_rows_kernel --param INFERNIX_CKT_COLUMNS=1,2,4,8
   ```

---

### Task 4: Shared-Memory Bank-Conflict & Multi-Stage `cp.async` Pipeline Tuning

**When to use**: Designing or debugging a tiled shared-memory layout (`__shared__`) or
choosing `cp.async` pipeline stage count (`stages = 1..4`).

1. **Compute exact bank-conflict degree and optimal padding (`ckt math smem`)**:
   ```bash
   ckt math smem --rows 128 --cols 64 --elem-bytes 2 --stages 2 --threads 256 --regs 96 \
       --access-stride 64
   ```
   - `ckt math smem` reports `conflict_way` (`1-way` conflict-free up to `32-way` serialized),
     recommends exact `pad_elements` (or XOR swizzle when `16 B` `LDS.128` vector alignment
     must be preserved), and shows the `stage_sweep` (`stages = 1..4`) impact on `blocks_per_sm`
     and `occupancy_%`.
2. **Check tile arithmetic intensity & register pressure (`ckt math tile`)**:
   ```bash
   ckt math tile --tile-m 64 --tile-n 96 --tile-k 64 --stages 2 --threads 384 \
       --a-bytes 2 --b-bytes 1
   ```

---

### Task 5: End-to-End NSYS SQLite Trace Hotspot & Amdahl Triage

**When to use**: Starting from a model-level NSYS `.sqlite` trace to decide which kernel
and launch geometry `(grid, block)` matters most for end-to-end tok/s.

1. **Configure or run optimal NSYS profiling with unlocked PMU/CPU counters (`ckt nsys`)**:
   ```bash
   ckt nsys --full --run -o profiles/nsys/trace ./build-native/bench/infernix_bench --weights ...
   ```
2. **Rank top kernels, host runtime overhead, sync stalls, and CPU hotspots (`ckt hotspots`)**:
   ```bash
   ckt hotspots profiles/nsys/trace.sqlite --binary build-native/infernix --top 15 --reps 10
   ```
3. **Project end-to-end Amdahl impact before and after tuning (`ckt math ab`)**:
   ```bash
   ckt math ab --a 99.2,99.1,99.3,99.2 --b 89.8,89.7,89.9,89.8 --e2e-weight 0.35
   ```
   - `ckt math ab` outputs `projected_e2e_speedup_x` and `projected_e2e_pct_gain` for the
     kernel's share of total decode/prefill time.

---

### Task 6: Staged Synthetic NCU Bottleneck Diagnosis & Roofline Attribution

**When to use**: Attributing whether a single kernel is Memory-Bound, L2/SMEM-Bound,
Tensor-Core/ALU-Bound, or Latency/Barrier-Bound on GB10.

> **Hard Policy (`AGENTS.md`)**: Never run `ncu` (`ckt profile` / `ckt analyze`) on a
> process that loads real model weights. Always use synthetic harnesses (`infernix_*_test --ckt`
> or `infernix_linear_bench`).

```bash
ckt profile build-native/tests/infernix_ggml_linear_test --args "--ckt" \
    --kernel mma_rows_kernel --sass --ncu
```
- `ckt profile` combines unprofiled CUDA-event timing (`us`), exact analytical workload
  bandwidth (`GB/s` vs GB10's `242.7 GB/s` ceiling, since `dram__*` hardware counters are
  unsupported on GB10 UMA), per-symbol SASS (`ckt static`), and NCU warp-stall counters
  using GB10's **40–70% utilization dead-zone rules** (`docs/limitations.md`).

---

### Task 7: Bit-Exactness (`0 ULP`) & Independent Numerical Oracle Qualification

**When to use**: Qualifying any kernel or `/shapes/` modification before completion.

1. **Verify route bit-exactness (`0 ULP` reduction invariance) for `/shapes/` changes**:
   ```bash
   ckt ab --csv-a cuda-kernel-toolkit/results/before.json \
          --csv-b cuda-kernel-toolkit/results/after.json --require-bit-exact
   ```
   - Must report `Bit-exact points: N/N` (`0` reduction changes) for `LinearRoutes::Exact`.
2. **Run the independent FP32/FP64 mathematical oracle in `ctest`**:
   ```bash
   ctest --test-dir build-native --output-on-failure -R '^infernix_linear(_add)?_q4_a16_test$'
   ```
3. **Run `compute-sanitizer` whenever pointers, shared-memory indexing, or barriers changed**:
   ```bash
   compute-sanitizer --tool memcheck ./build-native/tests/infernix_linear_q4_a16_test
   ```

---

### Task 8: Offline CUDA Autotuning & Dispatch Generation (`ckt autotune`)

**When to use**: Automatically discovering quantization families and schedule candidates,
running cold-L2 benchmark sweeps with adaptive crossover boundary refinement, enforcing
byte-identical test contracts, and generating deterministic C++ dispatch ladder code.

1. **Discover supported quantization families, operations, and test suites**:
   ```bash
   ckt autotune discover
   ```
2. **Benchmark candidates on a specific workload with right-sized cold-L2 flush**:
   ```bash
   ckt autotune benchmark --family nvfp4 -n 128 -k 2560 -t 1,8,16,32,64,128 --flush-mib 36
   ```
3. **Run full autotuning campaign with adaptive boundary refinement & dispatch generation**:
   ```bash
   ckt autotune tune --family nvfp4 -n 128 -k 2560 \
       --results results/autotune_nvfp4_n128_k2560.json \
       --generate results/n128_k2560_tuned.cu \
       --report results/autotune_nvfp4_n128_k2560.md
   ```
4. **Validate generated dispatch policy (compilation, byte identity, regression gate)**:
   ```bash
   ckt autotune validate-policy --policy results/n128_k2560_tuned.cu \
       --family nvfp4 -n 128 -k 2560 \
       --results results/autotune_nvfp4_n128_k2560.json
   ```
5. **Render human-readable Markdown tuning report from database**:
   ```bash
   ckt autotune report --results results/autotune_nvfp4_n128_k2560.json -o report.md
   ```

---

## 6. Unit Tests

```bash
PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m unittest discover -s cuda-kernel-toolkit/tests -v
```
