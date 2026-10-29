# Autonomous Agent Playbook: FP8 (`fp8`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul for the **FP8 (`fp8`)** linear family, interpret every benchmark/compiler/profiler signal, and choose the exact next action for each outcome.

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat kernel headers (`*_gemv.cuh`, `*_simt.cuh`, `*_sliced_k_mma.cuh`, `*_mma.cuh`, `*_tma_mma.cuh`, `*_a8_*.cuh`, `*_schedule.cuh`, `*_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking 2–3 shape files with pre-existing schedules.
> A complete family optimization on NVIDIA GB10 (`sm_121a`) requires:
> 1. **Static SASS & Register-Spill Audit (`ckt static ... --sass`)**: Identify and fix register spills (`STACK > 0`, `LDL`/`STL`) and SM occupancy bottlenecks in the kernel `.cuh` headers (for example, 9-warp TMA kernels have a 168-register hardware ceiling on GB10 because 3 warps land on 1 of the 4 warp schedulers: $\lfloor 512 / 3 \rfloor_8 = 168$ registers).
> 2. **GB10-Specific Schedule & Kernel Engineering**: Engineer and instantiate missing GB10 schedules in `*_schedule.cuh`, `*_instances.cuh`, and the sweep binary (such as `RowTiles = 2` Sliced-K, 2-blocks/SM `BlockK = 64` TMA schedules fitting inside `<= 50,176` B SMEM, and 2D L2-swizzled `Grouped` / `RowFast` TMA schedules for GB10's 24 MiB L2 cache).
> 3. **100% Shape Coverage Across All Policies (`a16` and `a8`)**: Audit and benchmark **every** shape file in `src/ops/linear/fp8/shapes/` (all 11 shapes, including `n248320_k5120.cu`) across decode ($T=1..8$), mid-band ($T=9..256$), and large prefill ($T=257..8192$).

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `regsPerMultiprocessor = 65536`, 4 warp schedulers/SM = 512 warp-register slots per scheduler).
- **9-Warp TMA Register Ceiling Trap**: A warp-specialized TMA CTA with 1 producer warp + 8 consumer warps = **9 warps (288 threads)** places 3 warps on one SM warp scheduler. Even at 1 block/SM, each thread is hardware-capped at $\lfloor 512 / 3 \rfloor_8 = \mathbf{168\text{ registers/thread}}$ (not 255!). Any `PingPong` fragment pipeline that uses $128\text{ accumulator regs} + 64\text{ fragment regs}$ will spill to local memory (`STACK > 0`, `STL`/`LDL` in inner loop) unless it streams B fragments on one plane when `kMmaTokens * kMmaRows >= 32`.
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory; never use the 256 MiB RTX 5090 default). At large $T \ge 1536$, `TokenFast` CTA rasterization streams 30–80 MiB of activations per weight row tile and thrashes GB10's 24 MiB L2 cache; use 2D L2-swizzled `Fp8MmaRaster::Grouped` (`RasterGroupRows = 8`) or `Fp8MmaRaster::RowFast` so TMA stays at 88–97 TFLOP/s instead of falling back to slow non-TMA `Fp8A16MmaSchedule` (~67–71 TFLOP/s).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **FP8 E4M3 Tensor Core Ceiling**: ~394 TFLOP/s (`policy=a8`), BF16 Tensor Core Ceiling: ~197 TFLOP/s (`policy=a16`).
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`50,176` B max for **2 blocks/SM**, `33,152` B max for **3 blocks/SM**).

### FP8 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/fp8/fp8_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_schedule.cuh): Defines `Fp8A16GemvSchedule`, `Fp8A16SimtSchedule`, `Fp8A16SlicedKMmaSchedule`, `Fp8A16MmaSchedule`, `Fp8A16TmaMmaSchedule` (including `Raster` and `RasterGroupRows`), `Fp8A8MmaSchedule`, `Fp8A8TmaMmaSchedule`, `Fp8A8TmaSplitKSchedule`, and `Fp8ScheduleInstance`.
  - [`src/ops/linear/fp8/fp8_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_gemv.cuh): $T=1$ GEMV (`ValuesPerLane = 8, 16, 32`).
  - [`src/ops/linear/fp8/fp8_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_simt.cuh): Small-$T$ SIMT (`TokenPacked` vs `SharedPhase`). Ensure `TokenTile` matches the `Simt` dispatch capacity (never leave `TokenTile = 4` when `Capacity = 8`, which launches 2 token tiles and reads all weights twice at $T=5..8$!).
  - [`src/ops/linear/fp8/fp8_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_sliced_k_mma.cuh): Inner-K split MMA (`KWarps = 2, 4, 8, 16`, `RowTiles = 1` or `2` $\to$ `BlockRows = 16` or `32`).
  - [`src/ops/linear/fp8/fp8_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_mma.cuh) & [`src/ops/linear/fp8/fp8_a16_tma_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a16_tma_mma.cuh): Full-K BF16 MMA and warp-specialized TMA MMA.
  - [`src/ops/linear/fp8/fp8_a8_mma_common.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_mma_common.cuh), [`src/ops/linear/fp8/fp8_a8_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_mma.cuh), [`src/ops/linear/fp8/fp8_a8_tma_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_tma_mma.cuh), & [`src/ops/linear/fp8/fp8_a8_pair_tma_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_a8_pair_tma_mma.cuh): Native FP8 (`E4M3 x E4M3`) Tensor Core MMA (`Terms = 1` or `2` residual profile).
  - [`src/ops/linear/fp8/fp8_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/fp8/fp8_instances.cuh).
- **Sweep & Test Binaries**:
  - Sweep binary: `build-native/bench/infernix_fp8_flash_next_sweep` ([`bench/ops/fp8_flash_next_sweep.cu`](file:///opt/etc/Infernix/bench/ops/fp8_flash_next_sweep.cu)).
  - Oracle tests: `build-native/tests/infernix_linear_fp8_a16_test` and `build-native/tests/infernix_linear_fp8_a8_test` (plus `ctest --test-dir build-native --output-on-failure -R fp8`).
- **Shape Dispatch Directory**: `src/ops/linear/fp8/shapes/` (11 `.cu` files).

---

## Step 1: Audit Autotuner Performance, Numerical Contracts & FP8 Coverage

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

$PY -m ckt shapes list --qtype fp8
```

1. **Read the Exact Oracle & Column-Invariance Contracts First ([`tests/ops/linear/test_fp8_a16.cpp`](file:///opt/etc/Infernix/tests/ops/linear/test_fp8_a16.cpp) and [`tests/ops/linear/test_fp8_a8.cpp`](file:///opt/etc/Infernix/tests/ops/linear/test_fp8_a8.cpp))**:
   - **Critical Pitfall to Avoid**: Do NOT assume that a `Sliced-K` band at $T \ge 9$ must preserve an old incumbent's `KWarps = 2` or `KWarps = 4` reduction-tree hash!
   - Inspect `fp8_a16_column_invariance()` in `test_fp8_a16.cpp`: default-route column invariance is required ONLY on the 5 Qwen3.8-Flash-Next shapes for their specified token windows:
     - `16384x2560`, `13312x2560`, `2560x6144`, `1280x2560` for $T \in [1..8]$ (`Gemv` at $T=1$ and `Simt` at $T=2..8$ must share identical `ValuesPerLane` and `AccumulatorChains`).
     - `2560x640` for $T \in [1..48]$.
     - `248320x5120` via its dedicated `launch_invariant` entry point (`A16ColumnInvariant` policy).
   - At $T \ge 9$ on all shapes (and at $T \ge 2$ on the 5 Qwen3.5 shapes `14336x5120`, `16384x5120`, `34816x5120`, `5120x6144`, `5120x17408`), you are completely free to switch `KWarps = 2` or `KWarps = 4` Sliced-K bands to faster `KWarps = 8` (`Fp8SlicedInstance<16, 8, 2>`, `Fp8SlicedInstance<32, 8, 2>`, `Fp8SlicedR2Instance<32, 8, 2>`)!
2. **Audit `bench/ops/fp8_flash_next_sweep.cu` and `cuda-kernel-toolkit/ckt/autotune/registry.py`**:
   - Ensure `fp8_flash_next_sweep.cu` defaults to `--flush-mib 32` and `--dram-gbps 246.0` for GB10, and includes `Fp8SlicedR2Instance` (`RowTiles = 2`), 2-blocks/SM `BlockK = 64` TMA schedules (`Fp8A16TmaT32R128S4B2K64`, `Fp8A16TmaT64R128S3B2K64`, `Fp8A16TmaT128R128S2B2K64`), and L2-swizzled `RowFast` / `Grouped` TMA schedules (`Fp8A16TmaT64R256RF`, `Fp8A16TmaT128R128RF`, `Fp8A16TmaT128R128G8`, `Fp8A16TmaT64R256G8`).

---

## Step 2: Run Autotuner (Stash Only) & Capture Cold-L2 GB10 Baselines (`a16` and `a8`)

Run boundary benchmarks across **all 11 shapes** for both `a16` and `a8` policies:

```bash
for f in src/ops/linear/fp8/shapes/n*.cu; do
  nk=$(basename "$f" .cu | sed -E 's/n([0-9]+)_k([0-9]+)/\1 \2/')
  set -- $nk
  $PY -m ckt shapes bench --qtype fp8 --n $1 --k $2 --policy a16 --mode boundaries --flush-mib 32
done

for nk in "14336 5120" "16384 5120" "34816 5120" "5120 6144" "5120 17408"; do
  set -- $nk
  $PY -m ckt shapes bench --qtype fp8 --n $1 --k $2 --policy a8 --mode boundaries --flush-mib 32
done
```

Also inspect `n248320_k5120.cu` (`lm_head`) at $T=1..64$ and $T=65..512$ (`ckt shapes bench --qtype fp8 --n 248320 --k 5120 --policy a16 --mode sweep --max-tokens 64 --flush-mib 32`).

---

## Step 3: Manual Audit of FP8 Kernels (`GEMV`, `SIMT`, `Sliced-K`, `A16 MMA`, `A16 TMA MMA`, `A8 TMA MMA`)

Inspect compiled kernel resources, register pressure, and SASS spills (note the positional binary argument and `--kernel` flag):

```bash
$PY -m ckt static build-native/bench/infernix_linear_bench --kernel fp8 --sass
```

Check these specific FP8 architectural mechanisms on GB10:

1. **Zero Register Spills in 9-Warp `Fp8A8TmaMmaSchedule` (`fp8_a8_mma_common.cuh`)**:
   - Check `cuobjdump -res-usage` / `ckt static` for every `fp8_a8_tma_mma_kernel` and `fp8_a16_tma_mma_kernel` instance.
   - If any 9-warp (288-thread) TMA kernel has `STACK > 0` (e.g., `128x256` with `kMmaTokens * kMmaRows = 8 * 4 = 32` uses 128 accumulator registers and exceeds the 168-register 9-warp ceiling when `PingPong` allocates 64 fragment registers), modify `fp8_mma_compute_stage` in `fp8_a8_mma_common.cuh` to hold `a_fragments[kMmaTokens][4]` once per `k_step` and stream `b0, b1` per `mma_row` whenever `kMmaTokens * kMmaRows >= 32`. Verify `STACK=0, LOCAL=0`.
2. **2-Blocks/SM `BlockK = 64` TMA Schedules (`Fp8A16TmaT128R128S2B2K64`, `Fp8A16TmaT64R128S3B2K64`)**:
   - Standard `Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1, 128>` uses `98,368` B of shared memory (**1 block/SM**, 116 regs/thread) and leaves half of each SM's 65,536 registers idle.
   - Using `BlockK = 64, WarpsTokens = 2, WarpsRows = 4, Stages = 2, MinBlocksPerSm = 2` (`Fp8A16TmaT128R128S2B2K64`) uses `2 * (128 * 128 + 128 * 64) + 64 = 49,216` B $\le 50,176$ B (**2 blocks/SM = 96 CTAs/wave on GB10**), boosting $T=384..3072$ throughput to **100–105.5 TFLOP/s** (up to **1.35x faster** than 1-block/SM TMA)!
3. **Eliminating Non-TMA `Fp8A16MmaSchedule` Tails via L2-Swizzled TMA (`Fp8MmaRaster::Grouped, 8` and `RowFast`)**:
   - Never fall back to `Fp8A16MmaSchedule<128, 256, 64, ...>` (`MmaR128T256K64`) or `MmaR64T128K64` at large $T$! Non-TMA `Fp8A16MmaSchedule` uses 1024 threads, stages FP8 weights into shared memory, widens them to BF16 in shared memory between two `__syncthreads()` barriers, and caps at ~67–71 TFLOP/s (causing +20% to +47% upward cliffs at $T=1537, 2049, 2305, 4097$).
   - Instead, use `Fp8A16TmaT128R128G8` (`Fp8MmaRaster::Grouped, 8`), `Fp8A16TmaT64R256G8`, or `Fp8A16TmaT64R256RF` at large $T \ge 2048..8192$: grouping 8 row tiles bounds concurrent working set to $8\text{ weight tiles} + \text{active token tiles} \approx 13\text{ MiB} < 24\text{ MiB}$ L2 cache, sustaining **88–97 TFLOP/s** with **0 ULP bit-identity** to all other TMA/MMA schedules!
4. **Small-$T$ `Simt` TokenTile Capacity Match (`n1280_k2560.cu`, `n2560_k6144.cu`)**:
   - Check every shape's `fp8_linear_a16_simt<Geometry, MaxTokens, Simt>` call: if `MaxTokens == 8` and `Simt::kTokenTile == 4`, then $T=5..8$ launches `token_tiles = 2` and reads the entire weight matrix twice! Change `Simt::kTokenTile` to `8` (keeping the same `ValuesPerLane` and `AccumulatorChains` as `Gemv` so column invariance holds with `0 ULP`) and `Fp8CodeCache::Streaming`.

---

## Step 4: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **`STACK > 0` (`spills(ldl/stl) > 0`) in `ckt static ... --sass` on 9-warp TMA kernel** | 9 warps (288 threads) puts 3 warps on one SM warp scheduler, capping registers at 168/thread. `PingPong` fragment pipeline exceeds 168 regs when accumulator regs $\ge 128$. | Stream `b0, b1` per `mma_row` instead of double-buffering `b_fragments[2][kMmaRows][2]` when `kMmaTokens * kMmaRows >= 32`. Verify `STACK=0`. |
| **`[UPWARD CLIFF]` (+20% to +47%) at $T=1537, 2049, 2305,$ or $4097$ when switching from TMA to `MmaR128T256K64` / `MmaR64T128K64`** | Old dispatch switched to slow non-TMA `Fp8A16MmaSchedule` (~68 TFLOP/s) because `TokenFast` TMA thrashed GB10's 24 MiB L2 cache at large $T$. | Replace the `fp8_linear_a16_mma` tail with `Fp8A16TmaT128R128S2B2K64` (up to $T=2304..4096$) and L2-swizzled `Fp8A16TmaT128R128G8` / `Fp8A16TmaT64R256G8` / `Fp8A16TmaT64R256RF` (1.22x–1.47x faster, bit-exact). |
| **Mid-$T$ ($T=384..2304$) TMA stuck at 75–85 TFLOP/s with 1 block/SM (`98 KiB` SMEM)** | 1-block/SM `BlockK=128` TMA leaves half the SM's registers idle and has low wave granularity on 48 SMs. | Switch mid-$T$ TMA to 2-blocks/SM `Fp8A16TmaT128R128S2B2K64` or `Fp8A16TmaT64R128S3B2K64` (`49.2 KiB` SMEM $\to$ 2 blocks/SM = 96 CTAs/wave, 100–105.5 TFLOP/s). |
| **`[INVERTED CLIFF]` or slow $T=9..32$ Sliced-K band using `KWarps=2` or `KWarps=4`** | `KWarps=2` or `4` under-utilizes SM warp slots at small/mid $T$, or Sliced-K extends to $T=96$ where TMA is 1.65x faster. | Replace `KWarps=2`/`4` at $T=9..32$ with `Fp8SlicedInstance<16, 8, 2>`, `Fp8SlicedInstance<32, 8, 2>`, or `Fp8SlicedR2Instance<32, 8, 2>`, and transition to TMA at $T=33$. |
| **`[INVERTED CLIFF]` at $T=8 \to 9$ (`Simt` slower at $T=5..8$ than `Sliced-K` at $T=9$)** | `Simt` uses `TokenTile=4` for a `Capacity=8` band, launching 2 token tiles at $T=5..8$ and reading weights twice. | Widen `Simt` `TokenTile` from `4` to `8` with `Fp8CodeCache::Streaming` (preserving `ValuesPerLane=16, AccumulatorChains=1` column invariance). |

---

## Steps 5, 6 & 7: Build, Qualify Against All FP8 Tests & Oracles (`a16` & `a8`) & Update `ckt`

1. Build benchmark, sweep, and all FP8 test binaries in one invocation:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_fp8_flash_next_sweep infernix_linear_fp8_a16_test infernix_linear_fp8_a8_test infernix_linear_add_fp8_test infernix_linear_swiglu_fp8_test infernix_offloaded_moe_fp8_test
   ctest --test-dir build-native --output-on-failure -R fp8
   ```
2. Re-run cold-L2 boundary benchmarks (`--flush-mib 32`) across **all 11 shapes** for both `a16` and `a8` policies to confirm every cliff is resolved and throughput is monotonically non-decreasing.
3. Verify `git diff` preserves all `uses_a8`, `partial_capacity_bytes`, `launch_invariant`, and `kPairMinTokens` contracts, and run `$PY -m pytest cuda-kernel-toolkit/tests -q` and `git diff --check`.
