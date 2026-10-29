# Autonomous Agent Playbook: BF16 (`bf16`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul for the **BF16 (`bf16`)** linear family, interpret every benchmark/compiler/profiler signal, and choose the exact next action for each outcome.

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat kernel headers (`bf16_a16_gemv.cuh`, `bf16_a16_simt.cuh`, `bf16_a16_sliced_k_mma.cuh`, `bf16_a16_mma.cuh`, `bf16_a16_tma_mma.cuh`, `bf16_schedule.cuh`, `bf16_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking only the subset of shapes hardcoded in `bf16_flash_next_sweep.cu`.
> A complete family optimization on NVIDIA GB10 (`sm_121a`) requires:
> 1. **Static SASS & Register-Spill Audit (`$PY -m ckt static build-native/bench/infernix_linear_bench --kernel bf16 --sass`)**: Check every BF16 kernel instance for `STACK > 0` local-memory spills (especially 9-warp TMA kernels, which have a 168-register hardware ceiling on GB10 because 3 warps land on 1 of the 4 warp schedulers: $\lfloor 512 / 3 \rfloor_8 = 168$ registers).
> 2. **GB10-Specific Schedule & Kernel Engineering**: Engineer and instantiate GB10-optimal schedules in `bf16_schedule.cuh`, `bf16_instances.cuh`, and `bf16_flash_next_sweep.cu` (such as 2-blocks/SM `<= 50,176` B SMEM schedules, `Simt` schedules with `TokenBatch == Capacity` so $T=5..8$ never reads weights twice, and L2-swizzled `Grouped` / `RowFast` TMA schedules for GB10's 24 MiB L2 cache at large $T \ge 1536$).
> 3. **Correct Interpretation of Numerical & Column-Invariance Contracts**: Read `tests/ops/linear/test_bf16_a16.cpp` first so you know which shapes/token ranges require column invariance (`0 ULP` across $T$) and where `KWarps=8` Sliced-K or TMA schedules can be freely substituted.
> 4. **100% Shape Coverage**: Benchmark and optimize **all 25 shape files** in `src/ops/linear/bf16/shapes/` (including LLM, GDN, and vision shapes `n1152_k1152.cu`, `n1152_k4304.cu`, `n4304_k1152.cu`).

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `regsPerMultiprocessor = 65536`).
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory for cold-L2 measurements; note that larger BF16 weights at `2.0` bytes/elem exceed 24 MiB even at moderate $[N, K]$).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **BF16 Tensor Core Ceiling**: ~197 TFLOP/s (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`).
- **Ridge Point (`TFLOP/s / GB/s`)**: $\approx 197,000 / 246 \approx 800\text{ FLOP/byte}$. Because each BF16 weight is 2 bytes, a BF16 linear layer is **memory-bandwidth bound up to $T \approx 800$ tokens** on GB10 unless weights fit in the 24 MiB L2 cache!
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`50,176` B max for **2 blocks/SM**).

### BF16 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/bf16/bf16_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_schedule.cuh): Defines `Bf16A16GemvSchedule`, `Bf16A16SimtSchedule`, `Bf16A16SlicedKMmaSchedule`, `Bf16A16MmaSchedule`, `Bf16A16TmaMmaSchedule`, `Bf16A16TmaTailMmaSchedule`, and `Bf16ScheduleInstance`.
  - [`src/ops/linear/bf16/bf16_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_a16_gemv.cuh): $T=1$ GEMV (`ValuesPerLane = 4, 8, 16`, `Direct` vs `Shared` activation access, `Sequential` vs `RowSwizzled`).
  - [`src/ops/linear/bf16/bf16_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_a16_simt.cuh): Small-$T$ SIMT (`DirectStream` vs `WarpPacked`, `TokenBatch = 1, 2, 4, 8`).
  - [`src/ops/linear/bf16/bf16_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_a16_sliced_k_mma.cuh) & [`src/ops/linear/bf16/bf16_split_k_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_split_k_launch.cuh): Inner-K / Split-K Tensor Core MMA.
  - [`src/ops/linear/bf16/bf16_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_a16_mma.cuh) & [`src/ops/linear/bf16/bf16_a16_tma_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_a16_tma_mma.cuh): Full-K `cp.async` and warp-specialized TMA (`mbarrier`) Tensor Core MMA (`Swizzle = Plain, Xor64, Tma128`).
  - [`src/ops/linear/bf16/bf16_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/bf16/bf16_instances.cuh).
- **Sweep & Test Binaries**:
  - Sweep binary: `build-native/bench/infernix_bf16_flash_next_sweep` ([`bench/ops/bf16_flash_next_sweep.cu`](file:///opt/etc/Infernix/bench/ops/bf16_flash_next_sweep.cu)).
  - Oracle test: `build-native/tests/infernix_linear_bf16_a16_test`.
- **Shape Dispatch Directory**: `src/ops/linear/bf16/shapes/` (25 `.cu` files, including LLM/GDN shapes and vision shapes `n1152_k1152.cu`, `n1152_k4304.cu`, `n4304_k1152.cu`).

---

## Step 1: Audit Autotuner Performance & BF16 Coverage

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

$PY -m ckt shapes list --qtype bf16
```

### Critical Check in `bench/ops/bf16_flash_next_sweep.cu` and `ckt/autotune/`
1. Inspect [`bench/ops/bf16_flash_next_sweep.cu`](file:///opt/etc/Infernix/bench/ops/bf16_flash_next_sweep.cu):
   - Which $[N, K]$ shapes does `infernix_bf16_flash_next_sweep` support?
   - **Check**: If `bf16_flash_next_sweep.cu` only supports a subset of Flash-Next shapes (or requires `StaticK`), any unsupported $[N, K]$ will cause `runner.py` to fall back to `infernix_linear_bench` (which only runs `ops::linear`, testing 0 alternative candidates!).
   - **Action**: Generalize `bf16_flash_next_sweep.cu` (or add dynamic-$K$ `Bf16A16GemvSchedule`, `Bf16A16SimtSchedule`, `Bf16A16SlicedKMmaSchedule`, `Bf16A16MmaSchedule`, `Bf16A16TmaMmaSchedule` candidates) so it can sweep **all 25 BF16 shapes** dynamically!
2. Inspect [`cuda-kernel-toolkit/ckt/autotune/registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py):
   - Check `_generate_family_template_candidates()` for `qtype == "bf16"`. Currently there is **no `elif qtype == "bf16":` branch** in `_generate_family_template_candidates()`! You must add BF16 candidate synthesis so `ckt autotune` evaluates all legal BF16 schedules across all 25 shapes.

---

## Step 2: Run Autotuner (Stash Only) & Capture Cold-L2 GB10 Baselines

1. Run `ckt autotune tune --family bf16` on a sample shape, inspect the output, and `git stash`:
   ```bash
   $PY -m ckt autotune tune --family bf16 -n 4096 -k 2048 --flush-mib 32
   git stash
   ```
2. Benchmark all 25 BF16 shapes at route boundaries (`--flush-mib 32`):
   ```bash
   for f in src/ops/linear/bf16/shapes/n*.cu; do
     nk=$(basename "$f" .cu | sed -E 's/n([0-9]+)_k([0-9]+)/\1 \2/')
     set -- $nk
     $PY -m ckt shapes bench --qtype bf16 --n $1 --k $2 --mode boundaries --flush-mib 32
   done
   ```

---

## Step 3: Manual Audit of BF16 GEMV, SIMT, Sliced-K, `cp.async` MMA & TMA MMA Kernels

Inspect compiled kernel resources and SASS:

```bash
$PY -m ckt static build-native/bench/infernix_linear_bench --kernel bf16 --sass
```

Check these specific BF16 architectural trade-offs on GB10 (`sm_121a`):

1. **Shared Memory Footprint in `Bf16A16MmaSchedule` vs `Bf16A16TmaMmaSchedule` (`bf16_schedule.cuh:150-211`)**:
   - In `Bf16A16MmaSchedule` (`bf16_schedule.cuh:150`):
     `kSharedBytes = Stages * (BlockRows + BlockTokens) * BlockK * 2`.
     Notice that both weights (`BlockRows * BlockK * 2`) AND activations (`BlockTokens * BlockK * 2`) are multiplied by `Stages` (`kStages >= 2`)!
   - Let's check when `kSharedBytes <= 50,176` B (**2 blocks/SM** on GB10):
     - For `BlockK = 64, Stages = 2`: `kSharedBytes = 4 * (BlockRows + BlockTokens) * 64 = 256 * (BlockRows + BlockTokens)`:
       - `BlockRows = 64, BlockTokens = 64`: `32,768` B (`32 KiB` $\to$ **3 blocks/SM**!).
       - `BlockRows = 128, BlockTokens = 64`: `49,152` B (`48 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**!).
       - `BlockRows = 128, BlockTokens = 128`: `65,536` B (`64 KiB` $> 50,176$ B $\to$ **1 block/SM**).
     - For `BlockK = 64, Stages = 3`: `kSharedBytes = 384 * (BlockRows + BlockTokens)`:
       - `BlockRows = 64, BlockTokens = 64`: `49,152` B (`48 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**!).
       - `BlockRows = 128, BlockTokens = 64`: `73,728` B (`72 KiB` $> 50,176$ B $\to$ **1 block/SM**).
   - **Critical GB10 Insight**: Because BF16 linear is DRAM-bandwidth bound up to $T \approx 800$, dropping from 2 blocks/SM (`48–64` KB/SM) to 1 block/SM (`> 50,176` B/block) at $T=32..256$ can *reduce* DRAM request concurrency on GB10's 48 SMs unless TMA (`Bf16A16TmaMmaSchedule` with `Stages = 3` or `4`) keeps enough bytes in flight per CTA!
2. **Vision Shapes (`n1152_k1152.cu`, `n1152_k1536.cu`, `n1152_k4304.cu`, `n4304_k1152.cu`)**:
   - Note that `4304 = 67 * 64 + 16` (not divisible by 64 or 128!). These shapes use `Bf16RowTailSchedule`, `Bf16KTailSchedule`, or `Bf16A16TmaTailMmaSchedule` and are heavily exercised at vision token counts ($T = 256..4096$).

---

## Step 4: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **`eff_GB/s < 180 GB/s` (`read_% < 73%`) at $T=1..8$** | `Simt` or `Gemv` schedule has too few warps in flight across 48 SMs or uses `DirectStream` instead of `WarpPacked` / `Shared` activation reuse. | For $T=1$, compare `Gemv` (`WarpsPerRow=4, 8` with `Bf16ActivationAccess::Shared`) vs `SlicedKMma` (`KWarps=8`). For $T=2..8$, compare `Simt` (`WarpPacked`) vs `SlicedKMma` (`BlockRows=32, BlockTokens=8, KWarps=4` or `8`). |
| **`[INVERTED CLIFF]` or **`[UPWARD CLIFF]`** when switching from `SlicedKMma` to `Mma` / `TmaMma` at $T=16..64$** | The full-K `Mma` tile (`BlockRows=128`) produces too few CTAs (`ceil(N/128) * ceil(T/64) < 96`) to fill GB10's 48 SMs at 2 blocks/SM, whereas `BlockRows=64` doubles the CTA count. | Use `BlockRows = 64, BlockTokens = 64, BlockK = 64, Stages = 2` or `3` (`<= 49,152` B $\to$ 2–3 blocks/SM) for mid-token bands ($T=17..96$) before transitioning to `BlockRows = 128` TMA schedules at larger $T$. |
| **`[UPWARD CLIFF]` at $T=16 \to 17$ followed by `[DOMINATED BAND]` at $T=32 \to 33$** | Sliced-K (`sliced_r32t16w4`) has `BlockTokens=16` and requires 2 full tile passes at $T=17..32$ (spiking latency by 1.8x–1.9x) before dropping when switching to tile-32 schedules. | Switch to `Bf16A16MmaR32T32K256S3` at $T=17$ for square/medium shapes (`n4608_k4608`, `n5120_k4608`). This delivers a **1.82x–1.91x speedup** at $T=17..32$ and completely flattens the transition. |
| **Severe latency drop ($>2\times$ slowdown) at $T=256$ on vision shapes** | Large `RowFast` tile schedules (`tail_r128t64s4_rows` taking ~595 $\mu$s) suffer from severe wave under-occupancy on GB10's 48 SMs at mid-range token counts. | Switch to `tail_r64t64s3` (`Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>`), which runs in **211 $\mu$s (2.81x speedup)**. |
| **L2-Resident Shape (`weight_bytes < 20 MiB`, e.g. `1152x1152` = 2.5 MiB) has low `TFLOP/s` at $T \ge 256$** | When `--flush-mib 32` is used, the first wave fetches weights from DRAM, and subsequent token tiles hit GB10's 24 MiB L2 cache if raster order (`Bf16MmaRaster`) reuses weight rows across token tiles! | Ensure `Bf16MmaRaster::TokenFast` or `Grouped` is used so CTAs sharing the same weight rows execute concurrently in the same wave and hit L2! |
| **Vision shape (`N=4304` or `K=4304`) fails or throws on non-divisible tile** | Non-multiple of 64/128 routed to an unpredicated `Bf16A16MmaSchedule` or `Bf16A16TmaMmaSchedule`. | Use `Bf16RowTailSchedule<...>` when $N \bmod \text{BlockRows} \ne 0$, `Bf16KTailSchedule<...>` when $K \bmod \text{BlockK} \ne 0$, or `Bf16A16TmaTailMmaSchedule`. |

---

## Steps 5, 6 & 7: Extend Sweep Binary, Update `ckt`, Tune & Verify

1. **Extend [`bench/ops/bf16_flash_next_sweep.cu`](file:///opt/etc/Infernix/bench/ops/bf16_flash_next_sweep.cu)** so it supports any $[N, K]$ shape (including `StaticK = 0` dynamic schedules and `Bf16RowTailSchedule` / `Bf16KTailSchedule` for $4304$ shapes).
2. **Add `qtype == "bf16"` candidate synthesis to [`cuda-kernel-toolkit/ckt/autotune/registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py)**.
3. **Build and Qualify**:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_bf16_flash_next_sweep infernix_linear_bf16_a16_test
   ./build-native/tests/infernix_linear_bf16_a16_test
   ```
4. **Verify Diffs & Tests**:
   - Inspect `git diff src/ops/linear/bf16/shapes/` to ensure no token guards or epilogue paths were stripped.
   - Run `$PY -m pytest cuda-kernel-toolkit/tests -q` and `git diff --check`.
