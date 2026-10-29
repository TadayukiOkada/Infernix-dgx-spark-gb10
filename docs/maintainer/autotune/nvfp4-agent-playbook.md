# Autonomous Agent Playbook: NVFP4 (`nvfp4`) Linear Kernel Analysis, Manual Optimization & Autotuner Overhaul on NVIDIA GB10 (`sm_121a`)

This playbook gives an LLM agent concrete, step-by-step instructions to execute the 7-step GB10 kernel optimization and `cuda-kernel-toolkit` (`ckt`) overhaul for the **NVFP4 (`nvfp4`)** linear family, interpret every benchmark/compiler/profiler signal, and choose the exact next action for each outcome.

> [!IMPORTANT]
> **Hard Gate — Reusing Existing Schedules on a Few Shapes Is Insufficient**:
> You MUST NOT treat kernel headers (`*_gemv.cuh`, `*_simt.cuh`, `*_sliced_k_mma.cuh`, `*_mma.cuh`, `*_a4_*.cuh`, `*_schedule.cuh`, `*_instances.cuh`) as read-only black boxes, and you MUST NOT stop after tweaking a handful of shape files with pre-existing schedules.
> A complete family optimization on NVIDIA GB10 (`sm_121a`) requires:
> 1. **Static SASS & Register-Spill Audit (`$PY -m ckt static build-native/bench/infernix_linear_bench --kernel nvfp4 --sass`)**: Identify and eliminate register spills (`STACK > 0`, `LDL`/`STL`) in both `a16` and `a4` TMA/MMA kernels (remembering that 9-warp TMA kernels have a 168-register hardware ceiling on GB10 because 3 warps land on 1 of the 4 warp schedulers: $\lfloor 512 / 3 \rfloor_8 = 168$ registers).
> 2. **GB10-Specific Schedule & Kernel Engineering**: Engineer and instantiate missing GB10 schedules in `nvfp4_schedule.cuh`, `nvfp4_instances.cuh`, and the sweep binary (such as `RowTiles = 2` Sliced-K, 2-blocks/SM `BlockK = 128, ActivationStages = 1` MMA schedules fitting inside `<= 50,176` B SMEM, `Simt` schedules where `TokenTile == Capacity` so $T=5..8$ never reads weights twice, and L2-swizzled `Grouped` / `RowFast` rasterization at large $T \ge 1536$ for GB10's 24 MiB L2 cache).
> 3. **Correct Interpretation of Numerical & Column-Invariance Contracts**: Read `tests/ops/linear/test_nvfp4_a16.cpp` and `test_nvfp4_a4.cpp` first. Never assume a mid-band `Sliced-K` route at $T \ge 9$ must preserve an old incumbent's `KWarps=2` or `KWarps=4` reduction-tree hash when column invariance only governs $T \le 8$ (or a dedicated `A16ColumnInvariant` entry point).
> 4. **100% Shape Coverage Across All Policies (`a16` and `a4`)**: Audit and benchmark **every** shape file in `src/ops/linear/nvfp4/shapes/` (all 15 shapes) across decode ($T=1..8$), mid-band ($T=9..256$), and large prefill ($T=257..8192$).

---

## 0. Target Hardware & Family Surface Area

### Hardware Constants (NVIDIA GB10, `sm_121a`)
- **SM Count**: 48 SMs (`maxThreadsPerMultiProcessor = 1536`, `maxBlocksPerMultiProcessor = 24`, `regsPerMultiprocessor = 65536`).
- **L2 Cache**: 24 MiB (`--flush-mib 32` is mandatory for all cold-L2 measurements; never use the 256 MiB RTX 5090 default).
- **Sustained DRAM Read Ceiling**: ~246 GB/s.
- **Shared Memory Ceiling**: 48 KiB static default, 99 KiB (`101,376` B) opt-in dynamic per block (`102,400` B per SM with `1,024` B driver overhead per CTA).
  - **Critical 2-Blocks/SM Cliff**: To keep **2 CTAs/SM** resident, a kernel MUST satisfy `kSharedBytes <= 50,176` bytes (`49.0 KiB`) AND `threads_per_block * 2 <= 1536` AND `ceil(threads_per_block * regs_per_thread, 256) * 2 <= 65536`.
  - **Critical 3-Blocks/SM Cliff**: `kSharedBytes <= 33,152` bytes (`32.375 KiB`).

### NVFP4 Source Files & Targets
- **Kernel & Schedule Headers**:
  - [`src/ops/linear/nvfp4/nvfp4_schedule.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_schedule.cuh): Defines `Nvfp4A16GemvSchedule`, `Nvfp4A16SimtSchedule`, `Nvfp4A16SlicedKMmaSchedule`, `Nvfp4A16MmaSchedule`, `Nvfp4A4MmaSchedule`, `Nvfp4A4TmaMmaSchedule`, and `Nvfp4ScheduleInstance`.
  - [`src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh): $T=1$ warp-specialized GEMV (`ValuesPerLane = 8, 16, 32`, `StagedRaw` vs `Direct` E4M3 scale access).
  - [`src/ops/linear/nvfp4/nvfp4_a16_simt.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_simt.cuh): Small-$T$ SIMT (`PairStream`, `TokenPacked`, `SharedPhase`).
  - [`src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh): Inner-K split across `KWarps` (`2, 4, 8, 16`), `BlockRows = 16`, `kTileKPerWarp = 64`.
  - [`src/ops/linear/nvfp4/nvfp4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh): Full-K BF16 MMA (`BlockK = 64` or `128`, `ActivationStages = 1` or `2`, dequantizes E2M1 codes + E4M3 group-16 scales to BF16).
  - [`src/ops/linear/nvfp4/nvfp4_a4_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_mma.cuh) & [`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh): Native Blackwell `m16n8k64` block-scaled NVFP4 Tensor Core routes (`policy=a4`, `Terms=1..3`).
  - [`src/ops/linear/nvfp4/nvfp4_instances.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_instances.cuh) & [`src/ops/linear/nvfp4/nvfp4_template_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_template_launch.cuh).
- **Shape Dispatch Directory**: `src/ops/linear/nvfp4/shapes/` (15 `.cu` files: `n128_k2560.cu`, `n2560_k2560.cu`, `n2560_k9216.cu`, `n3072_k1536.cu`, `n5120_k1536.cu`, `n5120_k2560.cu`, `n6144_k2560.cu`, `n9216_k2560.cu`, `n10240_k2560.cu`, `n12800_k2560.cu`, `n13952_k2560.cu`, `n14336_k5120.cu`, `n17408_k5120.cu`, `n18432_k2560.cu`, `n34816_k5120.cu`).
- **NVFP4 Physical Storage**:
  - 4-bit E2M1 codes (`0.5` bytes/elem) + 8-bit E4M3 scale per 16 elements (`1/16 = 0.0625` bytes/elem) = **`0.5625` bytes per weight** (`9` bytes per 16 weights: `8` code bytes + `1` scale byte).

---

## Step 1: Audit Autotuner Performance & NVFP4 Coverage

Run a fast inspection of how `ckt autotune` handles `--family nvfp4`:

```bash
export PYTHONPATH=cuda-kernel-toolkit
PY=/opt/etc/Infernix/.venv-convert/bin/python3

# Check discovered NVFP4 shapes and policies
$PY -m ckt shapes list --qtype nvfp4
```

### What to Check in `cuda-kernel-toolkit/ckt/autotune/`
1. **`runner.py`**:
   - Check `_find_sweep_binary("nvfp4")`: Does `infernix_nvfp4_route_sweep` exist in `bench/ops/`?
   - **Current State**: There is **no** `bench/ops/nvfp4_route_sweep.cu` binary yet! Without it, `runner.py` falls back to `infernix_linear_bench`, which only executes the already-compiled production dispatch (`ops::linear`) and cannot test alternative schedules without recompiling `infernix_ops` (~55s device link per candidate).
2. **`registry.py`**:
   - Check `_generate_family_template_candidates()` for `qtype == "nvfp4"`:
   - **Current State**: `registry.py` only harvests routes already present in the single shape file being tuned. Notice that NVFP4 shape files use template instantiations like:
     - `nvfp4_a16_gemv<Nvfp4ScheduleInstance<GemvW4R1V16C1, 2560>>(x, w, out, stream)`
     - `nvfp4_a16_simt<Nvfp4ScheduleInstance<SimtW4W1R1V16T4, 2560, 4, true>>(x, w, out, stream)`
     - `nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<SlicedW4T16, 2560>>(x, w, out, stream)`
     - `nvfp4_a16_mma<Nvfp4ScheduleInstance<MmaR64T64K64, 2560>>(x, w, out, stream)`

---

## Step 2: Run Autotuner (Stash Only) & Capture Cold-L2 GB10 Baselines

1. Run `ckt autotune tune --family nvfp4` on 2–3 representative shapes (`n6144_k2560`, `n14336_k5120`, `n34816_k5120`), inspect what it does, and `git stash` any changes:
   ```bash
   $PY -m ckt autotune tune --family nvfp4 -n 14336 -k 5120 --flush-mib 32
   git status
   git stash
   ```
2. Record the exact cold-L2 boundary baseline across all 15 NVFP4 shapes (and both `a16` and `a4` policies where supported):
   ```bash
   for f in src/ops/linear/nvfp4/shapes/n*.cu; do
     nk=$(basename "$f" .cu | sed -E 's/n([0-9]+)_k([0-9]+)/\1 \2/')
     set -- $nk
     $PY -m ckt shapes bench --qtype nvfp4 --n $1 --k $2 --policy a16 --mode boundaries --flush-mib 32
   done
   ```

---

## Step 3: Manual Audit of NVFP4 GEMV, SIMT, Sliced-K, A16 MMA & A4 TMA Kernels

Inspect the compiled resources (`REG`, `STACK`, `SHARED`, `LOCAL`) and SASS of all NVFP4 kernels:

```bash
$PY -m ckt static build-native/bench/infernix_linear_bench --kernel nvfp4 --sass
```

Audit each kernel family in `src/ops/linear/nvfp4/`:

1. **`nvfp4_a16_mma.cuh` & `Nvfp4A16MmaSchedule` (`nvfp4_schedule.cuh:187-224`)**:
   - Look at `kSharedBytes` in `Nvfp4A16MmaSchedule`:
     ```cpp
     static constexpr int kSharedBytes =
         kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
         kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
         kBlockRows * (kBlockK / 2 + kBlockK / 16);
     ```
   - Check `nvfp4_instances.cuh`: Which `Nvfp4A16MmaSchedule` tiles use `BlockK = 64` vs `BlockK = 128`, and `ActivationStages = 2` vs `ActivationStages = 1`?
   - Check how `nvfp4_a16_mma.cuh` stages and decodes weights:
     - Does it decode 2 E2M1 weights at a time or 8 E2M1 weights (`uint32_t` $\to$ 4 `__nv_bfloat162` $\to$ 16-byte `STS.128`)?
     - When `ActivationStages == 1`, does it prefetch the next weight tile (`stage_quant`) *before* the MMA loop and overlap `decode_weight` with `stage_activation` using `cp_wait<1>()`?
     - Check `kSharedBytes` for `BlockRows = 64, BlockTokens = 80, 96, 112, 128` with `BlockK = 128, ActivationStages = 1`:
       - Weight staging (`S=1`): `64 * (64 + 8) = 4,608` B (`4.5 KiB`).
       - Decoded `As` (`64 * 128 * 2`): `16,384` B (`16.0 KiB`).
       - Activation `Bs` (`1 * BN * 128 * 2`):
         - `BN = 64`: `16,384` B $\to$ Total = `37,376` B (`36.5 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**).
         - `BN = 80`: `20,480` B $\to$ Total = `41,472` B (`40.5 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**).
         - `BN = 96`: `24,576` B $\to$ Total = `45,568` B (`44.5 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**).
         - `BN = 112`: `28,672` B $\to$ Total = `49,664` B (`48.5 KiB` $\le 50,176$ B $\to$ **2 blocks/SM**!).
2. **`nvfp4_a16_sliced_k_mma.cuh` & `Nvfp4A16SlicedKMmaSchedule` (`nvfp4_schedule.cuh:228-257`)**:
   - Notice `Nvfp4A16SlicedKMmaSchedule` currently hardcodes `kBlockRows = 16` (`MT = 1`), whereas `Fp8A16SlicedKMmaSchedule` (`fp8_schedule.cuh:273`) supports `RowTiles = 1` or `2` (`kBlockRows = 16 * RowTiles = 32`), which halves activation L2/shared-memory re-reads at $T=8..32$!
   - Check if adding `RowTiles = 2` (`BlockRows = 32`) with hoisted `ldmatrix_x2` in `nvfp4_a16_sliced_k_mma.cuh` improves $T=8..48$ for large-$N$ shapes (`n14336_k5120`, `n17408_k5120`, `n34816_k5120`).
3. **`nvfp4_a4_mma.cuh` & `nvfp4_a4_tma.cuh` (`policy=a4`)**:
   - Check when `select_nvfp4_launch` selects `a4` vs `a16`, and verify `Nvfp4A4TmaMmaSchedule` occupancy (`kSharedBytes <= 50,176` B for 2 blocks/SM vs `kSharedBytes <= 99 KiB` for 1 block/SM).

---

## Step 4: Diagnostic Decision Matrix (How to Interpret Results & What to Do Next)

Use this exact decision table when reading `ckt shapes bench` and `ckt static` outputs:

| Observed Signal in `ckt shapes bench` / `ckt static` | Root Cause Diagnosis | Exact Action to Take Next |
|---|---|---|
| **`[INVERTED CLIFF / DOMINATED BAND]`**: Latency at $T=B+1$ (new route) is **lower** than at $T=B$ (old route). | The lower-token route has worse occupancy, higher barrier overhead, or excess K-split reduction compared to the destination route. | Lower the token threshold of the $T=B+1$ route to cover the dominated band (e.g., switch to `mma_r32_t64` or `mma_r64_t64` at $T=33$ instead of $T=65$), then re-benchmark $T \in [16, 24, 32, 33, 48]$. |
| **`[UPWARD CLIFF]` $+20\%\text{ to }+60\%$** at $T=64 \to 65$ or $T=96 \to 97$, and `eff_GB/s` drops from $>185\text{ GB/s}$ to $<140\text{ GB/s}$. | **Shared-Memory or Register Occupancy Cliff**: The destination MMA schedule crossed `50,176` B (`49.0 KiB`) of shared memory or `128` regs/thread (at 256 threads), dropping from **2 blocks/SM** to **1 block/SM** on GB10. | 1. Run `$PY -m ckt static --binary build-native/bench/infernix_linear_bench --filter <symbol>` to check `REG` and `SHARED`.<br>2. Switch the schedule to `BlockK = 128, ActivationStages = 1` (`kSharedBytes <= 49,664` B for $BN \le 112$) or `BlockRows = 32` (`MmaR32T64` / `MmaR32T128`) so `kSharedBytes <= 45 KiB` and **2 blocks/SM** remain resident. |
| **`wave_% < 40%`** at medium token counts ($T=33..128$) on mid-sized $N$ ($N \in [2560, 6144]$). | **CTA Grid Wave Quantization**: `CTAs = ceil(N / BlockRows) * ceil(T / BlockTokens)` is just above a multiple of `48 * blocks_per_SM` (e.g., 112 CTAs on 96 slots = 1 full wave + 16-CTA tail wave). | Choose `(BlockRows, BlockTokens)` so `CTAs` is a near-exact multiple of `96` (`2 blk/SM * 48 SMs`) or `144` (`3 blk/SM * 48 SMs`). For example, at $T=65..72$, `BlockTokens=72` (`1` tile) avoids launching `2` tiles of `BlockTokens=64`. |
| **`eff_GB/s < 160 GB/s` (`read_% < 65%`) at $T=2..16$** on `Simt` or `SlicedKMma`. | Under-utilized warps, small `BlockRows=16` re-reading activations from L2, or suboptimal `KWarps`. | Test `SlicedW8` (`KWarps=8` for $K=5120$) at $T=2..16$, and test `RowTiles=2` (`BlockRows=32`) in `Nvfp4A16SlicedKMmaSchedule` for $T=8..32$. |
| **`LOCAL > 0` or `STACK > 0`** in `ckt static`. | Register spilling to local memory (causes 30–70% slowdown). | Check `__launch_bounds__(kThreads, kMinBlocksPerSm)`: if `kMinBlocksPerSm` exceeds the shared-memory block limit `kMaxBlocksBySmem = 102400 / (aligned_smem + 1024)`, clamp `kMinBlocksPerSm` or reduce `WarpRows`/`WarpTokens`. |

---

## Steps 5, 6 & 7: Build `infernix_nvfp4_route_sweep`, Upgrade `ckt`, Tune & Verify

1. **Create `bench/ops/nvfp4_route_sweep.cu` and register `infernix_nvfp4_route_sweep` in `bench/ops/benchmarks.cmake`**:
   - Mirror [`bench/ops/q4_route_sweep.cu`](file:///opt/etc/Infernix/bench/ops/q4_route_sweep.cu): instantiate all shared `nvfp4_instances` schedules (`Gemv*`, `Simt*`, `Sliced*`, `Mma*`) with `StaticK = 0` (dynamic $K$) so a single binary can sweep any $[N, K]$ and `--tokens t1,t2,...` with `--flush-mib 32` without recompiling.
2. **Wire `infernix_nvfp4_route_sweep` into `cuda-kernel-toolkit`**:
   - In [`cuda-kernel-toolkit/ckt/autotune/runner.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/runner.py): add `"nvfp4": "infernix_nvfp4_route_sweep"` to `_find_sweep_binary()`.
   - In [`cuda-kernel-toolkit/ckt/autotune/registry.py`](file:///opt/etc/Infernix/cuda-kernel-toolkit/ckt/autotune/registry.py): add `elif qtype == "nvfp4":` in `_generate_family_template_candidates()` to synthesize all legal NVFP4 schedules with their `Nvfp4ScheduleInstance<..., K>` launch expressions.
3. **Compile and Qualify Against the Independent Oracle**:
   ```bash
   ninja -C build-native -j "$(nproc)" infernix_linear_bench infernix_nvfp4_route_sweep infernix_linear_nvfp4_a16_test infernix_linear_nvfp4_a4_test
   ./build-native/tests/infernix_linear_nvfp4_a16_test
   ./build-native/tests/infernix_linear_nvfp4_a4_test
   ```
4. **Verify Diff Integrity Before Reporting**:
   - Run `git diff src/ops/linear/nvfp4/shapes/` and confirm no precondition checks, `policy` branches (`LinearPolicy::Bf16` vs `LinearPolicy::Fp4`), or bounds checks were removed.
   - Run `$PY -m pytest cuda-kernel-toolkit/tests -q` and `git diff --check`.
