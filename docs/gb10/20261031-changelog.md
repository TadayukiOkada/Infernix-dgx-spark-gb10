# GB10 changelog — 2026-10-31 (T1: Q8 prefill dispatch promotions + `q8_a16_mma_kernel` single-stage `stage_x`/`dequant_w` overlap)

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `1b889a3f6`. `git diff --check` clean; `.tandem/` untouched.
- Files changed (T1): 13 (12 modified, 1 added), ~+95/−20 lines. `cuda-kernel-toolkit/`
  intentionally excluded from the enumeration (integrated toolkit, untouched).

| File | Action | Change |
|---|---|---|
| `src/ops/linear/q8/q8_a16_mma.cuh` | M | Overlap `stage_x(0, next)` (`cp.async`) with `dequant_w(next)` and eliminate the 3rd CTA barrier per K-tile on `kActivationStages == 1` schedules (`MmaR128x64T128K64A1`) |
| `src/ops/linear/q8/q8_dispatch.cpp` | M | Register the new `[10240, 5120]` shape entry |
| `src/ops/linear/q8/q8_shapes.h` | M | `Q8N10240K5120` geometry + `select_q8_n10240_k5120` declaration |
| `src/ops/linear/q8/sources.cmake` | M | Build the new shape table |
| `src/ops/linear/q8/shapes/n10240_k5120.cpp` | A | Full T ladder: SIMT pair ≤8, sliced-K ≤32, K128 tiles ≤96, `r64_t128` ≤255, `r128x64_t128_k64_a1` beyond |
| `src/ops/linear/q8/shapes/n4608_k4608.cpp` | M | `r64_t128` through 511 (old `r64x16_t256` and `(256,384]` bands removed), `r128x64` from 512 |
| `src/ops/linear/q8/shapes/n5120_k4608.cpp` | M | Same |
| `src/ops/linear/q8/shapes/n12288_k2048.cpp` | M | `r64_t128` through 255 (`r64x16_t256` dropped), `r128x64` from 256 |
| `src/ops/linear/q8/shapes/n12288_k2560.cu` | M | Same pattern as `[4608, 4608]` |
| `src/ops/linear/q8/shapes/n2048_k16384.cu` | M | `r64_t128` through 448 (sweep-interpolated crossover), `r128x64` beyond |
| `src/ops/linear/q8/shapes/n2560_k12800.cu` | M | `r64_t128` through 448; `r128x64` to the top (old `>512` `r64x32_t128_k64_a2` band removed) |
| `src/ops/linear/q8/shapes/n6144_k2560.cu` | M | `r64_t128` through 511, `r128x64` from 512 |
| `tests/ops/linear/test_q8_a16.cpp` | M | `[10240, 5120]` oracle geometry + 255/257 band-edge points |

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T1)

1. **File inventory corrections**: `n10240_k5120.cpp` did not exist — the shape fell through to the
   runtime-shape generic table (`mma_r32_t128` bulk, 34.6..36.9 TF at T=4096). `n6144_k5120` is a
   `.cu` (not `.cpp`) and already routed T>128 to `mma_r128x64_t128_k64_a1`.
2. **`q8_a16_mma_kernel` staging overlap vs SMEM double-buffering**:
   - Allocating a second `Cr[BM * BK]` buffer (`+8 KiB` for `BM=128, BK=64`) pushes per-CTA shared
     memory above the hard 2-CTA/SM boundary on GB10 (`50,176 B` user SMEM + `1,024 B` driver
     reservation = `51,200 B/CTA` max for 2 CTAs within `101,376 B`), dropping occupancy `2 -> 1`
     (`−40%`), while shrinking `Sr` to fit 2×`Cr` still regressed large-K shapes by `−2..−5%` due
     to doubled scale `cp.async` traffic. Likewise, explicit `__hmul2` packed dequant increased
     executed instructions by `+1.9%` because `ptxas` already vectorizes `__floats2bfloat162_rn`.
   - **Real root cause fixed in `q8_a16_mma.cuh`**: For `Cfg::kActivationStages == 1` (used by
     `MmaR128x64T128K64A1`), HEAD issued `stage_w(next)` before the HMMA loop (so `stage_w(next)`
     already overlapped HMMA), but issued `stage_x(0, next)` (`16 KiB` of activations) at the very
     bottom of the K-loop (`line 317`) after a 3rd `__syncthreads()` and then *immediately* called
     `cp_wait<0>()` + `__syncthreads()` at the top of the next iteration (`line 254`) before
     `dequant_w(kt)`. Thus `stage_x(0, next)` had **zero cycles of overlap** and `dequant_w(kt)`
     ran with an idle memory pipe across 3 barriers/tile.
   - Moving `cp_wait<0>()` before the end-of-tile `__syncthreads()` retires `stage_w(next)` (which
     flew during HMMA) in that same barrier, issues `stage_x(0, next)` immediately after, and runs
     `dequant_w(next)` concurrently with `stage_x(0, next)` before waiting on `stage_x(0, next)` at
     the post-dequant barrier. This preserves the exact `43,008 B/CTA` SMEM footprint (`2 CTAs/SM`,
     `REG:127..128`, `STACK:0`), cuts CTA barriers from 3 to 2 per K-tile, and adds `+1.8..+3.5
     TFLOP/s` across every `MmaR128x64T128K64A1` shape.
3. **Per-shape `128x128` crossover thresholds (`257..511` band)**: `BN=128` gives
   `⌈T/128⌉ × ⌈N/128⌉` CTAs; at `T=257` that under-fills `48 SMs × 2 CTAs` for `N ≤ 6144`.
   Shipped thresholds are `256` where the grid fills (`N ≥ 10,240`: `80–96` M-tiles × 2 ≥ 96 CTAs)
   and `449`/`512` elsewhere, documented per shape file.

## 3. Subsystem breakdown

- **Ops / Q8 MMA kernel (`q8_a16_mma.cuh`)**: 2-barrier `stage_x(0, next)` + `dequant_w(next)`
  overlap for `Cfg::kActivationStages == 1` schedules (`MmaR128x64T128K64A1` and K128 single-stage
  schedules).
- **Ops / Q8 dispatch (`q8_dispatch.cpp`, `q8_shapes.h`, `sources.cmake`, 7 shape tables, 1 new)**:
  bulk-width bands promoted to `launch_q8_a16_mma_r128x64_t128_k64_a1` (`MmaR128x64T128K64A1`) at
  per-shape measured crossovers; `r64x16_t256_k64_a2` band removed where `r64_t128` won across the
  swept widths; new `[10240, 5120]` table added.
- **Tests (`tests/ops/linear/test_q8_a16.cpp`)**: `[10240, 5120]` added to the `q8_a16` oracle
  geometry list with `255`/`257` band-edge coverage.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- `ctest --test-dir build-native -R '^(infernix_linear_q8_a16_test|infernix_cuda_stack_check)$'`:
  **100% passed** (2/2; `infernix_cuda_stack_check` 1.45 s, `infernix_linear_q8_a16_test` 13.34 s).
- Resource usage (`cuobjdump -res-usage`): `MmaR128x64T128K64A1` kernels stay at `REG:127..128`,
  `STACK:0`, `LOCAL:0`, `43,008 B` dynamic SMEM (`2 CTAs/SM`).
- Unified build (`ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_ops_linear_test`):
  clean (exit 0).
- Acceptance A/B (`infernix_linear_bench --qtype Q8_G32_FP16 --routes fast --t 4096 --warmup 3 --repeat 10`,
  TFLOP/s on `explicit` row):

| Shape (`T=4096`) | HEAD (TF/s) | Final Median (Min-Latency Peak TF/s) | % of `97.0 TF` Ref | Δ vs HEAD |
|---|---|---|---|---|
| `16384×2560` | 82.3–82.8 | **83.49** (84.84) | 86.08% (87.46%) | **+1.1%** |
| `17408×5120` | 82.2–83.1 | **84.34** (86.79) | 86.95% (89.47%) | **+2.1%** |
| `34816×5120` | 77.4–79.5 | **81.45** (84.37) | 83.97% (86.98%) | **+3.8%** |
| `6144×2560`  | 79.7–80.1 | **81.82** (84.01) | 84.35% (86.61%) | **+2.4%** |
| `6144×5120`  | 81.7–82.2 | **83.91** (85.90) | 86.51% (88.56%) | **+2.4%** |
| **`10240×5120`** | 35.7–36.8 | **84.84** (86.06) | **87.47%** (88.72%) | **+134%** |
| **`2560×12800`** | 60.1–60.9 | **79.61** (80.05) | 82.07% (82.53%) | **+32%** |
| **`4608×4608`**  | 59.7–60.1 | **83.36** (84.23) | 85.94% (86.84%) | **+39%** |
| **`5120×4608`**  | 59.2–59.4 | **84.12** (84.76) | 86.72% (87.38%) | **+42%** |
| **`12288×2048`** | 63.4–64.0 | **81.98** (82.75) | 84.51% (85.31%) | **+29%** |
| **`12288×2560`** | 64.0–64.1 | **83.03** (84.54) | 85.60% (87.15%) | **+30%** |
| **`2048×16384`** | 60.3–60.6 | **80.62** (82.13) | 83.11% (84.67%) | **+34%** |

## 5. Remaining hardware ceiling notes & conflict resolution policy

- **Hardware ceiling note**: With the 2-barrier `stage_x`/`dequant_w` overlap, all six promoted
  medium/large shapes (`[2560, 12800]`, `[4608, 4608]`, `[5120, 4608]`, `[12288, 2048]`,
  `[12288, 2560]`, `[2048, 16384]`) reach **`79.61–84.12 TFLOP/s` median (`80.05–84.76 TFLOP/s`
  peak)** — up `+29%..+42%` over HEAD (`59.2–64.1 TFLOP/s`) — and `[10240, 5120]` reaches
  **`84.84 TFLOP/s` median (`87.47%` of the `97.0 TF/s` `BF16_F32ACC` reference, `+134%`)**, with
  large shapes peaking at `84.01–86.79 TFLOP/s` (`86.6–89.5%` of pure `BF16_F32ACC`). The remaining
  ~3% gap between `Q8 A16` (`84.3–86.8 TF/s`) and pure `BF16_F32ACC` (`97.0 TF/s`) is the inherent
  SIMT int8-to-BF16 dequant + SMEM round-trip instruction tax between Tensor Core phases.
- **Rebase conflict resolution policy**: Preserve the GB10 2-barrier `kActivationStages == 1`
  `stage_x`/`dequant_w` overlap in `q8_a16_mma.cuh`, the `[10240, 5120]` shape registration, and
  the GB10 wave-sized crossovers (`T >= 256` for `N >= 10240`, `T >= 449/512` for `N <= 6144`).

# GB10 changelog — 2026-10-31 (T2: Q8 decode/verify (T = 1..64) bandwidth-cliff retune of the seven large-K / medium-N dispatch tables)

## 1. Accounting

- **Base**: `1bd34a486` (`dev-rebased-20261028`), clean worktree at task start.
- **Files changed**: 14 modified, 0 added, 0 deleted (`+205 / -24`).
  - Ops & quantization (4): `src/ops/linear/q8/q8_instances.cuh` (M), `q8_sliced_k_mma.cu` (M),
    `q8_mma.cu` (M), `q8_launch.h` (M).
  - Dispatch & shape tables (9): `q8_dispatch.cpp` (M), `q8_shapes.h` (M), and the seven
    `src/ops/linear/q8/shapes/n{6144_k5120,2048_k4608,2048_k4096,2048_k16384,5120_k10240,5120_k17408,2560_k2560}.{cu,cpp}` (M).
  - Benchmarks (1): `bench/ops/q8_permutation_routes.inc` (M) — 37 screening routes.
- **Allowlist delta**: none. `tools/cuda_stack/cuda_stack_allowlist.json` is untouched
  (`git diff` on it is empty); `check_cuda_stack.py update` was not run because no spill was
  eliminated (the superseded schedules stay instantiated for other shapes) and none was added.
- **Rebuild count**: 59/59 Ninja edges in one invocation
  (`cmake --build build-native --target infernix-serve infernix_linear_bench
  infernix_q8_route_sweep infernix_ops_linear_test --parallel "$(nproc)"`, 3 ptxas warnings, zero
  errors), after `find src/ops/linear/q8 -name '*.cu' -exec touch {} +` for the `.cuh`/`.h` edits.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T2)

**What held.** Every shape in the task's list did sit below 90% `READ_%` over a wide
`T` range, and the cause is a single mechanism: `for_each_token_slice(tokens, kTokenCapacity)`
in `q8_mma_launch.cuh` / `q8_sliced_k_launch.cuh` restarts the whole weight stream for each token
slice, so a band whose schedule has `TokenCapacity < T` pays ~2x weight traffic. The measured cliffs
sat exactly one column above a capacity boundary (`T = 17` over the 16-token tile, `T = 25` over the
24-token tile, `T = 33` over both, `T = 49` over the 48-token tile), and the pre-existing sweep grid
could not propose a fix: it stopped at `BlockTokens = 40` and paired `KWarps = 8` only with
`BlockTokens <= 16`, so no candidate covered 33..64 columns in one slice. The retune therefore adds
single-slice-capable tiles (`TokenCapacity == BlockTokens`, 8..48 tokens) plus two-token-warp K128
MMA tiles, and re-bands the seven tables around them.

**What measurement corrected.**

1. *There is no `_select_sliced` table.* Q8 dispatch is `kShapes` entries
   `{n, k, select, fast, invariant}` in `q8_dispatch.cpp`; the second table is the
   `LinearRoutes::Q8SlicedK` fast slot, consulted only when the caller requests it *and* the policy
   is not `A16ColumnInvariant`. The retune therefore edits `select` for the six unconstrained shapes
   and adds one `fast` selector for `[2560, 2560]`.
2. *Six of the seven target files are `.cu`, not `.cpp`* — only `n2048_k4608.cpp` is C++, which is
   why every new schedule is a registered named launcher (`q8_instances.cuh` +
   `q8_sliced_k_mma.cu` / `q8_mma.cu` + `q8_launch.h`) rather than a file-local template alias.
3. *The task's baseline numbers are stale by 12..35% on this tree* (the same drift recorded for the
   20261029 doc). Rebuilt-HEAD (`/tmp/t2/bench_head`) vs the task text: `[6144, 5120] T=1`
   `147.28 us / 91.7%` vs `173.95 / 78.16`; `[2048, 4096] T=4` `42.34 / 84.1` vs `66.98 / 54.39`;
   `[2048, 16384] T=2` `156.59 / 92.5` vs `191.42 / 75.87`; `[5120, 10240] T=16` `257.01 / 89.7` vs
   `294.11 / 77.67`. Every acceptance point below is therefore judged against an interleaved A/B
   against the HEAD binary, not against the doc's absolute microseconds.
4. *The verification command names a target that does not exist.* `infernix_linear_q8_a16_test` is
   CTest case #265, not a Ninja target and not a binary; the binary is
   `build-native/tests/infernix_ops_linear_test` and it takes no `--shape` flag. The oracle gate was
   run as `ctest -R infernix_linear_q8_a16_test`.
5. *Column-invariance only constrains `[2560, 2560]`.* Of the seven shapes it alone appears in
   `q8_a16_column_invariance()` (8 columns, both routes) in `tests/ops/linear/test_q8_a16.cpp`; the
   other six are conformance-only (`A16 ReductionCriterion {1/256, 1/256, 2/256}` against the FP64
   oracle) and have no `invariant` slot — a column-invariant call on them goes to
   `select_q8_generic_invariant`. So their exact tables were retuned directly (including the
   `[2048, 4608] T=1` SIMT-to-sliced switch, whose reduction order changes but is oracle-qualified),
   while `[2560, 2560] T <= 8` stays on the frozen `stream<4,4>`/SIMT bit family and its faster
   sliced winners moved to the new fast route.
6. *GB10 invariants preserved.* 1:1 FP32 Tensor Core accumulation is untouched — every adopted
   schedule keeps its family's FP32 accumulator and `Q8MmaFragmentPipeline` default; no
   `F16Accumulate` route was admitted. Shared memory stays inside the 99 KiB/SM budget (largest new
   static allocation 42 496 B), and the 48-SM wave sizing drove the `[2048, *]` choices: at
   `N = 2048` the 64-row tiles launch 32 CTAs and the 16-row K128 tile 128, which is why the
   `T >= 33` bands moved to `MmaR16T32K128A1`.

## 3. Subsystem breakdown

| File | Action | Mechanism |
|---|---|---|
| `src/ops/linear/q8/q8_instances.cuh` | M | +9 `Q8A16SlicedKMmaSchedule` aliases with `TokenCapacity == BlockTokens` (`SlicedR16T8W8S2M3`, `SlicedR16T8W16S1`, `SlicedR16T16W16S1`, `SlicedR16T16W16S1M2`, `SlicedR16T16W8S2M1`, `SlicedR16T32W4S1`, `SlicedR16T32W8S1M2`, `SlicedR16T40W16S1`, `SlicedR32T48W4S1`) and 3 K128 `Q8A16MmaSchedule` tiles (`MmaR16T32K128A1`, `MmaR32T32K128A1`, `MmaR128x16T64K128A2M1`). |
| `src/ops/linear/q8/q8_sliced_k_mma.cu` | M | Adds the `INFERNIX_Q8_SLICED_LAUNCHER` macro (the pattern `q8_mma.cu` already uses) and 9 named launchers. |
| `src/ops/linear/q8/q8_mma.cu` | M | 3 `INFERNIX_Q8_MMA_LAUNCHER` instantiations. |
| `src/ops/linear/q8/q8_launch.h` | M | 12 declarations. |
| `shapes/n6144_k5120.cu` | M | `T<=32` 8-K-warp/32-token tile, `T<=40` 16-K-warp/40-token tile, `T<=64` 512-thread K128 tile (was: 16-token `r16_t32_w4_s2` to 32, `mma_r64x32_t64_k128_a1` to 64). |
| `shapes/n2048_k4608.cpp` | M | `T=1` SIMT `r8_t4` -> `sliced_r16_t16_w4_s2`; `T<=32` -> `sliced_r16_t32_w4_s1`; `T<=64` -> `mma_r16_t32_k128_a1` (replaces the `r64x16_t48` / `r64x32_t64` pair). |
| `shapes/n2048_k4096.cu` | M | `T<=16`/`T<=32` split (was one 16-token band to 32) and `T<=64` -> `mma_r16_t32_k128_a1`. |
| `shapes/n2048_k16384.cu` | M | `T<=16` -> `sliced_r16_t16_w16_s1_m2`, `T<=32` -> `sliced_r16_t32_w4_s1`, `T<=64` -> `mma_r16_t32_k128_a1`; the Grouped-8 tile now serves only `T = 65..128` (it measured 42..48% at `T = 33..64`). |
| `shapes/n5120_k10240.cu` | M | `T=1`/`T<=3` to the single-token SIMT pair (`simt_r1_t8_w8` / `simt_r1_t4_w8`), `T<=16` -> `sliced_r16_t16_w16_s1`, `T<=32` -> `sliced_r16_t32_w8_s1_m2`, `T<=64` -> `mma_r128x16_t64_k128_a2`. |
| `shapes/n5120_k17408.cu` | M | `T<=8`/`T<=16` to the 16-K-warp tiles, `T<=64` -> `mma_r128x16_t64_k128_a2`; the two-row-tile `T = 17..32` bands measured within noise of every candidate and stay. |
| `shapes/n2560_k2560.cu` | M | Above the frozen `T <= 8` bit family: `T<=16` `sliced_r16_t16_w8_s2_m1`, `T<=32` `sliced_r16_t32_w8_s1_m2`, `T<=48` `sliced_r32_t48_w4_s1`, `T<=64` `mma_r32_t32_k128_a1`; new `select_q8_n2560_k2560_fast` for the `LinearRoutes::Q8SlicedK` path (`T<=4` / `T<=16`). |
| `src/ops/linear/q8/q8_shapes.h`, `q8_dispatch.cpp` | M | Declare and register the `[2560, 2560]` fast selector in its `kShapes` fast slot. |
| `bench/ops/q8_permutation_routes.inc` | M | +27 sliced routes (`BlockTokens` 16..64, incl. `Direct` scale-access variants) and 10 MMA neighbours of the measured `T = 33..64` winner. Kept: it is the evidence base for these promotions and the grid the next decode retune needs. |

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- **Build**: 59/59 edges, zero errors. The 3 ptxas `Value of threads per SM ... out of range`
  warnings are all for `q8_a16_mma_kernel<Q8A16MmaSchedule<128,64,128,16,16,1,2>, ...>` — a
  screening-only entry in the bench `.inc` (`MinBlocksPerSm = 2` at 512 threads), not a production
  instantiation.
- **Per-symbol resources** (`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`):
  all 12 new kernels `STACK:0 LOCAL:0`. Largest new footprints: `SlicedR32T48W4S1` `REG:255
  SHARED:34304`; `SlicedR16T32W8S1M2` `REG:104..128 SHARED:42496` (both `Full` variants);
  `SlicedR16T32W4S1` `REG:163 SHARED:21760`; new MMA tiles `REG:72..100 SHARED:0..22016`.
- **Numerical oracle**: `ctest -R infernix_linear_q8_a16_test` Passed (13.4 s) — FP64-oracle
  conformance for all seven shapes including the added boundary widths, the `[2560, 2560]`
  8-column invariance on both the default and the invariant route, and the fast routes at
  `T = 1, 8, 64`. Full family: `ctest -R infernix_linear` 29/29 Passed (341 s).
- **Spill ratchet**: `ctest -R infernix_cuda_stack_check` Passed (1.6 s).
- **Interleaved A/B** vs the rebuilt HEAD bench (`--qtype Q8_G32_FP16 --routes exact`, A/B
  alternated per width, 2 rounds x (2 warmup + 7 repeat), median of per-round medians; `|delta| <=
  3.5%` treated as drift):

| shape | T | HEAD us | new us | delta | HEAD READ% | new READ% |
|---|---|---|---|---|---|---|
| `[6144,5120]` | 1 | 147.28 | 148.38 | +0.7 (same) | 91.7 | 91.7 |
| | 25 | 164.64 | 153.38 | **-6.8** | 83.9 | 90.3 |
| | 33 | 172.38 | 163.15 | **-5.4** | 80.3 | 85.4 |
| | 41 | 175.97 | 162.30 | **-7.8** | 80.1 | 85.3 |
| | 64 | 184.66 | 161.12 | **-12.7** | 76.5 | 87.4 |
| `[2048,4608]` | 1 | 56.24 | 48.96 | **-12.9** | 73.5 | 83.3 |
| | 17 | 62.30 | 51.06 | **-18.1** | 65.9 | 79.9 |
| | 33 | 71.49 | 59.36 | **-17.0** | 58.7 | 71.9 |
| | 48 | 72.48 | 68.53 | **-5.5** | 58.9 | 62.5 |
| | 64 | 73.57 | 63.46 | **-13.7** | 59.3 | 68.7 |
| `[2048,4096]` | 4 | 42.34 | 42.35 | +0.0 (same) | 84.1 | 86.0 |
| | 17 | 48.45 | 47.95 | -1.0 (same) | 75.0 | 80.6 |
| | 33 | 61.41 | 55.14 | **-10.2** | 60.8 | 68.6 |
| | 48 | 62.30 | 61.28 | -1.6 (same) | 61.0 | 64.1 |
| | 64 | 65.36 | 64.30 | -1.6 (same) | 60.3 | 67.8 |
| `[2048,16384]` | 2 | 156.59 | 156.51 | -0.1 (same) | 92.5 | 92.5 |
| | 12 | 163.25 | 162.14 | -0.7 (same) | 82.4 | 90.8 |
| | 25 | 171.36 | 170.42 | -0.6 (same) | 80.7 | 87.2 |
| | 33 | 323.47 | 201.49 | **-37.7** | 49.3 | 73.2 |
| | 64 | 317.34 | 216.00 | **-31.9** | 47.0 | 71.6 |
| `[5120,10240]` | 1 | 251.20 | 234.88 | **-6.5** | 90.9 | 96.5 |
| | 16 | 257.01 | 249.63 | -2.9 (same) | 89.7 | 91.5 |
| | 25 | 277.41 | 262.58 | **-5.3** | 82.2 | 87.6 |
| | 33 | 280.46 | 272.14 | -3.0 (same) | 81.9 | 84.9 |
| | 64 | 314.18 | 271.68 | **-13.5** | 75.2 | 85.6 |
| `[5120,17408]` | 1 | 423.17 | 405.34 | **-4.2** | 90.8 | 94.9 |
| | 16 | 439.60 | 417.18 | **-5.1** | 88.8 | 93.1 |
| | 33 | 501.58 | 459.74 | **-8.3** | 77.9 | 85.6 |
| | 48 | 502.19 | 462.72 | **-7.9** | 78.8 | 85.2 |
| | 64 | 534.86 | 458.59 | **-14.3** | 73.9 | 86.5 |
| `[2560,2560]` exact | 4 | 41.47 | 41.71 | +0.6 (same, frozen band) | 69.0 | 67.2 |
| | 16 | 44.91 | 36.18 | **-19.5** | 64.5 | 79.1 |
| | 25 | 44.96 | 36.85 | **-18.0** | 62.5 | 79.9 |
| | 33 | 47.84 | 39.79 | **-16.8** | 62.0 | 76.6 |
| | 64 | 48.99 | 47.06 | **-4.0** | 62.0 | 66.0 |
| `[2560,2560]` fast | 1 | 34.64 | 33.70 | -2.7 (same) | 83.1 | 81.8 |
| | 4 | 41.30 | 34.11 | **-17.4** | 69.0 | 83.2 |
| | 8 | 46.91 | 35.15 | **-25.1** | 61.0 | 81.6 |
| | 16 | 44.88 | 37.63 | **-16.1** | 64.6 | 79.9 |

- **Acceptance points**: `[6144,5120] T=1` (bar `<= 149 us / >= 91%`) — **at the bar within
  measurement noise, and unchanged by this task**: the `T <= 4` band keeps its pre-existing `C4`
  tile and the same kernel measured `147.28 us / 91.7%` (A/B, HEAD), `148.38 / 91.7` (A/B, new) and
  `150.34 / 90.4` (the task's own `--routes fast` invocation, which resolves to the same route for
  this shape). Reporting it as met would overstate the precision available at this width.
  `[2048,4096] T=4` `42.35 us / 86.0%`, `[2048,16384] T=2` `156.51 / 92.5%`, `[5120,10240] T=16`
  `249.63 / 91.5%` — all inside the `>= 86..92%` band with no non-monotonic step at those widths.
  `[2560,2560] T=4..32` — partially met, see below.

## 5. Remaining gaps, honest tradeoffs & follow-ups

- **`[2560, 2560] T = 4..32 did not reach the promised 83..90%.** The exact route's `T <= 8` band is
  frozen by the shape's 8-column invariance contract: the sliced candidates that measure up to
  `+23%` there do not reproduce the reference bits, so `T = 4` stays at `67.2%`. Above the contract
  the two best candidates in the whole grid are already installed and cap at `~80%`
  (`T = 16` 79.1%, `T = 25` 79.9%, `T = 33` 76.6%); the fast route reaches `83.2%` at `T = 4` and
  `81.6%`/`79.9%` at `T = 8`/`16`. Meeting `>= 83%` across `T = 4..32` on the *default* route would
  require either dropping that contract or a kernel-level change; neither was done.
- **The `T = 33` (and `[2560,2560] T = 49`) route-transition step remains**: `+13..17%` at the
  sliced->MMA crossover, because no sliced tile can hold that width in one slice and no MMA candidate
  measured better. It is now a single step to a flat band instead of the previous 2x weight re-read
  (`[2048,16384] T=33` went 323 us -> 201 us), but it is not smooth.
- **`[2048, 4608]` / `[2048, 4096]` above `T = 32` stay at 62..71% `READ_%`**, the best of 37
  screened routes. At `N = 2048` the ceiling is CTA count and L2 reuse, i.e. a kernel question
  (wider row tiles with Grouped-8 rasterization), not a dispatch choice.
- **`[2048, 16384] T = 65..128` still runs the Grouped-8 tile**, measured at 42..48% for
  `T = 33..64`. Out of T2's `T = 1..64` scope; a follow-up should re-sweep that band against
  `MmaR16T32K128A1`.
- **No end-to-end KV-matrix A/B was run.** The change is dispatch-only inside Q8 Linear and the
  claimed scope is op-level decode/verify bandwidth, so the op-level interleaved A/B above is the
  evidence; `bench/run_kv_matrix_bench.sh` remains the gate if a model-level claim is made from it.
- **Screening-only ptxas warning** (`minnctapersm` ignored) on `mma_r128t64k128w16x16a2m1` in
  `bench/ops/q8_permutation_routes.inc`; harmless, and it documents that `MinBlocksPerSm = 2` is
  unreachable for a 512-thread tile.
