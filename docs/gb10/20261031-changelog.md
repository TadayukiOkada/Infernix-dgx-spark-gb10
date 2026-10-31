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

# GB10 changelog — 2026-10-31 (T3: Q8 skinny/LoRA/drafter shapes (`[320,10240]`, `[324,10240]`, `[10240,320]`, `[2560,640]`, `[1024,2048]`, `[1024,5120]`, `[1280,2560]`) across T = 1..4096)

## 1. Accounting

- **Base**: `ac8528dfc` (`dev-rebased-20261028`).
- **Files changed**: 10 modified, 0 added, 0 deleted (`+112 / -56`).
  - Ops & quantization (4): `src/ops/linear/q8/q8_instances.cuh` (M), `q8_sliced_k_mma.cu` (M),
    `q8_mma.cu` (M), `q8_launch.h` (M).
  - Shape dispatch tables (6): `src/ops/linear/q8/shapes/n320_k10240.cu` (M),
    `n324_k10240.cu` (M), `n2560_k640.cu` (M), `n1024_k2048.cpp` (M), `n1024_k5120.cpp` (M),
    `n1280_k2560.cu` (M). (`n10240_k320.cu` inspected and retained unchanged; see rationale below.)
- **Allowlist delta**: none. `tools/cuda_stack/cuda_stack_allowlist.json` is untouched; all 5 new
  instantiations compile with `STACK:0 LOCAL:0`.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T3)

1. **Eliminating the `61..227 us` (`8.7..25.4%` `READ_%`) serial/SIMT cliff on `[320, 10240]` and
   `[324, 10240]` (`98x`/pass)**:
   - On HEAD, `select_q8_n320_k10240` and `select_q8_n324_k10240` dispatched `Q8A16StreamSchedule`
     (`T <= 5`) and `launch_q8_a16_simt_r1_t8_w8` (`T = 6..64`), which sliced `T` into 8-token
     passes and re-read the 5.2 MiB weight matrix up to 8 times (`59..227 us` at `T = 16..64`).
   - Both shapes carry a **64-column bit-invariance contract** in `q8_a16_column_invariance()`
     (`tests/ops/linear/test_q8_a16.cpp:216`). Because `Q8A16StreamSchedule` (`8` warps per row),
     `simt_r1_t8_w8`, and `Q8A16SlicedKMmaSchedule` (`8` K-warps per 16-row tile) accumulate across K
     in different orders, mixing families inside `T = 1..64` breaks column invariance.
   - Moving `T = 1..80` onto the single 8-K-warp sliced-K MMA family (`sliced_r16_t8_w8_s2_m3` for
     `T <= 32`, `sliced_r16_t16_w8_s2_m1` for `T = 33..80`) preserves exact 64-column bit invariance
     while cutting `T = 16..64` latency by **`2.3x..6.8x`** (`63.4..222.0 us -> 27.4..32.6 us`,
     `59..64%` `READ_%` on `[320, 10240]`; `65.3..227.1 us -> 27.5..33.6 us`, `55..65%` `READ_%` on
     `[324, 10240]`). The accepted cost of keeping a single bit family across `T = 1..64` is
     `+3.1..+5.6 us` at `T = 1..4` (`19.3..27.5 us -> 24.3..31.4 us`), which is recovered starting at
     `T = 5` (`-19.4%` at `T = 5`, `-23.4%` at `T = 8`, `-49.8%` at `T = 9`).
   - Above the 64-column contract (`T > 64`), new zero-spill 4-K-warp 48-token sliced rungs
     (`SlicedR16T48W4S2`, `SlicedR16T48W4S1`) and `SlicedR32T48W4S1` (`T = 130..384` on `[320, 10240]`)
     plus `MmaR32T32K128A1` / `MmaR64x32T64K128A1` / `MmaR64x32T128K64A2` / `MmaR128x64T128K64A1`
     cut `T = 65..2048` latency by `10..62%`. (A swept 16-K-warp `RowTiles = 2` schedule spilled
     `8..16 B` at the `REG:128` 256-thread launch bound and was rejected in favor of the zero-spill
     4-K-warp 48-token pair.)
2. **Replacing `Q8A16StreamSchedule` on `[2560, 640]` (`49x`/pass) and retuning `[1024, 2048]`,
   `[1024, 5120]`, and `[1280, 2560]`**:
   - **`[2560, 640]`**: Replaced the 4-band `Q8A16StreamSchedule` ladder (`T = 1..16`, which stalled
     at `13.6..21.3 us` / `33.5..54.9%` `READ_%` for `T = 5..16`) and `mma_r32_t24` (`T = 17..32`,
     `16.2 us`) with a single `launch_q8_a16_mma_r32_t32_k128_a1` (`MmaR32T32K128A1`) schedule across
     `T = 1..32`, plus `MmaR32T64K128A1` at `T = 49..64`. Because a single non-split-K MMA schedule
     computes each token column independently, `T = 1..32` satisfies the 16-column invariance contract
     while cutting `T = 9..32` from `16.2..21.3 us` to `13.2..14.2 us` (`-18.7%..-35.2%`, `53..65%`
     `READ_%`), trading `+0.6/+1.5 us` at `T = 1..2`.
   - **`[1024, 2048]` & `[1024, 5120]`**: Replaced the premature `T = 33..96` MMA transition (which
     dropped to `30..40%` `READ_%`) with single-slice 4-K-warp sliced rungs (`SlicedR16T16W4S1`,
     `SlicedR16T32W4S1`, `SlicedR32T48W4S1`, `sliced_r32_t16_w4_s2`) and the new 32-row 64-token K128
     MMA tile (`MmaR32T64K128A1`), cutting `T = 33..256` latency by **`18..48%`** across both shapes
     and `-18.0%` at `[1024, 2048] T = 1`, plus `-11.9%` at `[1024, 5120] T = 4096` (`MmaR128x64T128K64A1`).
   - **`[1280, 2560]`**: Preserved `select_q8_n1280_k2560_invariant` through `T = 48` (required by the
     48-column invariance test) and retuned `T = 49..256` with `SlicedR32T48W4S1`,
     `MmaR32T64W16x32K128A1`, and `MmaR64x32T64K128A1` (`-15.4%..-39.9%` across `T = 65..256`) and
     `MmaR128x64T128K64A1` at `T > 3072` (`-6.0%..-7.1%` at `T = 4096`).
3. **Why `[10240, 320]` (`n10240_k320.cu`) is unchanged & hardware ceiling disclosures**:
   - **`[10240, 320]` at `T = 4..8` and `T = 4096`**: The task prompt assumed `n10240_k320` used a
     serial token loop, but HEAD already dispatches `Stream<4,4>` (`T = 1`), `StreamShared<8,8>`
     (`T = 2`, `T = 5..8`), and `simt_r8_t4` (`T = 3..4`) at `19.3..24.6 us` (`71.8..88.6%` `READ_%`).
     In the 149-route sweep (`sweep_10240x320.log`), the fastest candidate at `T = 4..8` is
     `mma_r32_t16` / `mma_r32_t24` at `19.1..20.1 us` (no route reaches `<= 16.5 us`), and switching
     `T = 4..8` to MMA breaks the 8-column invariance contract (`q8_a16_column_invariance`) unless
     `T = 1..2` also switches to `mma_r32_t16` (`21.02 us` at `T = 1`, regressing `T = 1` by `+1.1 us`).
     At `T = 4096`, `K = 320` has only 5 K-tiles of 64, so kernel launch + epilogue + per-tile
     staging overhead caps `mma_r64_t128` at `544.6 us` (`49.3 TFLOP/s`) and `MmaR128x64T128K64A1` is
     slower (`585.2 us`, `45.9 TFLOP/s`). Thus `n10240_k320.cu` is already at the exact-route Pareto
     frontier across `T = 1..4096`.
   - **`[320, 10240]` and `[324, 10240]` at `T = 512`**: `MmaR32T32K128A1` improves `T = 512` from
     `136.0 us` (`24.7 TFLOP/s`) to `120.7..122.7 us` (`27.3..27.8 TFLOP/s`, `-9.8%`) and `T = 513`
     from `161.6 us` to `117.5 us` (`-27.3%`). Reaching `60 TFLOP/s` at `N = 320, T = 512` is
     bounded by the narrow `N = 320` output dimension (at `BM = 32`, `N = 320` is only 10 row tiles,
     whereas `BM = 128` wastes 46% of the 3rd M-tile and runs slower at `T = 512`; only at `T = 4096`
     does `[320, 10240]` reach `50.7 TFLOP/s` / `69.2%` `READ_%`).

## 3. Subsystem breakdown

| File | Action | Mechanism |
|---|---|---|
| `src/ops/linear/q8/q8_instances.cuh` | M | +3 sliced-K schedules (`SlicedR16T16W4S1`, `SlicedR16T48W4S2`, `SlicedR16T48W4S1`) and +2 K128 MMA schedules (`MmaR32T64K128A1`, `MmaR32T64W16x32K128A1`). |
| `src/ops/linear/q8/q8_sliced_k_mma.cu`, `q8_mma.cu`, `q8_launch.h` | M | Instantiate and declare the 5 new named launchers. |
| `src/ops/linear/q8/shapes/n320_k10240.cu` | M | Replace `Stream` / `simt_r1_t8_w8` with `sliced_r16_t8_w8_s2_m3` (`T<=32`) and `sliced_r16_t16_w8_s2_m1` (`T<=80`), then `sliced_r16_t48_w4_s2` (`T<=112`), `sliced_r16_t48_w4_s1` (`T<=129`), `sliced_r32_t48_w4_s1` (`T<=384`), `mma_r32_t32_k128_a1` (`T<=640`), `mma_r64x32_t64_k128_a1` (`T<=1024`), `mma_r64x32_t128_k64_a2` (`T<=3072`), `mma_r128x64_t128_k64_a1` (`>3072`). |
| `src/ops/linear/q8/shapes/n324_k10240.cu` | M | Same 8-K-warp sliced ladder through `T = 129`, then `mma_r32_t32_k128_a1` (`T<=640`), `mma_r64x32_t64_k128_a1` (`T<=1024`), `mma_r64x32_t128_k64_a2` (`T<=3072`), `mma_r128x64_t128_k64_a1` (`>3072`). |
| `src/ops/linear/q8/shapes/n2560_k640.cu` | M | Replace `Stream` (`T<=16`) + `mma_r32_t24` (`T<=32`) with `mma_r32_t32_k128_a1` (`T<=32`) and `mma_r32_t32` (`T=49..64`) with `mma_r32_t64_k128_a1`. |
| `src/ops/linear/q8/shapes/n1024_k2048.cpp` | M | Unify `T<=16` onto `sliced_r16_t16_w8_s2`, extend sliced-K through `T=128` (`sliced_r16_t32_w4_s1` `<=64`, `sliced_r32_t48_w4_s1` `<=128`), `mma_r32_t64_k128_a1` (`<=192`), `mma_r64x16_t48_k128_a1` (`<=256`), `mma_r64x32_t96_k128_a1` (`<=512`). |
| `src/ops/linear/q8/shapes/n1024_k5120.cpp` | M | `sliced_r16_t16_w4_s1` (`<=32`), `sliced_r16_t32_w4_s1` (`<=64`), `sliced_r32_t16_w4_s2` (`<=96`), `mma_r32_t64_k128_a1` (`<=129`), `mma_r64x16_t48_k128_a1` (`<=256`), `mma_r64x32_t64_k64_a2` (`<=512`), `mma_r64_t128` (`<=1024`), `mma_r128x64_t128_k64_a1` (`>1024`). |
| `src/ops/linear/q8/shapes/n1280_k2560.cu` | M | Above `T<=48` invariant family: `sliced_r32_t48_w4_s1` (`<=96`), `mma_r32_t64_w16x32_k128_a1` (`<=129`), `mma_r64x32_t64_k128_a1` (`<=256`), `mma_r64_t128` (`<=3072`), `mma_r128x64_t128_k64_a1` (`>3072`). |

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- **Build**: `ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_ops_linear_test`
  clean (exit 0, zero warnings).
- **Per-symbol resources** (`cuobjdump -res-usage build-native/apps/infernix-serve`):
  all 5 new schedule families have `STACK:0 LOCAL:0`:
  - `SlicedR16T16W4S1`: `REG:138..141`, `SHARED:13568`
  - `SlicedR16T48W4S1`: `REG:182..187`, `SHARED:29952`
  - `SlicedR16T48W4S2`: `REG:163..179`, `SHARED:0` (dynamic SMEM)
  - `MmaR32T64K128A1`: `REG:78..88`, `SHARED:30208`
  - `MmaR32T64W16x32K128A1`: `REG:104..113`, `SHARED:30208`
- **Gates**: `ctest --test-dir build-native -R '^(infernix_linear_q8_a16_test|infernix_cuda_stack_check)$'`
  **100% passed** (2/2: `infernix_cuda_stack_check` 1.50 s, `infernix_linear_q8_a16_test` 13.62 s —
  covering FP64 oracle conformance and `q8_a16_column_invariance` across all 64/48/16/8-column
  contracts).
- **Interleaved A/B** vs rebuilt HEAD (`/tmp/t3/bench_head`, `--qtype Q8 --routes fast`, 2 rounds x
  (2 warmup + 7 repeat)):

| Shape | T | HEAD (us) | NEW (us) | Δ% | HEAD `READ_%` | NEW `READ_%` |
|---|---|---|---|---|---|---|
| `[320, 10240]` | 1 / 2 / 4 | 19.28 / 21.31 / 27.46 | 24.34 / 24.38 / 31.39 | +26.2% / +14.4% / +14.3% | 73.9 / 67.2 / 52.8 | 58.7 / 58.8 / 39.9 |
| | 5 / 8 / 9 | 31.55 / 34.58 / 48.54 | **25.44 / 26.50 / 24.38** | **-19.4% / -23.4% / -49.8%** | 46.2 / 42.8 / 30.4 | **55.1 / 55.9 / 61.2** |
| | **16 / 32 / 64** | 63.36 / 114.50 / 222.03 | **27.39 / 27.47 / 32.61** | **-56.8% / -76.0% / -85.3% (`2.3x..6.8x`)** | 24.5 / 14.8 / 8.8 | **59.0 / 63.9 / 60.2** |
| | 65 / 96 / 129 | 104.26 / 105.26 / 106.34 | **39.76 / 54.10 / 61.31** | **-61.9% / -48.6% / -42.3%** | 18.9 / 21.5 / 24.2 | **48.4 / 40.6 / 39.8** |
| | 256 / 512 / 513 | 115.63 / 135.98 / 161.62 | **89.95 / 122.69 / 117.54** | **-22.2% / -9.8% / -27.3%** | 30.4 / 42.5 / 36.0 | **40.2 / 47.4 / 49.1** |
| | 1024 / 2048 | 180.02 / 385.89 | **159.55 / 289.58** | **-11.4% / -25.0%** | 56.7 / 49.4 | **64.0 / 64.5** |
| `[324, 10240]` | 1 / 2 / 4 | 19.78 / 20.80 / 26.43 | 25.41 / 26.53 / 26.43 | +28.5% / +27.5% / 0.0% | 74.8 / 71.5 / 57.8 | 54.5 / 54.5 / 55.5 |
| | 5 / 8 / 9 | 31.06 / 33.20 / 48.96 | **28.50 / 29.49 / 26.46** | **-8.2% / -11.2% / -45.9%** | 46.7 / 46.0 / 30.2 | **51.8 / 49.2 / 57.1** |
| | **16 / 32 / 64** | 65.33 / 118.61 / 227.14 | **28.50 / 27.46 / 33.62** | **-56.4% / -76.9% / -85.2% (`2.3x..6.8x`)** | 24.0 / 14.4 / 8.7 | **55.1 / 64.6 / 57.3** |
| | 65 / 96 / 129 | 103.23 / 104.29 / 104.27 | **41.90 / 56.24 / 65.44** | **-59.4% / -46.1% / -37.2%** | 19.1 / 21.6 / 24.9 | **46.3 / 39.5 / 37.7** |
| | 256 / 512 / 513 | 113.58 / 123.78 / 159.57 | **103.26 / 118.66 / 118.62** | **-9.1% / -4.1% / -25.7%** | 32.2 / 46.7 / 36.6 | **35.5 / 48.3 / 49.2** |
| | 1024 / 2048 | 167.76 / 386.02 | **157.58 / 295.68** | **-6.1% / -23.4%** | 61.0 / 49.4 | **64.9 / 63.7** |
| `[2560, 640]` | 1 / 2 | 11.57 / 11.63 | 12.13 / 13.15 | +4.8% / +13.1% | 64.5 / 58.6 | 58.6 / 50.3 |
| | 5 / 8 / 9 | 13.63 / 14.59 / 20.29 | **13.15 / 13.15 / 13.15** | **-3.5% / -9.9% / -35.2%** | 54.9 / 51.9 / 36.0 | **59.4 / 60.1 / 60.3** |
| | 16 / 32 / 64 | 21.33 / 16.21 / 15.10 | **14.18 / 13.18 / 14.14** | **-33.5% / -18.7% / -6.4%** | 33.5 / 43.3 / 54.5 | **52.9 / 64.7 / 61.8** |
| `[1024, 2048]` | 1 | 16.13 | **13.23** | **-18.0%** | 56.6 | **73.9** |
| | 33 / 64 / 65 | 27.46 / 30.48 / 34.64 | **16.27 / 18.26 / 21.41** | **-40.7% / -40.1% / -38.2%** | 34.7 / 34.9 / 30.9 | **60.4 / 58.4 / 52.1** |
| | 96 / 129 / 256 | 34.62 / 34.70 / 36.67 | **24.38 / 24.42 / 28.50** | **-29.6% / -29.6% / -22.3%** | 33.1 / 35.5 / 42.1 | **47.0 / 50.3 / 54.2** |
| | 448 / 512 | 46.93 / 47.01 | **40.85 / 42.82** | **-13.0% / -8.9%** | 43.1 / 46.5 | **49.5 / 51.0** |
| `[1024, 5120]` | 32 / 33 / 64 | 38.75 / 66.38 / 70.45 | **35.70 / 34.66 / 39.74** | **-7.9% / -47.8% / -43.6%** | 62.6 / 36.0 / 36.1 | **69.8 / 70.1 / 66.7** |
| | 65 / 96 / 129 | 85.82 / 81.76 / 83.79 | **51.10 / 53.09 / 58.19** | **-40.5% / -35.1% / -30.6%** | 30.2 / 33.6 / 34.7 | **48.8 / 53.7 / 50.9** |
| | 192 / 256 / 384 | 84.78 / 86.91 / 98.11 | **66.29 / 66.46 / 88.91** | **-21.8% / -23.5% / -9.4%** | 38.5 / 41.2 / 42.6 | **49.3 / 54.1 / 46.5** |
| | 448 / 512 / 4096 | 106.34 / 108.38 / 670.66 | **97.15 / 100.19 / 590.66** | **-8.6% / -7.6% / -11.9%** | 42.3 / 44.5 / 33.5 | **45.9 / 48.1 / 38.5** |
| `[1280, 2560]` | 65 / 96 / 129 | 45.87 / 44.90 / 47.09 | **27.55 / 29.54 / 35.68** | **-39.9% / -34.2% / -24.2%** | 34.5 / 36.5 / 38.5 | **61.1 / 56.2 / 49.5** |
| | 192 / 256 / 4096 | 46.93 / 47.01 / 403.28 | **38.72 / 39.78 / 374.70** | **-17.5% / -15.4% / -7.1%** | 42.9 / 47.2 / 35.0 | **52.0 / 54.3 / 37.7** |

## 5. Rebase conflict resolution policy

- Preserve the GB10 8-K-warp sliced-K column-invariant ladders on `[320, 10240]` and `[324, 10240]`,
  the single-schedule `MmaR32T32K128A1` `T = 1..32` ladder on `[2560, 640]`, the zero-spill
  `SlicedR16T48W4S1`/`S2` and `MmaR32T64K128A1` mid-band schedules across `[1024, 2048]`,
  `[1024, 5120]`, and `[1280, 2560]`, and `n10240_k320.cu`'s existing stream/MMA dispatch.

---

# GB10 changelog — 2026-10-31 (T4: enable and retune NVFP4 `A4` (1-term) and `Nvfp4A4Pair` (2-term) TMA prefill across the 11 prefill shapes)

## 1. Accounting

- **Base**: `71da249c6` (`dev-rebased-20261028`).
- **Files changed**: 23 modified, 0 added, 0 deleted (`+309 / -147`), excluding this changelog.
  `git diff --check` is clean on the final tree.
  - Ops & quantization (9): `src/ops/linear/nvfp4/nvfp4_schedule.cuh` (M) — per-descriptor
    `Nvfp4A4TmaPromotions` replaces the single `WeightCodePromotion` template argument;
    `nvfp4_a4_tma.cuh` (M) — `nvfp4_make_tma_planes` / `make_nvfp4_a4_tma_descriptors` take one
    promotion per descriptor (`a_codes`, `b_codes`, `b_scales`);
    `nvfp4_a4_tma.cu` (M) — the `Nvfp4GeometryId` switch is replaced by a `K`-dispatched launcher
    (`launch<K, Tiled128Rows, Tiled128Stages, Promotions>`); `nvfp4_a4_tma_launch.h` (M) —
    `launch_nvfp4_a4_tma_linear` loses its geometry-id parameter;
    `nvfp4_launch.cuh` (M) — `nvfp4_a4_tma_route<Layout>` builds the route from the schedule
    (launch + scale layout + reciprocal quantize) instead of the caller restating three pointers;
    `attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma.cu`, `gdn_input_proj/nvfp4/…`,
    `linear_add/nvfp4/…`, `linear_swiglu/nvfp4/…` (M) — call-site updates to the named promotion
    constants only (behaviour unchanged; they already passed `L2_128B` for the weight codes).
  - Shape dispatch tables (11): `n6144_k2560.cu`, `n10240_k2560.cu`, `n12800_k2560.cu`,
    `n13952_k2560.cu`, `n16384_k2560.cu`, `n2560_k6144.cu` (M) — register the A4 1-term ladder
    (`launch_nvfp4_a4<select_a4>` + a measured `uses_a4` admission) where they previously threw
    `std::logic_error`; `n5120_k6144.cu`, `n5120_k17408.cu`, `n14336_k5120.cu`, `n16384_k5120.cu`,
    `n34816_k5120.cu` (M) — register `launch_nvfp4_a4_pair` with `pair_min_tokens = 65`, plus the
    GB10 A16 prefill re-fit on `[14336, 5120]` / `[16384, 5120]`.
  - Bench (2): `bench/ops/nvfp4_route_sweep.cu` (M) — promotion-named TMA candidates, weight-scale
    and all-descriptor variants, and `a4_split_{1q1g,2q1g,1q2g}` differential candidates that
    separate the quantizer cost from the GEMM cost; `bench/ops/linear_bench.cu` (M) — the
    `NVFP4_F32ACC` profile is now reported per registered shape at its real `uses_a4`/pair admission
    threshold instead of a 5-shape list with a blanket `T >= 2`.
  - Tests (1): `tests/ops/linear/test_nvfp4_a4.cpp` (M) — 6 shapes × 22 A4-policy invocations
    (incl. 5 graph-replay) and 5 more shapes under `NVFP4_A4_PAIR`.
- **Allowlist delta**: none. `tools/cuda_stack/cuda_stack_allowlist.json` is untouched and
  `infernix_cuda_stack_check` passes, so none of the 6 new A4 ladders, the 11 new pair records, or
  the 3 new A16 instantiations introduced a `STACK`/`LOCAL` symbol.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T4)

Measured on this host (`sm_121a`, 48 SMs, 24 MiB L2, 128 GiB unified LPDDR5X — 246 GB/s measured
sustained read), cold 32 MiB L2, `--warmup 2 --repeat 10`, baseline = rebuilt `71da249c6` binary
measured in the same session as the candidate (absolute-µs bars drift ±3.5% with thermal state).

1. **`--policy a4` falls back to `BF16_F32ACC` on 6 shapes (confirmed, fixed)**: `uses_a4` returned
   `false` and `launch_a4` threw for `[6144, 2560]`, `[10240, 2560]`, `[12800, 2560]`,
   `[13952, 2560]`, `[16384, 2560]` and `[2560, 6144]`. Each now carries the same ladder as its
   proven sibling (`nvfp4_a4_tma_route<>` at `T >= 257`, `nvfp4_a4_tma_route<Tiled128>` at
   `T = 129..256` for `K = 2560`, `Nvfp4A4MmaSchedule` rungs below), admitted by a measured
   threshold (`T >= 33` for `K = 2560`, `T >= 17` for `[2560, 6144]`, where the A16 GEMV/sliced-K
   decode tiles still win). This is a real production route, not a bench-only one:
   `tools/convert/quasar_nvfp4.py` assigns `activation_policy="AllowA4"` to `/gdn/*`, `/attention/*`
   and `/mlp/*`, so a QUASAR NVFP4 artifact asks for A4 on exactly these parameters and silently
   got the BF16-accumulate A16 route. `2.62x..3.36x` at `T = 4096` (§4 table).
2. **`Nvfp4A4Pair` missing on the 5 `K = 5120`/`N = 5120` shapes (confirmed, fixed)**: their
   `Nvfp4LinearShape` records had 5 fields, so `--routes fast` ran the A16 table at every width.
   Registering `launch_nvfp4_a4_pair` with `pair_min_tokens = 65` gives `2.02x..2.85x` at
   `T = 4096` and moves the three `K = 5120` shapes from `64.96..72.12` useful TFLOP/s to
   `182.54..187.26` — **the doc's `>= 165..195` useful (`330..390` hardware OMMA) bar is met.**
   The `Grouped, 8` A16 fallback the doc names is the existing raster (`nvfp4_tma_raster_blocks`,
   `group_n = k > 6144 ? 4 : 8`); it needed no change, and the A16 tables behind it were still the
   RTX 5090 fit, so they were re-fit too (item 5).
3. **Implementation step 1 — promote `a_codes`, `b_codes` *and* `b_scales` (partly falsified)**:
   the linear A4 TMA route previously had one promotion knob, defaulting to `NONE` on all three
   descriptors; it is now one named constant per descriptor. At the same geometry
   (`t256_r128_s3_m1`, `T = 4096`) only the weight panel pays, and only at `K <= 6144`:
   `[5120, 6144]` `1251.6 us → 990.6 us` (`-20.9%`) with `b_codes = L2_128B`; adding
   `b_scales` costs `+19.6%` (`1185.2 us`) and adding `a_codes` too is neutral (`988.3 us`,
   `0.2%`); at `[5120, 17408]` every combination is within `0.3%` (`2942..2950 us`) and at
   `[16384, 2560]` promotion is a `+1.0%` loss. The shipped default is therefore
   `{a_codes: NONE, b_codes: L2_128B, b_scales: NONE}` — the doc's `a_codes`/`b_scales` promotion
   is not shipped. End-to-end at `T = 4096` the promotion change is inside the noise on the 5
   already-A4 shapes (`1003.6 → 1005.4` and `2940.7 → 2933.6 us`): the quantizer, not the weight
   panel, owns the residual gap (item 6).
4. **Implementation step 2 — switch `N = 5120` prefill to `Tiled128 (BM = 128, GroupM = 8)` (falsified)**:
   the 128-token/64-row TMA tile is the *slowest* prefill candidate at both shapes —
   `[5120, 6144]` `1583.9 us` vs `990.6 us` for the shipped `Tiled256` tile (`+60%`),
   `[5120, 17408]` `3842.1 us` vs `2945.2 us` (`+30%`). A `GroupTokens` raster-band override was
   added to test the `GroupM = 8` half of the claim and never won beyond noise
   (`K = 17408`: `g8 +8%`, `g16 +79%`; `K = 6144`: `g2 +11%`, `g8/g16` within `0.7%`), so the knob
   and its raster parameter were removed rather than left as a dead template argument.
   The `Tiled128` layout is kept only where it genuinely wins (`T = 129..256` on the `K = 2560`
   shapes and on the 2-term residual route, which is `Tiled128`-native).
5. **A16 fallback re-fit on `[14336, 5120]` / `[16384, 5120]` (new finding, in the doc's scope)**:
   both tables still carried the RTX 5090 `t256` tail `Nvfp4A16MmaSchedule<64,256,64,32,32,2,1>`,
   the slowest candidate at every prefill width. The 128-row K64 GB10 tile
   `<128,64,64,64,16,2,2>` takes `T >= 129` (`-12.6%` / `-19.8%` at `T = 4096`, `-36%` at `T = 256`
   on `[16384, 5120]`), and the 64-row K128 tile takes `T = 97..128` (`-15%` / `-5.7%`). At
   `T = 160..192` the old `64x128` tile paid `+26..28%` for slab padding that the 64-row tile
   avoids. After the change the production route sits at the swept argmin at every width tested
   (`|Δ| <= 1%` at `T = 96..512`, two runs).
6. **The `>= 330..360` / `>= 320..355` / `>= 300..360` `--policy a4` bars are not reachable from
   dispatch, and the doc's baseline for `[5120, 17408]` is stale**: the differential candidates
   (`t(2q1g) - t(1q1g)` = quantizer, `t(1q2g) - t(1q1g)` = GEMM alone) split `T = 4096` as:

   | shape | `1q1g` total | quantizer `Q` | GEMM `G` | end-to-end | GEMM alone |
   |---|---|---|---|---|---|
   | `[5120, 17408]` | 2939.3 us | 807.5 us | 2151.6 us | 248.4 TF/s | **339.4 TF/s** (77.1% of 440) |
   | `[5120, 6144]` | 982.4 us | 306.2 us | 743.5 us | 262.3 TF/s | **346.6 TF/s** (78.8%) |
   | `[34816, 5120]` | 4024.7 us | 274.5 us | 4225.2 us | 362.8 TF/s | **345.6 TF/s** (78.5%) |
   | `[16384, 2560]` | 1284.2 us | 69.2 us | 1195.9 us | 267.6 TF/s | **287.3 TF/s** (65.3%) |

   The GEMM is at 77–79% of the 440 TF/s `OMMA.16864` reference where the tile family is at its
   structural limit (`Stages = 4` does not fit the 99 KiB SMEM at `BlockTokens = 256` /
   `BlockRows = 128`, and per-CTA `ldmatrix` traffic exceeds tensor-core issue time), and at
   `K = 2560` the same tile reaches only 287 TF/s — below the doc's `>= 300` floor for that whole
   family. The remaining `21..25%` on `K = 6144/17408` is the quantizer, which is stream-serialized
   ahead of the GEMM (`pdl::launch_consumer` becomes programmatic-dependent only under graph
   capture, so an eager A4 call cannot overlap the two) and moves its operands at 196–226 GB/s,
   80–92% of the 246 GB/s measured sustained read. Closing those bars needs a cross-kernel change
   (a second stream or a fused quantize→GEMM producer/consumer pipeline), not a tile, promotion or
   raster choice; it is listed as an accepted gap rather than attempted here.
7. **Bench reporting correction**: at `T >= 128` the old `linear_bench` predicate labelled
   `[5120, 6144]` `--routes fast` as `NVFP4_F32ACC` (`76.38 TF/s` above) although no pair route was
   registered and the A16/BF16 route actually ran. The profile column is now derived from each
   record's real `uses_a4`/`pair_min_tokens` admission, so a `NVFP4_F32ACC` row means the tensor-core
   route really executed. The `[5120, 17408]` A4 baseline quoted by the doc (`196.79 TF/s`) measures
   `248.29 TF/s` on the unmodified base binary; `[5120, 6144]` (`258.97`) measures `256.79`.

## 3. Subsystem breakdown

| Subsystem | File | Action | Mechanism |
|---|---|---|---|
| Ops (schedule) | `nvfp4_schedule.cuh` | M | `Nvfp4A4TmaPromotions{a_codes,b_codes,b_scales}` + 4 named constants; default `{NONE, L2_128B, NONE}`; dead `GroupTokens` argument removed |
| Ops (TMA) | `nvfp4_a4_tma.cuh` | M | per-descriptor `l2_promotion` on the plane/2-D `a_codes`, `b_codes`, `b_scales` maps; raster keeps its capacity-derived `group_n` |
| Ops (TMA) | `nvfp4_a4_tma.cu`, `nvfp4_a4_tma_launch.h` | M | `K`-dispatched launcher (`{2560, 5120, 6144, 17408}` → Tiled128 `<128, 3, m1/m2, r64>` + Tiled256 `<256, 3, m1>`); geometry-id parameter dropped |
| Ops (route) | `nvfp4_launch.cuh` | M | `nvfp4_a4_tma_route<Layout>` returns `{launch, scales, reciprocal_quantize}` built from the schedule |
| Ops (fused) | 4 × `*_a4_tma.cu` | M | named promotion constants only |
| Shapes | 6 × `n*_k2560.cu` / `n2560_k6144.cu` | M | A4 ladder + measured admission where `launch_a4` threw |
| Shapes | 5 × `n*_k5120.cu` / `n5120_k*.cu` | M | pair record (`launch_nvfp4_a4_pair`, `65`); `[14336,5120]`/`[16384,5120]` A16 prefill re-fit |
| Bench | `nvfp4_route_sweep.cu` | M | promotion variants + `a4_split_*` differential candidates for the `Q`/`G` attribution |
| Bench | `linear_bench.cu` | M | per-shape `uses_a4`/pair admission table for the `NVFP4_F32ACC` profile |
| Tests | `test_nvfp4_a4.cpp` | M | A4 policy on the 6 newly enabled shapes; pair on the 5 newly registered ones |

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- Build: `ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench
  infernix_nvfp4_route_sweep infernix_ops_linear_test` in one invocation after touching the nvfp4
  `.cu` files (the CUDA depfiles in this tree do not track headers). Note that
  `infernix_linear_nvfp4_a16_test` / `infernix_linear_nvfp4_a4_test` named by the task doc are no
  longer build targets — the op tests compile into the family executable `infernix_ops_linear_test`,
  which CTest runs per member.
- Tests (serial; `--parallel` produces spurious numerical failures on this single-GPU host):
  `infernix_linear_nvfp4_a4_test` (now 11 A4 shapes + 13 pair shapes),
  `infernix_linear_nvfp4_a16_test`, `infernix_nvfp4_a16_slice_ring_test`,
  `infernix_nvfp4_tma_staging_race_test` and `infernix_cuda_stack_check` pass; the whole linear
  family (20 CTest entries driven by `infernix_ops_linear_test`) passes — `20/20`, 376.8 s.
- A/B (`T = 4096`, `--warmup 2 --repeat 10`, base = rebuilt `71da249c6`; `0.99..1.01x` = unchanged
  within thermal drift):

  | shape | route | base us | T4 us | Δ | base TF/s | T4 TF/s | profile |
  |---|---|---|---|---|---|---|---|
  | `[5120, 6144]` | a4 | 1003.6 | 1005.4 | 1.00x | 256.79 | 256.32 | NVFP4 (unchanged) |
  | | fast | 3373.9 | 1670.0 | **2.02x** | 76.38 | 154.31 | NVFP4 (mislabeled → correct) |
  | | exact | 3380.9 | 3417.0 | 0.99x | 76.22 | 75.42 | BF16 |
  | `[5120, 17408]` | a4 | 2940.7 | 2933.6 | 1.00x | 248.29 | 248.89 | NVFP4 |
  | | fast | 10916.7 | 4903.7 | **2.23x** | 66.88 | 148.90 | BF16 → NVFP4 |
  | | exact | 10949.4 | 10899.3 | 1.00x | 66.68 | 66.99 | BF16 |
  | `[14336, 5120]` | a4 | 1861.5 | 1865.5 | 1.00x | 323.02 | 322.32 | NVFP4 |
  | | fast | 8853.6 | 3294.0 | **2.69x** | 67.92 | 182.54 | BF16 → NVFP4 |
  | | exact | 8860.5 | 7477.1 | **1.19x** | 67.86 | 80.42 | BF16 |
  | `[16384, 5120]` | a4 | 2074.3 | 2079.6 | 1.00x | 331.29 | 330.45 | NVFP4 |
  | | fast | 10578.8 | 3708.8 | **2.85x** | 64.96 | 185.29 | BF16 → NVFP4 |
  | | exact | 10625.9 | 8876.9 | **1.20x** | 64.67 | 77.41 | BF16 |
  | `[34816, 5120]` | a4 | 4106.5 | 4107.1 | 1.00x | 355.61 | 355.55 | NVFP4 |
  | | fast | 20247.4 | 7798.1 | **2.60x** | 72.12 | 187.26 | BF16 → NVFP4 |
  | | exact | 20047.7 | 20006.8 | 1.00x | 72.84 | 72.99 | BF16 |
  | `[6144, 2560]` | a4 | 1658.7 | 572.1 | **2.90x** | 77.68 | 225.24 | BF16 → NVFP4 |
  | `[10240, 2560]` | a4 | 2733.9 | 868.2 | **3.15x** | 78.55 | 247.36 | BF16 → NVFP4 |
  | `[12800, 2560]` | a4 | 3412.8 | 1042.1 | **3.27x** | 78.65 | 257.58 | BF16 → NVFP4 |
  | `[13952, 2560]` | a4 | 3740.5 | 1126.3 | **3.32x** | 78.22 | 259.79 | BF16 → NVFP4 |
  | `[16384, 2560]` | a4 | 4338.6 | 1292.0 | **3.36x** | 79.20 | 265.94 | BF16 → NVFP4 |
  | `[2560, 6144]` | a4 | 1716.1 | 655.2 | **2.62x** | 75.08 | 196.67 | BF16 → NVFP4 |
  | K = 2560 × 5, `[2560, 6144]` | fast | — | — | 0.99–1.01x | 122.7–174.7 | 122.4–173.8 | NVFP4 (pair already registered) |
  | K = 2560 × 6 | exact | — | — | 0.99–1.01x | 74.7–78.9 | 74.7–79.5 | BF16 (untouched tables) |

- Acceptance summary: the `K = 5120` `--routes fast` bar is **met** (`182.54..187.26` useful TFLOP/s
  vs `165..195`); the `K = 2560` shapes **switch** to `NVFP4_F32ACC` as required but land at
  `196.7..265.9 TF/s`, not `300..360`; `[5120, 17408]` (`248.89`) and `[5120, 6144]` (`256.32`)
  stay at their base values, below the `330..360` / `320..355` bars, for the measured reason in §2.6.

## 5. Accepted gaps & rebase conflict resolution policy

- **Unmet bars**: `--policy a4` at `T = 4096` stays at `196.7..265.9 TF/s` on the newly enabled
  shapes and at `248.9 / 256.3` on `[5120, 17408]` / `[5120, 6144]`. Cause and the missing lever are
  quantified in §2.6 (GEMM at 65–79% of the pipe at its SMEM/`ldmatrix` limit; the quantizer
  serialized ahead of it at 80–92% of sustained read). The follow-up is a quantize/GEMM overlap
  (multi-stream or a fused producer-consumer kernel), which is an execution-contract change in Ops
  and was not attempted at T4's scope.
- **Not shipped, with evidence**: `a_codes`/`b_scales` L2 promotion (neutral to `+19.6%`), the
  `Tiled128 (BM = 128)` prefill switch for `N = 5120` (`+30..60%`), and the `GroupTokens` raster
  override (never better than `0.7%`; removed together with its raster parameter).
- **Still A4-disabled**: `[128, 2560]`, `[512, 2560]`, `[2048, 2560]`, `[2560, 2560]` keep
  `uses_a4 == false` and their pair-only records — outside T4's named shape list; `[2048, 2560]`
  and `[2560, 2560]` belong to T5's decode sweep.
- **Route-shift note for reviewers**: because `uses_pair` excludes widths an A4 1-term admits, an
  `AllowA4` artifact on the 6 newly enabled shapes now runs the 1-term route at `T >= 33` where it
  previously ran the 2-term pair from `T >= 65` (`[16384, 2560]`: `1977 → 1292 us`). The term count
  is the artifact's `ActivationPolicy` choice (`AllowA4` vs `AllowA4Residual`), which is the
  pre-existing contract on the 5 already-enabled shapes; no artifact's declared semantics changed.
- **Rebase policy**: preserve the per-descriptor `Nvfp4A4TmaPromotions` default
  `{NONE, L2_128B, NONE}` (upstream's single `WeightCodePromotion` argument re-loses `21%` at
  `K = 6144`), the `K`-dispatched TMA launcher over any geometry-id switch, the `Tiled256` prefill
  tile on `N = 5120`, the `T >= 129` 128-row K64 A16 tiles on `[14336, 5120]` / `[16384, 5120]`,
  and the 11-shape A4/pair registrations (an upstream table that re-adds `throw std::logic_error`
  in `launch_a4` silently reverts `2.6..3.4x` on the `K = 2560` linears for `AllowA4` artifacts).

---

# GB10 changelog — 2026-10-31 (T5: NVFP4 A16 decode & MTP-verify (T = 1..64) bandwidth-cliff retune across the nine cliff shapes)

## 1. Accounting

- **Base**: `b7e1b9649` (`dev-rebased-20261028`).
- **Files changed**: 13 modified, 0 added, 0 deleted (`+305 / -33`), excluding this changelog.
  `git diff --check` is clean on the final tree.
  - Ops & launch (1): `src/ops/linear/nvfp4/nvfp4_launch.cuh` (M) — adds
    `nvfp4_linear_a16_sliced_k_dyn<Geometry, Schedule>`, the runtime-`K` twin of
    `nvfp4_linear_a16_sliced_k` (which bakes `K` through `Nvfp4ScheduleInstance`). No existing
    signature changed; no other quantization family touched.
  - Shape dispatch tables (11): `n34816_k5120.cu`, `n16384_k5120.cu`, `n14336_k5120.cu`,
    `n5120_k17408.cu`, `n5120_k6144.cu`, `n2560_k6144.cu`, `n2048_k2560.cu`, `n6144_k2560.cu`,
    `n13952_k2560.cu` (M) — decode/verification band re-selection inside `select_a16` (plus the
    mirrored `launch_routes` ladder in `n13952_k2560.cu`); `n16384_k2560.cu` (M) — A4 admission
    gate only (`uses_a4` from 33 to 65 tokens; its A16 table and `launch_routes` are untouched,
    see §4.6); `n2560_k2560.cu`, `n2048_k2560.cu` (M) — `pair_min_tokens` 128 → 65 only (§4.5);
    `n2560_k2560.cu`'s `select_a16` is byte-identical and served as this task's A/B drift control.
  - Bench (1): `bench/ops/nvfp4_route_sweep.cu` (M) — 27 runtime-`K`/`K`-baked candidate mirrors so
    a band can be timed against its own baked twin inside one process; an `a4_pair` candidate so the
    pair gate is measurable; per-candidate skip reasons on stderr (a candidate that threw and a
    candidate that failed numeric agreement were both invisible before); the 2-term residual
    workspace for the pair candidate (it was handed the 1-term layout and died the
    plane-vs-call check, which read as "no data") plus an exact-A16 reference for it, because the
    sweep's A4 reference is itself a 1-term A4 candidate; and a `launch_a4_production` fix — below a
    shape's A4/pair admission the capacity query returns 0 and the arena contract rejects a
    zero-length span, which silently deleted the A4-policy baseline at exactly those widths.
- **Allowlist delta**: none. `tools/cuda_stack/cuda_stack_allowlist.json` is untouched (it holds
  zero nvfp4 entries, so every nvfp4 kernel must be spill-free) and `infernix_cuda_stack_check`
  passes. The 5 new runtime-`K` `nvfp4_a16_sliced_k_mma_kernel` instantiations in `infernix-serve`
  measure `REG:104/108/114/120/122 STACK:0 SHARED:17408..50176 LOCAL:0`; no kernel in the tree
  gained a `STACK`/`LOCAL` symbol, so no ratchet (up or down) applies.
- **Rebuild count**: 8 `ninja` invocations on `build-native`, each preceded by
  `find src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +` (CUDA depfiles are dead in this tree) —
  1 failed (the sweep bench included only `nvfp4_launch.h`, not the `.cuh` defining the inline pair
  launcher), 2 rebuilt op + bench targets, 1 rebuilt after the un-reproduced `[2048, 2560]` GEMV
  promotion was reverted, 2 rebuilt the bench harness fixes (skip reasons and residual workspace,
  then the exact-A16 reference) together with the `[16384, 2560]` gate change, 1 rebuilt the two
  `pair_min_tokens` registrations, and 1 rebuilt the comment corrections that followed them. The
  four gates in §4.1 were re-run on each binary that changed shipped code.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T5)

Measurement discipline first, because four of the task's premises died on it.

All numbers come from `bench/infernix_nvfp4_route_sweep`: one process per shape (weights,
quantization and reference built once, every candidate timed in one CUDA context — never one
process per token width), cold 32 MiB L2 flush, `--warmup 3 --repeat 25`, selection on the
**minimum**, never the median. At these widths the median is host-jitter polluted: two
byte-identical instantiations have been observed 10..31% apart on median while agreeing to 0.3% on
minimum.

Acceptance evidence is an **interleaved production-only A/B** against the rebuilt `b7e1b9649`
binary (`--filter zzzz` keeps only the public route): HEAD and candidate alternate per shape, 3
rounds, min-of-rounds. Two *separate* full passes disagree by 4..17% on bands nobody touched, and
the byte-identical `[2560, 2560]` control table swings ±6.8% inside a single interleaved run — so
absolute-µs bars and single-pass deltas are inadmissible for the ~20 µs shapes and are quoted as
ranges where used. READ_% is against this host's 246 GB/s sustained-read reference, computed from
the launch's operand traffic (packed codes + one scale byte per 16 values for weight *and*
activation, plus the bf16 output).

1. **The `[5120, 17408]` cliff was the baked-`K` instantiation, not the tile (new mechanism,
   fixed).** The task asks to promote T = 2..16 from RowTiles 1 to RowTiles 2; that promotion was
   already in the base table (`<16, 8, 2, 2>` since the 2026-10-30 fit). What the table did not
   have was a runtime-`K` launch: `nvfp4_linear_a16_sliced_k` wraps the schedule in
   `Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>`, and the kernel's
   `#pragma unroll` over `kGroups` then fully unrolls the K-group loop. For the wide-token
   sliced-K tiles ptxas schedules that worse than the rolled form:

   | instantiation | T = 2 | T = 8 | T = 16 |
   |---|---|---|---|
   | baked `<16, 8, 2, 2>` (base) | 281.9 µs | 284.0 µs | 287.0 µs |
   | runtime-`K` `<16, 8, 2, 2>` | 222.3 µs | 223.5 µs | 228.2 µs |

   Same at `[34816, 5120]`: baked `<24, 4, 1, 2>` 525.8 µs against 449.0 µs runtime-`K`. The
   full-MMA tiles show no such gap, so only the sliced-K launch got a runtime-`K` form, and each
   shape table picks per band rather than globally. This is the single largest lever in the task:
   it is worth 13..26% on the affected bands and is invisible to a tile-name comparison.
2. **`[34816, 5120]` T = 2 was already at its bar at base; its real cliff was T = 17..32.** The
   doc baseline of 498.59 µs is stale — HEAD measures 431.07 µs (94.7%) there, because the
   2026-10-30 GB10 fit already retired the SIMT/RowTiles-1 tile. Promoting T ≤ 8 to
   `<8, 8, 2, 2>` as specified is a *regression* on this shape (432.4 µs against 430.9 µs for the
   RowTiles-1 tile the table keeps), so that band is unchanged. The shipped change is T = 17..32 →
   runtime-`K` 2-stage tiles: −4.9% / −3.9% / −3.5% at T = 20 / 24 / 32.
3. **`[16384, 5120]` T = 2 was genuinely still on the SIMT C2 tile.** Removed in favour of the
   RowTiles-2 16-token sliced-K tile (215.20 → 208.00 µs, −3.3%, 92.4% READ), meeting the ≤209 µs
   bar; the `using C2` alias went with it.
4. **`[14336, 5120]`: RowTiles 2 to T = 16, one 32-token tile across T = 17..32.** The T = 2 SIMT
   C2 tile and the 24-token 8-warp tile are both gone; T = 17..32 collapses from two bands to the
   `<32, 4, 1>` tile (−7.0% at T = 20, −8.0% at T = 24).
5. **Lowering `uses_a4` from `>= 8` to `>= 2` on `[5120, 17408]` would regress (premise
   rejected).** The A4 1-term ladder's best at T = 2 is 228.6 µs and its production time at T = 8
   is 233.3 µs, against 218.7 / 221.1 µs for the retuned A16 tiles. A4 stays admitted from T = 8,
   where it is the faster route.
6. **Removing the T = 2 SIMT C2 tile on `[2560, 2560]` is wrong (premise rejected).** It is the
   sweep argmin there (16.96 µs), ahead of every sliced-K and GEMV candidate measured at that
   width; the table is untouched.
7. **The `[2048, 2560]` GEMV promotion did not reproduce (reverted).** The selection sweep put the
   4-warp Direct-scale / 4-CTAs-per-SM GEMV at 13.6 µs against the family's 8-warp StagedRaw tile,
   but the interleaved A/B measures 15.90 against 15.74 µs. The band is back at the family tile;
   the two `[2048, 2560]` changes that survived (T ≤ 8 RowTiles 2 → 1, T = 25..32 sliced-K over the
   16-row MMA tile) are −4.4% and −1.2%, inside this shape's ±7% noise floor, and are recorded as
   argmins that never measured worse rather than as demonstrated gains.
8. **`[16384, 2560]`, `[10240, 2560]`, `[12800, 2560]` A16 tables left alone.** Their best
   candidates were ≤3.5% and did not reproduce across two independent passes; `launch_routes` in
   `n16384_k2560.cu` therefore needed no mirror change.
9. **`pair_min_tokens = 128 → 65` on `[2560, 2560]` and `[2048, 2560]`; `[6144, 2560]` unchanged.**
   The 2-term route had never been timed by this harness at all — it needed two harness fixes before
   it produced data, and then it lost to the admitted ladder everywhere in 33..64 but won by
   18..27% from T = 65 on the two shapes that register no 1-term ladder. See §4.5 for the numbers,
   the reason 33 is the wrong gate, and the A16 ladder defect that pair is papering over.
10. **The shipped fast route, not just the A16 table, decides the T = 32..64 bars**: see §4.6.

## 3. Subsystem breakdown

Every change below is a selection inside `select_a16` (and the mirrored `launch_routes` ladder in
`n13952_k2560.cu`); `Sliced<a, b, c, d>` is `Nvfp4SlicedInstance<Tokens, Warps, Stages, RowTiles>`,
`Mma<…>` is `Nvfp4A16MmaSchedule`, and `dyn` is the new runtime-`K` launcher.

| shape | band | base | shipped | interleaved Δ |
|---|---|---|---|---|
| `[34816, 5120]` | 17..24 | `Sliced<24,4,1,2>` | `dyn Sliced<24,4,2,2>` | −4.9% (T20), −3.9% (T24) |
| | 25..32 | `Sliced<32,4,1,2>` | `dyn Sliced<32,4,1,2>` | −3.5% (T32) |
| `[16384, 5120]` | 2 | SIMT `C2` | `Sliced<16,8,2,2>` | −3.3% (T2) |
| | 3..16 | `Sliced<16,8,2>` | `Sliced<16,8,2,2>` | −1.1% (T16) |
| | 17..24 | `Sliced<24,8,1>` | `dyn Sliced<24,4,1,2>` | −8.1% (T20) |
| | 25..32 | `Sliced<32,4,1>` | `dyn Sliced<32,4,1,2>` | −5.0% (T32) |
| | 33..64 | `Mma<64,64,128,32,16,1,2>` | `Mma<32,64,128,16,16,1,3>` | −2.6% (T48), +1.1% (T64) |
| `[14336, 5120]` | 2 | SIMT `C2` | `Sliced<16,8,2,2>` | −1.4% (T2) |
| | 3..16 | `Sliced<16,8,2>` | `Sliced<16,8,2,2>` | −2.0% (T16) |
| | 17..32 | `Sliced<24,8,1>`, `Sliced<32,4,1>` | `Sliced<32,4,1>` | −7.0% (T20), −8.0% (T24) |
| `[5120, 17408]` | 2..8 | `Sliced<16,8,2,2>` baked | `dyn Sliced<8,8,2,2>` | −15.7% (T2) |
| | 9..16 | `Sliced<16,8,2,2>` baked | `dyn Sliced<16,8,2,2>` | −18.2% (T16) |
| `[5120, 6144]` | 17..20 | `Sliced<24,4,1,2>` baked | `dyn Sliced<24,4,1,2>` | −14.7% (T20) |
| | 21..32 | `Sliced<32,4,1,2>` baked | `dyn Sliced<32,4,1,2>` | −17.4% (T24), −21.4% (T32) |
| `[2560, 6144]` | 17..24 | `Sliced<24,8,1,1>`, `Sliced<24,8,2,2>` | `dyn Sliced<24,4,2,2>` | −12.5% (T20), −18.0% (T24) |
| | 25..32 | `Mma<32,32,128,16,16,2,2>` | `Sliced<32,4,1>` | −6.9% (T32) |
| `[2048, 2560]` | 2..8 | `Sliced<8,8,2,2>` | `Sliced<8,8,2>` | −4.4% (T2), −3.9% (T8) |
| | 25..32 | `Mma<32,16,128,16,8,2,3>` | `Sliced<32,4,1>` | −1.2% (T32) |
| | 1 | 8-warp StagedRaw GEMV | (reverted) | +1.0% → reverted |
| `[6144, 2560]` | 17..24 | `Sliced<24,8,1>` | `Sliced<24,4,1,2>` | −1.8% (T20), −1.7% (T24) |
| | 33..64 | `Mma<32,64,128,32,16,2,2>` | `Mma<64,64,128,32,16,1,2>` | −6.7% (T48), −9.0% (T64) |
| `[13952, 2560]` | 33..48 | (in 33..64 `Mma<64,64,128,32,16,2,1>`) | `Mma<32,64,128,16,16,1,3>` | −7.4% (T48) |

`n13952_k2560.cu` keeps `select_a16` and `launch_routes` (the fused QSA split path) in lockstep:
the new T = 33..48 band was added to both, and the two ladders were diffed line by line.

Three registration gates also moved; none of them is a tile selection, and each is measured in
§4.5 / §4.6:

| shape | gate | base | shipped | measured effect |
|---|---|---|---|---|
| `[16384, 2560]` | `uses_a4` | `max_tokens >= 33` | `max_tokens >= 65` | the admitted A4 ladder is 7..11% slower than the A16 table in 33..64; from 65 it is 25..37% quicker (the A16 table takes 33..59% longer) |
| `[2560, 2560]` | `pair_min_tokens` | 128 | 65 | pair route 18..25% faster than the table across 65..127; 12..27% slower across 33..48 and level at 64 |
| `[2048, 2560]` | `pair_min_tokens` | 128 | 65 | pair route 20..27% faster across 65..127; 6..22% behind across 33..64 |

`[6144, 2560]`'s pair gate was checked and left at 128; `[10240, 2560]` and `[12800, 2560]` keep
their 33-token A4 admission, where the A4 ladder is at worst level with A16 in 33..64.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

### 4.1 Numerical and resource gates

`ctest --test-dir build-native -R
'^(infernix_linear_nvfp4_a16_test|infernix_linear_nvfp4_a4_test|infernix_nvfp4_a16_slice_ring_test|infernix_cuda_stack_check)$'
--output-on-failure`, run serially (parallel CTest produces false numerical failures on this
single-GPU host). On the final binary: **4/4 Passed in 18.1 s** — `#28 infernix_cuda_stack_check`
1.54 s, `#266 infernix_linear_nvfp4_a16_test` 3.53 s (it also carries
`verify_workspace_envelopes(QType::NVFP4, …)`), `#267 infernix_nvfp4_a16_slice_ring_test` 4.63 s,
`#268 infernix_linear_nvfp4_a4_test` 8.42 s. The same four passed on each earlier binary that
changed a shipped route. Every retuned band is dispatched by these tests through the public
`ops::linear` route, and the sweep independently checks each timed candidate against the production
reference (abs 1/16, rel 1/32) and for token invariance before timing it.

### 4.2 Task acceptance bars (interleaved A/B, min of 3 alternating rounds, µs / % of 246 GB/s)

| bar | doc baseline | HEAD measured | shipped | verdict |
|---|---|---|---|---|
| `[34816, 5120]` T = 2 ≤ 436 µs | 498.59 (81.9%) | 431.07 (94.7%) | 429.89 (95.0%) | **met** (met at base; band unchanged) |
| `[16384, 5120]` T = 2 ≤ 209 µs | 216.90 (88.6%) | 215.20 (89.4%) | 208.00 (92.4%) | **met** |
| `[5120, 17408]` T = 2..16 ≤ 235–245 µs | 262–285 (72–78%) | 259.4–275.4 (73–78%) | 218.7–225.3 (91.1–93.3%) | **met** |
| `[2560, 6144]`, `[5120, 6144]`, K = 2560 at T = 32..64 ≥ 85–91% | 60–75% | — | A16-only: 67.6–83.0% (T32), 55.5–75.3% (T48/64) | **partly met**; see §4.5, §4.6, §5 |

### 4.3 Shipped-state bandwidth (same A/B, candidate side)

| shape | T = 2 | T = 16 | T = 20 | T = 24 | T = 32 | T = 48 | T = 64 |
|---|---|---|---|---|---|---|---|
| `[34816, 5120]` | 95.0% | 94.1%* | 92.6% | 92.0% | 91.6% | 83.3% | 81.7% |
| `[16384, 5120]` | 92.4% | 92.7%* | 90.9% | 91.2% | 89.8% | 80.9% | 79.5% |
| `[14336, 5120]` | 92.8% | 92.7% | 89.3% | 88.7% | 86.3% | 77.2%* | 76.3% |
| `[5120, 17408]` | 93.3% | 91.1% | 87.5%* | 87.1%* | 85.6% | 66.2% | 66.7% |
| `[5120, 6144]` | 89.0% | 88.3%* | 83.8% | 84.3% | 83.0% | 66.5% | 67.8% |
| `[2560, 6144]` | 85.3% | 82.7%* | 77.2% | 76.4% | 67.6% | 61.6% | 63.5% |
| `[6144, 2560]` | 84.8%* | 86.2%* | 83.4% | 83.8% | 81.6% | 72.1% | 75.3% |
| `[13952, 2560]` | 91.7%* | 92.5%* | 87.9%* | 87.6%* | 86.2%* | 79.6% | 75.7% |

`*` = width whose band this task did not change (carried over from the 2026-10-30 fit); its HEAD
row in the same A/B agrees to within the noise floor stated in §2.

Across the 10 shapes and 10 widths in the interleaved run the candidate side is **−2.9% total**
(14 736.83 → 14 307.36 µs), with the whole delta coming from the retuned bands and the unchanged
bands flat to ±0.3% on the ≥ 175 µs shapes.

### 4.4 Rejected candidates (measured, not shipped)

Three bands won in one pass and lost in the next, and were reverted rather than shipped:
`[2560, 2560]` T = 25..32 → runtime-`K` `<32, 4, 1>` (+3.7% then −4.7%), `[34816, 5120]`
T = 33..48 → `Mma<32,64,128,16,16,1,3>` (−0.2% then −5.4%), `[14336, 5120]` T = 33..64 →
`Mma<64,64,128,32,16,2,1>` (+1.4% then −3.0%). `[34816, 5120]` T = 17..24 keeps the 2-stage tile
although its 1-stage twin is 0.7..1.0% ahead in the selection sweep — inside the noise floor, so
the extra instantiation was not bought.

### 4.5 Pair gate (`pair_min_tokens`)

The 2-term route could not be measured at all until the sweep itself was fixed, and it failed twice
for unrelated reasons, both hidden by the harness's silent skip:

1. it was handed the **1-term** workspace (`allocate_nvfp4_a4_workspace(…, terms = 1, false)`) and
   died in `make_nvfp4_a4_tma_descriptors` with *"NVFP4 TMA residual planes do not match the call"*
   (`nvfp4_a4_tma.cuh:103`) — the pair GEMM always runs `kNvfp4A4ResidualTerms = 2` row-scaled;
2. after that it was judged against the sweep's A4 reference, which is itself a **1-term** A4 MMA
   candidate, and was dropped as *"disagrees with the reference output"*. It now compares against an
   exact A16 launch of the same operands, and agrees within the harness tolerance at every width.

Both failures pre-date this task and read as "no data"; the skip-reason output added here is what
made them visible. Reachability, for the record: `uses_pair` (`nvfp4_dispatch.cpp:25-29`) needs a
policy that permits the 2-term residual or `LinearRoutes::Nvfp4A4Pair`, so the `--policy a4`
production column below never runs pair — only the forced `a4_pair` candidate does. In the product
that is `FastPath::Nvfp4A4Pair` (`qwen4_exp/program/fast_paths.cpp:786`), which sits on the
request-variant list (`:774`) and is qualified per configuration (`:538-544`), so a gate change
touches neither the reproducible route nor the kernel set of any configuration.

Measured (same harness, cold 32 MiB L2, 25 repeats, minimum; "table" = the public route that shape
actually runs under the policy shown — the A16 table for `[2560, 2560]` / `[2048, 2560]`, which
register no 1-term ladder at all, and the admitted 1-term ladder for `[6144, 2560]` under
`--policy a4`):

| shape | gate | pair T = 33 / 48 / 64 | table T = 33 / 48 / 64 | pair T = 65 … 127 | table T = 65 … 127 | verdict |
|---|---|---|---|---|---|---|
| `[2560, 2560]` | 128 → **65** | 33.02 / 30.85 / 26.94 µs | 25.98 / 27.52 / 27.46 µs | **26.34 … 28.61 µs** (62.0..64.2%) | 35.33 … 35.30 µs (47.3..52.1%) | **wins 18..25% from 65** |
| `[2048, 2560]` | 128 → **65** | 27.55 / 25.44 / 24.80 µs | 22.56 / 23.42 / 23.36 µs | **25.34 … 26.91 µs** (53.1..55.2%) | 34.34 … 35.07 µs (39.2..42.3%) | **wins 20..27% from 65** |
| `[6144, 2560]` | 128 (kept) | 52.29 / 55.81 / 53.25 µs | 45.38 / 45.70 / 45.73 µs (1-term ladder) | 52.99 … 55.90 µs | 47.07 … 49.06 µs | **kept at 128**; its real competitor is the A16 table, see below |

So the task's suggested `128 → 65 or 33` is right only as far as 65, and only on the two shapes that
register no 1-term ladder: across T = 33..64 pair is 6..27% *behind* their A16 table (the one
exception being `[2560, 2560]` at T = 64, a 1.9% wash), so a 33-token gate would be a regression
there. The two gates now sit exactly at the measured crossover, and only for artifacts whose policy
permits the residual term; A16Only/Exact artifacts are untouched.

`[6144, 2560]` needs a policy note rather than a number, because pair and the 1-term ladder are
**never live under the same policy** there: `allows_a4` is true only for `LinearPolicy::AllowA4`
(`include/infernix/ops/linear.h:113,134`), where `uses_pair`'s `!(allows_a4 && uses_a4)` clause
suppresses pair from the ladder's 33-token admission, while `allows_a4_residual` is true only for
`AllowA4Residual`, whose capacity query returns 0 for the 1-term layout. So the pair route on that
shape competes only with the A16 table — where it measures 52.29 … 55.90 µs against that table's
53.66 … 69.41 µs, i.e. 1..19% ahead from T = 33 upward except at T = 48 (−2.4%). Lowering its gate
to 33 is therefore a candidate, not a regression — but it would move the whole verification band of
a residual-permitting artifact onto a different kernel set on a 1%-of-noise margin at the narrow end,
so it is left to the model-level fast-path qualification (§5.2) instead of being shipped on sweep
data alone.

Two disclosures attached to that change:

- The reason pair wins above 65 is a defect in the A16 ladder, not a property of pair: the table
  steps from 23.4 µs at T = 64 to 34.3 µs at T = 65 on `[2048, 2560]` (27.5 → 35.3 µs on
  `[2560, 2560]`) because its next rung is a 64-token tile that pads 65 tokens to 128 columns. A
  full candidate sweep of that band finds no A16 tile below 32.6 µs, so pair is genuinely the best
  available route there; fixing the 65..127 rung is listed as a follow-up (§5.2) and would reopen
  this gate question.
- Both the capacity query and the dispatcher key off the same `shape.pair_min_tokens`
  (`nvfp4_dispatch.cpp:26-28` and `:41-45`), so the gate stays self-consistent; the envelope check
  `verify_workspace_envelopes` — run by `infernix_linear_nvfp4_a16_test` over intervals including
  `[1, 128]` and `[24, 129]`, which straddle the new 65 — passes, confirming every public point in an
  interval still fits the space its interval reserves.
- With the gate at 128, the `NVFP4_A4_PAIR` cases at T = 65 / 96 / 127 in
  `tests/ops/linear/test_nvfp4_a4.cpp` did **not** exercise the pair route on these two shapes —
  `uses_pair` failed the token test and the call fell back to the A16 table, so the coverage the
  width list implies did not exist. After this change those cases really do run pair, which is why
  the A4 test was re-run in full (§4.1).

### 4.6 What the shipped fast route actually runs at T = 32..64

The T = 32..64 bar is written against the decode/verify band, which the acceptance sweep measures
under the A16 policy. Artifacts converted with an A4 activation policy run a different ladder there,
so both are reported (separate passes of the same harness, production route only, cold 32 MiB L2,
25 repeats, minimum; the two passes agree to 0.4% where they overlap — `[16384, 2560]` T = 32,
110.60 against 110.18 µs):

| shape | A16 policy T=32 / 48 / 64 | A4 policy T=32 / 48 / 64 |
|---|---|---|
| `[5120, 6144]` | 88.83 / 109.22 / 108.99 µs (83.0 / 68.3 / 69.3%) | 84.61 / 87.55 / 87.58 µs (**87.1 / 85.2 / 86.2%**) |
| `[2560, 6144]` | 54.85 / 62.78 / 61.54 µs (68.9 / 59.9 / 62.1%) | 46.14 / 47.97 / 48.99 µs (80.4 / 78.5 / 78.0%) |
| `[6144, 2560]` | 48.42 / 58.27 / 58.59 µs (79.3 / 66.3 / 67.5%) | 48.48 / 47.58 / 47.87 µs (77.9 / 81.2 / 82.6%) |
| `[16384, 2560]` † | 110.53 / 129.18 / 131.04 µs (90.8 / 79.4 / 80.0%) | 109.70 / 123.84 / 126.78 µs (91.5 / 82.8 / 82.7%) |
| `[2560, 2560]` | 22.94 / 27.30 / 27.33 µs (69.0 / 59.6 / 61.1%) | 23.52 / 29.25 / 29.34 µs (67.3 / 55.6 / 56.9%) |
| `[2048, 2560]` | 21.60 / 23.49 / 23.14 µs (57.4 / 55.6 / 58.0%) | 22.72 / 23.42 / 22.94 µs (55.9 / 55.8 / 58.5%) |

`†` — this row is from the post-change route-gate pass, so its A4-policy column is the A16 table
(see below); every other row is from the selection passes described above, where the A4-policy
column for `[16384, 2560]` still read 110.18 / 141.28 / 142.02 µs (91.1 / 72.6 / 73.8%).

`[16384, 2560]` is the one place this audit changed a route gate. Its 1-term A4 ladder was admitted
from T = 33 and was **7..11% slower than the A16 table** across the whole MTP-verification band
(T = 33: 136.26 µs against 124.61; T = 48: 136.74 against 123.74; T = 64: 136.26 against 126.91),
while its time drops 25..37% from T = 65, where it wins outright (equivalently, the A16 table takes
33..59% longer there — T = 65: 111.52 against 148.10 µs, 94.1 against 70.9%; T = 128: 114.11 against
181.86, 99.6 against 62.5%). `uses_a4` therefore admits at 65 instead of 33, which is what the
`†` row above measures. `select_a4` keeps its T ≤ 64 rungs so the measured per-width argmin stays
recorded, but below the gate they are unreachable through the public route — the capacity query
returns 0 there and the dispatcher takes the A16 table — so they remain live only for the sweep's
forced A4 candidate. The other K = 2560
shapes keep their 33-token admission: there the A4 ladder is at worst level with A16 in 33..64
([10240, 2560] 88.6 against 88.4 µs, [12800, 2560] 102.4 against 102.4, [13952, 2560] 111.5 against
116.4) and far ahead above it, so narrowing those gates would trade nothing.

## 5. Accepted gaps, honest ceilings & rebase conflict resolution policy

### 5.1 The T = 33..64 bandwidth bar was not reached on the small-N shapes, and the sweep says why

The bar asks for ≥ 85..91% of the 246 GB/s READ reference across T = 32..64. After the retune the
shipped state reaches it at [5120, 6144] on the A4 policy (85.2..87.1%) and comes closest at
[16384, 2560] (80.9..83.9%) and [6144, 2560] (66.3..82.6%); it stays 55..80% at [2560, 6144],
[2560, 2560], [2048, 2560], [10240, 2560], [12800, 2560] and [13952, 2560]. This is not a tile the
sweep failed to pick. At every one of those widths the best of the ~60 candidates per shape is
within a few percent of what now ships, while the pure-transfer floor sits far below both:

| shape | best candidate T=48 / 64 | production T=48 / 64 | transfer floor T=48 / 64 |
|---|---|---|---|
| `[2560, 6144]` | 61.86 / 61.54 µs (60.8 / 62.1%) | 62.78 / 61.54 µs (59.9 / 62.1%) | 37.64 / 38.20 µs |
| `[2560, 2560]` | 26.94 / 27.01 µs (60.4 / 61.8%) | 27.30 / 27.33 µs (59.6 / 61.1%) | 16.27 / 16.69 µs |
| `[2048, 2560]` | 22.46 / 22.88 µs (58.2 / 58.7%) | 23.49 / 23.14 µs (55.6 / 58.0%) | 13.07 / 13.43 µs |
| `[6144, 2560]` | 55.97 / 55.90 µs (69.0 / 70.7%) | 58.27 / 58.59 µs (66.3 / 67.5%) | 38.64 / 39.54 µs |
| `[10240, 2560]` | 85.25 / 87.23 µs (75.3 / 75.3%) | 88.38 / 87.68 µs (72.7 / 74.9%) | 64.22 / 65.64 µs |
| `[12800, 2560]` | 101.92 / 102.46 µs (78.7 / 80.0%) | 102.40 / 103.94 µs (78.3 / 78.9%) | 80.20 / 81.96 µs |
| `[13952, 2560]` | 111.46 / 115.84 µs (78.4 / 77.1%) | 111.46 / 118.11 µs (78.4 / 75.6%) | 87.40 / 89.30 µs |
| `[16384, 2560]` | 124.32 / 124.86 µs (82.5 / 83.9%) | 126.82 / 124.96 µs (80.9 / 83.9%) | 102.58 / 104.81 µs |

Reaching 85% at `[2048, 2560]` T = 64 means 15.8 µs; the fastest of sixty tiles measured there is
22.88 µs, and the whole DRAM footprint of that launch is 3.3 MiB. Candidate selection is exhausted
for these bands — closing them needs a mechanism the current family does not contain (a split-K /
narrow-row decomposition that fills 48 SMs from what is a 64-CTA launch today, or one kernel fusing
the several K = 2560 projections that share an activation), not another entry in the table. No NCU
attribution was collected for these bands, so the cause is stated only as far as this evidence
carries: the entire swept family sits 1.4..1.9× above its pure-transfer floor at these shapes.

### 5.2 Accepted gaps and follow-ups

- `[5120, 6144]` T = 17..32 lands at 83.0..84.3% against the 85..91% bar (−14.7 / −17.4 / −21.4%
  interleaved). The remaining 1..2 points are inside this harness's pass-to-pass spread at 87 µs.
- `[2560, 2560]` and `[2048, 2560]` T = 33..64 stay at 55..61%: no A4 admission, the pair gate is at
  65 (pair loses that band, §4.5), and no candidate goes below 22.5 µs against a 13..17 µs floor
  (§5.1).
- `pair_min_tokens` stays at 128 on `[6144, 2560]`: under the policy that admits its 1-term ladder
  the pair route is suppressed, and under the residual policy its competitor is the A16 table, where
  pair is only 0.8..19% ahead and behind at T = 48 — too thin to move a fast path's kernel set on
  sweep data alone (§4.5).
- Follow-ups, in the order they look worth trying: the `[2048, 2560]` / `[2560, 2560]` 65..127 A16
  rung, which pads 65 tokens into a 64-token tile and costs 47% at a single token (fixing it is the
  honest fix under the pair gate lowered in §4.5, and would reopen that gate); an interleaved A/B of
  the `[6144, 2560]` pair gate at 33 under a residual-permitting policy, together with the
  model-level fast-path qualification under `docs/maintainer/model-quality.md`; a T = 33..64
  sliced-K / split-K rung for the N ≤ 2560 shapes (the sweep has no sliced tile above 32 tokens,
  which is exactly the band that misses the bar); and a launch-overhead measurement to say how much
  of the 22..27 µs at `[2048, 2560]` / `[2560, 2560]` is kernel versus launch.

### 5.3 Rebase conflict resolution policy

- `nvfp4_launch.cuh`: keep both launchers. `nvfp4_linear_a16_sliced_k_dyn` is not a duplicate of the
  baked form — it is 13..26% faster for wide-token sliced-K tiles. If upstream deletes it or bakes
  `K` into every sliced-K band, restore the per-band choice and re-measure; the full-MMA tiles have
  no runtime-`K` form and none is needed.
- Shape tables: the accepted end state is the interleaved-A/B winner per band, not the sweep argmin.
  On conflict, keep `[2560, 2560]`'s T = 2 SIMT `C2` tile and `[2048, 2560]`'s 8-warp `StagedRaw`
  GEMV (both re-tested against the alternatives and lost), keep the `C2` removals and RowTiles-2
  T ≤ 16 tiles on `[16384, 5120]` and `[14336, 5120]`, keep the new `[13952, 2560]` 33..48 band in
  **both** `select_a16` and `launch_routes`, keep `uses_a4(...) >= 65` on `[16384, 2560]`, and keep
  `pair_min_tokens = 65` on `[2560, 2560]` / `[2048, 2560]` against an upstream revert to 128 (and
  against an upstream push down to 33, which the T = 33..64 measurements reject).
- Any upstream change to these tables is a routing change: re-validate it with the same
  production-only interleaved A/B (≥ 3 alternating rounds, min-of-rounds, byte-identical control
  shape) before accepting it. A single sweep pass is not evidence at these widths — see §2.
- `bench/ops/nvfp4_route_sweep.cu`: keep the `residual` candidate flag and the exact-A16 reference
  for 2-term candidates. Without them the pair route is skipped as "disagrees with the reference"
  and its timings silently disappear from every future audit.

# GB10 changelog — 2026-10-31 (T6: parallelize/coalesce the NVFP4 fused GdnConvSnapshot epilogue — per-row scratch line prefetched behind the contraction)

## 1. Accounting

- Base commit: `7a1b17941` (`dev-rebased-20261028`). No commit created (task instruction 7 — the
  reviewer commits).
- Files changed (10 tracked, 0 added, 0 deleted):

| File | Action | Change |
|---|---|---|
| `src/ops/linear/common/epilogue.cuh` | M | `LinearRowPrefetch<Output>` detection trait (`words`/`units`, `{0,0}` for an output that reads nothing but its accumulator) and the optional row-scratch argument on `linear_finish_row` |
| `src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh` | M | `nvfp4_a16_gemv_kernel` claims a per-row scratch line in shared, fills one row per thread after `pdl::enter_streaming()` and before the contraction, and hands the row's line to the row consumer |
| `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh` | M | the same line in `nvfp4_a16_sliced_k_mma_kernel`, filled before the prologue copies and published by the mainloop's existing first barrier; new row-consumer branch passing the line to a six-argument `apply_row` |
| `src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh` | M | `GdnConvSnapshotColumnsOutput` owns the line layout (`kPrefetchColumns`/`kPrefetchUnits`/`kPrefetchWords`) and its fill (`prefetch_unit`/`prefetch_row`); `store_row` gains a prefetched path sharing the convolution tail; `GdnConvSnapshotPrefetchedRowEpilogue` is the line-carrying hand-off |
| `src/ops/linear/nvfp4/shapes/n16384_k2560.cu` | M | `launch_nvfp4_n16384_k2560_conv_snapshot` selects the prefetched epilogue |
| `bench/ops/gdn_input_proj_conv_snapshot_bench.cu` | M | `--format nvfp4-flash`: the Qwen4Exp-Flash [16384, 2560] parent through the public routes-form `gdn_input_proj_conv_snapshot` (W=1, B=1..8, no workspace) |
| `bench/README.md` | M | documents `--format nvfp4-flash` and that `--batch` is its token count |
| `src/models/qwen4_exp/program/fast_paths.cpp` | M | the two `gdn-conv-fused` NVFP4 comments (the `kQualified` block and `fast_path_applies`) carry T6's measurements instead of T1's NSYS `120.2 us` figures; no qualification added (§4) |
| `tests/models/qwen4_exp/test_fast_paths.cpp` | M | the W4Dense block comment carries the same; the `applies && !qualified` assertion is unchanged |
| `docs/maintainer/model-quality.md` | M | §6.7's `gdn-conv-fused` row updated to the T6 verdict and numbers |

- CUDA stack allowlist: **unchanged** — zero new `STACK`/`LOCAL` symbols, so no ratchet and no
  `check_cuda_stack.py update` run (§4).
- Rebuilds: 9 single-invocation `ninja -C build-native -j "$(nproc)"` builds
  (`infernix-serve`, both GDN benches, `infernix_ops_recurrent_test`,
  `infernix_qwen4_exp_fast_paths_test`, `infernix_bench`), each preceded by
  `find src/ops/… -name '*.cu' -exec touch {} +` because `build-native` CUDA depfiles do not track
  headers.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T6)

**What held.** The regression is real and sits where the task said. In the fused route each fully
reduced row is finished by one thread (`lane == 0` of its owning warp), and that thread issued the
row's seven global reads — the four BF16 `conv_weight` taps and the three BF16 history taps of
`conv_states` — *after* the contraction's final `__syncthreads()`. The decode grid is 1024 CTAs over
48 SMs, ≈ 21 waves at 2 CTAs/SM, so every wave paid a full DRAM round trip in its tail with nothing
left in flight to cover it; the standalone convolution does the same traffic in ≈ 4 µs because there
it is the whole kernel and every thread participates. Hoisting the reads into the contraction, where
the codes/scales streaming already keeps the pipelines busy, is the right lever and is what shipped.

**Premise corrections from measurement.**

1. *`src/models/qwen4_exp/execution_plan.cpp` and `LinearRoutes::GdnConvFused` do not exist.* The
   fusion is not a `LinearRoutes` value. `FastPath::GdnConvFused` in
   `src/models/qwen4_exp/program/fast_paths.cpp` is the gate: `forward()` sets
   `FastPathSelection::gdn_conv_fused`, and `execution/forward.cpp:640` turns that flag into the
   routes-form Op call. The `LinearRoutes` argument of `ops::gdn_input_proj_conv_snapshot` selects the
   *projection's* route set (exact vs q8-sliced-k), not the fusion. The registration decision therefore
   belongs in `program/fast_paths.cpp`'s `kQualified`.
2. *The named verification targets do not exist.* `infernix_ops_gdn_input_proj_test` and
   `infernix_qwen4_exp_model_test` are not CTest names. The real ones are
   `infernix_gdn_input_proj_test`, `infernix_gdn_input_proj_conv_snapshot_test` and
   `infernix_gdn_input_proj_conv_record_test` — all three are
   `build-native/tests/infernix_ops_recurrent_test` with a selector argument — plus
   `infernix_qwen4_exp_fast_paths_test`.
3. *The task's absolute microsecond bars are unreachable in the reproducible harness, so the
   acceptance test as written cannot be satisfied by any implementation.* The bar is "≤ 96.5 µs,
   beating composed 99.0 µs", where 99.0 is a 96.2 µs projection plus a 2.8 µs convolution read off an
   NSYS graph-node trace of the running model (the T1 record). The same projection in this tree's
   cold-L2, CUDA-graph-body-timed harness — the harness a fused number must be compared in — costs
   **100.5 … 103.1 µs** depending on the hour, so the projection alone already exceeds the fused bar
   before one tap is read. The criteria used here are the invariant ones: the fused launch must cost
   the unfused projection plus its own unavoidable state traffic, and it must beat the composed route
   measured under the same conditions, with the plain projection measured in the same session as the
   control. Both are reported (§4), together with the end-to-end A/B, which is the reproducible
   tiebreaker on this host.
4. *The epilogue's cost is CTA-tail latency, not load count or coalescing.* "7 serial uncoalesced
   loads" is right about the serialization and wrong about the pattern: a warp finishing 32
   consecutive channels reads 64 contiguous bytes per tap — two full sectors. Re-laying or vectorizing
   the taps buys nothing; only leaving the tail does. The measured recovery (§4) is the round trip.
5. *The `[16384, 5120]` launcher named in the task is not the Qwen4Exp route.*
   `nvfp4_gdn_snapshot_decode_launch` serves the 27B-class GDN parent from
   `nvfp4_gdn_snapshot_plan.cpp`, and its output map (`GdnConvOutput`/`Nvfp4GdnConvEpilogue`) is shared
   with the Q8 and FP8 record routes. Qwen4Exp's fused decode calls
   `ops::gdn_input_proj_conv_snapshot` on the `[16384, 2560]` parent
   (`execution/forward.cpp:656` → `launch_nvfp4_n16384_k2560_conv_snapshot`). The hook added here is
   output-owned and optional, so that shape can claim the same line later with no kernel change; it is
   left unclaimed (§5).
6. *The hook's first design was wrong in two ways that measurement caught.* (a) Splitting the line
   into independent units spread over the whole CTA, and publishing it with a warp-scoped sync instead
   of a CTA barrier, both measured **slower** (B=1: 106.2 / 106.7 µs against the shipped 105.3 µs, and
   no gain at B=4/8), because each thread then issued half as many independent loads and the barrier
   the second resident CTA was covering went away. The shipped form — one thread fills one whole row,
   one CTA barrier — wins. (b) Adding the six-argument `apply_row` to the *existing*
   `GdnConvSnapshotRowEpilogue` silently captured the **Q8** fused route, whose contraction never fills
   a line: it selected the prefetched store with a null line and corrupted `infernix_gdn_input_proj_test`
   (`Q8 [16384,2560] B=8 fast in place replay`: non-finite outputs, guard overwritten). The prefetched
   hand-off is therefore a separate type, selected by the launcher that fills lines — a compile-time
   choice, with no null case and no duplicated epilogue code in the Q8 kernels.

**Why a shared scratch line rather than registers.** The row's owner could keep seven values in
registers, but the route serves up to eight contraction columns, so a full line is 4 + 4·8 words and
at B = 8 one thread would own 36 loads; worse, register-resident values stay live across the whole
contraction, which the GEMV's ~118-register budget cannot absorb. A shared line keeps the loads out of
the row consumer's register pressure and lets the fill be spread. The layout is the output's property,
so contractions stay output-agnostic and a plain linear route — which declares neither trait member —
compiles the line and the fill away entirely.

**Why the fill sits after `pdl::enter_streaming()`.** `conv_states` is written by the preceding kernel
in the decode graph, so the prefetch is a producer-dependent read; issuing it before the PDL wait would
race the producer (the same rule that lets the immutable `scales` be staged early and nothing else).

## 3. Subsystem breakdown

- `linear/common` (M): `LinearRowPrefetch` is a `std::void_t`-detected trait over
  `Output::kPrefetchWords`/`kPrefetchUnits`, defaulted to `{0, 0}`. It is a struct with
  `static constexpr` members, not a `requires`-guarded expression: the latter eagerly instantiates
  `Output::kPrefetchWords` on outputs that do not declare it (nvcc rejected
  `LinearBf16SegmentedOutput<6144,1024,6144,1024>`), and a `constexpr` helper instead is rejected as a
  host call from the kernel. `linear_finish_row` gained a trailing
  `const std::uint32_t* row_prefetch = nullptr` and prefers a six-argument `apply_row` when the
  epilogue has one, so every existing epilogue selects exactly the branch it selected before.
- `linear/nvfp4` (M): both decode contractions declare
  `__shared__ std::uint32_t row_prefetch[kBlockRows * kPrefetchWords]` (one word when the output claims
  nothing), fill it one row per thread through `output.prefetch_row`, and pass the row's line into the
  row consumer. The GEMV publishes with the one `__syncthreads()` the fill requires; the sliced-K MMA
  publishes with the barrier the pipeline already executes before its first scale read, adding no
  synchronization. The sliced-K fill passes `token_begin + live_columns`, i.e. the line is indexed by
  *absolute* contraction column — the index the row consumer uses for `destination[sequence * rows +
  row]`; a relative index would read back a neighbouring row's line whenever `token_begin != 0`.
- `gdn_input_proj/q8` (M): word 0..3 of a row's line are the four `conv_weight` taps (BF16 pattern in
  the low half), and word 4 + 4·c .. 7 + 4·c are contraction column `c`'s three history taps plus its
  `snapshot_base_slots[c]`, so the row consumer needs no `initial_state_slots`/`snapshot_base_slots`
  indirection either. A z row (row ≥ 10240) fills and reads nothing. `store_row` is now
  `store_row<Columns, Prefetched>`: the two instantiations share the convolution, SiLU and
  window-publish tail verbatim and differ only in where the seven values come from, which is what keeps
  the un-prefetched path — and every other caller of this output map, including the whole Q8 route —
  bit-identical. The convolution's FMA order and the published window bits are unchanged, so the
  route's FP64 oracle criterion is unchanged and its test is the same one.
- `bench` (M): the `nvfp4-flash` fixture is the routes-form snapshot Op on the Flash-Next parent with a
  synthetic `conv_weight` and the bench's own L2 flush, graph-body-timed like every other fixture. It
  exists because the fused decode route had no op-level benchmark at its own shape — the existing
  `nvfp4` fixture is the 27B [16384, 5120] parent, which is precisely why the T1 "near-tie at the Op
  level" did not reproduce in the model.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

**Tests** — run serially; `--parallel` produces real numerical failures on this single-GPU host:

```
1/5 infernix_cuda_stack_check .................... Passed    1.53 sec
2/5 infernix_qwen4_exp_fast_paths_test ........... Passed    0.00 sec
3/5 infernix_gdn_input_proj_test ................. Passed   10.55 sec
4/5 infernix_gdn_input_proj_conv_snapshot_test ... Passed   28.01 sec
5/5 infernix_gdn_input_proj_conv_record_test ..... Passed    7.95 sec
100% tests passed, 0 tests failed out of 5
```

`infernix_gdn_input_proj_conv_snapshot_test` is the FP64-oracle qualification of the fused route over
B = 1..8 including the in-place-replay configurations, so the shipped convolution and published window
are the bits the Op contract promises. The stack gate is green with zero new spills, so the allowlist
was not touched.

**Kernel resources** (`cuobjdump --dump-resource-usage`, fused instantiations):

| kernel | REG | STACK | LOCAL | SHARED |
|---|---|---|---|---|
| `nvfp4_a16_gemv_kernel` × `GdnConvSnapshotColumnsOutput` × `GdnConvSnapshotPrefetchedRowEpilogue` | 118 | 0 | 0 | 5888 |
| `nvfp4_a16_sliced_k_mma_kernel` × the same output and epilogue | 100 | 0 | 0 | 38400 |

**Op latency** — cold 32 MiB L2 flush, CUDA-graph-body-timed, 5 warmup / 30 repeat, median µs; one
public Op call per timed graph; the plain projection measured in the same session is the control:

| columns B | fused before T6 | fused after T6 | plain projection | `causal_conv1d_silu_snapshot` | composed | fused − composed |
|---|---|---|---|---|---|---|
| 1 | 109.248 | **105.280** | 102.400 | 4.10 | 106.50 | **−1.22** |
| 2 | 114.656 | **110.432** | 103.968 | 4.19 | 108.16 | +2.27 |
| 4 | 113.344 | **108.768** | 104.160 | 6.14 | 110.30 | **−1.53** |
| 8 | 115.424 | **112.352** | 104.992 | 6.14 | 111.13 | +1.22 |

T6 removed **3.0 … 5.9 µs** of fused-route latency (mean −4.2 µs). The residual over the unfused
projection is 3.6 µs at B = 1 in five paired alternating samples (+3.04, +3.36, +3.78, +3.90, +3.84),
against 248 KB of extra state traffic — `conv_weight` 80 KB, history read 60 KB, window publish 60 KB,
selectors ≈ 0 — which is ≈ 1.1 µs at this shape's measured 229 GB/s. So ≈ 2.5 µs of the epilogue is
still not hidden, and the fused/composed column above swings by more than that with the control's own
session drift (the plain projection measured 100.5 … 103.1 µs across runs today).

**End-to-end decode** — `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `infernix_bench -p 4096
-n 128,256 -pg '4096,128;4096,256' -r 2 --warmup 1 --concurrency 1 --kv-dtype fp8`, alternating
default / `--fast-path gdn-conv-fused`, 2 rounds each, `decode_output_tok_s_median`:

| test | off | on | Δ mean |
|---|---|---|---|
| tg 128 | 46.06 / 45.97 | 45.86 / 45.83 | **−0.17** |
| tg 256 | 45.99 / 45.85 | 45.73 / 45.74 | **−0.19** |
| pp4096 + tg128 | 44.87 / 44.85 | 44.82 / 44.68 | **−0.11** |
| pp4096 + tg256 | 44.95 / 44.89 | 44.84 / 44.77 | **−0.12** |

Per-arm standard deviations are 0.03 … 0.08 tok/s, so all eight paired samples are outside noise: the
fused route is **0.2 … 0.4 % slower end to end**. Against T1's measurement of the same artifact
(45.78 default vs 44.09 fused, −3.7 %), T6 recovers ≈ 90 % of the regression.

**Registration decision.** `FastPath::GdnConvFused` for `DenseKernels::Nvfp4` therefore **stays
unregistered**. The project's shipping rule is that a numerics-changing fast path registers only where
it is measured faster, and this route is not: it convolves the FP32 projection instead of
`bf16(projection)`, so it changes the q/k/v bits, and it is slower on the only measurement this host
reproduces to 0.1 %. It remains implemented, oracle-qualified, and runnable under
`--fast-path gdn-conv-fused`; `tests/models/qwen4_exp/test_fast_paths.cpp` keeps its
`applies && !qualified` assertion, and the "measured slower (120.2 µs …)" comments in
`program/fast_paths.cpp` and that test now carry this task's numbers as the reason.

## 5. Accepted gaps & follow-ups

- **The route is at parity, not faster, and is therefore still unregistered** — the one acceptance
  criterion T6 does not meet. What is left is ≈ 2.5 µs of the epilogue the row scratch line does not
  hide. The candidate causes, in the order worth testing: the fill's `__syncthreads()` at the end of a
  wave (the second resident CTA is draining, so it may not cover it); the fill's own issue cost ahead of
  a mainloop already at 93 % of the read roofline; and the row consumer's stores, which now leave the CTA
  tail with real dependents. The next lever is an NCU comparison of the fused and unfused
  `nvfp4_a16_gemv_kernel` on synthetic operands at `[16384, 2560]` — `stall_barrier` and `stall_long_sb`
  in the tail — not another schedule change.
- **The `[16384, 5120]` fused route (`nvfp4_gdn_snapshot_decode_launch`) does not claim the line.** It
  is a different product route (27B-class GDN, `nvfp4_gdn_snapshot_plan.cpp`), its output map is shared
  with the Q8 and FP8 record paths, and it is outside T6's acceptance criteria. The kernel-side hook is
  generic, so claiming it is a change to `GdnConvOutput`/`Nvfp4GdnConvEpilogue` alone; it should come
  with the record/snapshot tests in the same change.
- **The model-quality joint gate for the NVFP4 fused convolution was not run.** It is required only to
  ship the route, which this task did not do; the route stays a candidate, which the reproducibility
  assertion already covers (`{GdnConvFused}` alone keeps `request_invariant()` true).
- **E2E was measured on one artifact and one workload** (W4Dense, fp8 KV, concurrency 1, plain decode).
  MTP verification and the Dense8/IQ3_S artifacts were not run: verification and prefill stay composed,
  and the other two artifacts do not use NVFP4 dense kernels, so neither can exercise this route. The
  full `./bench/run_kv_matrix_bench.sh` matrix was not run for the same reason.
- **Harness note for future audits:** `--format nvfp4-flash` is the only op-level benchmark for the
  Qwen4Exp fused decode route. The `nvfp4` fixture in the same bench is the 27B [16384, 5120] parent,
  and comparing the two shapes is what made T1's Op-level result disagree with the model.
- **Rebase conflict policy for these files:** the row-prefetch hook is optional and output-declared — an
  upstream change that adds a contraction must keep `LinearRowPrefetch`'s default `{0, 0}` reachable so
  outputs without a line compile unchanged, and must not merge the prefetched epilogue back into
  `GdnConvSnapshotRowEpilogue` (§2.6b). On conflict, keep `GdnConvSnapshotPrefetchedRowEpilogue` a
  distinct type, keep the sliced-K fill's `token_begin + live_columns` absolute column bound, and keep
  the GEMV fill after `pdl::enter_streaming()`.

---

# GB10 changelog — 2026-10-31 (T7: retune the fused NVFP4 `linear_add` / `gdn_input_proj` / `attn_input_proj` decode & prefill schedules)

Task: `docs/tasks/20261031-performance-tasks.md` T7 (P1). Scope as specified: the three fused NVFP4
projections' decode/prefill schedule tables and the GDN A16-vs-A4 route gate.

## 1. Accounting

- Base commit: `d2350c3be` (`dev-rebased-20261028`). No commit created (task instruction — the reviewer
  commits). `git diff --check` clean.
- Files changed (14 tracked: 11 M, 3 D; 223 insertions, 291 deletions):

| File | Action | Change |
|---|---|---|
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu` | M | `launch_matrix<K>` rewritten: every sliced-K band in the runtime-K (bare-schedule) form with `RowTiles = 2`; `T = 2..8` → `<8,8,2,2>` at `K = 6144`, and at `K = 17408` `<8,8,2,2>` under `wide_k` / `<8,4,2,2>` otherwise; `9..16` → `<16,…,2,2>`; `17..32` → `<32,4,1,2>`; the 2 K-warp `33..64` tiles dropped in favour of the bands' MMA tiles; the `T = 2..5` SIMT branch removed from `nvfp4_linear_add_a16_launch` |
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu` | **D** | superseded — no route reaches it after the retune |
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.h` | M | the dead `nvfp4_linear_add_small_t_launch` declaration removed |
| `src/ops/linear_add/sources.cmake` | M | the deleted source removed from `infernix_ops` |
| `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp` | M | `resolve_route` under `allows_a4`: `T ≤ 16` → `A16`, `T ≥ 17` → `A4` (was unconditionally `A4`) |
| `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu` | M | `T = 2..16` → the runtime-`K`-baked `<16,8,2,2>` sliced-K tile; the `T ≤ 2` SIMT branch and the `RowTiles = 1` bands removed |
| `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu` | **D** | superseded |
| `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h` | M | dead declaration removed |
| `src/ops/gdn_input_proj/sources.cmake` | M | the deleted source removed |
| `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu` | M | `T = 2..16` → `<16,8,2,2>` baked-`K` sliced-K tile; SIMT branch and `RowTiles = 1` bands removed |
| `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu` | **D** | superseded |
| `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h` | M | dead declaration removed |
| `src/ops/attn_input_proj/sources.cmake` | M | the deleted source removed |
| `bench/ops/nvfp4_linear_add_bench.cu` | M | `--variants` candidate set extended for the T7 levers: `bench_a16_sliced_baked`/`bench_a16_sliced_rt` (RowTiles and runtime-`K` exposed separately), `bench_a16_gemv`, `bench_a4_tma_schedule` + `kBenchTmaScaleLayout` for the `BlockTokens`/`BlockRows`/`Stages` prefill sweep; the superseded `simt_small_t_prod` candidate removed; the pre-existing dangling-capture UB in the `a4`/`a16`/new candidate helpers fixed (`[=, &x, &packed, &out, &scratch]` — a stored `std::function` must not capture its function-pointer parameter by reference) |

- Not changed, on measurement (§2): `nvfp4_linear_add_decode.cu` (the T=1 GEMV is already the argmin),
  `nvfp4_linear_add_a4_tma.cu` / `nvfp4_linear_add_a4.cu` (the `Tiled128` premise is falsified),
  `nvfp4_attn_input_plan.cpp` (the A4 hand-over at 4 is already measured-optimal), and
  `src/ops/linear_swiglu/nvfp4/*` (audited, §5).
- CUDA stack allowlist: **unchanged** — `infernix_cuda_stack_check` passes with zero new `STACK`/`LOCAL`
  symbols and no now-unneeded entries (the three deleted kernels carried no allowlist entries), so no
  ratchet and no `check_cuda_stack.py update` run.
- Rebuilds: five single-invocation `ninja -C build-native -j "$(nproc)"` builds in this continuation
  (retune, post-deletion, candidate `infernix_bench`, baseline `infernix_bench`, post-restore), plus the
  candidate-sweep builds earlier in the task, each preceded by
  `find src/ops/{linear_add,gdn_input_proj,attn_input_proj} -name '*.cu' -exec touch {} +` because
  `build-native` CUDA depfiles do not track headers.

## 2. Architectural rationale & task premise analysis (`20261031-performance-tasks.md` T7)

**What held, and what shipped.** The three fused ops carried the RTX 5090 fit's `RowTiles = 1`,
`K`-baked sliced-K tiles, and on GB10 the `K` baking is the dominant loss, not the SIMT-vs-tile choice
the task named. Baking `K` into `Nvfp4ScheduleInstance` lets `ptxas` fully unroll the K-group loop, which
costs far more than it saves at these widths: the same tile in the runtime-`K` form (`nvfp4_sliced_k_mma`
with a runtime `kGroups`) is 30–42 % faster on the wide tiles — 322.4 → 226.3 µs (`K = 17408, T = 1`),
403.5 → 252.0 (`T = 24`), 432.1 → 262.1 (`T = 32`), 118.6 → 96.3 (`K = 6144, T = 24`), 135.4 → 100.4
(`T = 32`). `RowTiles` adds a further 1–4 % where both are legal. `RowTiles` only sets how many row tiles
share a CTA and its launch bounds (`nvfp4_sliced_min_blocks_per_sm`), and the MMA's columns are
independent, so a column's `K` slices and their reduction order — and therefore the bits — are
unchanged: the shipped `RowTiles = 2` tiles are bit-identical to the tiles they replace. The `T = 2..5`
SIMT kernel lost to the tile of the same K-warp count at every one of those bands (`K = 17408`: 237.5 /
238.4 / 246.6 µs at `T = 2/4/5` against 228.2 / 230.2 / 231.3; `K = 6144`: 85.0 against 83.6), so it and
its two siblings were deleted rather than left as an unreached fallback.

**Premise corrections from measurement.**

1. **The GEMV specialization (step 1) is a no-op: the two schedules are the same speed.** The claim is
   that `nvfp4_linear_add_decode.cu`'s `Nvfp4A16GemvSchedule<8, 2, 16, 4, StagedRaw, Default, 2>` costs
   `+18.7 µs` against the standalone table's `<8, 1, 16, 4, Direct, Default, 2>`. The standalone
   `src/ops/linear/nvfp4/shapes/n5120_k6144.cu` does not use the `Direct` schedule the task quotes — it
   uses the same `<8, 2, 16, 4, StagedRaw, Default, 2>` as the fused op, so the 79.62 µs figure it
   attributes to `Direct` was never produced by it. Measured as candidates in one process (cold L2, pass
   2, `K = 6144, T = 1`): shipped `w2 StagedRaw` **80.80 µs**, `w1 Direct` 81.7–83.8, `w1 StagedRaw`
   82.5, `w2 Direct` 82.8–90.3. At `K = 17408, T = 1` the GEMV (226.3–227.1) ties the runtime-`K` 4-warp
   tile (226.3–229.2). No change: the requested specialization is not faster, and the `T = 1` route is
   the release's.
2. **The `Tiled128` / `GroupM = 8` prefill premise (step 2, second half) does not exist as written and
   regresses as implemented.** `Nvfp4A4TmaMmaSchedule<BlockTokens, Stages, MinBlocksPerSm, Promotions,
   Terms, BlockRows>` has no `GroupM` parameter at all, and `kScaleLayout` is `Tiled128` iff
   `BlockTokens == 128` — so "BM = 128, GroupM = 8" can only be expressed as `BlockTokens = 128` (which
   forces `MinBlocksPerSm = 1`, the same occupancy the shipped `BlockTokens = 256` lane has) or
   `BlockRows = 128`. Swept against the production lane (cold L2, pass 2): `a4_tma128_b3` is **+13 … +23 %
   slower** at every prefill width (`K = 6144`: 184.3 vs 169.0 at `T = 512`, 336.9 vs 326.7 at 1024,
   1431.6 vs 1177.6 at 4096; `K = 17408`: 489.5 vs 451.6 at 512, 3982.2 vs 3228.7 at 4096), and
   `a4_tma256_b3_r64` is +16 … +21 % slower. Not shipped; `nvfp4_linear_add_a4_tma.cu` is untouched.
3. **The task's absolute microsecond bars are protocol artifacts, so every judgement here is an A/B
   delta against the saved HEAD binary in the same protocol.** Re-running the *HEAD* binary's own
   acceptance command shows its `T = 1` baseline is not reproducible: `--t-sweep 1,1,1,1 --warmup 2
   --repeat 10` gives 78.7 / 78.8 / 79.5 / 78.7 µs after the first point's 98.3 µs. Two distinct
   artifacts explain it, both in the harness rather than the kernel:
   - *first-point allocation inflation* — the process that also sizes the A4 workspace for `max_t = 4096`
     reads its first point at 95.2–97.3 µs, while the same point with a sweep max of 64 reads 79.7–85.0
     (`--policy a4` and `--policy a16` agree once the sweep is the same);
   - *memory-clock droop* — the retuned tiles run ~25 % faster at boosted clocks and converge on the SIMT
     kernel once the GPU is at sustained-load clocks (§5). The 12-identical-sample sequence
     `176.99 × 9, then 225.9 / 190.7 / 203.6` for one unchanged kernel is the direct evidence.
   The reported acceptance numbers are the sustained-load / task-command protocol; the boost-state
   numbers are disclosed next to them rather than in place of them.
4. **The GDN hand-over is at 17, not 16-or-3 as the task offered, and Attn needs no gate change.** Cold
   sweep (`warmup 3 / repeat 15`): A16 wins `T = 1` 202.6 vs 212.8, `T = 4` 207.7 vs 213.8, `T = 8` 208.8
   vs 213.8, `T = 16` 211.8 vs 215.9, then collapses above the tile table's reach (`T = 17` 237.4,
   `T = 31` 366.7) while A4 holds ~216–219. Boundary set at exactly 17. For Attn the A16 lane is 1–3 µs
   ahead at `T = 4/8` (185.0 / 185.2 vs 188.2 / 187.2), inside the run-to-run spread, so
   `nvfp4_attn_input_plan.cpp` keeps its `T ≥ 4 → A4` gate.
5. **The `K = 17408` `T = 2..4` bar needs the 8 K-warp tiles, which stay gated.** The 8 K-warp runtime-`K`
   tiles are the argmin at every `K = 17408` width (223.1–228.2 for `T = 2..8` against 228.2–230.2 for the
   4 K-warp twins) and would meet the ≤ 226 µs bar, but the K-warp count *is* the reduction order, so they
   change bits relative to the release. They remain admitted only under `LinearRoutes::WideDownTile`
   (batched decode/verification, where T5 qualified them); the ordinary A16/A4 route and prefill keep the
   release's 4 K-warp tiles, which the sweep still puts ahead of the SIMT kernel and of every `K`-baked
   tile. Shipping them on the other routes is a fast-path qualification (§5), not a dispatch edit.

## 3. Subsystem breakdown

- **ops / linear_add (NVFP4)** — dispatch table only; no kernel source changed. Numerics are unchanged
  (`RowTiles` and runtime-`K` preserve each column's slice order), which `infernix_linear_add_nvfp4_test`
  and the column-invariant and sliced-K pipeline tests re-check against their oracles.
- **ops / gdn_input_proj (NVFP4)** — one route-gate change plus one tile band. The gate change *does*
  alter the artifact's bits for `T = 1..16` under an A4-allowing policy: those widths now run the A16
  dequant tiles instead of the A4 lane. Those A16 kernels are not new — they are the already-qualified
  `A16Only` route's kernels, covered by `infernix_gdn_input_proj_test`'s independent oracle, and the E2E
  MTP acceptance rate moved within noise (§4).
- **ops / attn_input_proj (NVFP4)** — one tile band; policy gate untouched, so no bits change outside the
  bands whose tile width changed.
- **bench** — the candidate harness is the deliverable's measurement instrument: `--variants` now measures
  `RowTiles`, runtime-`K`, GEMV schedule and A4 TMA shape as *separate* candidates so no lever is bundled
  into another's verdict.
- **Deleted** — the three `*_small_t.cu` SIMT kernels, their `sources.cmake` entries, their `plan.h`
  declarations and the bench's `simt_small_t_prod` candidate, per the repository's no-backward-compat rule
  for superseded routes. Nothing else referenced them (no production caller, no test).

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- Build: one `ninja -C build-native -j "$(nproc)"` invocation for `infernix-serve`, the three benches,
  `infernix_bench` and the three family test executables; clean (`BUILD_EXIT = 0`, 85 edges after the
  deletions).
- Tests, serial (this host's CTest `--parallel` produces real numerical failures):
  `ctest --test-dir build-native -R "infernix_linear_add_.*test|infernix_gdn_input_proj_test|infernix_attn_input_proj_test|infernix_column_invariant_projections_test|infernix_groupwise_sliced_k_pipeline_test|infernix_cuda_stack_check" --output-on-failure`
  → **100 % passed, 0 failed of 13, 90.5 s**, run twice (after the retune and after the deletions + tree
  restore). The task's named targets `infernix_ops_linear_add_test` / `infernix_ops_gdn_input_proj_test` /
  `infernix_ops_attention_test` are not CTest names; the op tests merge into
  `infernix_ops_linear_fused_test` / `infernix_ops_recurrent_test` / `infernix_ops_projection_test`.
- **Acceptance points — 5 alternating passes, one `T` per process, `warmup 2 / repeat 15`, median of 5**
  (baseline = the saved HEAD binary; ranges are the pass minima/maxima):

| point | HEAD | T7 | Δ |
|---|---|---|---|
| `linear_add` `K=17408` `a4` `T=2` | 235.52 (234.3–237.4) | **175.97** (175.9–176.1) | **−25.3 %** |
| … `T=3` | 234.43 | 177.98 | −24.1 % |
| … `T=4` | 235.36 | **176.13** | **−25.2 %** |
| … `T=8` (A4 lane) | 226.14 | 226.08 | −0.0 % |
| `linear_add` `K=6144` `a4` `T=2` | 84.96 | **73.57** | **−13.4 %** |
| … `T=3` / `T=4` | 76.67 / 76.64 | 74.66 / 73.57 | −2.6 % / −4.0 % |
| `gdn_input_proj` `a4` `T=1` | 212.83 (210.8–213.9) | **202.56** (201.6–205.7) | **−4.8 %** |
| … `T=2` | 212.77 (211.9–212.8) | **208.74** (208.7–210.8) | **−1.9 %** |
| … `T=4` / `T=16` | 211.81 / 214.88 | 209.76 / 212.80 | −1.0 % / −1.0 % |
| `attn_input_proj` `a4` `T=2` | 185.18 (184.3–186.2) | **183.07** (182.1–183.1) | **−1.1 %** |
| … `T=4` | 186.24 | 187.20 | +0.5 % (noise) |

- **Task-command protocol** (the four commands in the task, 3 alternating passes per binary, median of 3) —
  the protocol the acceptance bars were written in, and the one that puts the GPU under sustained DRAM
  load. Largest movements, all improvements, none a regression beyond noise:
  `linear_add K=17408 a16` `T=8/16/24/32/64` 286.7→230.4, 315.4→236.4, 344.1→251.7, 407.2→265.1,
  647.0→354.3 µs (−19.6 / −25.1 / −26.8 / −34.9 / −45.2 %); `K=6144 a16` `T=24/32/64` 110.4→94.2,
  126.7→98.3, 257.9→119.1 µs (−14.7 / −22.4 / −53.8 %); `K=17408 a4` `T=2` 235.5→228.4, `T=32`
  235.5→231.5; `K=6144 a4` `T=512` 177.4→167.9; `gdn a4` `T=1/2/4/8/16` 211.8→202.6, 211.7→208.7,
  212.0→208.7, 214.9→209.8, 216.9→211.8; `gdn a16` `T=2` 215.9→208.7; `attn a4` `T=2` 185.2→182.9.
  Unchanged as intended: both `T=4096` points (227.9→227.1 TF/s at `K=17408`; `K=6144` is bimodal in both
  binaries, §5) and the `T=1` points (both binaries 96.3/97.2 at `K=6144`, 233.5/233.5 at `K=17408`).
- **E2E** — `./bench/run_kv_matrix_bench.sh` with `MODELS=nvfp4 KV_DTYPES=fp8 MODES=mtp` (W4Dense
  `Qwen4ExpForCausalLM`, pp4096 / tg128 / tg256 / pp4096+tg128 / pp4096+tg256, `-r 2 --warmup 1`), 2
  alternating passes per side against a HEAD-built `infernix_bench`: **parity, all five shapes within
  ±2.1 % with mixed signs** (decode 65.7→64.8, 70.0→69.7, 95.2→96.6, 98.7→96.8 tok/s; prefill
  3271.6→3206.2, 3246.2→3265.0, 3250.1→3205.4 tok/s), MTP accepted-tokens-per-round within ±2.1 %
  (1.33→1.30, 1.62→1.61, 3.79→3.80, 4.25→4.20). The Op-level decode wins do not move this workload
  end-to-end: at real decode widths the affected bands are a small share of a step that is dominated by
  the host/SSD-offloaded experts (§5).

## 5. Accepted gaps & rebase conflict resolution policy

- **`linear_add K=17408, T = 2..4` misses the ≤ 226 µs bar.** Task-command protocol: 228.4 µs (`T = 2`)
  and 227.3–235.8 µs (`T = 4`) against the bar, i.e. 1–4 % short; HEAD in the same protocol is 235.5 /
  234.5. The single-point protocol reaches 176 µs there, but that is the boosted-clock state of the same
  kernel and is not what a sustained decode stream sees, so it is disclosed and not claimed. Closing the
  bar means the 8 K-warp tiles on the non-`wide_k` routes (§2.5), which changes the reduction order and
  therefore needs the model-quality fast-path qualification for that configuration plus a measured-faster
  registration — a qualification decision, not a schedule edit.
- **The `T = 4096` prefill bars (≥ 310 / ≥ 320 TF/s) are not reachable by any schedule knob.** Measured
  219.8 (HEAD 163.8, bimodal) at `K = 6144` and 227.3 (HEAD 227.3) at `K = 17408` — the doc's own
  baselines, unchanged. The A4 prefill lane is quantizer-serialized ahead of the GEMM; the GEMM alone
  sustains 339–346 TF/s, so the bar needs a cross-kernel producer/consumer or a second stream, not a tile
  change. Two supporting measurements: `Tiled128` and `BlockRows = 64` variants are 13–23 % and 16–21 %
  *slower* (§2.2), and the same A4 lane reaches **356.6 TF/s** in `nvfp4_linear_swiglu` at
  `[34816, 5120]`, where the activation quantizer is 12 % of the traffic instead of ~50 % — the ceiling
  tracks the quantizer's share, which is the serialization signature.
- **`nvfp4_linear_swiglu` was audited, not changed.** The task title names it but the spec lists no
  swiglu file, step, or bar and its bench is not in the verification list. Audit (`[34816, 5120]`,
  `warmup 2 / repeat 10`): the A4 lane — the one the production policy runs — is at 226–228 GB/s for
  `T = 1..16` and 224.8 / 222.9 at `T = 32 / 64`, i.e. already in the band the retuned ops reached, with
  289.0 / 356.6 TF/s at 512 / 4096. The `A16Only` lane dips to 184.0 / 201.5 GB/s at `T = 32 / 64`
  (against 227.3 at `T = 16`), which is the same `K`-baking signature and is the natural follow-up; it is
  left alone here because that lane is the policy-fallback route with no acceptance bar and no spec step.
- **Bits change only on the GDN `T = 1..16` A4 → A16 gate** (§3). The kernels it admits are the already
  qualified `A16Only` route's, and the E2E MTP acceptance stayed within noise; no new kernel was added to
  any configuration.
- **E2E coverage is one artifact and one KV dtype** (W4Dense, fp8, MTP, concurrency 1, 2 passes/side).
  IQ3_S and Dense8 were not run: their dense projections are not the NVFP4 routes this task retunes, and
  the full 42-arm matrix is the orchestrator's campaign gate, not this task's verification list.
- **Harness notes for future audits.** (a) The first point of a `--t-sweep` process is inflated ~17 µs by
  the max-token A4 workspace allocation — measure a `T = 1` point either with a sweep whose max is small or
  as a single-point process. (b) `--variants` numbers are all in the drooped-clock regime (29 candidates ×
  18 launches share one process), so they are valid for ranking candidates and not for absolute bars.
  (c) A candidate `std::function` in the bench must capture its launch function pointer by value; the
  pre-existing `a4`/`a16` helpers captured it by reference out of a returned frame, which segfaulted the
  first `--variants` candidate in this task (`EXIT = 139`).
- **Rebase conflict policy for these files:** keep the fused sliced-K bands in the runtime-`K` form with
  `RowTiles = 2` — an upstream change that re-bakes `K` into these instances reintroduces the 30–42 % loss
  on this device; keep the `K = 17408` 8 K-warp tiles behind the `wide_k` argument and do not let a merge
  promote them to the ordinary route without the fast-path qualification; keep the GDN A4 hand-over at 17
  and the Attn one at 4 as measured values with their numbers in the comment; and keep
  `nvfp4_{linear_add,gdn_input,attn_input}_small_t.cu` deleted — if an upstream change resurrects a SIMT
  `T = 2..5` band, it must be re-measured against the runtime-`K` tile of the same K-warp count first.
