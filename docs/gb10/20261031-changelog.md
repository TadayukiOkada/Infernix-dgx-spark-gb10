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
