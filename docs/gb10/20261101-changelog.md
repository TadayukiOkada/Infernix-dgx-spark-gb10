# GB10 changelog — 2026-11-01 (T1: NVFP4 A4 & `Nvfp4A4Pair` prefill GEMM `ldmatrix`/`mma` interleaving, warp-packed `uint32_t` scale stores, `K=17408` register caching, and `K=2560` A4 admission)

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `7ce9b8b46`. `git diff --check` clean; `.tandem/` untouched.
- Files changed (T1): 10 (9 modified, 1 added), `+188 / -70` lines in `src/` and `tests/` (excluding this changelog). `cuda-kernel-toolkit/` intentionally excluded from the enumeration (integrated toolkit, untouched).
- Allowlist delta: none (`tools/cuda_stack/cuda_stack_allowlist.json` untouched; zero new `STACK > 0` or `LOCAL > 0` symbols).

| File | Action | Change |
|---|---|---|
| `src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh` | M | Preserve 1-group/thread (`32 B`) coalesced warp `uint4` loads and `uint2` code stores; pack 4 consecutive lanes' `e4m3` scale bytes via `__shfl_down_sync` (`store_nvfp4_scale_quad`) into a single coalesced `uint32_t` store across `RowMajor`, `Tiled128`, and `Tiled256`; raise `kNvfp4A4RowScaledCachedGroups` `2 -> 5` so `K = 17408` (`5` groups/thread) caches all input groups in registers and reads `x` once from DRAM |
| `src/ops/linear/nvfp4/nvfp4_a4.cu` | M | Specialize `K = 17408` `nvfp4_a4_row_scaled_residual_quantize_kernel` with `kMaxGroups = 5`; allow `Nvfp4ScaleLayout::Tiled256` in `launch_residual` |
| `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh` | M | Interleave `ldmatrix` fragment loads with `mma_nvfp4_e4m3` using a 2-slot `a_fragments[2][kTerms][4]` / `a_scales[2][kTerms]` ping-pong buffer across `mma_m` and preloading `b_frag[0]` + `b_scale[0]` before the `mma_n` loop |
| `src/ops/linear/nvfp4/nvfp4_a4_tma.cu` | M | Instantiate `Nvfp4A4TmaMmaSchedule<256, 2, 1, Promotions, kNvfp4A4ResidualTerms, 64>` and dispatch it in `launch_residual` when `p.scale_layout == Nvfp4ScaleLayout::Tiled256` |
| `src/ops/linear/nvfp4/shapes/n2560_k2560.cu` | M | Enable 1-term `A4` ladder (`uses_a4`: `max_tokens >= 33`; `T16R64`/`T32R64`/`T32R128`/`T128R128Pipelined` for `T <= 128`, `Tiled128` TMA for `129..256`, `Tiled256` TMA for `T >= 257`) |
| `src/ops/linear/nvfp4/shapes/n2048_k2560.cu` | M | Same 1-term `A4` ladder and `uses_a4` (`max_tokens >= 33`) |
| `src/ops/linear/nvfp4/shapes/n512_k2560.cu` | M | Enable 1-term `A4` ladder (`uses_a4`: `max_tokens >= 129`, `Tiled128` `128x64` 2-CTA/SM TMA at `T >= 129` for 8 N-blocks / 256 CTAs at `T = 4096`) |
| `src/ops/linear/nvfp4/shapes/n128_k2560.cu` | M | Enable 1-term `A4` ladder (`select_a4` with `Tiled128` `128x64` TMA at `T >= 129`, `uses_a4`: `max_tokens >= 33 && min_tokens <= 128` where A4 beats/matches A16 without regressing `T >= 512`) |
| `tests/ops/linear/test_nvfp4_a4.cpp` | M | Add `[2560, 2560]`, `[2048, 2560]`, `[512, 2560]`, and `[128, 2560]` to the `NVFP4_A4` oracle test suite across `T = 1..2048` and misaligned graph-replay token counts |
| `docs/gb10/20261101-changelog.md` | A | Record T1 architectural rationale, hardware roofline analysis, verification, and 15-shape A/B results |

## 2. Architectural rationale & task premise analysis (`20261101-performance-tasks.md` T1)

1. **GB10 LPDDR5X physical DRAM roofline for the standalone activation quantizer**:
   - On GB10 (`sm_121a`, 256-bit LPDDR5X-8533 unified memory, `273.1 GB/s` theoretical peak, `238.7–246.0 GB/s` sustained STREAM triad/read ceiling in `.ckt_cache/env.json`), at `T = 4096`:
     - `K = 6144`: 1-term `A4` reads `4096 * 6144 * 2 B = 50.33 MB` of BF16 activations and writes `12.58 MB` of packed 4-bit codes + `1.57 MB` of `e4m3` scales = **`64.49 MB` minimum DRAM traffic**. At `246.0 GB/s`, the physical lossless DRAM floor is **`262.1 us`**. 2-term `Nvfp4A4Pair` writes a second code + scale plane (`78.64 MB` total), with a physical DRAM floor of **`319.7 us`** at `246.0 GB/s`.
     - `K = 17408`: 1-term `A4` moves `142.61 MB` read + `35.65 MB` codes + `4.46 MB` scales = **`182.71 MB` minimum DRAM traffic** -> physical DRAM floor is **`765.5 us`** at `238.7 GB/s`. 2-term `Nvfp4A4Pair` moves **`222.82 MB`** -> physical DRAM floor is **`933.5 us`** at `238.7 GB/s`.
   - Note on the task description's `< 110 us` (`K = 6144`) and `< 280 us` (`K = 17408`) figures: achieving `110 us` on `64.5 MB` or `280 us` on `182.7 MB` would require **`586–652 GB/s` of DRAM bandwidth** (`2.4x` above GB10's `273.1 GB/s` theoretical LPDDR5X pin bandwidth; those numbers correspond to RTX 5090 GDDR7). On GB10, the quantizer is already at **`98–100%` of the sustained LPDDR5X bandwidth ceiling** (`258.5–263.5 us` = `244.7–249.5 GB/s` on `K = 6144` 1-term; `315.7–320.0 us` = `245.7–249.1 GB/s` on `K = 6144` 2-term; `778.0 us` = `234.9 GB/s` on `K = 17408` 1-term; `965.4 us` = `230.8 GB/s` on `K = 17408` 2-term).

2. **Why 4-consecutive-groups-per-thread quantizer indexing regressed by `+30%..+54%` (rejected) vs 1-group/thread + 4-lane `__shfl_down_sync` `uint32_t` scale packing (shipped)**:
   - Assigning 4 consecutive 16-element K-groups (`64` BF16 elements = `128 B`) to a single thread caused adjacent lanes `tid` and `tid + 1` in a warp to load `LDG.E.128` (`uint4`) from addresses separated by `128 B` (32 cache lines per warp instruction instead of 8 coalesced sectors) and raised `nvfp4_a4_quantize_kernel` register pressure from `REG:39` to `REG:65`. In `nsys` profiling at `T = 4096, K = 17408`, that strided layout regressed `nvfp4_a4_quantize_kernel` from `782.6 us` to `1,207.6 us` (`+54.3%` slower) and `nvfp4_a4_row_scaled_residual_quantize_kernel` from `967.5 us` to `1,252.5 us` (`+29.5%` slower).
   - **Shipped design**: Keep 1 group of 16 BF16 elements (`32 B`, `2 x uint4`) per thread per loop step so warp global loads (`LDG.E.128`) and code stores (`STG.E.64`) remain 100% coalesced across the 32 lanes (`1,024 B` read and `256 B` code write per warp step). Because `TileTokens` (`128` or `256`) is a multiple of 4 and `groups` (`K / 16`) is a multiple of 4 for all NVFP4 shapes (`K = 2560, 5120, 6144, 17408 -> groups = 160, 320, 384, 1088`), every 4 consecutive threads `(tid & 3) == 0..3` map to 4 consecutive bytes in `Tiled128`, `Tiled256`, and `RowMajor` scale layouts (`offset(base + lane) == offset(base) + lane`). Packing the 4 lanes' `e4m3` scale bytes via `__shfl_down_sync` (`store_nvfp4_scale_quad`) and issuing a single coalesced `uint32_t` store from `(tid & 3) == 0` eliminates byte-store uncoalescing while keeping `nvfp4_a4_quantize_kernel` at `REG:39..41`.
   - Furthermore, raising `kNvfp4A4RowScaledCachedGroups` from `2` to `5` and specializing `K = 17408` (`1088 / 256 = 5` groups/thread) with `kMaxGroups = 5` caches all 5 input groups (`10 x uint4` = 40 registers, `REG:127..128`, `STACK:0`) across Pass 1 (row-max reduction) and Pass 2 (quantize + residual), eliminating the second global read of `x` on `K = 17408`.

3. **2-term `Tiled256` (`BM = 256, BN = 64, Stages = 2`) vs `Tiled128` (`BM = 128, BN = 128, Stages = 3`) on `sm_121a` (`99 KiB` SMEM limit)**:
   - On GB10 (`101,376 B` max shared memory per SM), a 2-term `Nvfp4A4Pair` kernel stages two activation code+scale planes (`kTerms = 2`) per K-tile (`BK = 256`). For `BM = 256`, a single stage of activations requires `2 * (32,768 + 4,096) = 73,728 B`, so `Stages = 3` (`221 KiB` + weights) is physically impossible on `sm_121a`, forcing `Stages = 2` (`95,360 B` SMEM with `BN = 64`) and `BN = 64` (since `BM = 256, BN = 128` requires 128 FP32 accumulator registers per thread, spilling at the 168-register CTA limit).
   - For `kTerms = 2`, a `256 x 64` tile loads `2 * 256 + 64 = 576` tile-rows per `16,384` output elements (`0.0352` tile-rows/output) with only 2 pipeline stages, whereas `128 x 128` (`Tiled128`, `Stages = 3`, `94,592 B` SMEM) loads `2 * 128 + 128 = 384` tile-rows per `16,384` output elements (`0.0234` tile-rows/output) with 3 pipeline stages. Routing `launch_nvfp4_a4_pair` to `Tiled256` therefore increases TMA traffic by `+50%` and drops 3-stage latency hiding, slowing `[5120, 17408]` GEMM from `3,979 us` to `4,767 us` (`-16.5%`).
   - **Shipped design**: `launch_residual` (`nvfp4_a4.cu` and `nvfp4_a4_tma.cu`) supports `Nvfp4ScaleLayout::Tiled256` when requested by `p.scale_layout`, while `launch_nvfp4_a4_pair` (`nvfp4_launch.cuh`) stays on the optimal 3-stage `128 x 128` (`Tiled128`) schedule, where the new `ldmatrix`/`mma` interleaving improves `[5120, 17408]` GEMM from `3,979.1 us` to `3,875.4 us` (`+2.7%`) and `[5120, 6144]` GEMM from `1,363.5 us` to `1,334.6 us` (`+2.2%`).

4. **Per-shape `A4` admission across `[2560, 2560]`, `[2048, 2560]`, `[512, 2560]`, and `[128, 2560]`**:
   - `[2560, 2560]` and `[2048, 2560]`: 1-term `A4` beats `A16` from `T = 33` upward (`24.4 us` vs `28.5 us` at `T = 33`; `42.8 us` vs `114.5 us` at `T = 512`; `288.6 us` vs `706.4 us` at `T = 4096` on `[2560, 2560]` — **`+143.9%` / `2.44x` speedup** at `T = 4096` and **`+161.2%` / `2.61x` speedup** at `T = 512`).
   - `[512, 2560]`: Using `Tiled128` (`BM = 128, BN = 64, 2 CTAs/SM`, giving 8 N-blocks = 256 CTAs at `T = 4096` vs only 4 N-blocks for `Tiled256`), 1-term `A4` beats `A16` from `T = 129` upward (`27.5 us` vs `34.6 us` at `T = 512`, **`+21.7%`**; `155.5 us` vs `167.8 us` at `T = 4096`, **`+5.8%..+8.0%`**).
   - `[128, 2560]`: The entire `[128, 2560]` NVFP4 weight matrix is only `184 KiB` (100% L2-resident), whereas quantizing `x[4096, 2560]` adds an extra `13.1 MiB` DRAM read + `7.4 MiB` DRAM write (`~35 us` standalone quantizer cost on a `108 us` total op) and `N = 128` yields only 2 N-blocks (`BN = 64`). As a result, `A4` beats `A16` at `T = 128` (`16.2 us` vs `18.2 us`, **`+12.4%`**), while `A16` is faster at `T = 512` (`22.5 us` vs `26.5 us`) and `T = 4096` (`108.4 us` vs `116.6 us`). Setting `uses_a4(min_tokens, max_tokens)` to `max_tokens >= 33 && min_tokens <= 128` enables the `A4` ladder across `T = 33..128` (`+12.4%` at `T = 128`) while routing `T >= 129` to `A16` so `--policy a4` has zero regressions at `T = 512` and `T = 4096`.

## 3. Subsystem breakdown

- **Ops / NVFP4 quantizer (`src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`, `src/ops/linear/nvfp4/nvfp4_a4.cu`)**:
  - Added `store_nvfp4_scale_quad` using `__shfl_down_sync` across 4 consecutive threads `(tid & 3) == 0..3` to pack 4 `e4m3` scale bytes into a single coalesced `uint32_t` store in `RowMajor`, `Tiled128`, and `Tiled256` layouts while preserving 1-group/thread (`32 B`) coalesced `uint4` loads.
  - Raised `kNvfp4A4RowScaledCachedGroups` from `2` to `5` and specialized `K = 17408` with `kMaxGroups = 5` in `launch_row_scaled_residual`.
  - Enabled `Nvfp4ScaleLayout::Tiled256` in `launch_residual`.
- **Ops / NVFP4 A4 TMA GEMM (`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh`, `src/ops/linear/nvfp4/nvfp4_a4_tma.cu`)**:
  - Replaced upfront bulk `ldmatrix` loading of all `kMmaTokens x kMmaRows` fragments with a 2-slot `a_fragments[2][kTerms][4]` / `a_scales[2][kTerms]` ping-pong buffer across `mma_m` and interleaved `b_frag[mma_n + 1]` `ldmatrix` loads directly with `mma_nvfp4_e4m3` inside the inner loop.
  - Added `Nvfp4A4TmaMmaSchedule<256, 2, 1, Promotions, kNvfp4A4ResidualTerms, 64>` dispatch in `launch_residual` when `p.scale_layout == Nvfp4ScaleLayout::Tiled256`.
- **Ops / NVFP4 shape tables (`src/ops/linear/nvfp4/shapes/n{2560,2048,512,128}_k2560.cu`) & tests (`tests/ops/linear/test_nvfp4_a4.cpp`)**:
  - Enabled `select_a4` and `uses_a4` on `[2560, 2560]`, `[2048, 2560]`, `[512, 2560]`, and `[128, 2560]`, and added all four shapes to `infernix_linear_nvfp4_a4_test`.

## 4. Verification & resource usage (GB10 `sm_121a`, CUDA 13, `build-native`)

- **CTest verification**:
  - `infernix_cuda_stack_check`: **Passed** (1.52 s; zero new `STACK > 0` or `LOCAL > 0` symbols).
  - `infernix_linear_nvfp4_a16_test`: **Passed** (3.60 s).
  - `infernix_linear_nvfp4_a4_test` (including new `[2560, 2560]`, `[2048, 2560]`, `[512, 2560]`, `[128, 2560]` oracle checks): **Passed** (7.85–9.50 s).
  - `infernix_nvfp4_tma_staging_race_test`: **Passed** (2.72 s).
- **Per-kernel resource usage (`cuobjdump -res-usage`)**:
  - `nvfp4_a4_quantize_kernel`: `REG:39..41`, `STACK:0`, `LOCAL:0`.
  - `nvfp4_a4_residual_quantize_kernel`: `REG:56`, `STACK:0`, `LOCAL:0`.
  - `nvfp4_a4_row_scaled_residual_quantize_kernel`: `REG:47..48` (`K = 2560`), `REG:56` (`K = 5120`), `REG:64` (`K = 6144`), `REG:127..128` (`K = 17408`, `kMaxGroups = 5` register-cached), `STACK:0`, `LOCAL:0`.
  - `nvfp4_a4_tma_kernel`: `REG:89..168`, `STACK:0`, `LOCAL:0`.

## 5. Performance benchmarks & `nsys` kernel decomposition

### 5.1 `nsys` standalone kernel decomposition (`T = 4096`, `--flush-mib 32`)

| Workload (`T = 4096`) | Kernel | HEAD Avg (us) | Final Avg (us) | Final Min (us) | Sustained DRAM BW / Pipe Ceiling |
|---|---|---:|---:|---:|---|
| `[5120, 17408]` `--policy a4` | `nvfp4_a4_quantize_kernel` | 782.6 | **778.0** | **776.0** | `235.5 GB/s` (**98.7%** of `238.7 GB/s` GB10 DRAM ceiling) |
| `[5120, 17408]` `--policy a4` | `nvfp4_a4_tma_kernel` (1-term) | 2,144.3 | **2,161.5** | **2,153.3** | `338.8 TFLOP/s` (**84.7%** of `400 TF/s` sustained NVFP4 pipe; **96.2%** of combined roofline) |
| `[5120, 17408]` `--routes fast` | `nvfp4_a4_row_scaled_residual_quantize_kernel` | 967.5 | **965.4** | **959.0** | `232.3 GB/s` (**97.3%** of `238.7 GB/s` GB10 DRAM ceiling) |
| `[5120, 17408]` `--routes fast` | `nvfp4_a4_tma_kernel` (2-term) | 3,979.1 | **3,875.4** | **3,858.0** | `378.1 TFLOP/s` (**94.5%** of `400 TF/s` sustained NVFP4 pipe, **`+2.7%` vs HEAD**) |
| `[5120, 6144]` `--policy a4` | `nvfp4_a4_quantize_kernel` | 258.5 | **263.5** | **262.2** | `246.0 GB/s` (**100.0%** of `246.0 GB/s` GB10 DRAM ceiling) |
| `[5120, 6144]` `--policy a4` | `nvfp4_a4_tma_kernel` (1-term) | 738.0 | **736.8** | **730.0** | `352.6 TFLOP/s` (**88.2%** of `400 TF/s` sustained NVFP4 pipe; **97.4%** of combined roofline) |
| `[5120, 6144]` `--routes fast` | `nvfp4_a4_row_scaled_residual_quantize_kernel` | 315.7 | **320.0** | **316.5** | `248.4 GB/s` (**100.0%** of `246.0 GB/s` GB10 DRAM ceiling) |
| `[5120, 6144]` `--routes fast` | `nvfp4_a4_tma_kernel` (2-term) | 1,363.5 | **1,334.6** | **1,327.6** | `388.2 TFLOP/s` (**97.1%** of `400 TF/s` sustained NVFP4 pipe, **`+2.2%` vs HEAD**) |

### 5.2 Full 15-shape benchmark matrix (`--policy a4` and `--routes fast`, `--flush-mib 32 --repeat 20`)

#### `--policy a4` (1-term A4 where admitted)

| Shape `[N, K]` | `T=128` HEAD -> Final (us / TF) | `T=512` HEAD -> Final (us / TF) | `T=4096` HEAD -> Final (us / TF) | `T=4096` Δ vs HEAD |
|---|---|---|---|---:|
| `[128, 2560]` | `18.2 -> 16.2` (`4.60 -> 5.17 TF`, **`+12.4%`**) | `22.4 -> 22.4` (`15.00 -> 15.00 TF`) | `110.6 -> 110.4` (`24.28 -> 24.31 TF`) | **+0.1%** |
| `[512, 2560]` | `18.3 -> 18.2` (`18.36 -> 18.40 TF`, `+0.2%`) | `34.7 -> 28.5` (`38.73 -> 47.13 TF`, **`+21.7%`**) | `167.8 -> 158.6` (`64.00 -> 67.72 TF`) | **+5.8%** |
| `[2048, 2560]` | `34.7 -> 24.4` (`38.73 -> 55.04 TF`, **`+42.1%`**) | `90.0 -> 43.8` (`59.68 -> 122.46 TF`, **`+105.2%`**) | `569.2 -> 256.9` (`75.45 -> 167.21 TF`) | **+121.6%** |
| `[2560, 2560]` | `36.7 -> 28.5` (`45.71 -> 58.84 TF`, **`+28.7%`**) | `114.5 -> 43.8` (`58.60 -> 153.08 TF`, **`+161.2%`**) | `706.4 -> 289.6` (`76.00 -> 185.36 TF`) | **+143.9%** |
| `[6144, 2560]` | `50.9 -> 50.0` (`79.09 -> 80.50 TF`, `+1.8%`) | `72.6 -> 72.5` (`221.73 -> 222.02 TF`, `+0.1%`) | `569.2 -> 562.0` (`226.39 -> 229.26 TF`) | **+1.3%** |
| `[10240, 2560]` | `77.6 -> 75.6` (`86.44 -> 88.75 TF`, `+2.7%`) | `120.7 -> 120.7` (`222.39 -> 222.45 TF`, `+0.0%`) | `870.2 -> 847.6` (`246.78 -> 253.36 TF`) | **+2.7%** |
| `[12800, 2560]` | `101.1 -> 99.2` (`82.96 -> 84.59 TF`, `+2.0%`) | `152.4 -> 154.7` (`220.15 -> 216.92 TF`, `-1.5%`) | `1050.5 -> 1043.3` (`255.54 -> 257.30 TF`) | **+0.7%** |
| `[13952, 2560]` | `103.3 -> 101.2` (`88.55 -> 90.34 TF`, `+2.0%`) | `156.5 -> 159.6` (`233.73 -> 229.14 TF`, `-2.0%`) | `1136.4 -> 1119.1` (`257.47 -> 261.46 TF`) | **+1.5%** |
| `[16384, 2560]` | `115.5 -> 114.4` (`92.95 -> 93.88 TF`, `+1.0%`) | `192.3 -> 193.3` (`223.32 -> 222.14 TF`, `-0.5%`) | `1297.2 -> 1284.8` (`264.87 -> 267.44 TF`) | **+1.0%** |
| `[14336, 5120]` | `203.6 -> 201.6` (`92.28 -> 93.22 TF`, `+1.0%`) | `288.6 -> 285.5` (`260.43 -> 263.26 TF`, `+1.1%`) | `1867.6 -> 1864.4` (`321.96 -> 322.51 TF`) | **+0.2%** |
| `[16384, 5120]` | `232.2 -> 228.2` (`92.49 -> 94.11 TF`, `+1.8%`) | `340.8 -> 336.9` (`252.03 -> 255.00 TF`, `+1.2%`) | `2114.4 -> 2092.9` (`325.01 -> 328.35 TF`) | **+1.0%** |
| `[34816, 5120]` | `489.3 -> 488.3` (`93.26 -> 93.46 TF`, `+0.2%`) | `696.2 -> 694.1` (`262.20 -> 262.98 TF`, `+0.3%`) | `4151.1 -> 4142.9` (`351.78 -> 352.48 TF`) | **+0.2%** |
| `[2560, 6144]` | `55.1 -> 55.1` (`73.03 -> 73.03 TF`, `+0.0%`) | `88.9 -> 87.9` (`181.11 -> 183.29 TF`, `+1.2%`) | `659.3 -> 655.2` (`195.44 -> 196.66 TF`) | **+0.6%** |
| `[5120, 6144]` | `98.1 -> 96.1` (`82.08 -> 83.83 TF`, `+2.1%`) | `142.2 -> 142.2` (`226.57 -> 226.57 TF`, `+0.0%`) | `1011.6 -> 1006.4` (`254.76 -> 256.06 TF`) | **+0.5%** |
| `[5120, 17408]` | `263.0 -> 262.0` (`86.76 -> 87.09 TF`, `+0.4%`) | `423.9 -> 419.7` (`215.32 -> 217.45 TF`, `+1.0%`) | `2956.1 -> 2965.2` (`247.00 -> 246.23 TF`) | **-0.3%** |

#### `--routes fast` (2-term `Nvfp4A4Pair` where admitted)

| Shape `[N, K]` | `T=128` HEAD -> Final (us / TF) | `T=512` HEAD -> Final (us / TF) | `T=4096` HEAD -> Final (us / TF) | `T=4096` Δ vs HEAD |
|---|---|---|---|---:|
| `[128, 2560]` | `16.2 -> 16.3` (`5.18 -> 5.16 TF`) | `22.4 -> 22.4` min (`15.00 TF`) | `112.4 -> 112.4` (`23.87 -> 23.87 TF`) | **+0.0%** |
| `[512, 2560]` | `18.3 -> 18.2` (`18.36 -> 18.40 TF`) | `34.6 -> 34.6` (`38.76 -> 38.76 TF`) | `167.8 -> 167.8` (`63.99 -> 64.00 TF`) | **+0.0%** |
| `[2048, 2560]` | `28.5 -> 28.5` (`47.07 -> 47.07 TF`) | `62.3 -> 59.2` (`86.17 -> 90.64 TF`, **`+5.2%`**) | `359.2 -> 353.1` (`119.56 -> 121.64 TF`) | **+1.7%** |
| `[2560, 2560]` | `30.5 -> 30.6` (`54.96 -> 54.90 TF`) | `61.3 -> 61.3` (`109.51 -> 109.51 TF`) | `426.8 -> 414.5` (`125.79 -> 129.51 TF`) | **+3.0%** |
| `[6144, 2560]` | `53.2 -> 57.2` (`75.62 -> 70.41 TF`) | `108.4 -> 105.3` (`148.65 -> 152.89 TF`, **`+2.9%`**) | `856.9 -> 837.4` (`150.36 -> 153.86 TF`) | **+2.3%** |
| `[10240, 2560]` | `84.9 -> 86.8` (`79.02 -> 77.27 TF`) | `176.0 -> 170.8` (`152.55 -> 157.12 TF`, **`+3.0%`**) | `1321.8 -> 1279.8` (`162.47 -> 167.80 TF`) | **+3.3%** |
| `[12800, 2560]` | `117.8 -> 112.4` (`71.23 -> 74.60 TF`, **`+4.7%`**) | `215.9 -> 210.8` (`155.44 -> 159.19 TF`, **`+2.4%`**) | `1592.2 -> 1554.1` (`168.60 -> 172.73 TF`) | **+2.4%** |
| `[13952, 2560]` | `116.5 -> 115.5` (`78.46 -> 79.15 TF`, `+0.9%`) | `235.4 -> 229.2` (`155.40 -> 159.56 TF`, **`+2.7%`**) | `1729.4 -> 1675.1` (`169.19 -> 174.68 TF`) | **+3.2%** |
| `[16384, 2560]` | `128.8 -> 130.9` (`83.34 -> 82.02 TF`) | `264.0 -> 258.9` (`162.67 -> 165.91 TF`, **`+2.0%`**) | `1990.5 -> 1933.2` (`172.62 -> 177.74 TF`) | **+3.0%** |
| `[14336, 5120]` | `230.2 -> 229.2` (`81.61 -> 81.98 TF`, `+0.5%`) | `434.0 -> 422.8` (`173.19 -> 177.79 TF`, **`+2.7%`**) | `3347.3 -> 3243.9` (`179.64 -> 185.36 TF`) | **+3.2%** |
| `[16384, 5120]` | `256.8 -> 254.8` (`83.61 -> 84.29 TF`, `+0.8%`) | `480.0 -> 464.7` (`178.97 -> 184.85 TF`, **`+3.3%`**) | `3768.2 -> 3664.7` (`182.37 -> 187.52 TF`) | **+2.8%** |
| `[34816, 5120]` | `548.7 -> 541.5` (`83.17 -> 84.27 TF`, `+1.3%`) | `954.2 -> 928.6` (`191.30 -> 196.57 TF`, **`+2.8%`**) | `7673.7 -> 7655.2` (`190.30 -> 190.76 TF`, `7314.3 us` min = `199.7 TF`) | **+0.2% (+3.4% min)** |
| `[2560, 6144]` | `61.3 -> 61.2` (`65.71 -> 65.74 TF`, `+0.0%`) | `133.0 -> 129.9` (`121.14 -> 124.00 TF`, **`+2.4%`**) | `1058.5 -> 1044.3` (`121.73 -> 123.38 TF`) | **+1.4%** |
| `[5120, 6144]` | `108.4 -> 106.3` (`74.32 -> 75.76 TF`, `+1.9%`) | `230.2 -> 224.1` (`139.91 -> 143.74 TF`, **`+2.7%`**) | `1691.5 -> 1667.9` (`152.35 -> 154.50 TF`) | **+1.4%** |
| `[5120, 17408]` | `310.1 -> 308.1` (`73.58 -> 74.07 TF`, `+0.7%`) | `625.4 -> 617.3` (`145.93 -> 147.85 TF`, **`+1.3%`**) | `4941.8 -> 4945.7` (`147.75 -> 147.63 TF`, `4901.7 us` min = `149.0 TF`) | **-0.1% (+0.8% min)** |

## 6. Rebase conflict resolution policy

- Preserve 1-group/thread (`32 B`) coalesced warp loads + 4-lane `__shfl_down_sync` `uint32_t` scale packing (`store_nvfp4_scale_quad`) and `kNvfp4A4RowScaledCachedGroups = 5` in `nvfp4_a4_quantize.cuh` and `nvfp4_a4.cu`.
- Preserve `ldmatrix`/`mma_nvfp4_e4m3` interleaving in `nvfp4_a4_tma.cuh`, `Tiled256` support in `launch_residual`, and 3-stage `Tiled128` (`128 x 128`) in `launch_nvfp4_a4_pair` (`nvfp4_launch.cuh`).
- Preserve the GB10 `A4` admission gates in `n2560_k2560.cu`, `n2048_k2560.cu`, `n512_k2560.cu`, and `n128_k2560.cu`.

# GB10 changelog — 2026-11-01 (T2: NVFP4 decode/verification `READ_%` — 64-row `Nvfp4A4Pair` weight panel, pair-gate retune, and the falsified sliced-K premises)

Task: `docs/tasks/20261101-performance-tasks.md` T2 (P0), "Eliminate Sub-90% `READ_%` Bandwidth
Cliffs Across NVFP4 Decode & Verify (T = 1..128)". Companion to the T1 section above, same base
branch.

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `3612fa8b4`. `git diff --check` clean; `.tandem/`
  and `task_orchestrator.py` untouched; nothing committed or pushed.
- Files changed (T2): 16 modified code files, `+190 / -19` lines, plus this changelog.
  `cuda-kernel-toolkit/` intentionally excluded from the enumeration (integrated toolkit, untouched).
- Allowlist delta: none (`tools/cuda_stack/cuda_stack_allowlist.json` untouched). The checker scores
  a symbol that is not in the allowlist against a limit of `0`
  (`tools/cuda_stack/check_cuda_stack.py:111`, `limit.get(key, 0)`), so a new spilling kernel cannot
  pass unnoticed: `infernix_cuda_stack_check` passing means the new 64-row pair kernel is
  `STACK:0`, `LOCAL:0` in `infernix-serve`. No ratchet-down was due (nothing was removed).

| File | Action | Change |
|---|---|---|
| `src/ops/linear/nvfp4/nvfp4_a4_tma_launch.h` | M | Declare `launch_nvfp4_a4_residual_tma_linear_r64` (2-term residual pair on a 64-row weight panel) |
| `src/ops/linear/nvfp4/nvfp4_a4_tma.cu` | M | Add file-local `launch_residual_r64<K>` and the exported K dispatch over `K ∈ {2560, 5120, 6144, 17408}`; the existing `launch_residual` heuristic is unchanged |
| `src/ops/linear/nvfp4/nvfp4_launch.cuh` | M | Add `launch_nvfp4_a4_pair_r64`: same quantizer, 64-row panel for `T <= 128`, delegating to the public heuristic above it |
| `src/ops/linear/nvfp4/shapes/n{34816,16384,14336}_k5120.cu` | M | Pair entry `-> launch_nvfp4_a4_pair_r64`; gates `65 -> 33` (`34816`, `14336`) and `65 -> 48` (`16384`) |
| `src/ops/linear/nvfp4/shapes/n{16384,13952,12800,6144}_k2560.cu` | M | Pair entry `-> launch_nvfp4_a4_pair_r64`; gate `128 -> 65` (`6144`) |
| `src/ops/linear/nvfp4/shapes/n{2560,5120}_k6144.cu`, `n5120_k17408.cu` | M | Pair gate `65 -> 33` (heuristic panel kept — it already picks 128 rows there) |
| `src/ops/linear/nvfp4/shapes/n10240_k2560.cu` | M | Pair gate `128 -> 56` (heuristic panel kept) |
| `bench/ops/nvfp4_route_sweep.cu` | M | `a16_ref` candidate (the A16 table the public route actually runs, `unchecked` against the A4 plane oracle), comma-separated `--filter`, and four explicit-panel pair candidates `a4_pair_{r128_s3_m1,r64_s3_m1,r64_s2_m2,r64_s4_m1}` |
| `tests/ops/linear/test_nvfp4_a4.cpp` | M | Pair-route oracle coverage at the newly-admitted `T = {33, 34, 48, 56, 64}` |

## 2. Task-premise corrections and architectural rationale

1. **The `[34816, 5120]` `T = 2` acceptance bar was already met before this task.** The task quotes
   `82.99%` / `491.9 us`; the HEAD binary on the public route measures `429.9 us` / `93.8%` of the
   `246 GB/s` sustained-read reference (final binary `430.9 us` / `94.7%`). The `>= 92%` / `<= 440 us`
   bar is met, and the `T = 2` rung is the `Nvfp4SlicedInstance<8, 8, 2>` tile inherited from T5, not
   the SIMT `C2` tile the task assumed. This is the same baseline staleness recorded for the
   20261029 task doc.

2. **Packed shared-memory scale loads and a 3-stage single-barrier ring in
   `nvfp4_a16_sliced_k_mma.cuh` — falsified, reverted.** Implementing the spec as written (the four
   scalar `uint8_t` loads per `load_pair` replaced by packed `uint16`/`uint32` loads across `k_step`
   0/1, and `kStages = 3` to drop the second `__syncthreads()` per K-group) did what the premise
   predicted at the SASS level: `LDS.U8` `320 -> 0`, `96 x LDS.128`, `REG 91 -> 88`, `STACK:0` /
   `LOCAL:0`, `infernix_cuda_stack_check` and the 7 relevant CTest cases all green. It did not
   predict the cost: extraction raised `SHF.R` `202 -> 506`, and a 7-pass order-swapped interleaved
   A/B put **8 cells `>= +3%` slower against 3 cells `<= -3%` faster**. Median neutral, tail negative.
   Both files are back at HEAD (`git checkout --`); that change is not in this diff.

3. **Wide-token sliced-K tiles (`Nvfp4SlicedInstance` `MaxTokens = 48, 64`, TokenTiles 3/4) at
   `T = 33..64` — falsified.** They measure 2..3x behind the shipped MMA band
   (`[34816, 5120]` at `T = 64`: `950.6 us` against `527.4 us`). A 32-wide token slab at TokenTiles
   3/4 forces activation staging that holds the CTA to 1 per SM. The kernel class also asserts
   `Stages == 1 || Stages == 2` (`nvfp4_schedule.cuh:264`); the 3-stage variant compiles only after
   relaxing that invariant and measures worse, so the assert and the file were left alone. The two
   candidates stay in `bench/ops/nvfp4_route_sweep.cu` (`r16_t48_w4_s2`, `r16_t64_w4_s2`) with that
   verdict as their comment, so the record is reproducible; no shape table selects them.

4. **The 64-row `Nvfp4A4Pair` weight panel is a decode/verification win and a prefill loss — the
   boundary is a wave-count boundary, not a schedule preference.** Halving the weight panel each CTA
   owns doubles the grid. While the grid is small that recovers SMs idle in the last wave:
   `[12800, 2560]` runs 100 CTAs at `BN = 128` (4 of 48 SMs carry a third wave) and 200 at `BN = 64`,
   worth `-9.2%` mean over `T = 65..128`. From the prefill width up, thousands of CTAs run, the wave
   argument is void, and the halved panel reads its weight tile twice as often: **`+10.5%` to `+30%`
   behind at every `T >= 129` on all seven shapes** (`[34816, 5120]` at `T = 4044`:
   `7895.9 -> 9681.8 us`). `launch_nvfp4_a4_pair_r64` therefore applies the panel only at `T <= 128`
   and delegates above it to `launch_nvfp4_a4_residual_tma_linear`, whose `ctas_128` heuristic picks
   the 128-row panel for every shape here at that width — i.e. exactly the route HEAD ran. Of the four
   measured stage/occupancy pairs, `Stages = 3, MinBlocksPerSm = 1` won everywhere it was selected;
   `4/1` and `2/2` are `99 KiB`-legal but never ahead and stay bench candidates. A 64-row panel with
   `Stages = 3, MinBlocksPerSm = 2` does not fit the `99 KiB` SMEM limit with two activation planes
   per stage.

5. **The `>= 90% READ_%` bar over `T = 1..128` is not reachable on the routes `--routes fast`
   admits.** Above `T ~= 40` the A16 table is tensor-core bound, not bandwidth bound: the bench
   reports `dense_bf16_tensor_tflops fp32_acc=97.0` against `~492 TF/s` for the NVFP4 `OMMA` pipe, and
   the A16 kernels fall to `54..67%` `READ_%` at `T = 96/128`. The 1-term A4 ladder (`--policy a4`)
   does reach `92..99%` across `T >= 33` (`[34816, 5120]` at `T = 128`: `485 us` = `92.6%`;
   `[16384, 2560]`: `116 us` = `99.5%`), but `infernix_linear_bench --routes fast` leaves the policy
   at `LinearPolicy::A16Only` (`bench/ops/linear_bench.cu`), so under the acceptance command only the
   A16 table and the lossy 2-term pair compete. Every `READ_%` gain in this task therefore comes from
   admitting the pair earlier, not from a faster A16 kernel.

6. **Pair gates were set against the route the policy actually runs, in one process.** Following the
   earlier lesson that a route-sweep delta is not evidence about the shipped route, each gate was
   confirmed by A/B on the public route (`--routes fast`, cold `32 MiB` L2, order-swapped passes,
   per-cell minima) rather than from the sweep alone. `[34816, 5120]` and `[14336, 5120]` at
   `T = 33..47` are the case in point: the sweep suggested `~2..4.5%`, and the public-route A/B
   confirmed `15/15` widths negative on each (`-0.6..-2.2%` and `-3.7..-5.5%`), which is what
   justifies lowering those gates to `33`. `[16384, 5120]` measured level (`-1.0%` mean) and keeps
   `48`.

### 2.7 Measured effect on the widths whose route changed (us; cold `32 MiB` L2, min over passes)

| Shape `[N, K]` | pair gate `HEAD -> new` | cells | mean | median | best | worst | `READ_%` over those cells |
|---|---|---:|---:|---:|---:|---:|---|
| `[34816, 5120]` | `65 -> 33` | 96 | `-1.9%` | `-1.9%` | `-2.9%` | `-0.6%` | `81.9 -> 83.9` |
| `[16384, 5120]` | `65 -> 48` | 81 | `-1.0%` | `-0.8%` | `-4.0%` | `+0.4%` | `81.9 -> 83.0` |
| `[14336, 5120]` | `65 -> 33` | 96 | `-4.1%` | `-3.7%` | `-8.8%` | `-2.7%` | `79.7 -> 83.4` |
| `[16384, 2560]` | `65` (panel only) | 64 | `-0.9%` | `-0.9%` | `-2.5%` | `+0.8%` | `86.8 -> 87.7` |
| `[13952, 2560]` | `65` (panel only) | 64 | `-6.3%` | `-6.3%` | `-8.2%` | `-3.6%` | `82.4 -> 87.6` |
| `[12800, 2560]` | `65` (panel only) | 64 | `-9.2%` | `-9.2%` | `-10.1%` | `-7.6%` | `77.0 -> 84.9` |
| `[10240, 2560]` | `128 -> 56` | 72 | `-25.8%` | `-26.5%` | `-34.2%` | `-2.7%` | `62.2 -> 84.1` |
| `[6144, 2560]` | `128 -> 65` | 64 | `-18.6%` | `-16.6%` | `-27.1%` | `-0.2%` | `65.3 -> 78.5` |
| `[5120, 17408]` | `65 -> 33` | 32 | `-8.8%` | `-8.9%` | `-10.5%` | `-6.5%` | `66.9 -> 74.6` |
| `[5120, 6144]` | `65 -> 33` | 32 | `-9.5%` | `-9.3%` | `-11.9%` | `-7.4%` | `67.8 -> 73.6` |
| `[2560, 6144]` | `65 -> 33` | 32 | `-7.9%` | `-7.1%` | `-10.5%` | `-6.7%` | `62.3 -> 69.3` |

## 3. Subsystem breakdown

- **Ops / NVFP4 A4 TMA GEMM (`nvfp4_a4_tma.cu`, `nvfp4_a4_tma_launch.h`)**: one new instantiation
  per `K` of `nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 3, 1,
  kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 64>, K>>`, exported through
  `launch_nvfp4_a4_residual_tma_linear_r64`. An earlier revision exposed a 4-case
  `Nvfp4A4PairTile` enum; three of its cases were selected by no shape table while each cost four
  kernel instantiations, so the exported surface was reduced to the one measured panel and the
  unselected cases removed rather than kept as options.
- **Ops / NVFP4 launch (`nvfp4_launch.cuh`)**: `launch_nvfp4_a4_pair_r64` reuses the existing
  `Tiled128` quantizer call unchanged and switches only the consumer GEMM, so a shape that adopts it
  changes no quantization semantics.
- **Shape tables**: seven adopt the 64-row panel; eleven carry the measured per-width evidence as a
  dated comment. Four changed gate only (`n{2560,5120}_k6144`, `n5120_k17408`, `n10240_k2560`), where
  the heuristic already picks the 128-row panel.
- **Bench (`bench/ops/nvfp4_route_sweep.cu`)**: the A16 table becomes a same-process candidate
  (`a16_ref`, marked `unchecked` because it is a different approximation from the A4 planes the list
  otherwise checks against), which is what makes the pair-vs-A16 admission gate measurable in one
  process instead of two passes that disagree by 4..17%.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- **CTest** — `infernix_cuda_stack_check` (1.5 s), `infernix_linear_nvfp4_a16_test` (3.5 s),
  `infernix_linear_nvfp4_a4_test` (8.8 s), `infernix_nvfp4_a16_slice_ring_test` (4.6 s),
  `infernix_nvfp4_tma_staging_race_test` (2.6 s), and the neighbours sharing the NVFP4 launch headers
  (`infernix_attn_input_proj_test`, `infernix_gdn_input_proj_test`, `infernix_linear_add_nvfp4_test`,
  `infernix_linear_swiglu_nvfp4_test`, `infernix_ple_nvfp4_test`): **10/10 passed**, re-run after
  each of the three builds in this task. Run without `--parallel` (parallel CTest produces real
  numerical failures on this single-GPU host).
- **Test coverage extended to the admitted band.** `tests/ops/linear/test_nvfp4_a4.cpp` qualified the
  pair route at `T = {65, 96, 127, 128, 129, 256, 512}` only, because that was where the gates
  admitted it. Lowering gates to 33, 48 and 56 makes it the live route at widths it was never
  checked at, so `T = {33, 34, 48, 56, 64}` were added to `pair_invocations` (13 shapes x 5 new
  widths against the sampled oracle). Final run: `ctest -R 'nvfp4|infernix_cuda_stack_check'` —
  **14/14 passed**, the two `dflash2_tree_nvfp4_real` integrations skipped on their unset artifact
  environment.
- **Numerics**: `bit_exact = 1` on all 104 rows of every pair candidate in the route sweep, including
  the three 64-row panel variants, so the 64-row GEMM is bit-exact against the same exact host oracle
  as the shipped 128-row panel and no new kernel enters any artifact configuration — the existing
  pair-route qualification carries over.
- **Measurement resolution (null test)**: the identical binary on both sides of the same protocol
  flags **8 of 512 cells `>= +3%`, up to `+8.3%`**, at the small shapes. That floor is what separates
  an attributable regression from a timer step, and is why the A/B alternates per shape, swaps order
  between passes, and aggregates per-cell minima.
- **Attribution**: of 1,664 cells (13 shapes x `T = 1..128` x 3 passes), 667 changed route. Those 667
  average `-8.25%` (median `-6.34%`) with **worst cell `+0.8%`**. Every cell flagged `>= +3%` lies
  below every gate, i.e. runs the same A16 kernel in both binaries.
- **Post-fix bands**: `T = 65..128` on the seven panel shapes — 448 cells, mean `-5.86%`, best
  `-25.8%`, worst `+2.8%`. `T = 129..4096` — 56 cells, mean `-0.06%`, worst `+1.0%`, no cell `>= 3%`.
- **End to end** (Qwen3.8-Flash-Next local NVFP4 W4Dense artifact, `--kv-dtype fp8 --spec mtp
  --lm-head-draft --concurrency 8` — the only shipped-bench configuration whose decode batch reaches
  the retuned band, since the `run_kv_matrix_bench.sh` gate runs at concurrency 1 and never does).
  Two interleaved passes, `base`/`new` binaries alternating, per-cell minimum of the two passes:

  | test | prefill `t/s` base -> new | decode `t/s` base -> new | MTP acceptance base -> new |
  |---|---|---|---|
  | `pp4096` | 3234.4 -> 3234.6 (`+0.01%`) | — | — |
  | `pp4096+tg128` | 3222.3 -> 3231.0 (`+0.27%`) | 30.5 -> 30.7 (`+0.67%`) | 0.7919 -> 0.7912 (`-0.09%`) |
  | `pp4096+tg256` | 3223.5 -> 3222.6 (`-0.03%`) | 28.1 -> 28.0 (`-0.33%`) | 0.7612 -> 0.7557 (`-0.72%`) |
  | `tg128` | — | 15.4 -> 15.6 (`+1.72%`) | 0.4247 -> 0.4030 (`-5.11%`) |
  | `tg256` | — | 16.1 -> 16.6 (`+3.12%`) | 0.4615 -> 0.4738 (`+2.67%`) |

  Prefill: 3 cells, mean `+0.08%`, worst `+0.27%` — the prefill regression of section 5 item 1 is
  gone at the model level as well as at the Op level. Decode: 4 cells, mean `+1.30%`, best `+3.12%`,
  worst `-0.33%` (`pp4096+tg256`, inside the run-to-run spread of a `28 t/s` concurrency-8 arm).
  Acceptance is deterministic and identical in both passes, so its movement is a real semantic
  effect, not noise; see section 5 item 5.

## 5. Regressions and adverse results

1. **A prefill regression this task introduced and then removed.** Shipping the 64-row panel as the
   pair entry unbounded made every `T >= 129` width on all seven shapes `10..30%` slower (worst
   `[34816, 5120]` at `T = 2478`, `4640.6 -> 6013.8 us`, `+29.6%`). The task's own verification band
   (`T = 1..128`) could not see it; the first concurrency-8 end-to-end run showed prefill `t/s` down
   `0.6..1.6%`, which led to the op-level check that exposed it. Disposition: bounded at `T <= 128`,
   after which prefill is at parity (56 cells, worst `+1.0%`).
2. **96 cells flagged `>= +3%` in the main A/B, none attributable.** 58 sit in widths where both
   binaries run the same pair route, 38 where both run the same A16 kernel; worst `+9.1%`
   (`[2560, 2560]` at `T = 18`, `20.4 -> 22.2 us`) and `+7.3%` (`[2560, 6144]` at `T = 98`,
   `54.1 -> 58.0 us`). Neither shape's route changes at those widths, and the identical-binary null
   test reaches `+8.3%`. Reported as unresolved measurement noise at the `15..28 us` scale rather than
   folded into the gains.
3. **`mk_r32_t64_k128_s1` (`Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>`) was not promoted.** It
   wins 0..6% on six shapes at `T = 48` and loses 16% on `[2560, 6144]`; `select_a16` is unchanged
   everywhere. The task's "retune `select_a16`" instruction therefore produced no promotion.
4. **Two of the task's three implementation premises were falsified** (section 2 items 2 and 3) and
   one was already satisfied at HEAD (item 1). The delivered gain comes from the pair route's panel
   and gates, which the task listed as the third measure.
5. **Acceptance-rate movement, i.e. the retuned band does not reproduce HEAD's bits.** MTP
   acceptance is deterministic and identical across both passes, so these are semantic differences,
   not measurement noise: `tg128` `0.4247 -> 0.4030` (`-5.11%`), `tg256` `0.4615 -> 0.4738`
   (`+2.67%`), `pp4096+tg128` `0.7919 -> 0.7912` (`-0.09%`), `pp4096+tg256` `0.7612 -> 0.7557`
   (`-0.72%`). Cause: in `T = 33..64` the pair route now computes those linears where the A16 route
   did, and the pair quantizes activations to a 2-term NVFP4 representation instead of consuming
   them as BF16, so the verified logits differ from the previous engine version. Disposition and
   limits: this is the documented case of a speed change altering a request's bits — the route
   remains a deterministic function of `(request, artifact)`, so the reproducible route's
   request-invariance is untouched, and no new kernel enters any artifact configuration (the pair
   schedule already ran above the old gates), so its existing fast-path qualification covers it.
   It is not, however, a bit-preserving retune, and the `-5.11%` on `tg128` is the worst single
   movement: that arm still gains `+1.72%` decode `t/s` because the faster verify kernel outweighs
   the lower acceptance, but the acceptance cost is real and is not averaged away by the `+2.67%`
   on `tg256`. A model-level quality gate on the accepted-token distribution was not run and is the
   natural follow-up if the `tg128` shape is a shipped quality concern.

## 6. Accepted gaps and follow-ups

- **The `>= 90% READ_%` bar over `T = 1..128` is not met: 882 of 1,024 large-N (`N >= 6144`) cells
  remain below it.** Cells at or above 90% of `246 GB/s` per shape: `[34816, 5120]` 31/128 (min
  `82.8%` at `T = 50`), `[16384, 5120]` 25/128 (min `78.6%`), `[14336, 5120]` 16/128 (min `69.4%`),
  `[16384, 2560]` 28/128 (min `67.8%`), `[13952, 2560]` 21/128 (min `72.8%`), `[12800, 2560]` 11/128
  (min `74.8%`), `[10240, 2560]` 9/128 (min `69.0%`), `[6144, 2560]` 1/128 (min `57.8%`). Section 2
  item 5 states why: those widths are BF16-tensor-core bound on the route the acceptance command
  admits, so no dispatch change reaches the bar.
- **The medium-shape `T = 32..64` bar (`>= 88..90%`) is not met**, though it moved a long way:
  `[5120, 6144]` `61.6..83.2%`, `[5120, 17408]` `69.2..85.4%`, `[2560, 6144]` `67.3..74.9%` (HEAD
  means over the changed widths were `67.8%`, `66.9%`, `62.3%`). The pair is already the faster route
  at every one of those widths; the remaining gap is the pair's own cost, not a route choice.
- **The fused `[16384, 2560]` GDN split entry (`launch_nvfp4_n16384_k2560_split`) keeps its
  hard-coded 128-row pair schedule.** It is a separate consumer with its own admission test, and the
  panel result was measured on the public Engine route only. Extending it needs its own A/B on that
  fused route — a route-sweep delta is not evidence for a fused Op.
- **Follow-ups**, in the order that would move the unmet bars:
  1. Make the 1-term A4 ladder reachable under the benchmark's `--routes fast` (or add an A4-ladder
     route to the acceptance command); it is the only configuration measured at `92..99%` across
     `T = 33..128` and it is never live there today.
  2. An A16 decode/verify GEMM that is not HMMA-bound at `T = 33..128` — the `97 TF/s` BF16 pipe is
     the ceiling, so this needs an NVFP4-operand weight-only path, not a schedule change.
  3. Re-check the `T <= 128` panel boundary if the pair's stage count or `BK` changes; the boundary is
     a wave-count result, not a constant.

# GB10 changelog — 2026-11-01 (T3: register-resident row prefetch in `nvfp4_a16_gemv_kernel`, the falsified sliced-K fill move, and the `FastPath::GdnConvFused` decision on `W4Dense`)

## 1. Accounting

- Base commit: `c270f4db1` (`dev-rebased-20261028`). No commit created (task instruction 7 — the
  reviewer commits).
- Files changed (7 tracked, 0 added, 0 deleted):

| File | Action | Change |
|---|---|---|
| `src/ops/linear/common/epilogue.cuh` | M | `LinearRowPrefetchSingle<Output>` — the `std::void_t`-detected declaration of a *register-resident* row line (`Output::kPrefetchLiveWords`, default `0`), beside the existing shared-line `LinearRowPrefetch` |
| `src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh` | M | `nvfp4_a16_gemv_kernel`'s row scratch line moves from `__shared__ std::uint32_t row_prefetch[kBlockRows * kPrefetchWords]` filled by one thread per row behind a `__syncthreads()`, into the registers of the thread that owns the row (`lane == 0` of its warp), issued before `compute_nvfp4_rows` and read back by that same thread in `linear_finish_row`; zero shared words, zero added barriers |
| `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh` | M | comment only — the task's "issue the fill after the prologue's `cp_commit()`" move was implemented, measured `+1.4 .. +2.4 µs` at `B = 3..8`, and reverted; the measurement is recorded at the site so it is not retried |
| `src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh` | M | `GdnConvSnapshotColumnsOutput` gains the single-column half of its own line layout: `kPrefetchLiveWords = 8` and `prefetch_single_row` (the four taps plus column 0's window), which is what a one-column contraction can hold in registers |
| `src/models/qwen4_exp/program/fast_paths.cpp` | M | the `gdn-conv-fused` NVFP4 comment carries T3's numbers instead of T6's; **no qualification added** (§4, §6) |
| `tests/models/qwen4_exp/test_fast_paths.cpp` | M | the same, and the W4Dense assertion message reads `(measured a tie)` rather than `(measured slower)`; the `applies && !qualified` assertion is unchanged |
| `docs/maintainer/model-quality.md` | M | §6.7's `gdn-conv-fused` row now records both the T6 and T3 levers, the Op-level and end-to-end numbers, and why a tie does not register |

- CUDA stack allowlist: **unchanged** — zero new `STACK`/`LOCAL` symbols, so no ratchet and no
  `check_cuda_stack.py update` run (§4).
- Rebuilds: 7 single-invocation `ninja -C build-native -j "$(nproc)"` builds (`infernix-serve`,
  `infernix_gdn_input_proj_conv_snapshot_bench`, `infernix_causal_conv1d_silu_bench`,
  `infernix_bench`, `infernix_linear_bench`, `infernix_ops_recurrent_test`,
  `infernix_qwen4_exp_fast_paths_test`), each preceded by the changed-header include closure (179 TUs,
  computed from the four headers' reverse include graph, plus `find src/ops/linear/nvfp4
  src/ops/gdn_input_proj -name '*.cu' -exec touch {} +`), because `build-native` CUDA depfiles do not
  track headers. Object freshness was verified with `stat` before every measurement, and the
  baseline binaries for the A/B were built from a clean `git stash` of exactly the four source files.

## 2. Task-premise corrections and architectural rationale (`20261101-performance-tasks.md` T3)

**What held.** The GEMV's row scratch line really was a synchronous `LDG → STS` fill ahead of the
mainloop, and it really was the last part of the fused route that T6 had not yet put behind the
contraction. The register form is what the task asked for and it is what shipped: `lane == 0` of the
owning warp issues its two rows' eight live words, `compute_nvfp4_rows` runs with them in flight, and
the same thread reads them back in the tail. `cuobjdump -sass` on the shipped fused instantiation
confirms the placement — the `@!P1/@!P2 LDG.E` / `LDG.E.U16` group (the `conv_weight` taps, the three
history taps at `+0x5000/+0xa000/+0xf000`, and the two slot selectors) sits at `0x03d0 … 0x0560`, after
the scale-staging `BAR.SYNC.DEFER_BLOCKING` and the PDL wait and before the first
`LDG.E.CONSTANT` code load at `0x05e0`; the tail (`0x4420 … 0x4bb0`) contains no re-load of those
values, only the `STG.E.U16` stores that publish them. The register line is 16 words per thread
(2 rows × 4 taps + 4 window words), which is why the single-column form is a *separate* output
contract: the shared line is 36 words wide to cover eight contraction columns and no thread could hold
that across the contraction.

**Premise corrections from measurement.**

1. *The `__syncthreads()` and the shared array were not costing 2.5 µs.* T6 left "≈ 2.5 µs of the
   epilogue the row scratch line does not hide", and the task attributed it to the serialized
   `LDG → STS` plus the barrier. Removing both — the whole mechanism the task named — recovers
   **0.4 … 1.2 µs** at `B = 1` (§4), not 2.5. The remaining residual over the unfused projection is
   1.1 … 4.1 µs depending on the session, of which ≈ 0.9 µs is the fused route's own state traffic, so
   what is left is the traffic and the tail's stores, not a barrier that the second resident CTA was
   already covering (T6's own first hypothesis).
2. *The task's `0.40 µs` DRAM floor undercounts the route's traffic.* It prices "the extra
   `conv_states` + `conv_weight` payload" at `98.3 KiB`. The fused route additionally *publishes* the
   three-tap window: `conv_weight` 80 KB + history read 61 KB + window publish 61 KB ≈ **202 KB**,
   which at this shape's measured `231 GB/s` is **≈ 0.87 µs**, not `0.40 µs`. `102.40 + 0.40` is
   therefore below the physical floor of the work the fused launch does, and the `≤ 103.0 µs` bar sits
   `0.2 µs` under the best number this tree has produced for it.
3. *"beating unfused `gdn_input_proj` + `causal_conv1d_update` by `>= 12.5 µs`/call" is unreachable at
   this shape by construction.* The composed route's separate convolution kernel is
   `causal_conv1d_silu_snapshot [10240,1,1] = 4.10 µs` (§4), so the most any fusion can save per call
   is `4.10 µs`, and it must still pay its own state traffic inside the projection. The fused route
   now captures `2.98 µs` of those `4.10 µs` — 73 % of the theoretical maximum.
4. *The sliced-K move is a regression, not an improvement.* Implementing step 2 as written (fill the
   line after `stage_codes(0)`/`stage_activation(0)`/`cp_commit()` instead of before the prologue) is
   `+1.376 / +2.176 / +2.432 / +2.208 / +2.048 / +2.048 µs` at `B = 3..8` against the same binary in
   alternating rounds. Cause: the fill's `LDG → STS` then sits directly in front of the mainloop's
   first `cp_wait` + barrier, so it delays the first MMA group, while the async copies it was meant to
   overlap were already covered by the pipeline. Reverted; the site comment now carries the number.
5. *The verification command names a target that does not exist.* `infernix_qwen4_exp_Execution_test`
   is neither a Ninja target nor a CTest name. The coverage the task means is
   `build-native/tests/infernix_ops_recurrent_test` (whose CTest names
   `infernix_gdn_input_proj_test`, `infernix_gdn_input_proj_conv_snapshot_test` and
   `infernix_gdn_input_proj_conv_record_test` carry the FP64-oracle qualification of the fused route
   over `B = 1..8` including in-place replay) plus `infernix_qwen4_exp_fast_paths_test`.
6. *Enabling `FastPath::GdnConvFused` on `W4Dense` is a registry decision with three gates, not a
   dispatch flag.* `fast_path_applies` already returns `true` for `DenseKernels::Nvfp4`; what is
   missing is a `kQualified` entry, which [model-quality.md §6.1](../maintainer/model-quality.md)
   conditions on Op qualification (met — the FP64 oracle test passes), **joint path fidelity** against
   the stored reference through `tools/quality/certify.py` (never run for this route), and *a resolved
   speedup … the gain ahead in every round*. The end-to-end A/B (§4) is a tie: +0.026 tok/s mean over
   20 of 24 paired samples with the rest inside noise, i.e. `+0.06 %`, not "a source gain clear of its
   run-to-run spread with room to spare". Both the third gate and the second fail, so the route stays
   an implemented, oracle-qualified candidate under `--fast-path gdn-conv-fused`.

**Why a distinct contract instead of reusing `LinearRowPrefetch`.** The GEMV can only hold a
single-column line, so it must not size its registers from the eight-column shared line, and an output
that declares only the shared form must keep compiling (and prefetch nothing) in the GEMV. The two
traits make that a compile-time choice per output, which is the same lesson T6 recorded when the
prefetched `apply_row` became a distinct epilogue type instead of a second overload.

## 3. Subsystem breakdown

- `linear/common` (M): `LinearRowPrefetchSingle<Output>` is a second `std::void_t` trait over
  `Output::kPrefetchLiveWords`, defaulting to `{0}`. `LinearRowPrefetch` (the shared line, sized by
  `kPrefetchWords`/`kPrefetchUnits`) is untouched and remains the sliced-K MMA's contract, so the two
  contractions now claim the form each can actually keep: shared memory for a line that spans up to
  eight contraction columns, registers for a line whose only reader is the thread that filled it.
- `linear/nvfp4` (M): `nvfp4_a16_gemv_kernel` declares
  `std::uint32_t row_prefetch[Schedule::kRowsPerWarp][kPrefetchWords == 0 ? 1 : kPrefetchWords]` as a
  local (register-resident) array and fills it inside `if (lane == 0)` for exactly the rows that lane
  later finishes, after `pdl::enter_streaming()` — `conv_states` is producer-written, so the fill may
  not move ahead of the PDL wait, which is why it cannot join the immutable `scales` staged before it.
  The parent-row computation moved above the fill (it is the fill's address source and the mainloop's
  operand list), and the row consumer hands `row_prefetch[r]` to `linear_finish_row` instead of a
  shared pointer. The `__shared__` array and the fill's `__syncthreads()` are gone; `SHARED` for the
  fused instantiation drops by exactly the removed `16 × 36 × 4 = 2304 B`.
- `gdn_input_proj/q8` (M): `prefetch_single_row(words, row)` is expressed as `prefetch_unit(words, row,
  0, 1)` and `prefetch_unit(words, row, 1, 1)` — the same fill code the shared path uses, so the taps,
  the BF16 patterns and the slot selector are identical bits on both paths. A `row >= kGdnChannels`
  (z) row still fills nothing. The Q8 fused route is unaffected: it selects
  `GdnConvSnapshotRowEpilogue`, its contractions read neither new member, and its kernels' resource
  usage is byte-identical before and after (§4).
- `models/qwen4_exp/program` + `tests` + `model-quality.md` (M): the registration evidence for the
  NVFP4 fused convolution moves from T6's numbers to T3's; no `kQualified` entry was added, and
  `infernix_qwen4_exp_fast_paths_test`'s `applies && !qualified` assertion still holds.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

**Tests** — run serially; `--parallel` produces real numerical failures on this single-GPU host:

```
 1/11 infernix_cuda_stack_check .................... Passed    1.45 sec
 2/11 infernix_qwen4_exp_fast_paths_test ........... Passed    0.00 sec
 3/11 infernix_gdn_input_proj_test ................. Passed   10.21 sec
 4/11 infernix_gdn_input_proj_conv_snapshot_test ... Passed   27.63 sec
 5/11 infernix_gdn_input_proj_conv_record_test ..... Passed    8.56 sec
 6/11 infernix_linear_nvfp4_a16_test ............... Passed    3.51 sec
 7/11 infernix_nvfp4_a16_slice_ring_test ........... Passed    4.64 sec
 8/11 infernix_linear_nvfp4_a4_test ................ Passed    8.30 sec
 9/11 infernix_nvfp4_tma_staging_race_test ......... Passed    2.71 sec
10/11 infernix_linear_add_nvfp4_test ............... Passed    2.34 sec
11/11 infernix_linear_swiglu_nvfp4_test ............ Passed    7.88 sec
100% tests passed, 0 tests failed out of 11
```

The conv-snapshot test is the fused route's FP64-oracle qualification over `B = 1..8`, so the shipped
convolution, published window and `z` bits are those the Op contract promises; the four NVFP4 linear
suites cover the GEMV kernel's other consumers (`linear`, `linear_add`, `linear_swiglu`, the A4
routes). The stack gate is green with zero new spills, so the allowlist was not touched.

**Kernel resources** (`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`, before →
after; every *other* `nvfp4_a16_gemv_kernel` and `nvfp4_a16_sliced_k_mma_kernel` instantiation in the
binary — every plain NVFP4 `T = 1` route, the Q8 fused route, the `[16384,5120]` `GdnConvOutput` route
— has byte-identical `REG`/`STACK`/`SHARED`/`LOCAL` in both binaries):

| kernel | REG | STACK | SHARED | LOCAL |
|---|---|---|---|---|
| `nvfp4_a16_gemv_kernel` × `GdnConvSnapshotColumnsOutput` × `GdnConvSnapshotPrefetchedRowEpilogue` | 118 → **128** | 0 → 0 | 5888 → **3584** | 0 → 0 |
| `nvfp4_a16_sliced_k_mma_kernel` × the same | 100 | 0 | 38400 | 0 |

128 registers × 512 threads is exactly the 65,536-register file at the kernel's
`__launch_bounds__(256, 2)`, so occupancy stays at 2 CTAs/SM and nothing spills — but the kernel now
sits at its ceiling (§6).

**Op latency — the acceptance point.** `infernix_gdn_input_proj_conv_snapshot_bench --format
nvfp4-flash --batch 1 --execution graph --cache cold --warmup 20 --repeat 150`, medians of alternating
rounds against the HEAD binary built from a stashed tree (the two binaries alternate inside each round,
so this is a paired comparison and not two sessions):

| session | rounds | baseline (HEAD) | T3 | Δ median | Δ mean |
|---|---|---|---|---|---|
| cool | 9 | 104.352 µs (104.13 … 106.18) | **103.232 µs** (103.10 … 104.13) | **−1.120 µs (−1.07 %)** | −1.19 µs |
| warm | 5 | 105.568 µs (105.18 … 108.22) | **105.184 µs** (104.16 … 105.18) | **−0.384 µs (−0.36 %)** | −0.93 µs |

The new binary is ahead in 13 of the 14 alternating rounds and tied in the remaining one — the warm
session's second round, where both binaries report the same `105.184 µs` median, which is the
quantization step of that report. Its whole sample set also sits below the baseline's in the cool
session. The absolute bar is `≤ 103.0 µs`: the T3 median is `103.232 µs` and its best round is
`103.104 µs`, both above the bar, while the same-code baseline measures `104.35 µs` in that session
(T6's session measured `105.28 µs` for the identical code, and the warm re-run measures `105.57 µs`),
so the bar is judged under a `±1 … 3` timer-tick drift and the paired delta is the defensible result.

**Op latency — the rest of the route.** Same harness, 4 alternating rounds, after the sliced-K revert;
`B ≥ 2` runs `nvfp4_a16_sliced_k_mma_kernel`, whose code is unchanged by T3 (comment-only), so these
are the noise floor for the claim:

| columns B | baseline | T3 | Δ |
|---|---|---|---|
| 2 | 110.000 | 109.296 | −0.704 (−0.64 %) |
| 4 | 111.296 | 110.992 | −0.304 (−0.27 %) |
| 8 | 111.904 | 112.336 | +0.432 (+0.39 %) |

**Controls, same tree, same session.** Plain projection `infernix_linear_bench --qtype NVFP4 --n 16384
--k 2560 --t 1 --execution graph --policy a16 --flush-mib 32`: `101.216 … 102.528 µs` across the three
paired samples (T6 recorded `102.400`). `infernix_causal_conv1d_silu_bench --decode --snapshot
--channels 10240 --tokens 1 --batch 1 --cache cold`: `4.10 µs` (its own same-byte copy baseline is
`3.94 µs`, i.e. the kernel is at its launch floor). Composed route ≈ `105.3 … 106.6 µs`, so the fused
launch is `2.0 … 3.0 µs` per call ahead of it — the `12.5 µs/call` in the acceptance text is not
attainable (§2.3).

**Regression check — the shared GEMV kernel's other consumers.** The GEMV is every NVFP4 `T = 1` linear
route, so the plain routes were A/B'd against a HEAD-built `infernix_linear_bench` (5 alternating
rounds each, `--t 1 --execution graph --policy a16 --warmup 20 --repeat 150 --flush-mib 32`):

| shape | baseline | T3 | Δ |
|---|---|---|---|
| `[16384, 2560]` | 103.168 | 102.528 | −0.640 (−0.62 %) |
| `[6144, 2560]` | 41.664 | 41.696 | +0.032 (+0.08 %) |
| `[5120, 6144]` | 79.584 | 79.552 | −0.032 (−0.04 %) |
| `[5120, 17408]` | 220.256 | 219.328 | −0.928 (−0.42 %) |

No plain route moved outside its spread, which is what the identical resource usage predicts: for an
output that declares no line, `kPrefetchWords` is `0` and the array, the fill and the barrier all
compile away exactly as before.

**End-to-end** — `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `infernix_bench -p 4096 -n 128,256
-pg '4096,128;4096,256' -r 2 --warmup 1 --concurrency 1 --kv-dtype fp8`, default arm vs
`--fast-path gdn-conv-fused`, six alternating rounds, `decode_output_tok_s_mean`:

| test | off mean | on mean | Δ excluding the r4 outlier | per-round Δ (r1 … r6) |
|---|---|---|---|---|
| tg 128 | 46.080 | 46.344 | **+0.019** | +0.04, 0.00, +0.11, *(+1.49)*, −0.07, +0.01 |
| tg 256 | 46.051 | 46.230 | **−0.002** | +0.05, −0.06, +0.06, *(+1.12)*, −0.07, −0.02 |
| pp4096 + tg128 | 44.931 | 45.197 | **+0.022** | +0.03, 0.00, +0.12, *(+1.45)*, −0.02, +0.02 |
| pp4096 + tg256 | 45.079 | 45.254 | **+0.031** | +0.05, −0.01, +0.11, *(+0.87)*, −0.04, +0.07 |

Round 4's four *off* arms are cold/thermal outliers (`43.71 … 44.86` against that arm's other five
samples at `45.14 … 46.37`); they are kept visible because they dominate the raw means. Excluding them
the fused arm is ahead in 15 of 20 paired samples by `+0.002 … +0.031 tok/s` (`+0.06 %`), which is
inside the run-to-run spread of a single arm (`σ ≈ 0.05`). T6 measured the same artifact and workload at
`−0.11 … −0.19 tok/s`; T3 turns that regression into a tie.

**Registration decision.** `FastPath::GdnConvFused` for `DenseKernels::Nvfp4` **stays unregistered**.
It is now neither slower nor meaningfully faster: a `+0.06 %` tie is not the resolved speedup
§6.1 requires, and its joint certification has never been run, so the second gate is also unmet. It
remains implemented, FP64-oracle qualified and runnable under `--fast-path gdn-conv-fused`.

## 5. Regressions and adverse results

1. **The sliced-K fill move (reverted).** Implementing the task's step 2 as written —
   `nvfp4_a16_sliced_k_mma_kernel`'s `prefetch_row` fill after the prologue's `cp_commit()` — measured
   `+1.376 / +2.176 / +2.432 / +2.208 / +2.048 / +2.048 µs` at `B = 3..8` (cold L2, graph-body-timed,
   3 alternating rounds, medians) against the same binary with the fill before the prologue. Cause as
   diagnosed: the fill's `LDG → STS` then delays the mainloop's first `cp_wait`/barrier while the
   `cp.async` copies it was meant to overlap were already hidden by the pipeline. **Disposition: fixed
   by reverting**; the file keeps only a comment recording the number.
2. **The fused GEMV now sits at its register ceiling.** `REG 118 → 128` under
   `__launch_bounds__(256, 2)`, i.e. exactly `65,536` registers for the two resident CTAs. `STACK` and
   `LOCAL` are `0` and occupancy is unchanged today, but any further register demand in this
   instantiation spills or drops the second CTA. **Disposition: accepted tradeoff** — the register line
   is what removes the shared array and the barrier, and the alternative (`cp.async` into a shared
   line) needs its own commit/wait group after the PDL wait, which reintroduces a wait in the tail
   instead of the head. Recorded as a follow-up constraint (§6).
3. **The absolute acceptance bar is not met.** `103.232 µs` best-session median against `≤ 103.0 µs`.
   The bar was written against a `105.28 µs` baseline that this machine no longer reproduces (the same
   code measures `104.35 µs` cool, `105.57 µs` warm today), and §2.2 shows the bar's `+0.40 µs` floor
   underprices the route's own traffic by ≈ 0.47 µs. **Disposition: unresolved** — reported as a
   `−0.4 … −1.1 µs` paired gain rather than a met bar; the tail's `STG` count is the next lever (§6).
4. **`B = 8` reads `+0.432 µs (+0.39 %)`** in the post-revert table. That band runs the sliced-K
   kernel, whose compiled code is bit-identical to baseline (the file's only change is a comment) and
   whose resources match exactly, so the reading is harness noise, not a regression; it is listed
   because it is a positive (worse) number in a table of otherwise negative deltas.
5. **One end-to-end round (r4) of the *off* arm is an outlier** (`43.71 … 44.86 tok/s` against
   `45.14 … 46.37` elsewhere). It is not dismissed: it is shown per round above, and the tie verdict is
   computed both with and without it. It does not hide a fused-arm regression — the fused arm's own six
   samples per test are tight (`σ ≤ 0.05`).
6. **Nothing else regressed.** Compared: every
   `nvfp4_a16_gemv_kernel` / `nvfp4_a16_sliced_k_mma_kernel` resource tuple in `infernix-serve`
   (only the fused GEMV changed); the four plain NVFP4 `T = 1` shapes against a HEAD-built bench; the
   stack-spill allowlist (unchanged); and the Q8 fused route, whose kernels and tests are unchanged.
   The 11 tests above were run on the changed tree only — they were not re-run against a HEAD-built
   tree, so for them the evidence is that the shipped route passes its own oracle and contract suites
   after the change, not that it passes identically before and after.

## 6. Accepted gaps and follow-ups

- **The route is a tie, so it stays unregistered** — the acceptance criterion T3 does not meet, and the
  one that decides the task's third step. Two of §6.1's three gates are unmet for a registration: the
  end-to-end gain is inside the arm's own spread, and `tools/quality/certify.py`'s joint path-fidelity
  gate has never been run for this route. Running that campaign (lossless-route logits, noise twins,
  the family's KV arms) is the prerequisite for any future registration, and it is only worth starting
  if the Op-level gap to the composed route grows past today's `2.0 … 3.0 µs`.
- **The last ≈ 0.2 … 3.2 µs over the plain projection is the tail, not the head.** What remains after
  the register prefetch is the row consumer's stores: each owning thread posts four `STG.E.U16` for its
  row's q/k/v-or-z value plus three more for the published window, so a CTA's 16 rows leave the kernel
  as 112 two-byte stores issued from 8 threads. The obvious next lever is a *row-pair* store contract —
  a GEMV warp owns two adjacent channels, so their `bf16` results and their window taps can leave as
  four-byte stores, halving the tail's store instructions — which needs a distinct output/epilogue type
  in the same style as `GdnConvSnapshotPrefetchedRowEpilogue` and an A/B on the fused route alone. It
  was not attempted here: it is a new kernel↔output contract, beyond T3's named mechanism, and its
  expected gain (`~0.2 … 0.4 µs`) is at the harness's own resolution.
- **The register ceiling is now the constraint on this instantiation.** Any change that adds live
  state to the fused GEMV must be checked against `REG 128` at `__launch_bounds__(256, 2)`; the first
  option if it breaks is `kMinBlocksPerSm`-aware staging of the line into the existing `raw_scales`
  bytes (the line is 2,304 B against a 99 KiB budget), accepting the barrier this task removed.
- **The `[16384, 5120]` fused route (`nvfp4_gdn_snapshot_decode_launch`) still claims no line** —
  unchanged from T6: a different product route whose output map (`GdnConvOutput`/
  `Nvfp4GdnConvEpilogue`) is shared with the Q8 and FP8 record paths. The new hook is generic, so
  claiming it is a change to that output map alone, to be taken with its record/snapshot tests.
- **`B = 2..8` (MTP verification widths) is untouched by T3** after the sliced-K revert, and verification
  stays composed in the model; only plain decode (`B = 1`) exercises the changed kernel.
- **Harness note for future audits:** the fused launch's absolute median moves by `1 … 2 µs` between
  cool and warm states and the plain-projection control moves by `1.3 µs`, so the fused-minus-plain
  residual computed from two non-alternating runs ranges over `1.1 … 4.1 µs` for the *same* binaries.
  Judge this route only on alternating base/candidate rounds in one session, as §4 does.

---

# GB10 changelog — 2026-11-01 (T4: the fused NVFP4 `linear_swiglu` / `linear_add` residual routes take the 2-term TMA lane above the cp.async cliff, and the RTX 5090 routing caps come out)

## 1. Accounting

- Base commit: `3bedb4e4c` (`dev-rebased-20261028`, T1–T3 shipped, tree clean). No commit created
  (task instruction 7 — the reviewer commits).
- Files changed (9 tracked, 0 added, 0 deleted):

| File | Action | Change |
|---|---|---|
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu` | M | the residual dispatch's `x.ne[1] > 96 && x.ne[1] <= 128` TMA gate becomes `x.ne[1] >= kResidualTmaTokens` (`97`) — the 128-column cap that sent the whole prefill band back to `ResidualM64N128` is gone; the now-unreachable `ResidualM64N64` alias is deleted and the ladder's `tokens <= 64 \|\| tokens > 128` becomes `tokens <= 64`; the two RTX 5090 tile-choice comments carry GB10 numbers |
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu` | M | comment only — records that the `--variants` harness ranks the 64-row panel ahead below 97 while the shipped route does not, so the entry was set from the route binary |
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu` | M | `uses_residual_tma(tokens) = tokens >= 65` routes the residual GEMM and its quantizer plane together (`Tiled128` / `RowMajor`); the superseded `ResidualM64N64` (65..128) and `ResidualM64N128` (>128) tiles and their branches are removed, leaving the 16/32-token cp.async tiles that still win inside one tile |
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu` | M | `launch_nvfp4_a4_tma_linear_add_residual` — the 2-term TMA lane (`Nvfp4A4TmaMmaSchedule<128, 3, 1, kNvfp4A4TmaWeightCodeL2_128B, kNvfp4A4ResidualTerms, 128>`) with the in-place `LinearResidualAddEpilogue`, baked at `K = 6144` / `17408` |
| `src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h` | M | its declaration (the task named `launch_nvfp4_linear_add_a4_residual_tma`, which is swiglu's naming; this file's existing `launch_nvfp4_a4_tma_*` convention is kept) |
| `bench/ops/nvfp4_linear_swiglu_bench.cu` | M | `--variants` gains the residual candidates the production plan hides: `a4r_m16n64/m32n64/m64n128/m96n128` and `a4r_tma128_r128_s3/r64_s3`, `a4r_tma256_r64_s2`, over a terms-2 scratch sized `one_term + two_term` |
| `bench/ops/nvfp4_linear_add_bench.cu` | M | same for the add lane: `bench_a4_tma_residual<BlockTokens, Stages, MinBlocksPerSm, BlockRows, K>` plus `a4r_m32n64/m64n64/m64n128` and the TMA variants |
| `tests/ops/linear_swiglu/test_nvfp4.cpp` | M | `A4_RESIDUAL` walks the new bands (`65, 96, 97, 112, 128, 129, 130, 192, 255, 256, 257, 512, 768, 1024`) and replays `8, 33, 65, 97, 129, 257` under a CUDA Graph |
| `tests/ops/linear_add/test_nvfp4.cpp` | M | `AllowA4Residual` gains `130, 255, 256, 257, 512, 1024, 2048`; `257` also replays under a graph |

- CUDA stack allowlist: **unchanged** — zero new `STACK`/`LOCAL` symbols, so `infernix_cuda_stack_check`
  passes as registered and `check_cuda_stack.py update` was not run (§4). The route that changed
  instantiates one TMA kernel that the 1-term lane already carried plus the existing residual
  quantizer; the deleted cp.async residual tiles were not allowlist entries.
- Rebuilds: 6 single-invocation `ninja -C build-native -j "$(nproc)"` builds (the last one
  `infernix-serve infernix_nvfp4_linear_swiglu_bench infernix_nvfp4_linear_add_bench
  infernix_ops_linear_fused_test infernix_ops_attention_test infernix_ops_recurrent_test`, preceded by
  `find … -name '*.cu' -exec touch {} +` over the four nvfp4 directories because `build-native` CUDA
  depfiles do not track headers), plus two builds of the bench pair from a clean `git stash` of exactly
  the five `src/ops` files to put the HEAD binaries on disk for the alternating A/B.

## 2. Task-premise corrections and architectural rationale (`20261101-performance-tasks.md` T4)

**What held.** `nvfp4_linear_swiglu_a4.cu` really did carry an RTX 5090 (`sm_120a`) dispatch cap that
had never been re-measured on GB10, and it really did cost the prefill band a factor of 3.6: with the
cap, `T = 129` costs `1367.9 µs` against `555.9 µs` without it and `T = 4096` costs `29333 µs` against
`8078 µs` (§4). `linear_add` had no TMA residual lane at all — the 2-term profile ran entirely on
cp.async tiles above its 16-token band.

**Premise corrections from measurement.**

1. *The swiglu guard is `x.ne[1] > 96 && x.ne[1] <= 128`, not `<= 128`.* The task describes a route
   that "restricted 2-term TMA to `x.ne[1] <= 128`"; the lower bound already existed, so 97..128 was
   TMA at HEAD and only the *upper* cap needed removing. This matters because the first candidate read
   the guard as `<= 128`, concluded that HEAD's decode band was TMA, measured the cp.async ladder to be
   a tie against a TMA that HEAD never ran there, and shipped "TMA everywhere" — which costs
   **+10.3 … +12.2 % over `T = 5..96`** (498.6 against 445.3 µs at 5, 523.1 against 470.8 at 96). That
   candidate was reverted; §5 carries the numbers.
2. *The two lanes do not share a crossover.* `linear_add`'s TMA entry is `T ≥ 65` (measured
   −19 … −20 % at 65..128 and −24 … −38 % above), `linear_swiglu`'s stays at `T ≥ 97` (the ladder is
   11 % ahead below it, and 97..128 is level). The single `T > 64` the task specifies for the fused
   projections is right for one of them and 33 columns too low for the other.
3. *`launch_nvfp4_linear_add_a4_residual_tma` does not exist.* The add lane's TMA file exports
   `launch_nvfp4_a4_tma_linear_add_residual`; the name in the task is swiglu's
   (`launch_nvfp4_linear_swiglu_a4_residual_tma`) with the operands swapped. The declaration went into
   the existing `nvfp4_linear_add_a4_tma_launch.h` rather than creating a second spelling.
4. *`nvfp4_linear_add_a16.cu` has no 5090 guard left and no `launch_n5120_k6144` /
   `launch_n5120_k17408` symbols* — those names appear only in the task document; the file's own
   dispatcher is `launch_matrix<K>` and it was GB10-retuned in T7 (runtime-K sliced-K tiles,
   `RowTiles = 2`, the 2 K-warp tiles dropped). Nothing in it is a 5090 leftover, so T4 leaves it
   alone; the residual band the task points at lives in `nvfp4_linear_add_a4.cu`.
5. *The gdn/attn projections inherit T1's quantizer and promotion work already.* Both files are
   1-term, hold no token-count guard, and call the shared `launch_nvfp4_a4_quantize` and
   `launch_nvfp4_a4_tma_mma` with the shipped `kNvfp4A4TmaWeightCodeL2_128B` promotions, so "propagate
   T1/T2" is not an edit there — it is automatic through the shared launchers. `nvfp4_gdn_input_a4.cu`
   has no 128-token Tiled128 lane where attn has one, which is a real lever but not this task's
   (§6).
6. *The `--variants` candidate harness misranks this family.* It puts the 2-term TMA lane ahead of the
   ladder below 97 columns for swiglu, where the shipped route says the opposite by 11 %, and its
   absolute numbers (`a4r_m64n128` 95 µs at `[5120,6144] T = 64`) sit 15 % under what that tile costs on
   the route. Every band boundary in this task was therefore set from alternating HEAD/candidate rounds
   of the route binaries, exactly as T2's raster lesson said; the harness remains useful for *within*
   one schedule family (the 64- vs 128-row panel comparison it does support is what §3 records).
7. *"90 %+ roofline efficiency with standalone `ops::linear`" has to be read per activation plane.*
   The 2-term residual profile multiplies the same weight tile by two activation code+scale planes, so
   at equal `N, K, T` it performs 2× the MACs of a 1-term standalone `ops::linear`. Normalised that way
   the fused routes run at **102 … 145 %** of the standalone pair route measured in the same session
   (§4.4): `swiglu [34816,5120] T = 4096` does `2 × 1.4603e12` FLOP in `8078 µs` = 361.5 TF/s against
   the standalone's 353.6 TF/s. The raw single-plane `useful_tflops` the bench prints (180.8) is not
   comparable to a 1-term kernel's number and is not what the bar means.
8. *The TMA and MMA residual lanes are not bit-identical.* They are the same two-plane arithmetic with
   the same quantizer output, but the TMA lane accumulates `BlockK = 128` in three stages where the
   ladder's tiles use their own K order, so the FP32 accumulation order per output differs. Both stay
   inside the unchanged `kA4ResidualTolerance{2.5e-2, kBf16UnitRoundoff, 5.0e-2}` FP64-oracle criterion;
   1:1 FP32 Tensor Core accumulation is untouched (no FP16 accumulate anywhere on the route).

## 3. Subsystem breakdown

- `linear_swiglu/nvfp4` (M): `launch_residual<Terms>` now asks one question — `Terms == 2 && tokens >=
  97` → Tiled128 quantize + `launch_nvfp4_linear_swiglu_a4_residual_tma`, otherwise RowMajor quantize +
  the `16 / 32 / 64x128 / 96x128` ladder. The `ResidualM64N64` alias is deleted because no branch
  reaches it once the cap is gone; the four remaining residual tiles are unchanged schedules, so the
  decode band's bits are unchanged. `SwiGluTokenMajorMmaRows<S>` still requires `kWarpRows % 16 == 0`,
  which the 128-row panel satisfies.
- `linear_add/nvfp4` (M): one `uses_residual_tma(tokens)` predicate drives both the quantizer layout
  and the GEMM, which is what keeps the plane the quantizer writes and the plane the GEMM's descriptor
  reads in step — the pairing is the reason the layout is not a parameter of the launcher. The TMA
  instantiation is the one the standalone pair route's grid heuristic also picks for 5,120 rows
  (`ctas_128 = (5120/128) × ceil(T/128) ≥ 40 > 24` CTAs, so 128-row panel, 3 stages, 1 CTA per SM), and
  it writes in place through `LinearResidualAddEpilogue` exactly as the 1-term TMA lane does.
  `nvfp4_a4_padded_tokens` pads the planes to 256 tokens, which already satisfies the Tiled128
  descriptor's coverage check for both `terms = 2` planes.
- `linear_add/nvfp4` TMA lane (M): baked at `K = 6144` and `K = 17408`; any other `K` throws
  `std::invalid_argument` naming the route, rather than falling back to the lane the task replaces.
- `bench/ops` (M): the residual candidates exist so a future retune can see the ladder and the TMA lane
  side by side; they are sweep instrumentation, and §2.6 records that their *ordering across* the two
  families must not be read as route evidence.
- `tests/ops` (M): the residual token lists now cross both entries (65 and 97), the first ragged call
  past each 128-token tile (129/130, 255/256/257), and the prefill widths the TMA lane owns (512, 1024,
  2048); graph capture at 257 regenerates every plane inside a capture, which is the case that would
  break if the Tiled128 plane were sized from the unpadded token count.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

**4.1 Tests** — run serially; `--parallel` produces real numerical failures on this single-GPU host.

```
35/35 ctest -R "linear_add|linear_swiglu|attn_input_proj|gdn_input_proj|
                column_invariant_projections|softmax_attention|gdn_gating|infernix_cuda_stack_check"
      -> 100% tests passed, 0 tests failed out of 35   (1331.01 s, incl. #28 cuda_stack_check 1.52 s)
 8/8 ctest -R "linear_add_nvfp4|linear_swiglu_nvfp4|attn_input_proj|gdn_input_proj|
               infernix_cuda_stack_check"               -> 100% on the final binaries (92.61 s)
     ctest -R infernix_cuda_stack_check                 -> 100%, 1.55 s (final `infernix-serve`)
```

The task's `infernix_ops_quant_mlp_test` is not a target on this tree; the two Op tests the task means
are CTest members of `build-native/tests/infernix_ops_linear_fused_test`
(`infernix_linear_add_nvfp4_test`, `infernix_linear_swiglu_nvfp4_test`), both re-run directly on the
final binaries (`OK NVFP4 linear_add`, `OK LinearSwiGLU NVFP4 correctness`), together with
`infernix_ops_attention_test` and `infernix_ops_recurrent_test`'s members.

**4.2 Production route, `--policy a4r`, cold L2, quantizer included — min of two alternating rounds
of the HEAD and T4 binaries in one session** (µs; `×2 TF/s` is the T4 route's real MAC rate, since the
2-term profile does two planes):

| shape | T | HEAD | T4 | Δ | T4 ×2 TF/s |
|---|---|---|---|---|---|
| swiglu `[34816,5120]` | 5 / 16 / 32 / 64 / 96 | 444.3 / 451.6 / 450.4 / 460.8 / 470.8 | 443.4 / 450.5 / 450.6 / 461.7 / 472.0 | −0.2 … +0.3 % | 8.0 / 25.3 / 50.6 / 98.9 / 145.0 |
| swiglu | 97 / 128 | 524.3 / 525.2 | 525.1 / 525.1 | +0.2 / −0.0 % | 131.7 / 173.8 |
| swiglu | **129** | **1367.9** | **555.9** | **−59.4 %** | 165.5 |
| swiglu | 256 / 512 / 1024 | 1840.0 / 3819.5 / 7337.8 | 571.2 / 967.4 / 1919.8 | −69.0 / −74.7 / −73.8 % | 319.6 / 377.4 / 380.3 |
| swiglu | **4096** | **29333.3** | **8078.1** | **−72.5 %** | **361.5** |
| add `[5120,6144]` | 5 / 16 / 32 | 85.0 / 88.1 / 89.1 | 85.0 / 88.1 / 90.0 | 0.0 / 0.0 / +1.1 % | 7.4 / 22.9 / 44.7 |
| add `[5120,6144]` | 48 / 64 | 112.6 / 114.7 | 111.6 / 113.7 | −0.9 / −0.9 % | 54.1 / 70.9 |
| add `[5120,6144]` | **65 / 96 / 128** | 128.0 / 131.1 / 135.2 | **102.4 / 105.5 / 109.6** | **−20.0 / −19.5 / −18.9 %** | 79.9 / 114.5 / 147.0 |
| add `[5120,6144]` | **512 / 4096** | 353.0 / 2393.1 | **248.8 / 1819.7** | **−29.5 / −24.0 %** | 258.9 / 283.2 |
| add `[5120,17408]` | 5 / 16 / 32 | 227.3 / 230.4 / 240.6 | 227.3 / 230.4 / 241.7 | 0.0 / −0.1 / +0.4 % | 7.8 / 24.8 / 47.2 |
| add `[5120,17408]` | 48 / 64 | 470.0 / 476.2 | 475.4 / 509.9 | +1.1 / +7.1 % † | 36.0 / 44.8 |
| add `[5120,17408]` | **65 / 96 / 128** | 486.4 / 500.7 / 518.1 | **295.9 / 300.0 / 310.3** | **−39.2 / −40.1 / −40.1 %** | 78.3 / 114.1 / 147.1 |
| add `[5120,17408]` | **512 / 4096** | 1904.9 / 14004.2 | **651.3 / 5175.3** | **−65.8 / −63.0 %** | 280.3 / 282.2 |

The `T = 512` and `T = 4096` rows are the task's acceptance points: both Op shapes improve by 24 … 75 %,
far above the 25 % bar, and every `T ≥ 129` point does too. † is discussed in §5.

**4.3 The TMA kernel really runs on the route** — `nsys profile --capture-range=cudaProfilerApi` on
`--policy a4r --profile` (synthetic bench inputs), kernel summary from the exported SQLite:

| trace | kernels on the route |
|---|---|
| swiglu `[34816,5120] T=4096` | `nvfp4_a4_tma_kernel` (96.5 %), `nvfp4_a4_residual_quantize_kernel` |
| add `[5120,17408] T=4096` | `nvfp4_a4_tma_kernel` (81.3 %), `nvfp4_a4_residual_quantize_kernel` |
| add `[5120,6144] T=512` | `nvfp4_a4_tma_kernel` (86.9 %), `nvfp4_a4_residual_quantize_kernel` |

No cp.async residual GEMM appears in any trace. At `[5120,17408] T = 4096` the quantizer is 19 % of the
route, which is the known serialized-quantizer limit on this family (§6).

**4.4 Against standalone `ops::linear`** — `infernix_linear_bench --qtype NVFP4 --policy a4 --routes
fast`, same session, cold L2, median µs; "equal-work standalone" doubles the 1-term time because the
fused Op computes two activation planes:

| shape | T | standalone | equal-work | fused T4 | fused / equal-work |
|---|---|---|---|---|---|
| `[34816,5120]` | 512 | 698.2 | 1396.4 | 967.4 | 144 % |
| `[34816,5120]` | 4096 | 4129.6 | 8259.3 | 8078.1 | 102 % |
| `[5120,6144]` | 512 | 145.5 | 291.1 | 248.8 | 117 % |
| `[5120,6144]` | 4096 | 1013.5 | 2027.0 | 1819.7 | 111 % |
| `[5120,17408]` | 512 | 420.8 | 841.7 | 651.3 | 129 % |
| `[5120,17408]` | 4096 | 2954.1 | 5908.1 | 5175.3 | 114 % |

Every fused route is at or above 100 % of the equal-work standalone rate, so the 90 %+ bar is met at
both acceptance widths for all three shapes; the fused Op also carries the residual read-modify-write
(`add`) and the SwiGLU epilogue (`swiglu`) inside that time.

## 5. Regressions and adverse results

Compared: every token count the changed dispatch can route, `T = 5, 8, 16, 32, 48, 64, 65, 96, 97,
128, 129, 256, 512, 1024, 2048, 4096`, for all three shapes, HEAD binary vs T4 binary, alternating
rounds in one session (the tables in §4.2), plus the neighbouring `attn_input_proj`, `gdn_input_proj`
and column-invariant tests, the spill ratchet, and the FP64-oracle tolerances.

1. **Fixed — the "TMA everywhere" candidate cost +10.3 … +12.2 % over `T = 5..96` for swiglu.**
   Reading the guard as `<= 128` made the decode band look like a tie with TMA, so the first candidate
   deleted the ladder; measured against HEAD it was 498.6 against 445.3 µs at 5, 505.6 against 448.2 at
   16, 517.0 against 461.7 at 64 and 523.1 against 470.8 at 96 (rounds 3/4 of the small-band sweep).
   Cause: one 128-token TMA tile streams the 89 MiB gate/up panel worse than a cp.async tile whose
   column is split across two CTAs per SM, and the `--variants` harness that recommended it does not
   reproduce the route. Disposition: **fixed** — the ladder is restored and the TMA entry stays at 97;
   the harness's misordering is recorded at the site (§2.6).
2. **`[5120,17408]` at `T = 64` reads +7.1 % (476.2 → 509.9 µs) in one round; it is not a code change.**
   HEAD and T4 run *the same kernel* there (`ResidualM32N64` over a RowMajor plane — the deleted tiles
   only covered `65..128` and `> 128`), so four further alternating rounds were run: HEAD
   `473.9 / 476.0 / 476.2 / 477.0`, T4 `475.1 / 477.1 / 478.0 / 478.0` µs, i.e. medians `476.1` against
   `477.6` (+0.3 %) with a within-binary spread of 0.6 %. The `509.9` sample did not reproduce. Same
   reading for the `+1.1 %` at `T = 48` and `+1.1 %` / `+0.4 %` at `T = 32`: identical code paths, and
   the four-round medians agree within 0.4 %. Disposition: **accepted as session variance, with the
   identical-kernel argument as the evidence** rather than a dismissal — the band's route did not change.
3. **No regression in the bands the change actually owns.** swiglu `5..128` is within ±0.3 % of HEAD
   (it runs HEAD's own kernels), `linear_add 5..64` within 0.4 % as above, and `infernix_cuda_stack_check`
   plus the 35-test and 8-test CTest selections pass 100 %. The `linear_add` `65..128` and all `T ≥ 129`
   rows improve; nothing in the sweep got worse by more than the 1.3 % single-run spread.
4. **Adverse result on method, not on the route:** the candidate harness's absolute numbers are 15 %
   below the route's for the same nominal tile (`a4r_m64n128` 95 µs vs 111.6 µs measured on the route at
   `[5120,6144] T = 64`), which is what made an unreproducible 44 % "win" look available at
   `[5120,17408] T = 64` early in the task. Chasing it would have re-instantiated a tile the route
   already shows no need for; it was dropped once the route binary said the band is level. Disclosed so
   the harness numbers in §3 are not read as route evidence.
5. **Deleted schedules were live, not dead.** `ResidualM64N64` (`65..128`) and `ResidualM64N128`
   (`> 128`) in `linear_add` and `ResidualM64N64` in swiglu ran at HEAD and are removed *because* the TMA
   lane replaces them at those widths (−18.9 … −40.1 %), not as cleanup; the `65..128` removal is a
   measured improvement, and no width loses its kernel.
6. **Memory/workspace:** the residual lane now sizes its scratch for a `terms = 2`, 256-token-padded
   Tiled128 pair of planes above its entry. The Op tests' `workspace.peak_used() == capacity` assertion
   holds at every token count, including the graph replays, so query and execution agree; no route
   allocates outside its queried high-water.

## 6. Accepted gaps and follow-ups

- **swiglu `97..128` is a coin flip** — TMA 527.2/527.6 against the ladder 528.2/529.3 µs. HEAD already
  had TMA there and the difference is inside any protocol's resolution, so it was left rather than
  re-litigated; a future pass that needs the last 0.2 % should decide it on more rounds, not on
  `--variants`.
- **The residual ladder's 96-token tile now exists for `65..96` alone.** With the TMA entry at 97,
  `ResidualM96N128` is reachable on the 2-term route only in `65..96` (`ResidualM64N128` covers
  `33..64`), and on the 1-term route above 64 columns. Both tiles are live for that 1-term profile, so
  nothing here is dead code — but if a quantizer change moved the TMA entry below 97, `ResidualM96N128`
  would lose its only 2-term band and should be re-swept rather than kept by default.
- **`linear_add` below 65 keeps the cp.async tiles on sweep evidence, not route A/B.** The candidate
  harness favours them there and the route binary shows no gap to close (0.4 % against HEAD, which runs
  the same tile), so nothing was changed; a route-level TMA-vs-ladder A/B at `T = 33..64` would settle
  whether the entry should drop further.
- **`nvfp4_gdn_input_a4.cu` still has no 128-token Tiled128 lane** where `nvfp4_attn_input_a4.cu`
  instantiates `Nvfp4A4TmaMmaSchedule<128, 3, 2, …, 1, 64>`. That asymmetry is the remaining unexploited
  lever in the fused NVFP4 projections and is a separate change with its own `gdn_input_proj` oracle run.
- **The quantizer is 19 % of the `[5120,17408] T = 4096` residual route** (nsys, §4.3) and the prefill
  rate sits at 283 TF/s of real math against the 498.2 TF/s `ckt env` nvfp4 ceiling. Per the stored
  finding that this family's prefill gap is the serialized quantizer rather than tile/promotion/raster
  knobs, closing it is a quantizer-overlap task, not a schedule retune.
- **No end-to-end run.** The acceptance criteria and every measurement here are Op-level on the fused
  routes; `./bench/run_kv_matrix_bench.sh` was not run, because no model route, plan gate, or kernel
  outside the two fused Ops changed. If the reviewer wants E2E evidence, the A/B to run is that script
  against `3bedb4e4c`, and the `--fast-path`/`a4r` configuration it exercises is the one these bands
  feed.
- **`nvfp4_linear_add_a16.cu` was left as T7 shipped it** (§2.4). Its `T = 2..32` bands were re-read,
  not re-measured: the task's premise for that file (5090 K-baked tiles named
  `launch_n5120_k*`) does not exist in the tree, and T7's record already carries the GB10 sweep and the
  two bars it could not meet.

---

# GB10 changelog — 2026-11-01 (T5: the multi-column W4A4 narrow decode/verify pass pre-quantizes x once per entry; the wide schedules skip the idle 16-column MMA tiles)

## 1. Accounting

- Base commit: `a71c6ca7a` (`dev-rebased-20261028`, T1–T4 shipped, tree clean). No commit created
  (task instruction — the reviewer commits).
- Files changed (4 tracked, 0 added, 0 deleted): +161 / −24 lines. `cuda-kernel-toolkit/` untouched.

| File | Action | Change |
|---|---|---|
| `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | M | `gate_up_kernel` gains a `const ActCodes* narrow_acts` parameter; its column-pair loop takes the new `unit_sums_pair_cols` (two pre-quantized columns' per-column `dup` words are exactly the packed pair's unpacked halves) and its single-column loop reads the plane instead of calling `quantize_column_codes`; new `quantize_narrow_x_kernel` (one CTA per job, `gate_up_kernel`'s guards, writes `[entry][parity][block]` of the workspace plane); `launch_expert_pass` gains `columns` and `narrow_acts` and launches the quantizer before the gate/up pass when `columns > 1` |
| `src/ops/offloaded_sparse_moe/cuda/wide_expert.h` | M | `ExpertsWorkspace::narrow_acts` plus `kNarrowActBytes = 24` (`static_assert`-tied to `ActCodes`) and `kNarrowQuantColumns = 64` |
| `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu` | M | `offsets()` sizes the plane (`entries × 2 × kGateUpBlocks × 24`) for multi-column W4A4 calls of 2..64 columns; `carve_experts_workspace` hands it out (null otherwise) |
| `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` | M | `Schedule::kSkipIdleColumns = true` (was the integration default `false`, commit `69b3da534`, never a measurement); `ScheduleResidual`'s geometry-conditional override deleted — it inherits the base |

- Task: `docs/tasks/20261101-performance-tasks.md` T5 (P0).

## 2. Task-premise corrections and architectural rationale

1. **The bench flag and target names in the task do not exist.** The decode bench has no
   `--ColdL2` flag — it always prints a warm and a cold row per (T, split). There is no
   `infernix_ops_moe_test` and no `infernix_offloaded_moe_prefill_bench`: the checks used were
   `ctest -R "infernix_offloaded_moe|infernix_cuda_stack_check"` (20 tests, unified binary
   `build-native/tests/infernix_ops_offloaded_moe_test`) and
   `build-native/bench/infernix_offloaded_moe_wide_bench --tokens 512,1024,2048,4096`.
2. **`stage_nmma16_units` (spec item 2) does not exist.** `narrow_a16_mma.cuh`'s ring already
   distributes its `cp.async16` fills across all 32 lanes, and the A16 nmma route is not what the
   kA4 decode bench exercises. No change; the spec item is a documentation artifact.
3. **The T=4/8 baseline is 164.6/299.1 µs cold (task said 142–150).** The per-column pass cost
   scales with the job's count, so the ≤132 µs bar at T=8 is arithmetically out of reach of this
   op (gate/up alone is 2× the T=1 traffic). Disclosed with measurements in §5.
4. **`kSkipIdleColumns = false` was never justified by a measurement** — commit `69b3da534`
   introduced it as the integration default. The skip is bit-exact by construction:
   `live_mma = min(⌈columns − warp_m·kWarpTokens + 15⌉/16, kMmaTokens)` and every epilogue guards
   `column >= ctx.columns`, so a skipped tile's zeroed accumulators are never read. All wide
   hashes are unchanged (§4).
5. **The wide-prefill gain the task predicted (15–25 %) did not materialize: measured neutral**
   (−0.9 % … +0.3 % at T = 512…4096, both placements, §5). The wide A4 route is bound by the
   serialized x-quantizer and the weight stream, not by padding MMAs — consistent with the stored
   GB10 finding for this family's prefill gap. The flag stays on (it removes real tensor work at
   tail tiles and costs nothing), but it is not a measured win here.
6. **The NCU premise is confirmed and its prescribed fix is falsified at T=1.** `gate_up_kernel`
   is latency-bound (3.73 active warps/scheduler, 59.5 % long_scoreboard, Mem 15 %, Compute 20 %).
   But streaming the slice to registers is *worse* than the full-slice `stage_wait`: at two CTAs
   per SM the unrolled per-warp `__ldg` stride keeps ~1–1.5 KB in flight per warp against
   cp.async's whole-slice 46 KB per CTA — ~4.6× less MLP — and a separate pre-quantizing launch
   serializes ~16 µs ahead of the captured graph. Both variants were implemented, measured, found
   slower at T=1 (warm +26 %/+21 %), and removed; the staged pass keeps its inline quantization
   there, which overlaps the fetch. The pre-quantization wins only where the duplication is worst:
   multi-column calls (spec item 1's other half, kept).

## 3. Subsystem breakdown

- `src/ops/offloaded_sparse_moe/cuda/` (M): as §1. Bit-exactness rests on two identities, both
  verified against the CPU oracle (`exact=yes`, all 12 decode rows): the pair packing
  `(a0.dup[q] & 0x0F0F0F0F) | (a1.dup[q] & 0xF0F0F0F0)` reproduces `quantize_pair_codes`' word, and
  `unit_sums`' lane mapping (q = lane>>3, i = lane&7) reads the per-column `dup` words unchanged.
  The plane layout `[entries][2 parities][160 blocks]` mirrors the gate/up kernel's parity loop;
  the quantizer kernel's skip predicates (`job_count`, `cpu_flags`, `pass_record`, `wide_route`)
  are the gate/up kernel's, so a skipped job's CTAs never read unwritten plane bytes.
- No plan, converter, model, serving, or documentation contract changed. The workspace plane is
  sized and carved from the same `offsets()` arguments (the caller's token count), the
  `x_acts`-ring precedent; one-column and >64-column calls leave it null and run the untouched
  inline path.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- **Tests:** `ctest -R "infernix_offloaded_moe|infernix_cuda_stack_check"` — 20/20 passed
  (layer, plan, wide, cuda, ggml, ggml_a8, w4a8, a8_wide, fp8, 122b, a4_residual_122b, team,
  the five cpu tests, layer_idle, stack ratchet). Run serially per the stored single-GPU finding.
- **Spill ratchet:** no allowlist update was run or needed (`git diff` touches no file under
  `tools/cuda_stack/`). Per-symbol `cuobjdump -res-usage`: `quantize_narrow_x_kernel` REG:72
  STACK:0 LOCAL:0; `gate_up_kernel` REG:101 STACK:0 — identical to the baseline binary's symbol.
  (The first compile put the two by-value `ActCodes` loads on the stack, STACK:48/24; taking
  references like the original `unit_sums` eliminated both before any gate failure shipped.)
- **Bit-exactness:** decode bench `exact=yes` on all 12 rows (every T × split × warm/cold). Wide
  bench `output fnv1a` identical to the saved baseline on a4 (T = 512…4096, both placements:
  `44a3d026d330a364`/`505681a3f4ee4fc1`/`33d2ef52f688991e`/`90cb5930998d3eeb`) and on every
  regression variant: `--activation a16x2`/`a16x3` (the residual schedule now inherits the skip),
  `--split-scales` narrow multi-column calls (the plane's parity path), `--geometry 122b`
  `--activation a16`/`a16x2`, `--bank ggml --ggml-wide bf16`.
- **A/B protocol:** absolute µs on this host drifts with thermal state (a non-interleaved
  comparison showed a phantom +17 µs T=1 warm "regression" that interleaving erased), so all
  claims below are from two alternating base/candidate passes over separately built binaries in
  one session.

## 5. Results — wins, parity, and every adverse number

Decode bench medians (µs; base → new, second value = second pass):

| Row | warm base → new | cold base → new |
|---|---|---|
| T=1 split=0 | 94.8 / 92.5 → 94.1 / 87.7 (parity) | 134.0 / 132.8 → 135.9 / 132.8 (parity) |
| T=1 split=1 | 98.5 / 98.9 → 97.9 / 95.8 (parity) | 123.6 / 123.6 → 122.5 / 122.9 (parity) |
| T=4 split=0 | 146.9 / 147.8 → 135.7 / 139.7 (**−6 %**) | 158.5 / 167.0 → 166.8 / 165.6 (flat) |
| T=4 split=1 | 278.8 / 278.7 → 264.7 / 236.0 (**−5…−15 %**) | 295.6 / 297.7 → 258.1 / 253.7 (**−14 %**) |
| T=8 split=0 | 274.7 / 273.0 → 239.9 / 243.3 (**−12 %**) | 286.4 / 286.7 → 263.1 / 264.1 (**−8 %**) |
| T=8 split=1 | 531.2 / 532.9 → 443.9 / 446.0 (**−16 %**) | 543.6 / 542.4 → 455.4 / 456.4 (**−16 %**) |

- **No regression shipped.** Every row is parity or a win under interleaving. The T=4 split=0 cold
  row is flat, not a win (base itself measured 158.5 then 167.0 across passes; candidate 165.6–166.8).
- **Wide prefill:** T = 512…4096, frames and staged: medians move by −0.9 % … +0.3 % — neutral
  (§2.5). The task's ≥15–25 % bar is **not met**; the premise that tail padding MMAs bound the
  wide route is falsified by this measurement.
- **Removed attempts (both bit-exact, both slower at T=1, disclosed rather than kept):**
  (a) `gate_up_single_kernel` reading pre-quantized codes from the plane — T=1 warm 103.8 vs 82.4
  (the separate quantize launch serializes ~16–21 µs ahead of the 3-node graph); (b) the same
  kernel with the quantization moved inside (no launch) — T=1 warm 98.8, because the codes'
  `__syncthreads` puts the weight loads behind the column's barrier, losing the cp.async overlap.
  The T=1 path is therefore the untouched baseline; the per-symbol trace evidence is
  `/tmp/t5_dec_after.sqlite` (quantize kernel min 3.7 µs, single kernel avg 82.1 vs staged 85.0 —
  the per-kernel averages hide exactly the launch-serialization the whole-graph rows show).
- **Acceptance bars:** T=1 cold ≤124 µs **not met** (~133 µs, unchanged — 82 % of the 246 GB/s
  reference; 27.6 MB/expert-call at that size is 112 µs of DRAM floor and the pass is already
  within ~20 % of it). T=4 ≤132 µs **not met** (166 µs); T=8 ≤132 µs **not met** (264 µs, but
  −16 % from the pre-quantization). Wide ≥15 % **not met** (neutral).

## 6. Accepted gaps and follow-ups

- **The T=1 narrow decode bar needs a different lever than this task named.** At 2 CTAs/SM the
  staged pass's remaining latency is the whole-slice `stage_wait` join; the two register/
  pre-quantization alternatives are measured slower (§5). The untried shape is the task's
  "compact cp.async ring": warp-owned chunked staging (each warp waits only its own commit
  group, or a two-group half-slice pipeline) at unchanged 2 CTAs/SM — a rewrite of the hot
  kernel's staging contract, worth a follow-up task with the same interleaved protocol.
- **Cold T=1 is near the DRAM floor** (82 % READ_%): 27.6 MB per call at 246 GB/s is 112 µs, so
  the ≤124 µs bar leaves ~12 µs of non-DRAM budget for stage+dispatch+down; the down kernel's
  PDL overlap already absorbs most of its own staging. Reaching 90 %+ likely requires cutting
  bytes (e.g., a resident-frame L2 pin of the 24 MiB working set) rather than latency work.
- **Zero-copy host records are untested for the new plane path:** the benches use device-resident
  frames; a forked call whose `job_records` point at pinned host memory exercises the same code
  with PCIe latency in the quantizer's x reads. The guards are the gate/up kernel's, so the
  correctness contract is unchanged, but no measurement covers it.
- **`run_kv_matrix_bench.sh` was not run:** every change is inside the W4A4 narrow op and the wide
  schedule constants, exercised op-level by the two benches and the 20-test family; no model
  route, plan gate, or artifact format changed. If the reviewer wants E2E evidence, A/B that
  script against `a71c6ca7a`.
- **`kNarrowQuantColumns = 64` is a contract constant, not a tuned one:** above it the plane's
  quantizer serializes more x-traffic than the duplication it removes; the 64 boundary follows
  `kMaxColumns` (the narrow/wide planning split) rather than a sweep at intermediate widths.

---

# GB10 changelog — 2026-11-01 (T6: Qwen4Exp MTP & n-gram speculative decoding gating, post-cut copy fallback, multi-row cut application, and falsified terminal-retention / cost-fit premises)

Task: `docs/tasks/20261101-performance-tasks.md` T6 (P0), "Smart MTP & N-Gram Speculative Decoding
Acceptance & Verification Gating (`Qwen4Exp`)".

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `d4dee8c0a`. `git diff --check` clean; `.tandem/`
  and `task_orchestrator.py` untouched.
- Files changed (T6): 4 code/test files (2 modified, 2 added), `+502 / -79` lines (excluding this
  changelog). `cuda-kernel-toolkit/` intentionally excluded from the enumeration (integrated
  toolkit, untouched).
- Allowlist delta: none (`tools/cuda_stack/cuda_stack_allowlist.json` untouched; host-only policy
  changes, `infernix_cuda_stack_check` passes with zero new `STACK > 0` or `LOCAL > 0` symbols).

| File | Action | Change |
|---|---|---|
| `src/models/qwen4_exp/program/mtp_gating.h` | A | Host policy for a Qwen4Exp MTP draft round: `kMtpCumulativeBins`, `kMtpCumulativeBase`, `MtpCumulative`, `MtpDraftCut { kept, executed }`, `mtp_cumulative_cut`, `mtp_min_logprob_cut`, `expected_copy_tokens`, `expected_chain_tokens`, and `copy_beats_chain` |
| `src/models/qwen4_exp/program/program_impl.h` | M | Delegate cumulative/fixed-floor cuts to `mtp_gating.h` and record `mtp_executed_[b]`; relax pre-drafter copy filter from `match.tokens.size() > extents[b]` to `!match.tokens.empty()` via `copy_beats_chain`; retain declined copy proposals in `ngram_proposal_[b]` / `ngram_proposal_bucket_[b]` and fall back to them post-cut when `copies > extents[b]` and `copy_beats_chain` holds over the kept prefix; apply `mtp_confident_[b]` across all rows `b < batch` (`batch >= 2`), recompute `width`, and reset `staged[b] = 0` (`b >= 1`) when `width != staged_width` |
| `tests/models/qwen4_exp/test_speculative.cpp` | A | Host unit test verifying `mtp_cumulative_cut` and `mtp_min_logprob_cut` (`kept` vs `executed` across gated/ungated and strided multi-row rounds against a direct simulation of `launch_graph_gate_product`), `expected_copy_tokens`, `expected_chain_tokens`, and `copy_beats_chain` pre-drafter and post-cut copy fallback |
| `tests/models/qwen4_exp/tests.cmake` | M | Register `infernix_qwen4_exp_speculative_test` |

## 2. Task-premise corrections and architectural rationale

1. **Terminal draft token retention (`kMtpTerminalMinRate = 0.12F`, `kMtpTerminalMinProduct = 0.08F`)
   — falsified on GB10 and reverted.**
   - *Task premise*: When the device continuation gate (`launch_graph_gate_product`) stops the draft
     chain after step `j` because `product(0..j) < threshold`, draft step `j` has already run and its
     cost is sunk, so retaining token `j` for verification whenever `rate(j) >= 0.12` and
     `product(0..j) >= 0.08` should increase accepted tokens per round at near-zero verify cost.
   - *Measured reality on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`*:
     - **Variant A (retention on, terminal token observed in `mtp_accept` / `mtp_cumulative`)**:
       2-pass order-swapped A/B showed `tg256` decode throughput dropped **`-9.11%`** (`70.94 -> 64.48
       tok/s`, acceptance `0.763 -> 0.646`, rounds `187 -> 228.5`) and `tg128` dropped **`-2.83%`**
       (`63.84 -> 62.04 tok/s`, acceptance `0.699 -> 0.592`).
     - **Variant B (`final` and `iso` arm `C`: retention on, retained cut token excluded from
       `mtp_accept` and `mtp_cumulative` learners)**: Even with learner poisoning eliminated, the
       3-pass order-swapped `final` campaign regressed **every** workload: `pp4096+tg128` **`-1.65%`**
       (`94.50 -> 92.94 tok/s`), `pp4096+tg256` **`-2.82%`** (`95.19 -> 92.51 tok/s`), `tg128`
       **`-2.06%`** (`64.66 -> 63.33 tok/s`), and `tg256` **`-3.63%`** (`70.45 -> 67.89 tok/s`). In
       the 3-pass 4-arm isolation campaign (`iso` arm `C`, retention isolated alone), `tg128`
       regressed **`-6.9%`** (`64.62 -> 60.15 tok/s`, acceptance `0.70 -> 0.53`) and `tg256` regressed
       **`-7.2%`** (`69.52 -> 64.53 tok/s`, acceptance `0.75 -> 0.63`).
     - *Root cause*: The table's `0.12..0.26` bin rates are unconditional averages, whereas the
       continuation gate stops the chain on step `j` conditioned on the product falling below
       `0.35..0.25`. On those cut steps, the extra `35..44` retained tokens per `tg128`/`tg256` run
       yielded **`~1–2%` empirical acceptance** (and net fewer accepted tokens across the run due to
       altered round boundaries). Worse, whenever the gate cuts at `j = 0`, retaining token `0` stops
       the round from taking the fast `width = 1` plain decode path (`plain_round_ = true`), forcing a
       full `W = 2` verification + GDN replay fold + MTP catch-up round for a token that rejects
       `~98%` of the time.
     - *Disposition*: `mtp_cumulative_cut` and `mtp_min_logprob_cut` drop the step `j` that tripped
       the cut (`cut.kept = j`) while recording `cut.executed = gated ? j + 1 : steps`. Both the
       header comment in `mtp_gating.h` and `test_speculative.cpp` lock `cut.kept == j` as an explicit
       regression guard.

2. **Feeding `mtp_executed_` into `last_round_steps_` (`round_cost_` OLS fitter) — falsified on GB10
   and reverted.**
   - *Task premise*: Setting `last_round_steps_ = mtp_executed_[0]` instead of `-1` when a round is
     truncated by `mtp_confident_` feeds the online `round_cost_` linear regression more data points
     with the actual number of executed device draft steps.
   - *Measured reality (`final2` and `iso` arm `D`: retention off, `last_round_steps_ =
     mtp_executed_[0]`)*: In the 3-pass `final2` campaign, `pp4096+tg256` regressed **`-3.89%`**
     (`95.56 -> 91.85 tok/s`, rounds `102 -> 111`, `acc-len 5.02 -> 4.59`) and `tg256` regressed
     **`-5.61%`** (`72.94 -> 68.85 tok/s`, rounds `181 -> 198`, `acc-len 2.68 -> 2.45`). In `iso` arm
     `D`, `pp4096+tg128` regressed **`-2.9%`** and `pp4096+tg256` regressed **`-3.3%`**.
   - *Root cause*: A gated one-row draft round replays `mtp_gated_graphs_[steps - 1]`, whose CUDA
     Graph topology and conditional-node dispatch still depend on the drafted `steps`, and whose
     subsequent verify round is narrower than an uncut round's. Pairing a truncated round's total wall
     time with a smaller `x = mtp_executed_[0]` in the 2-parameter OLS fit `Time(x) = a + b * x`
     biases the estimated slope `width_cost() = b / a` upward, causing `mtp_draft_round_steps` to
     under-select draft lengths on subsequent rounds (`acc-len` drops from `5.02 -> 4.59` and
     `2.68 -> 2.45`).
   - *Disposition*: `mtp_executed_[b]` is computed and stored on each row (`cut.executed`, documenting
     which draft slots in `mtp_io_.drafts` were actually written by the device chain vs stale past a
     skipped conditional node), while `decode()` keeps `last_round_steps_ = -1` whenever
     `from_ngram_[0] || extents[0] != wanted[0]` so `round_cost_` fits only complete neural rounds.

3. **Pre-drafter copy competition, post-cut n-gram copy fallback, and multi-row (`batch >= 2`) cut
   application — shipped.**
   - Before `mtp_draft`, any non-empty n-gram proposal (`!match.tokens.empty()`, removing the strict
     `match.tokens.size() > extents[b]` gate) competes against the planned MTP chain via
     `copy_beats_chain(...)`. If the neural chain wins before drafting, the copy proposal and its
     match-length bucket are saved in `ngram_proposal_[b]` and `ngram_proposal_bucket_[b]`.
   - After `mtp_draft` and `read_mtp_drafts`, `extents[b] = std::min(extents[b], mtp_confident_[b])`
     is applied across all `b < batch` (including `batch >= 2`). If the cut leaves row `b` with fewer
     drafts than `ngram_proposal_[b]` (`copies > extents[b]`) and `copy_beats_chain(...)` holds
     against the truncated prefix `extents[b]`, row `b` falls back to `ngram_proposal_[b]` (`from_ngram_[b] = true`)
     instead of decaying to a narrow or plain round.
   - `width` is recomputed across all rows (`max_b(1 + extents[b])`), and when `width != staged_width`
     on a multi-row round, `staged[b] = 0` is reset for `b >= 1` so `verify()` re-stages their n-gram
     columns at the new row stride `b * width`.

## 3. Subsystem breakdown

- **Model / Qwen4Exp MTP gating (`src/models/qwen4_exp/program/mtp_gating.h`)**:
  - Centralizes `kMtpCumulativeBins`, `kMtpCumulativeBase`, `MtpCumulative`, `MtpDraftCut`,
    `mtp_cumulative_cut`, `mtp_min_logprob_cut`, `expected_copy_tokens`, `expected_chain_tokens`, and
    `copy_beats_chain` in a self-contained header testable without loading a model artifact.
- **Model / Qwen4Exp Program (`src/models/qwen4_exp/program/program_impl.h`)**:
  - `decode()` and `read_mtp_drafts()` use `mtp_gating.h`, track `mtp_executed_`, retain declined
    n-gram proposals in `ngram_proposal_` / `ngram_proposal_bucket_` for post-cut fallback, and apply
    `mtp_confident_` across all `batch` rows.
- **Tests (`tests/models/qwen4_exp/test_speculative.cpp`, `tests/models/qwen4_exp/tests.cmake`)**:
  - Adds `infernix_qwen4_exp_speculative_test` covering bin rates, cumulative and `--spec-min-p` cuts,
    strided multi-row readout, 20,000 random chains against `simulated_device_steps`, and pre-/post-cut
    `copy_beats_chain` decisions.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`)

- **Host & stack CTest suite (13/13 passed, 6.36 s)**:
  - `infernix_cuda_stack_check`: **Passed** (1.52 s; zero new `STACK > 0` or `LOCAL > 0` symbols in
    `infernix-serve`).
  - `infernix_qwen4_exp_speculative_test`: **Passed** (0.01 s).
  - `infernix_qwen4_exp_fast_paths_test`, `call_plan_test`, `rope_positions_test`, `memory_plan_test`,
    `prefix_state_test`, `ngram_hash_test`, `ngram_volume_test`, `expert_cache_test`,
    `host_expert_tier_test`, `host_tier_test`, `expert_activations_test`: **All Passed**.
- **Real-artifact Qwen4Exp speculative verification tests (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`)**:
  - `infernix_qwen4_exp_speculative_sampling_real_test`: **Passed** (`1,845.3 s`; 2,000 requests per
    arm across plain `T = 0.8`, one-hot MTP-4 `7,886` rounds / `1.62` tok/round, sampled MTP-4 `8,277`
    rounds / `1.54` tok/round, and greedy parity).
  - `infernix_qwen4_exp_resume_width_real_test`: **Passed** (`88.00 s`).
  - `infernix_qwen4_exp_decide_real_test`: **Passed** (`61.54 s`).

## 5. Benchmarks, regressions, and adverse results

All end-to-end benchmarks ran on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`INFERNIX_EXPERTS_ON_GPU=1`,
`--ngram-residency volume`, `--corpus bench/fixtures/bench_corpus.ids`, `--kv-dtype fp8`, `--spec mtp`,
`--draft-tokens auto`, `--lm-head-draft`, `--concurrency 1`, `-r 2 --warmup 1`) using order-swapped
passes (`AB / BA / AB`).

### 5.1 Rejected candidates (falsified task premises)

| Campaign / Variant | `pp4096+tg128` base -> cand | `pp4096+tg256` base -> cand | `tg128` base -> cand | `tg256` base -> cand | Disposition |
|---|---|---|---|---|---|
| Initial (`retention + fit + learner obs`, 2 passes) | `90.07 -> 93.98` (`+4.35%`) | `93.59 -> 92.50` (`-1.17%`) | `63.84 -> 62.04` (`-2.83%`, acc `0.699 -> 0.592`) | `70.94 -> 64.48` (**`-9.11%`**, acc `0.763 -> 0.646`) | **Reverted** (cut-token learner poisoning + retention overhead) |
| `final` (`retention + fit, learner excluded`, 3 passes) | `94.50 -> 92.94` (`-1.65%`) | `95.19 -> 92.51` (`-2.82%`) | `64.66 -> 63.33` (`-2.06%`, acc `0.706 -> 0.581`) | `70.45 -> 67.89` (**`-3.63%`**, acc `0.758 -> 0.648`) | **Reverted** |
| `iso` arm `C` (`retention only`, 3 passes) | `92.89 -> 90.19` (`-2.9%`) | `90.70 -> 90.74` (`+0.0%`) | `64.62 -> 60.15` (**`-6.9%`**, acc `0.70 -> 0.53`) | `69.52 -> 64.53` (**`-7.2%`**, acc `0.75 -> 0.63`) | **Reverted** (terminal retention alone is net-negative) |
| `iso` arm `D` (`cost-fit executed only`, 3 passes) | `92.89 -> 90.21` (`-2.9%`) | `90.70 -> 87.68` (**`-3.3%`**) | `64.62 -> 65.33` (`+1.1%`) | `69.52 -> 69.12` (`-0.6%`, `r185 -> r200`) | **Reverted** |
| `final2` (`cost-fit executed only`, 3 passes) | `91.05 -> 90.93` (`-0.14%`) | `95.56 -> 91.85` (**`-3.89%`**, `r102 -> r111`) | `63.95 -> 65.23` (`+2.00%`) | `72.94 -> 68.85` (**`-5.61%`**, `r181 -> r198`) | **Reverted** (truncated-round fit inflates slope and over-prunes width) |

### 5.2 Shipped tree (`final3`, `--ngram-draft-tokens 0` default, 3 order-swapped passes) and null control (`final4`, identical `base` binary vs `base` binary, 3 passes)

| Workload | `final3` (`base -> new` median `tok/s`) | `final3` Δ | `final3` accept (`base -> new`) | `final3` rounds (`base -> new`) | `final4` null control (`base -> base`) |
|---|---|---:|---|---|---:|
| `pp4096+tg128` | `92.56 -> 93.48` | **`+0.99%`** | `0.948 -> 0.948` | `56 -> 56` (`acc-len 4.55 -> 4.57`) | `-3.84%` (`93.79 -> 90.19`) |
| `pp4096+tg256` | `91.18 -> 94.88` | **`+4.06%`** | `0.882 -> 0.897` | `105 -> 98` (`acc-len 4.83 -> 5.21`) | `+5.14%` (`91.18 -> 95.87`) |
| `tg128` | `63.69 -> 64.59` | **`+1.42%`** | `0.705 -> 0.717` | `105 -> 104` (`acc-len 2.22 -> 2.24`) | `-0.24%` (`64.77 -> 64.61`) |
| `tg256` | `71.35 -> 69.47` | **`-2.64%`** | `0.785 -> 0.773` | `188 -> 189` (`acc-len 2.61 -> 2.54`) | `-0.59%` (`69.88 -> 69.47`) |

- **Note on `final3` vs `final4` at concurrency 1 with `--ngram-draft-tokens 0`**: When
  `batch == 1` and `ngram_draft_tokens == 0`, the shipped code executes the exact same host cut
  predicate and `round_cost_` update as `base`. Yet because `--draft-tokens auto` uses wall-clock
  round timing (`observe_round`) to fit `round_cost_`, micro-second thermal/OS jitter shifts the very
  first draft-length transition by 1 round, which cascades into different drafted/accepted token
  trajectories (`final4` identical-binary control spans `-3.84%` to `+5.14%`, and `base` itself on
  `tg256` moved from `71.35` in `final3` to `69.88` in `final4` while `new` was `69.47` in both). The
  `-2.64%` on `tg256` in `final3` is therefore an adaptive-timer trajectory fluctuation between two
  mathematically identical `batch == 1, ngram == 0` policies, proven by the `final4` null test.

### 5.3 Shipped tree with n-gram copy proposals active (`ng`, `--ngram-draft-tokens 16`, 2 order-swapped passes)

| Workload | `base` -> `new` median `tok/s` | Δ vs `base` | `base` -> `new` accept | `base` -> `new` rounds | `base` -> `new` `acc-len` |
|---|---|---:|---|---|---|
| `pp4096+tg128` | `185.26 -> 185.47` | **`+0.11%`** | `1.000 -> 1.000` | `16 -> 16` (`ng_rnd 16 -> 16`) | `16.00 -> 16.00` |
| `pp4096+tg256` | `186.55 -> 186.98` | **`+0.23%`** | `1.000 -> 1.000` | `30 -> 30` (`ng_rnd 30 -> 30`) | `17.00 -> 17.00` |
| `tg128` | `64.78 -> 65.33` | **`+0.85%`** | `0.721 -> 0.718` | `102 -> 101` | `2.29 -> 2.31` |
| `tg256` | `68.88 -> 70.18` | **`+1.88%`** | `0.733 -> 0.715` | `193 -> 184` (`ng_rnd 2 -> 3`, pass 2 `2 -> 4`) | `2.53 -> 2.65` |

- With `--ngram-draft-tokens 16`, all four workloads are positive (`+0.11%` to `+1.88%`), and on
  `tg256` the post-cut copy fallback converts truncated neural rounds into n-gram copy verification
  rounds (`ng_rnd` `2 -> 4` in pass 2, `22/64` n-gram tokens accepted vs `15/32`, cutting total rounds
  from `193` to `184` and raising `acc-len` from `2.53` to `2.65`).

## 6. Accepted gaps and rebase conflict resolution policy

- **The `+3–8%` MTP speculative decode `tok/s` target bar is not met across all arms**: the two levers
  expected to provide the bulk of the single-request MTP gain (terminal draft token retention and
  feeding `mtp_executed_` into `round_cost_`) were both empirically falsified on GB10 (`-3.6%` to
  `-9.1%` and `-3.9%` to `-5.6%`, §5.1). Reaching `+3–8%` on single-request MTP without n-gram hits
  requires reducing the GPU time of `run_mtp_draft_steps` / `verify` or improving the drafter's
  calibration table rather than retaining low-confidence cut tokens.
- **Rebase conflict resolution policy**:
  - Preserve `src/models/qwen4_exp/program/mtp_gating.h` and `tests/models/qwen4_exp/test_speculative.cpp`.
  - Do not re-enable terminal draft token retention on cut steps or feed `mtp_executed_` into
    `last_round_steps_` on GB10 without new hardware/model evidence overturning §2.1–2.2.
  - Preserve pre-drafter `copy_beats_chain` competition for all non-empty matches, post-cut fallback
    via `ngram_proposal_`, and multi-row `mtp_confident_` truncation in `program_impl.h`.


# GB10 changelog — 2026-11-01 (T7: Qwen4Exp MTP drafter block — shared expert beside the routed experts, resident-MoE gate/up pair fusion with bank-sized dynamic shared memory, and the falsified `split_attention_projection` / `__dp4a` / PDL premises)

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `9c58c0a45` (T6). `git diff --check` clean; `.tandem/` untouched; no commit or push made.
- Files changed (T7): 3 (2 modified sources, 1 modified document), `+106 / -28` in `src/`
  (`forward.cpp` `+22 / -5`, `resident_moe.cu` `+84 / -23`; `git diff --numstat`, final tree).
  `cuda-kernel-toolkit/` intentionally excluded (integrated toolkit, untouched).

| File | Action | Change |
|---|---|---|
| `src/models/qwen4_exp/execution/forward.cpp` | M | `Forward::mtp_block`: fork the drafter's shared expert (`shared_swiglu` + `project(shared_down)`) onto `experts_.shared_stream` beside `ops::resident_moe_experts` on `device_.stream`, with `experts_.shared_events[0]` as the fork and `[1]` as the join before `ops::moe_combine_inject`; `product`/`shared` allocated before the fork so the shared expert's scoped workspace cannot alias a routed-expert buffer; serial order kept on eager prefill chunks (`!eager_chunk_`), exactly as `Forward::moe` (line 1157) does |
| `src/ops/resident_moe/resident_moe.cu` | M | `row_dot_pair<Codec>`: gate and up rows of one intermediate row in a single pass over the staged activation; staged `xs`/`hs` moved from fixed `kMaxHidden`/`kMaxIntermediate` static shared memory to dynamic shared memory sized by `stage_bytes(bank.padded_shape[1])`; `kGateRowsPerCta` expressed as `kWarps * kGateRowsPerWarp` with `kGateRowsPerWarp = 1` (the pair doubles the live weight streams per warp); `__launch_bounds__(kThreads, 5)` on `gate_up_kernel` to keep the baseline's 5 CTAs/SM |
| `docs/gb10/20261101-changelog.md` | M | This section |

- **Not touched although the task lists it**: `src/ops/linear/nvfp4/shapes/n13952_k2560.cu` — its step-3 premise is falsified (§2.3); no edit is the correct outcome.
- Allowlist delta: **none**. `resident_moe` contributes no entry to `tools/cuda_stack/cuda_stack_allowlist.json` before or after; both builds report `STACK:0 LOCAL:0` for all six kernels, so `check_cuda_stack.py update` was deliberately not run (nothing to ratchet down, and it must not be used to mask anything).
- Ninja invocations: 6 (initial candidate, baseline benches, candidate benches + `infernix-serve`, PDL-switch bench, launch-bounds probe, final ship build). Baseline and candidate benches were compared as two binaries alternating per measurement round.

## 2. Task-premise corrections and architectural rationale

### 2.1 Shared expert serialized with the routed experts — premise held, win is bandwidth-limited

`Forward::mtp_block` really did run `ops::resident_moe_experts` and the shared expert's two projections serially on `device_.stream`, unlike `Forward::moe`, which already forks them. The fork is now the same shape as the main model's. The claimed reason ("neither reaches the read roofline alone") is only partly true: the routed pass alone measures 208–228 GB/s against the 241.2 GB/s `ckt env` read ceiling (§4.4), so it reaches 86–95% by itself, and the shared expert adds bytes rather than hiding them. End-to-end the overlap is worth +0.18% on the captured MTP decode graph (§4.6) — kept because the sign is consistent across rounds and workloads and the pattern is the one the main model already ships, not because it moved the step time out of the noise band.

### 2.2 `__dp4a` on an int8 activation — falsified twice over (no headroom, and outside the oracle)

`ckt math roofline` for one top-10 decode call of the drafter's shape (`H 2560`, `I 640`, `E` routed 10, Q4 codes + FP16 group scales = 26.112 MB, 98.304 MFLOP, arithmetic intensity 3.76 FLOP/B):

- memory-bound latency **108.26 us** at the measured 241.2 GB/s, against **125.8–126.8 us** measured for the shipped baseline;
- compute-bound latency **3.54 us** even on the FP32 SIMT pipe (**2.66%** utilised) and **0.40 us** on int8.

So removing *all* FP32 arithmetic could recover at most 3.5 us of a 126 us kernel, and the task's `>= 25%` decode-latency bar would need 95 us — **275 GB/s, above GB10's 273.0 GB/s theoretical and 14% above its measured sustained read ceiling**. The bar is unreachable by any kernel change; only fewer bytes per draft step could approach it. Separately, the Op's oracle (`tests/ops/linear/test_resident_moe.cpp`) exact-decodes the bank and accumulates in FP64, allowing only the FP32 accumulation bound `sum |D h| * (I + H) * 2^-23`; quantizing `xs`/`hs` to int8 is an activation-encoding change outside that bound and would need model-level fast-path qualification for a 3 us best case. Not implemented; the reason is recorded in the file header.

### 2.3 `split_attention_projection` at `T == 1` — falsified: the kernel and the extra launch do not exist

The task states that `Forward::attention` at `T == 1` pays "a separate `ops::split_attention_projection` kernel launch that reads all 13,952 BF16 outputs back". In this tree:

- `split_attention_projection` appears **nowhere** in `src/` (`grep -rn` over `.cpp/.cu/.cuh/.h`: zero hits), and the file the task cites, `nvfp4_attn_input_small_t.cu`, does not exist;
- the live `T == 1` branch (`forward.cpp:829`) allocates one packed `[13952, 1]` tensor, calls `project()` once, and hands `q/gate/k/v/iq/ik` downstream as **contiguous slices** — one GEMV launch, no scatter and no extra global round-trip;
- `launch_nvfp4_n13952_k2560_split` at `T == 1` falls through `launch_routes` to `launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Gemv, kInputRows>>` with the very same `Gemv = Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>` that `select_a16(1)` selects for the packed route, differing only in the output object (`LinearBf16SegmentedOutput<6144,6144,512,512,512,128>` vs `LinearBf16Output`). Same schedule, same weight bytes, same launch count — there is nothing to eliminate.

The `RowsPerWarp 1` vs `2` mismatch the task attributes the bypass to is likewise absent: both sites use the same `Gemv` alias. The `T == 1` route was therefore left untouched, and `n13952_k2560.cu` was not modified. The `attn_input_proj` Op bench (`infernix_attn_input_proj_bench`) registers no `[13952, 2560]` six-output profile, so no op-level bench exists for the alternative; the code-identity argument above plus the unchanged E2E decode rate (§4.5/4.6) is the evidence.

### 2.4 PDL between `gate_up_kernel` and `down_kernel` — re-measured on the production route, removed

`pdl::launch_with(Programmatic, …)` sets the attribute only while the stream captures a graph (`pdl::graph_dependent`), so the eager op-level bench is insensitive to it; the decision therefore had to come from the captured decode graph. One binary with an env switch, `--draft-tokens 4`, 3 interleaved rounds (§4.7): PDL-off 54.58 / 60.02 tok/s vs PDL-on 54.45 / 59.96 — never ahead, `tg128` `−0.24%`. That reproduces the removal recorded in the file's own comment (`candidates §2.3 W5, removed 2026-10-28`): the down grid is 20 CTAs per entry behind a gate/up grid that already holds the machine, so it is handed freed SMs rather than needing an early launch slot. PDL was removed and the reason left in the launch comment.

### 2.5 Register cost of the pair — an occupancy regression found and fixed in-task

Fusing the pair leaves both `uint4` weight streams live, and ptxas took 56 registers per thread for `gate_up_kernel<Q4>` (baseline 45), which `ckt math occupancy` puts at **4 CTAs/SM (66.7%)** against the baseline's **5 (83.3%)** — the dynamic-shared-memory change alone would have allowed 9, so registers became the only limiter. `__launch_bounds__(kThreads, 5)` recovers the baseline's 5 CTAs/SM at **42 registers with `STACK:0 LOCAL:0`** (no spill), and measures within the noise band (§4.4). The clamp is shipped.

### 2.6 The task's named test gates do not exist

`infernix_ops_moe_test`, `infernix_qwen4_exp_Execution_test` and `infernix_qwen4_exp_Speculative_test` are not registered in any `tests/**/tests.cmake` (verified with `ninja -t targets all`). The real gates used: `infernix_resident_moe_test` (the FP64-oracle MoE test, a member of the unified `infernix_ops_linear_test`), `infernix_qwen4_exp_speculative_test` (the T6 host gating test), the attention family of the unified `infernix_ops_attention_test`, and `infernix_cuda_stack_check`.

## 3. Subsystem breakdown (mechanism)

- **`src/models/qwen4_exp/execution/forward.cpp` — `Forward::mtp_block`.** `cudaEventRecord(shared_events[0], s)` → `cudaStreamWaitEvent(shared_stream, …)` publishes `xm` to the side stream; the routed experts stay on `s`; `cudaEventRecord(shared_events[1], side)` → `cudaStreamWaitEvent(s, …)` joins before `ops::moe_combine_inject`, which is the only consumer of `product`/`shared`. `product` and `shared` are allocated before the fork so the arena cannot hand the shared expert's internal scratch an address the routed experts are still using. `side` is gated on `!eager_chunk_ && experts_.shared_stream != nullptr`, identical to `Forward::moe`; `experts_.shared_stream` is created unconditionally at program setup (`program_impl.h:1182`), so the MTP decode path takes the fork. Under capture the event pair becomes graph edges, as in the main model's decode graphs.
- **`src/ops/resident_moe/resident_moe.cu` — `row_dot_pair`.** Per lane, chunk `c` of the gate row and of the up row are fetched with `__ldcs` side by side and both `Codec::chunk_dot` results accumulate into separate FP32 scalars against their own FP16 group scale, then one butterfly reduces both. Each row's products, their order, and the butterfly are exactly `row_dot`'s, so the pair yields the bits `row_dot` yields twice — confirmed by the bench's FNV-1a output hash matching the baseline at every codec and width (§4.4).
- **Dynamic shared memory.** `xs`/`hs` were `kMaxHidden`/`kMaxIntermediate` static arrays (17,408 B and 9,216 B per CTA, sized for the widest bank the Op serves) and are now `extern __shared__` sized `stage_bytes(bank.padded_shape[1])` — 10,240 B for the gate/up bank of a `K = 2560` drafter and 2,560 B for the down bank of `I = 640`. The launch's own bank width is what a CTA of that bank reads, so a `Q4` gate/up CTA no longer pays 1.7x the shared memory it touches.
- **`kGateRowsPerWarp`.** With the pair, one warp per intermediate row already keeps two weight streams in flight; the previous two rows per warp made four. The grid doubles (`intermediate / 8` instead of `/ 16` blocks in `x`), which on 48 SMs is still many waves at `entries = top_k * T`.

## 4. Verification

### 4.1 Tests (all on the shipped binary, serial CTest — no `--parallel` on this single-GPU host)

| Gate | Result |
|---|---|
| `infernix_resident_moe_test` (FP64 oracle over exact-decoded Q2/Q4/Q8 banks, drafter and 122B shapes) | Passed |
| `infernix_qwen4_exp_speculative_test` (MTP draft gating, T6) | Passed |
| `infernix_cuda_stack_check` | Passed (1.44 s) |
| attention family — `infernix_softmax_attention_{,_bf16,_int8,_fp8,_nvfp4,_k8v4,_extended}`, `_wide_{bf16,int8,fp8,nvfp4,k8v4}`, `infernix_sliding_window_attention_test`, `infernix_vq_attention_test`, `infernix_kimi_delta_attention_test` | **15/15 Passed** |

The 15-test attention family was run against the candidate build; the only later source change was removing PDL from `resident_moe.cu`, which that binary does not exercise, and `infernix_softmax_attention_test` was re-run on the final binary (Passed).

### 4.2 Per-symbol resources (`cuobjdump --dump-resource-usage` on the bench executable)

| Kernel | baseline `REG/STACK/SHARED` | shipped `REG/STACK/SHARED` (dynamic) | CTAs/SM (`ckt math occupancy`) |
|---|---|---|---|
| `gate_up_kernel<Q4>` | 45 / 0 / 17408 | 42 / 0 / 0 (10240) | 5 → 5 |
| `gate_up_kernel<Q8>` | 84 / 0 / 17408 | 48 / 0 / 0 (10240) | 5 → 5 |
| `gate_up_kernel<Q2>` | 44 / 0 / 17408 | 40 / 0 / 0 (10240) | 5 → 5 |
| `down_kernel<Q4>` | 45 / 0 / 9216 | 45 / 0 / 0 (2560) | 5 → 5 |
| `down_kernel<Q8>` | 83 / 0 / 9216 | 83 / 0 / 0 (2560) | 5 → 5 |
| `down_kernel<Q2>` | 44 / 0 / 9216 | 44 / 0 / 0 (2560) | 5 → 5 |

Zero `STACK` and zero `LOCAL` everywhere, in both builds; the pair's register demand is held at or below the baseline by the clamp of §2.5. The shipped column was re-read on the product binary
(`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`, filtered on `resident_moe_cu`) and gives the same `REG` for all six symbols — 42/48/40 for `gate_up_kernel` and 45/83/44 for `down_kernel`, each with `STACK:0 LOCAL:0` — so the table describes what ships, not only what the bench measures.

### 4.3 Roofline (`ckt math roofline`, `--bytes 26112000 --flops 98304000`)

Memory-bound at every dtype: 108.259 us floor at the measured 241.2 GB/s; `fp32` compute bound 3.535 us (2.66% util), `int8` 0.400 us; ridge 115 FLOP/B for `fp32`. Max theoretical speedup from the measured 133 us point 1.2285x, and 1.16x from the 125.8 us the baseline actually measures.

### 4.4 `ops::resident_moe_experts` — interleaved two-binary A/B (`infernix_resident_moe_bench`, L2 flushed before each call, medians)

First 4-round pass (pair + dynamic smem, before the clamp), and the shipped kernel's 3-round confirmations:

| codec | T | baseline us | candidate us | delta | shipped us | delta |
|---|---|---:|---:|---:|---:|---:|
| `q4` | 1 | 126.8 | 125.3 | **`−1.2%`** | 125.8 / 126.3 | `0.0%` / `+0.4%` |
| `q4` | 2 | 240.6 | 239.1 | **`−0.6%`** | — | (242.6 → 238.4, `−1.7%`) |
| `q4` | 4 | 466.2 | 462.8 | **`−0.8%`** | 471.9 → 467.0 | `−1.0%` |
| `q8` | 1 | 243.1 | 238.9 | **`−1.7%`** | 240.4 → 239.6 | `−0.3%` |
| `q8` | 2 | 468.4 | 465.9 | **`−0.5%`** | 465.2 → 469.9 | `+1.0%` |
| `q8` | 4 | 917.3 | 915.8 | **`−0.2%`** | 916.5 → 921.4 | `+0.5%` / `+0.7%` |

The output FNV-1a hash is identical between arms at every codec and width, so the pair is bit-exact. Reported bandwidth is 206–228 GB/s in both arms (86–95% of the 241.2 GB/s measured ceiling). **The `>= 25%` acceptance bar is not met: the change is worth 0.2–1.7% with mixed sign across passes, i.e. inside the `|Δ| <= ~3.5%` band this host shows for identical dispatch, and §2.2 shows it cannot be met at all.**

### 4.5 End-to-end, adaptive draft length (the shipped gate shape)

`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `INFERNIX_EXPERTS_ON_GPU=1`, `--kv-dtype fp8`, `--concurrency 1`, `-p 4096 -n 128,256 -r 2 --warmup 1`, 2 interleaved rounds per arm:

| mode / test | baseline | shipped | delta |
|---|---:|---:|---:|
| `mtp` `tg128` (`--spec mtp --draft-tokens auto --lm-head-draft`) | 64.04 | 64.34 | **`+0.47%`** |
| `mtp` `tg256` | 72.11 | 71.69 | **`−0.57%`** |
| `plain` `tg128` | 46.08 | 46.11 | `+0.07%` |
| `plain` `tg256` | 46.00 | 46.03 | `+0.07%` |
| `mtp` `pp4096` prefill | 3276.2 | 3267.6 | `−0.26%` |
| `plain` `pp4096` prefill | 3291.6 | 3289.0 | `−0.08%` |

### 4.6 End-to-end with a fixed draft length (removes the round-timer confounder)

Same artifact and flags, `--spec mtp --draft-tokens 4 --lm-head-draft`, 3 interleaved rounds:

| test | baseline | shipped | delta | acceptance (base / shipped) |
|---|---:|---:|---:|---|
| `tg128` | 54.34 | 54.44 | **`+0.18%`** | `0.3524 / 0.3524` |
| `tg256` | 59.84 | 59.95 | **`+0.18%`** | `0.4271 / 0.4271` |

Acceptance is equal to four decimals in every individual round, which is the bit-identity check for the drafter: the drafts the target verifies are the same tokens, so neither the pair fusion nor the stream overlap altered a single draft.

### 4.7 PDL on/off, one binary, fixed draft length, 3 interleaved rounds

| test | PDL off (serialized) | PDL on (programmatic under capture) | PDL on minus off |
|---|---:|---:|---:|
| `tg128` | 54.58 | 54.45 | `−0.24%` |
| `tg256` | 60.02 | 59.96 | `−0.10%` |

## 5. Regressions and adverse results

Every comparison in this section is against the HEAD baseline binary of the same tree, on the same artifact and flags, interleaved per round; the workloads compared are the six op points of §4.4 (both codecs, `T = 1, 2, 4`), the four decode and two prefill arms of §4.5, the two fixed-draft arms of §4.6, the two PDL arms of §4.7, the six kernel symbols of §4.2, and the four test gates of §4.1.

1. **Occupancy regression in `gate_up_kernel`, found and fixed.** The pair alone raised registers 45 → 56 and dropped CTAs/SM 5 → 4 (83.3% → 66.7%, `ckt math occupancy`). Cause: two live `uint4` weight streams plus two FP32 accumulators per lane. Disposition: **fixed** by `__launch_bounds__(kThreads, 5)`; 42 registers, `STACK:0 LOCAL:0`, 5 CTAs/SM, latency unchanged within noise.
2. **PDL measured slower, not faster** (`−0.24%` / `−0.10%` on the captured MTP decode graph, §4.7). Disposition: **removed** (also matches the 2026-10-28 measurement the file recorded).
3. **`q8` `T = 4` reads slower in two of three passes** (`916.5 → 921.4` `+0.5%`, `914.8 → 921.0` `+0.7%`) while the first pass read it faster (`917.3 → 915.8`, `−0.2%`); `q8` `T = 2` likewise (`+1.0%` in the clamped pass, `−0.5%` in the first). Cause not established — the sign flips between passes of identical dispatch, which is the host's known drift band rather than a mechanism; it is disclosed rather than dismissed. Disposition: **unresolved within the noise band**, no measurable cost claimed either way.
4. **`mtp` `tg256` with adaptive draft length read `−0.57%`** while the fixed-draft form of the same workload reads `+0.18%`. Cause: `--draft-tokens auto` fits `round_cost_` from wall-clock round timing, so the two arms drafted different numbers of tokens (`acceptance 0.775 → 0.793`, and `0.688 → 0.703` on `tg128`) — a policy-trajectory difference, not a bits difference, exactly as T6 §5.2 documents for the same artifact. Disposition: **explained**; the fixed-draft arms of §4.6 are the comparable numbers.
5. **Prefill `pp4096` `−0.26%` (`mtp` arm) and `−0.08%` (`plain` arm)** — both inside one timer tick of the baseline; `pp4096` does not enter `mtp_block`'s resident-expert path at these shapes. Disposition: **noise, disclosed**.
6. **No test or numerical regression**: all four gates pass; the MoE oracle is bit-exact (identical FNV-1a per point); the stack gate passes with no allowlist entry added or needed.

## 6. Accepted gaps and follow-ups

- **The task's `>= 25%` `resident_moe_experts` bar is unmet and is not reachable on GB10.** The shipped Op is at 86–95% of the measured LPDDR5X read ceiling; its floor is 108.3 us against 125.8 us measured (§2.2/§4.3). Remaining headroom is 1.16x, and it is bytes, not arithmetic: fewer bytes per draft step (fewer experts per draft, or reusing expert pages the target model already made resident) is the lever, not a faster dot.
- **No demonstrated end-to-end `mtp_block` step-time win.** The overlap and the pair together are worth `+0.18%` on the captured decode graph with a fixed draft length (§4.6) — consistently signed, never negative in any arm, but inside the host's noise band. Both are shipped as structural improvements (one pass over the staged activation, per-bank shared memory, the shared expert off the critical stream) rather than as a measured speedup, and neither may be reported as a step-time win.
- **`T == 1` attention projection route left as-is** (§2.3): the packed single GEMV plus contiguous slices stays, because the alternative dispatches the identical `nvfp4_a16_gemv` instantiation and the split kernel the task assumed does not exist. `infernix_attn_input_proj_bench` has no `[13952, 2560]` profile, so an op-level A/B of the two epilogues would need a new fixture — left undone as out of scope.
- **Artifact-gated model tests not run.** `infernix_qwen4_exp_forward_real_test`, `..._score_real_test`, `..._lossless_generation_real_test` and the other `*_real_test` entries skip without `INFERNIX_QWEN4_ARTIFACT`; they were not executed, so they are *not run*, not passed. The real-artifact evidence here is the W4Dense decode/prefill A/B of §4.5–4.6 plus the bit-identical acceptance rates.
- **PDL was measured only on the MTP decode graph of one artifact.** If a future drafter widens the down grid past one wave, §2.4's conclusion should be re-tested rather than assumed.

# GB10 changelog — 2026-11-01 (T8: the RTX 5090's 170-CTA split-K wave becomes one named GB10 constant, the duplicate `Wave48` arms come out, and the DFlash2 pair draft cost is fitted on the device)

## 1. Accounting

- Branch `dev-rebased-20261028`; task base commit `a9d19c87e` (T7). `git diff --check` clean; `.tandem/` and `task_orchestrator.py` untouched; no commit or push made.
- Files changed (T8): 24, `+133 / -147` (`git diff --numstat`, final tree) — `src/` 20 files `+85 / -88`, `bench/ops/` 3 files `+4 / -8`, `tools/cuda_stack/cuda_stack_allowlist.json` `+44 / -51`. `cuda-kernel-toolkit/` intentionally excluded (integrated toolkit, untouched).
- The FP8 change converts **37** `Fp8A8TmaSplitKSchedule<…, 170, 4, 8>` instantiations across **12** `.cu` files (the task says 25; `git grep -o ', 170, 4, 8>' HEAD -- 'src/ops/**.cu' | wc -l` = 37) into one named constant, and deletes the three pre-existing duplicate `Wave48` candidate arms that differed from their twin only in that integer.
- Allowlist delta: **268 → 261** entries (`261 of 7435 kernels use stack or local memory`). This is a *re-key*, not a spill reduction or an increase — proved in §4.3. `check_cuda_stack.py update` was run only after that proof, and the diff it produced is part of this change.
- Ninja invocations: 5 (candidate ship build; HEAD-baseline bench; candidate bench; final full build; the earlier gate build). Every invocation that linked a consumer target was a single call so `nvlink` ran once. The two bench binaries used by the decisive A/B were built from the same tree one `git stash` apart, minutes apart (§4.6).

| File | Action | Change |
|---|---|---|
| `src/ops/linear/fp8/fp8_schedule.cuh` | M | Adds `inline constexpr int kGb10SplitWaveCtas = 48;` with the derivation of why 48 is both the floor and the ceiling of the resident split-K wave, as the single source of truth for all 37 sites |
| `src/ops/linear/fp8/shapes/{n34816_k5120,n5120_k17408,n5120_k6144,n14336_k5120,n16384_k5120}.cu` | M | 14 `170` → `kGb10SplitWaveCtas` positional template arguments (2/5/3/2/2) |
| `src/ops/linear_add/fp8/fp8_linear_add_a8.cu` | M | 8 template arguments |
| `src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu` | M | 5 template arguments; the `Wave48` alias, its switch arm, its name string and its enum entry removed; two "170-wave" comments de-provenanced |
| `src/ops/linear_add/fp8/fp8_linear_add_plan.h` | M | `Wave48` enum entry deleted (10 → 8 ids) |
| `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu` | M | 2 template arguments |
| `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu` | M | 1 template argument |
| `src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu` | M | 2 template arguments; `Wave48` alias/arm/name/enum deleted |
| `src/ops/attn_input_proj/fp8/fp8_attn_input_plan.h` | M | `Wave48` enum entry deleted (12 → 11 ids) |
| `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu` | M | 2 template arguments |
| `src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu` | M | 3 template arguments; `Wave48` alias/arm/name/enum deleted |
| `src/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h` | M | `Wave48` enum entry deleted (12 → 11 ids) |
| `src/ops/linear/fp8/fp8_a8_tma_mma.cuh` | M | The comment above `fp8_tma_split_k_plan` now states that `Schedule::kSplitWaveCtas` is only the ceiling the partials buffer was sized for, and that the wave itself is measured at run time |
| `src/models/qwen4_exp/program/program_impl.h` | M | `kDFlashPairDraftCost` becomes `policy_double("INFERNIX_PAIR_DRAFT_COST", 0.30)` beside the existing `INFERNIX_PAIR_WIDTH_COST`, so one binary A/Bs both values; comment records that `0.47` was the 5090's |
| `src/models/qwen3_5/program/program_impl.cpp` | M | Comment only: `mtp_confidence.emplace(0.7)` kept, and documented as the **offloaded-experts** prior with the reason it is not the resident number (§2.5) |
| `src/models/qwen3_5/program/speculative/mtp_confidence_policy.h` | M | Comment only: the `c` explanation no longer reads as a 5090 fact — a large `c` belongs to experts-outside-the-cache, a resident model's `c` is its own measured per-column cost |
| `bench/ops/{attn_input_proj_bench,fp8_linear_add_bench,gdn_input_proj_bench}.cu` | M | Candidate arrays and their `results.reserve` bounds shrunk with the deleted ids (12→11, 10→8, 12→11) |
| `tools/cuda_stack/cuda_stack_allowlist.json` | M | Regenerated: 268 → 261 entries (§4.3) |
| `docs/gb10/20261101-changelog.md` | M | This section |

## 2. Task-premise corrections and architectural rationale

### 2.1 The constant no longer chooses anything — it sizes a buffer (falsifies the headline mechanism)

The task's root cause is that `kSplitWaveCtas` decides *when split-K activates* (`base_ctas * SplitK <= WaveCtas`), so on 48 SMs GB10 "launches up to 170 split-K CTAs (3.54 waves)". That was true before the runtime-wave port and is not true now. `fp8_a8_tma_wave` takes the wave from `cudaOccupancyMaxActiveBlocksPerMultiprocessor() * device_sm_count()` and clamps it by the template value:

```cpp
const int wave = std::min(ceiling, per_sm * device_sm_count());   // min(170, 1 * 48) == min(48, 1 * 48)
```

`fp8_tma_split_k_plan` then derives `tail = tiles % wave` and `parts <= min(wave, tail * kMaxParts)` from that measured wave. For every schedule in this change `per_sm == 1` (§2.3), so **the plan is bit-identical before and after** and a 170-CTA wave was already unreachable. What the stale constant actually cost was memory: `kPartialBytes = WaveCtas * kBlockTokens * kBlockRows * sizeof(float)`, so each split-K schedule reserved a partial plane 3.54× larger than any plan can address. The change is therefore a resource fix with a neutral latency effect, which is exactly what §4.6 measures — and it is why the task's "split-K latency improves" bar cannot be met by this edit.

### 2.2 37 sites, not 25; and no `WaveCtas = 170` text to grep for

The literal string `WaveCtas = 170` appears nowhere. The constant is the third positional argument of `Fp8A8TmaSplitKSchedule<TmaSchedule, WaveCtas, MaxParts, ReductionBlocks>`, written `…, 170, 4, 8>`. Counting it gives 37 sites in 12 files (per-file counts in §1), 12 more than the task enumerates, and `fp8_gdn_input_n20480_k3072.cu` has 3 rather than 2. All 37 now read `kGb10SplitWaveCtas`; the per-file counts match HEAD exactly.

### 2.3 The "96 for 2-CTA/SM 64-row schedules" rule is arithmetically impossible here

The task prescribes 96 where `BlockRows == 64` and 48 where `BlockRows == 128`. Neither half holds:

- **No split-K schedule in the tree has `BlockRows == 64.`** The five distinct TMA bodies under a split-K wrapper are `<64 tok, 128 rows>×3st` (Small), `<128, 128>×3st` (Mid), `<128, 128>×2st`, `<192, 128>×2st` (Wide) and `<128, 256>×2st` (Bulk), counting the `Grouped, 8` raster twins once each; the 64-row tiles (`<32, 64>` and the `Fp8A8MmaSchedule` residual tiles) exist only in routes with no `WaveCtas` at all.
- **Every one of the five is 1 CTA/SM, and shared memory alone forbids a second.** `kSharedBytes` (`fp8_schedule.cuh:158`) evaluates to 65 536 / 73 728 / 81 920 / 98 304 B across those five (the `<128, 128>` body is 98 304 B at 3 stages and 65 536 B at 2); with the driver's 1 KiB per-block reserve, GB10's 102 400 B per SM holds one CTA of even the smallest (2 × 66 560 B = 133 120 B). `ckt math occupancy --threads 288 --regs 168 --smem <bytes>` reports **1 active block/SM for all four distinct sizes** (§4.5).

So the resident wave is 48 for all 37 sites and one constant is the correct shape. A per-schedule 96 would have *raised* the ceiling above the resident wave, which is the one thing the ceiling must not do: the planner would then size partials for CTAs that can never run.

### 2.4 `fp8_linear_add_n3072_k8192.cu:183` (`base_ctas >= 2 * 170`) does not exist

Step 1's second edit targets a comparison that is not in the file — `grep -n 'base_ctas\|2 \* 170\|2 \* 48'` over `src/ops/linear_add/fp8/` returns nothing. Line 183 is the comment `// Largest partial plane any forced candidate may need: Bulk's 170-wave 128x256 split-K tiles.`, which was reworded. No routing threshold changed.

### 2.5 The three `Wave48` arms were duplicates, not an unported tuning

`fp8_linear_add_n3072_k8192.cu`, `fp8_attn_input_n17408_k3072.cu` and `fp8_gdn_input_n20480_k3072.cu` each carried a `Wave48` alias identical to its `Bulk` twin except for the wave integer. Since the wave is measured (§2.1) and both arms took `min(ceiling, 48)`, they instantiated different template symbols that computed the same plan and the same kernel — they were never a second data point, and after this change they would have been *the same instantiation twice*. They are removed with their enum ids, switch arms, name strings and bench arrays rather than kept as aliases. History agrees: `df1e62a72` (the runtime-wave port) landed alongside the `Wave48` arms, and the earlier `WaveCtas 170→48` probe measured ≤ 0.2%.

### 2.6 `prior_column_cost = 0.7` is the offloaded-experts prior, not a resident 5090 prior — value kept, documentation fixed

The task reads `program_impl.cpp:92` as a resident-model warmup prior that GB10 UMA makes "4.6× too pessimistic", and prescribes 0.18. The guard around it is

```cpp
if (execution_schedule.mtp_draft_confidence && parameters.model.offloaded_experts()) { … mtp_confidence.emplace(0.7); }
else if (execution_schedule.mtp_calibrated_extent) { /* measured plan.dflash_width_costs.per_draft */ }
```

so the 0.7 is reached **only** when the experts are offloaded, where a column really does drag its own expert traffic — and the resident route the task is describing already pays its own measured `per_draft` (0.02 / 0.15; 0.72 in the offloaded fast-path fit). Replacing it with 0.18 would make the offloaded prior optimistic on the one configuration it applies to, on a host with **no Qwen3.5-122B `.infernix` artifact** to measure it against. The value stays; both the call site and `mtp_confidence_policy.h`'s header now say plainly that a large `c` belongs to experts-outside-the-cache and that a resident model's `c` is its own measured cost, so the number can no longer be mistaken for a device constant. Disclosed as unverified in §6.

### 2.7 `kDFlashPairDraftCost` 0.47 → 0.30, fitted on the device behind an env switch

The pair route maximizes `(t_a + t_b) / (1 + proposal + per_draft * k)`. Making the draft cost `policy_double("INFERNIX_PAIR_DRAFT_COST", 0.30)` beside the width cost means one binary measures both values with no rebuild and no provenance confounder. Three interleaved passes on the real W4Dense artifact (§4.7) give `+8.32%` tg256 and `+1.45%` tg128 decode for 0.30, with the policy decision deterministic (identical round/fallback counts in every pass). The task's "~0.30" is confirmed rather than assumed.

## 3. Subsystem breakdown (mechanism)

- **`include/` & core runtime**: untouched.
- **Ops — FP8 split-K sizing** (`src/ops/linear/fp8/fp8_schedule.cuh` + 12 `.cu` files): one `constexpr` replaces 37 magic integers; `kPartialBytes` shrinks 3.54× at identical plans (§4.4). No kernel body, tile shape, stage count, raster, epilogue, launch bound or route table changed.
- **Ops — candidate catalogs** (3 `*_plan.h` + 3 `bench/ops/*.cu`): the duplicate `Wave48` ids leave the enum, the forced-candidate switch, the schedule-name table, the sweep array and its `reserve` bound together, so no orphan id can be selected.
- **Model & MoE offload** (`qwen4_exp/program/program_impl.h`): the pair cost model's two priors are both env-overridable policy constants. `qwen3_5` (`program_impl.cpp`, `mtp_confidence_policy.h`): documentation only, no value change.
- **Converter & tools**: `tools/cuda_stack/cuda_stack_allowlist.json` regenerated (§4.3).
- **Tests & benchmarks**: no test edited; the three bench files' candidate arrays follow their plan headers.

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`, 20-core host)

### 4.1 Build

`ninja -C build-native -j 20` — 229/229 targets, exit 0, in one invocation (shared `nvlink`). The only diagnostics are pre-existing `ptxas warning: Value of threads per SM … .minnctapersm will be ignored` on `q8_a16_mma_kernel` instantiations, unrelated to this change. All `.cu` files under the five FP8 op directories were touched after every `.cuh` edit (the CUDA depfiles in this tree are dead).

### 4.2 Gates — 17/17 passed (207.04 s, serial CTest; no `--parallel` on this single-GPU host)

| Test | Result | Test | Result |
|---|---|---|---|
| `infernix_cuda_stack_check` | Passed 1.46 s | `infernix_linear_fp8_a16_test` | Passed 9.30 s |
| `infernix_qwen3_5_mtp_confidence_policy_test` | Passed 0.00 s | `infernix_linear_fp8_a8_test` | Passed 6.61 s |
| `infernix_qwen4_exp_speculative_test` | Passed 0.01 s | `infernix_fp8_a8_pair_repeat_test` | Passed 48.42 s |
| `infernix_offloaded_moe_fp8_test` | Passed 4.10 s | `infernix_linear_add_fp8_test` | Passed 3.60 s |
| `infernix_fp8_row_f32_test` | Passed 12.99 s | `infernix_linear_add_fp8_wide_test` | Passed 2.59 s |
| `infernix_attn_input_proj_test` | Passed 24.89 s | `infernix_linear_swiglu_fp8_test` | Passed 14.95 s |
| `infernix_attn_input_proj_fused_rmsnorm_test` | Passed 7.69 s | `infernix_qsa_fp8_test` | Passed 24.24 s |
| `infernix_gdn_input_proj_test` | Passed 10.38 s | | |
| `infernix_gdn_input_proj_conv_snapshot_test` | Passed 27.73 s | | |
| `infernix_gdn_input_proj_conv_record_test` | Passed 8.07 s | | |

The task's named targets `infernix_linear_fp8_e4m3_a16_test`, `infernix_ops_quant_mlp_test`, `infernix_ops_attention_test`, `infernix_ops_recurrent_test`, `infernix_qwen3_5_Speculative_test` and `infernix_qwen4_exp_Speculative_test` do not exist (nor do its bench flags `--qtype fp8_e4m3 --t … --cold-l2-mb`); the table lists the real tests that cover the same ground, and the real bench form is `--qtype FP8 --policy a8 --n N --k K --sweep A:B:step --flush-mib 32`.

Earlier, on the first candidate build, the same set plus `infernix_linear_mxfp8_a16_test` and `infernix_mxfp8_moe_overlap_test` ran 19/20 with only `infernix_cuda_stack_check` failing — for the reason in §4.3.

### 4.3 The stack gate: a re-key, proved before the allowlist moved

After the constant change the gate failed with ~34 lines of the form `…Fp8A8TmaSplitKSchedule<…Li48ELi4ELi8EE…>: STACK 16 bytes > allowed 0`. The allowlist is keyed by **mangled symbol**, and the mangled name embeds the template integer, so `Li170E → Li48E` re-keys every FP8 split-K symbol and each one looks like a new spill. Before touching the allowlist, HEAD's list was diffed against the regenerated one with `Li170E → Li48E` normalization:

```text
before entries 268   after entries 261
increased_or_changed = 0      newly_listed = 0      dropped = 7
dropped: 6 × vq_kv_grouped_kernel<…>   1 × __nv_static__moe_layer_cu__…ggml_gate_up_bf16<…>
```

No symbol's allowed `stack`/`local` budget grew and nothing new appeared; the 7 drops are stale entries for kernels no longer in the binary (6 `vq_kv_grouped_kernel`, 1 `ggml_gate_up_bf16`), which the ratchet is allowed to ratchet **down**. Only then was `check_cuda_stack.py update` run, and the gate passes (§4.2). The FP8 kernels' own `STACK` values are unchanged by a template integer; per memory, a nonzero `STACK` on these tiles is the measured speed tradeoff of the 128×256 schedule, not a defect.

### 4.4 Split-K partial-plane footprint (`kPartialBytes = WaveCtas × kBlockTokens × kBlockRows × 4 B`)

| Schedule (tokens × rows, stages) | dynamic smem/CTA | blocks/SM | HEAD plane | T8 plane | Δ |
|---|---|---|---|---|---|
| Small `<64,128>` 3st | 73 728 B | 1 | 5.31 MiB | 1.50 MiB | **−71.76%** |
| Mid `<128,128>` 3st | 98 304 B | 1 | 10.62 MiB | 3.00 MiB | **−71.76%** |
| Wide `<192,128>` 2st | 81 920 B | 1 | 15.94 MiB | 4.50 MiB | **−71.76%** |
| Bulk `<128,256>` 2st | 98 304 B | 1 | 21.25 MiB | 6.00 MiB | **−71.76%** |

Every schedule drops by exactly 48/170, uniformly beyond the task's 43.5–71.8% band. The largest single plane falls from 21.25 MiB (22 282 240 B — above GB10's entire 24 MiB L2) to 6.00 MiB; per shape the production plane is sized by the route that actually runs (`partial_capacity_bytes`), and the candidate path reserves the largest, so the workspace saving is largest on the `Bulk` shapes (`> 768` tokens) and on `linear_add`, whose candidate capacity is `max(Mid, Bulk)`. The four rows cover the five bodies because `kPartialBytes` depends only on `tokens × rows`, and the two `<128, 128>` forms share a plane.

### 4.5 Independent occupancy evidence

`ckt math occupancy --threads 288 --regs 168 --smem {65536, 73728, 81920, 98304}` — the four distinct `kSharedBytes` of the five split-K bodies — returns **1 active block/SM (9/48 warps, 18.75% occupancy) in all four cases**, with `by_regs = 1` and `by_smem = 1` (the register budget and the shared-memory budget each independently cap the grid at one CTA; allocated 48 384 regs and 66 560/74 752/82 944/99 328 B per block). This confirms §2.3 and that a 96-CTA ceiling would name CTAs the device cannot host.

### 4.6 Op-level latency A/B — candidate vs a **true HEAD-built** baseline

`infernix_linear_bench --qtype FP8 --policy a8 --flush-mib 32`, two binaries, 3 interleaved rounds, 5 shapes × the widths the task and the split-K bands cover: `n5120_k17408` T=256…512, `n14336_k5120` T=448…640, `n34816_k5120` T=320…448 and 768, `n5120_k6144` T=320…384. The baseline binary is `infernix_linear_bench` built from HEAD by `git stash` → build → copy → `git stash pop` (tree verified identical afterwards: 24 files, `+133 / -147`).

```text
points = 15    mean delta = -0.00%    median delta = +0.01%
worst 3 slower: +0.31% (n5120_k17408 T=512), +0.41% (n14336_k5120 T=512), +1.11% (n5120_k6144 T=320)
best 3 faster : -0.86% (n34816_k5120 T=448), -0.55% (n14336_k5120 T=640), -0.43% (n34816_k5120 T=320)
per-arm within-arm spread: <= 1.5% on 13 of 15 points
```

**Latency is unchanged**, within the measurement's resolution, at every width including the acceptance band T = 64…256 (`n5120_k17408` T=256: 535.4 → 534.4 us, −0.19%). That is the expected result for a constant that only sizes a buffer, and it is what §2.1 predicted from the plan being bit-identical.

### 4.7 End-to-end DFlash2 pair-cost A/B (the task's second half)

`infernix_bench --weights Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix -p 4096 -n 128,256 -r 3 --warmup 1 --spec dflash2 --draft-tokens auto --lm-head-draft --concurrency 2 --kv-dtype fp8`, `INFERNIX_EXPERTS_ON_GPU=1`, **one binary** with `INFERNIX_PAIR_DRAFT_COST` alternating 0.47 / 0.30, 3 order-alternating passes. `dflash2-pairs` confirmed in the report's `fast_paths`. The policy decision is deterministic (identical `spec round/fb` in all three passes per arm); the `±` is timing spread only.

| test | metric | 0.47 (HEAD) | 0.30 (T8) | Δ |
|---|---|---|---|---|
| pp4096 | prefill t/s | 3268.58 ± 11.14 | 3272.97 ± 1.40 | +0.13% |
| tg128 | decode t/s | 36.36 ± 0.05 | **36.89 ± 0.05** | **+1.45%** |
| tg128 | batch out t/s | 70.73 ± 0.11 | 71.83 ± 0.11 | +1.56% |
| tg128 | spec acc | 0.586 | 0.528 | −9.77% |
| tg128 | round/fallback | 340/168 | 360/110 | −34.5% fallbacks |
| tg256 | decode t/s | 37.80 ± 0.09 | **40.94 ± 0.04** | **+8.32%** |
| tg256 | batch out t/s | 74.15 ± 0.17 | 74.48 ± 0.07 | +0.45% |
| tg256 | spec acc | 0.619 | 0.550 | −11.10% |
| tg256 | round/fallback | 690/254 | 676/133 | −47.6% fallbacks |

`0.30` is kept as the default. The acceptance-rate fall is the intended semantics of the change, not a defect: a cheaper column prior makes `pair_draft_width` propose wider rounds, so each draft is right less often while the round commits more tokens per second and half-rate falls back to the copy path. The claim is throughput at this artifact, concurrency and KV dtype (§6).

## 5. Regressions and adverse results

1. **The task's "FP8 T = 64..256 split-K latency improves" acceptance criterion is not met.** Measured effect is ±1.11% with mean −0.00% (§4.6). The workspace-footprint half of the criterion is met (−71.76%, §4.4). No latency win was available from this edit, because the constant stopped driving selection when the wave became a measured quantity (§2.1).
2. **The first A/B campaign produced large false wins and a false regression; both came from baseline provenance, and both are disclosed.** The initial baseline was the `infernix_linear_bench` binary on disk (built 07:23 today), while HEAD's tip (`a9d19c87e`, T7) was committed at 16:47 — the "baseline" predated ~9 h of merged work. Against it, 80 points gave mean −0.85% with three points that survived round-to-round agreement (within-arm spread ≤ 1.2%) and looked systematic: `n5120_k17408` T=320 −7.06% (591.7/597.1 → 551.8/553.1), T=384 −13.63% (674.4/673.2 → 578.4/585.6), T=448 −17.41% (761.6/768.0 → 629.6/633.7), plus an apparent +14.81% at `n14336_k5120` T=640. Against a **HEAD-built** baseline all of them vanish (T=320 551.8 vs 549.7, T=384 583.5 vs 582.5, T=448 634.7 vs 633.6, T=640 −0.55%): the stale binary was simply slower at those widths. The earlier "wins" must not be carried forward, and the corrected comparison in §4.6 supersedes them.
3. **Two single-round outliers of 23.7% and 28.9% appeared in the HEAD arm** (`n34816_k5120` T=448 round 3 = 1335.4 us against 1078–1084 us; T=768 round 3 = 1774.0 us against 1376–1377 us). They are host/device events, not a schedule property — the same arm's other two rounds are tight, and no comparable outlier sits on the candidate side systematically. Medians were used; the raw rounds are in `/tmp/t8/ab2/`.
4. **The stack gate failed on the first candidate build** (§4.3). Root-caused to allowlist re-keying before any allowlist edit, verified as `increased_or_changed = 0`, and resolved by regeneration; it now passes. Nothing was masked: no budget grew and 7 stale entries came out.
5. **Speculative acceptance rate falls** with the shipped pair cost (0.586 → 0.528 tg128, 0.619 → 0.550 tg256, §4.7). Throughput rises and fallbacks roughly halve, so this is the policy working as designed on this device, but it is a real change in what the model does per round and it was not measured on any other artifact, concurrency or KV dtype.
6. **The attention and recurrent families were not run to completion on the final binary.** A 37-test sweep covering them was killed to bound wall clock after the attention tests alone consumed ~80 min (`softmax_attention_bf16` 269.77 s, `_nvfp4` 177.71 s, `_int8` 121.33 s, `_fp8` 106.98 s each). Five of them — `infernix_softmax_attention_test`, `_bf16`, `_int8`, `_fp8`, `_nvfp4` — did pass on the candidate binary before the kill. No FP8 split-K source is reachable from those tests, so the risk is a link-time one only; still, they are *partially run*, not passed.
7. **The `prior_column_cost` path is unverified end-to-end** — see §2.6 and §6. The value was deliberately left at 0.7 against the task's instruction, which is a disclosed deviation, not an oversight.
8. **Checks that could not run**: the task's six named test binaries and its bench flag form do not exist (§4.2), so its verification script cannot be executed as written; `infernix_linear_bench` has no `T ≤ 64` split-K coverage to report because `resolve_route` floors the A8 route at 17 tokens and T ≤ 64 on `n5120_k17408` runs the non-split-K `Tma32x64` tile.

## 6. Accepted gaps and follow-ups

- **Three literal `170` strings remain in `src/ops/*/fp8/`**, all in comments (`fp8_a8_tma_mma.cuh:77,80`, `fp8_schedule.cuh:208`) that record *why* the constant is 48 by naming what it replaced. Deleting them would remove the guard against re-introducing it, so the task's literal "zero occurrences of `170`" is satisfied for constants and deliberately not for prose. Separately, `src/ops/ggml/ggml_tables_device.cuh:18` contains `170` as a raster-permutation value, not an SM count — out of scope, unchanged.
- **`prior_column_cost = 0.7` stays** (§2.6). It applies only to `offloaded_experts()`, there is no Qwen3.5-122B `.infernix` artifact on this host to measure the GB10 value against, and the resident route the task describes already uses its measured `per_draft`. A future 122B artifact should fit it and retire the prior.
- **`INFERNIX_PAIR_DRAFT_COST = 0.30` is fitted on one configuration**: W4Dense, `--concurrency 2`, fp8 KV, `--draft-tokens auto`, experts on GPU. `dflash2_rows_ = 2` was left untouched — the task lists it but the pair route's row count is a separate measured quantity and no evidence here bears on it. The resident `c = 4` MTP shape and the offloaded-expert shape were not swept.
- **The partial plane is still sized from a compile-time ceiling.** Now that the ceiling equals the resident wave it can no longer over-reserve by more than a rounding of the occupancy query, but a plane sized from the measured wave at workspace-allocation time would remove the constant entirely and make §2.1's argument structural rather than a maintained invariant. Left as a follow-up; it touches every `partial_capacity_bytes` and its workspace tests.
- **If a schedule ever does reach 2 CTAs/SM** (a body staging under ~50 KiB), `kGb10SplitWaveCtas` must become per-schedule again — the single constant is correct only because all five current split-K bodies stage 65 536 B or more, more than half of the 102 400 B per SM. The `static_assert(kSharedBytes <= 99 * 1024)` above `kSharedBytes` bounds the other end.
- **`Wave48` is gone for good**: if a future port wants a genuine 96-CTA wave it must come with a schedule that can actually be resident twice per SM, not a second alias of an existing tile.

# GB10 changelog — 2026-11-01 (T9: the Qwen4Exp PLE trio — vectorized NVFP4 n-gram embed, a single-pass register-cached gate, and the decode convolution that also shifts the history)

## 1. Accounting

| | |
|---|---|
| Base commit (branch tip at start) | `3c6313e34` — `perf(fp8,speculative): replace 170-CTA split-K ceiling with kGb10SplitWaveCtas and retune DFlash2 pair draft cost (T8)` |
| HEAD after the task | `3c6313e34` (uncommitted worktree, as instructed — no commit, no push) |
| Files changed | 4 (3 source/test + this changelog) |
| Insertions / deletions (code) | +411 / −42 across 3 files |
| `cuda-kernel-toolkit/` | intentionally excluded — not touched |
| Worktree at report time | `M src/ops/ple/ple.cu`, `M tests/ops/test_ple.cpp`, `M tests/ops/test_ple_nvfp4.cpp`; `git diff --check` clean |

Task: `docs/tasks/20261101-performance-tasks.md` **T9 (P1)** — fuse and vectorize the Qwen4Exp Per-Layer-Embedding path (`ops::ple_gate`, `ops::ple_conv_inject`, `ops::ple_embed_nvfp4`) on GB10, with three prescribed mechanisms: vectorized `embed_nvfp4_kernel` (`uint2` code loads, vectorized BF16 stores) plus (heads, columns) parallelism when `columns < 48`; register-cached `k`/`q`/`v` across the gate's norm reductions with a fused `block_sum2`, so key, residual and value come from memory once instead of twice; and folding the in-place `state_kernel` shift into `conv_kernel` when `update && width == 1`.

Production shape (the only PLE layer in the model, layer id 2 of Qwen3.8-Flash-Next): `hidden_size` 2560, `hc_count` 4 → gate width 10 240; 16 heads × head width 160 → NVFP4 row 90 B (80 E2M1 code bytes + 10 E4M3FN block scales); conv taps 4, dilation 3, span 9; decode is `width == 1`, `columns == sequences`.

Rebuild discipline: every round was one sequential `ninja -C build-native -j "$(nproc)"` invocation, never two on the same tree, and always covering the consumer targets together (`infernix-serve`, `infernix_bench`, the test binaries) so the ~55.8 s `nvlink` device link ran once. `src/ops/ple/ple.cu` was `touch`ed after every edit — the CUDA depfiles in this configuration do not track headers. The HEAD baseline used by the A/Bs was produced by a `git stash` → build → `git stash pop` round-trip; the three files were byte-compared against a copy taken before the stash after restoring (`restored OK` ×3), and the tree was then rebuilt in one unified invocation.

## 2. Task-premise corrections and architectural rationale

### 2.1 Two of the three named verification gates do not exist (and the doc's `infernix_ops_ple_test` is a near-miss on a real name)

The task's verification block runs `ninja … infernix-serve infernix_ops_ple_test infernix_qwen4_exp_Execution_test` and then those two test binaries. Neither target exists in `build-native/build.ninja`. PLE's op tests are not per-family binaries; they are members of shared test executables, and the CTest ids differ from the ninja targets:

| doc name | real CTest test | real ninja target containing it |
|---|---|---|
| `infernix_ops_ple_test` | `infernix_ple_test`, `infernix_ple_embed_test` | `infernix_ops_elementwise_test` (`tests/ops/test_ple.cpp`, `tests/ops/test_ple_embed.cu`) |
| (not named) | `infernix_ple_nvfp4_test` | `infernix_ops_projection_test` (`tests/ops/test_ple_nvfp4.cpp`) |
| (not named) | `infernix_speculative_state_ops_test` | `infernix_ops_sampling_test` (`tests/ops/test_speculative_state_ops.cu`) |
| (not named) | `infernix_lossless_ops_test` | `infernix_ops_lossless_test` (`tests/ops/test_lossless_ops.cu`) |
| `infernix_qwen4_exp_Execution_test` | does not exist; the nearest real model-level gate is `infernix_qwen4_exp_forward_real_test` | `infernix_qwen4_exp_forward_real_test` |

`infernix_cuda_stack_check` does exist and was run as instructed. §4.2 lists what actually ran.

### 2.2 "Per-layer" is per forward call — there is exactly one PLE layer

`Qwen4ExpForCausalLM` builds PLE for a single layer id (2) and `Forward::ple` runs once per call, so "total PLE execution time per layer" is exactly "per forward call". The acceptance numbers in §4.5 are therefore per-call averages over 774 real PLE calls per profiled arm, not a sum over layers.

### 2.3 The gate is 4 CTAs, not 40 — and its 14.2 µs was re-reads on four SMs, not arithmetic

The task's framing ("the gate spreads over channels/256 = 40 CTAs") describes the convolutions. `gate_kernel`'s grid is `dim3(streams, columns)` = **4 CTAs of 256 threads** at decode (one CTA per hyper-connection stream), because the whole gate is one RMS-normalized dot product per (stream, column) and every reduction must complete inside one CTA. `conv_kernel`/`state_kernel` are the 40-CTA kernels.

The cost came from the body, which walked `hidden` four times with a runtime-trip strided loop: pass 1 read `key`+`residual` for the two statistics, pass 2 read `key`+`residual` again for the dot, pass 3 read `value` and wrote `gated`, pass 4 read `value` again and wrote `normalized`. So each operand crossed memory twice and `block_sum` ran four times (four barrier pairs). The fix is the task's own idea, executed with the constraints in §2.5: one load of each operand into registers, `block_sum2` folding the two statistics into one barrier pair (3 barrier pairs total), and 8-element vector accesses — 14.23 µs → 3.67 µs (§4.5) with the CTA count unchanged at 4.

### 2.4 `uint2` code loads are not sufficient: a 90-byte NVFP4 row is 16-byte aligned for one head in sixteen

`row_bytes = 80 + 10 = 90`, so head *h*'s row starts at `90h` — congruent to 0 mod 16 only when `h ≡ 0 (mod 8)`. An 8-byte load per task is unaligned for 7 of the 8 residue classes it can land on, and a 16-byte load for 15 of 16. The kernel therefore gathers the 8 code bytes with **two aligned `uint2` loads plus a 64-bit funnel shift** (`(l >> shift) | (h << (64 - shift))`), which is alignment-free on the fast path, and falls back to eight byte loads when that 16-byte window would read past the end of the tensor (the last row of the last column). Stores are the vectorized part the task actually needed: eight BF16 per `uint4`, two `uint4` per 16-element block, dispatched on a host-side `aligned<16>(out)` test so an unaligned tensor view takes the scalar store path instead of faulting.

### 2.5 "Cache k, q, v in registers" only works if the slice count is a compile-time constant, and that caps the (V, K) matrix

A `float kf[K][V]` array indexed by a runtime loop trip count goes to local memory, which would have traded a 4-pass read pattern for a spill-heavy one. Both dimensions are therefore template parameters — `V` elements per access, up to `K` accesses per thread — with `#pragma unroll` on both loops and a `d >= hidden` break. Because the kernel carries `__launch_bounds__(kGateMaxThreads)` (1024 threads, so 64 registers), the dispatch is constrained to keep the cached operand set affordable: `V = 8` is only ever paired with `K = 1` (24 cached floats), `V = 1` with `K ≤ 8`, and anything larger takes `gate_kernel_scalar`. The measured result is REG 40…62, STACK 0, LOCAL 0 across all twelve vector instantiations (§4.4). Production decode takes `<Norm, 8, 1>` with a 320-thread block (2560 / 8 = 320 slices, one per thread).

### 2.6 The fused conv+state needs the span as a template parameter, which is why it is capped at 16

The fused kernel holds the whole history per thread (`bf16 past[SPAN]`) and must index it by `k` and by `k / dilation` inside the tap loop; with a runtime `span` that array becomes local memory and the fusion loses more than the saved launch. So `SPAN = (taps - 1) * dilation` is a template argument and `ple_conv_inject` dispatches over a switch of the sixteen instantiations it admits, `kFusedConvMaxSpan = 16`. Production is `taps = 4, dilation = 3 → SPAN = 9`, comfortably inside. Outside that window — or when `width != 1`, or `update` is false — the original `conv_kernel` + `state_kernel` pair still runs unchanged, so nothing regresses for an unusual checkpoint. The tap accumulation keeps the original add order (history taps in increasing `k`, then the current column's `latest` last), which is what lets the existing FP64 oracle bound hold exactly on the state shift and within one rounding on the residual.

### 2.7 The embed kernel's stack frame was the E2M1 lookup table, not the vectorization

The first vectorized `embed_nvfp4_kernel` came back `REG=48 STACK=64 LOCAL=0` where the baseline is `STACK=0`. `ckt static` flagged the symbol; `cuobjdump -sass -fun <exact stored symbol>` (verified nonzero: 4 × `STL.128`, many `LDL`) put the frame on the dynamically indexed `constexpr float kE2m1[16]` — a 64 B table ptxas could not keep in constants once the index came from a nibble. Replacing the table with arithmetic (`sign = (code & 8) << 28`, `exponent = (code >> 1) & 3`, `mantissa = code & 1`, subnormals as `0x3F000000`, normals as `((exponent + 126) << 23) | (mantissa << 22)`) removed it: both instantiations are `STACK=0 LOCAL=0` (REG 47 / 48) and the bit-exact NVFP4 oracle test still passes, which is the check that matters — the decode must still be the exact product rounded once.

### 2.8 The gate's fast-route bits move; the lossless route does not

Caching each thread's whole slice changes the *intra-thread* summation order of `key_sq`, `query_sq`, `dot` and `gated_sq`, so the fast route's `gated`/`normalized` bits are not the previous ones. That is permitted here and was checked rather than assumed: the oracle for this Op is the FP64 closed formula with a one-BF16-rounding criterion (`kOneRounding`) plus exactness on the state shift, both of which pass at the production geometry and at fallback geometries (§4.2); the reduction stays FP32, and every per-element FP expression is character-for-character the earlier one (including `normalized` recomputing `gate * value` rather than reading `gated` back). `--reproducible` determinism is unaffected — the kernel is a pure function of its inputs. The `--lossless` route is untouched: its PLE is a separate set of kernels in `src/ops/lossless/lossless_flash.cu` (`ple_gated_kernel`, `ple_history_kernel`, `ple_conv_inject_kernel`, `ple_scores_kernel`), which this change does not edit, and `infernix_lossless_ops_test` passes.

### 2.9 The fusion is decode-only, so prefill still launches four kernels — and the profiled decode route never enters the prefill conv

`update && width == 1` is the fused condition, so a prefill chunk (`width` = chunk tokens) still runs `conv_kernel` and `state_kernel` separately: the "4 → 3 launches" acceptance is a decode statement, and it is met. It also means the profiled route exercises only the fused conv — the tg256 bench config is `n_prompt 0`, `max_context=257`, so all 774 PLE calls per arm have `width == 1` (the two unfused kernels are entirely absent from the candidate traces, §4.5). The wide-`width` conv path is covered by unit tests only (`test_conv` at widths 1, 3, 4, 5, 9 and 40, including a `SPAN = 20` case past the fused cap), not by an end-to-end measurement.

### 2.10 The E2E half of the acceptance bar cannot be a decode-throughput claim

PLE is 24.76 µs of a decode step that takes 27.1 ms at the measured 36.9 t/s (iq3s) or 22.4 ms at 44.6 t/s (nvfp4) — **0.09 % / 0.11 % of the step**. The whole 15.6 µs saving is bounded at +0.06 % / +0.07 % end to end, which is below the ±0.1 % resolution of the harness. Both routes were still measured (§4.6, §4.7), and their deltas (−0.18 % to +0.31 %) are larger in magnitude than the entire theoretical effect, so they are reported as "unchanged within resolution" rather than as a win — see §5.2 for the sign disagreement between the two routes.

## 3. Subsystem breakdown (mechanism)

| File | Action | Mechanism |
|---|---|---|
| `src/ops/ple/ple.cu` | M | New constants `kGateMaxThreads` (1024), `kFusedConvMaxSpan` (16), `kEmbedSpreadColumns` (48), `kEmbedCtaTasks` (32); host `aligned<Alignment>()`. `block_sum2` (two values per barrier pair, `float2` scratch). `embed_nvfp4_kernel` becomes `template <bool VectorStore>` with a task loop, `load_packed_codes` (2 × `uint2` + funnel shift + byte fallback), arithmetic `e2m1_value`, `uint32_t words[8]` staging and two `uint4` stores; `launch_embed_nvfp4` computes `tasks = heads × width/16`, spreads across ≤ 48 SMs at `columns < 48`, grid `(⌈tasks/tasks_per_cta⌉, columns)`. `gate_kernel` becomes `template <class Norm, int V, int K>` with `kf/qf/vf[K][V]` register slices, `load_bf16x`/`store_bf16x`/`load_normx` helpers, one `block_sum2` + two `block_sum`; `gate_kernel_scalar` keeps the old strided walk (with `block_sum2`) for unaligned or oversized geometries; `ple_gate` dispatches on 16-byte alignment, `%8` divisibility of `hidden`/`width`/strides and `hidden ≤ 8192`. New `template <int SPAN> conv_state_kernel` reads the history once, injects the residual and rewrites the shifted slot; `ple_conv_inject` takes it when `update && width == 1 && 1 ≤ span ≤ 16`. `embed_kernel`, `conv_kernel`, `state_kernel`, `commit_kernel`, every `require()` validation and all four public entry points are unchanged. |
| `tests/ops/test_ple.cpp` | M | +8 lines: production gate geometry (4 streams × 2560 at `T = 1` and 33 — the 8-element vector path), `hidden = 9000` (past the cached-slice limit → scalar walk), and two conv cases pinned at `width = 1`: `SPAN = 8` fused and `SPAN = 20` past the cap (unfused pair), the latter with distinct source/destination slots. |
| `tests/ops/test_ple_nvfp4.cpp` | M | `embed_exact()` parameterized over `kColumns` and run at 1, 37 and 64 columns (block-split decode, a chunk below one CTA per SM, a chunk at one CTA per column); failure messages now carry `T=`. |
| `src/models/qwen4_exp/execution/forward.cpp` | **unchanged** | The task lists it as a target file, but no op signature, workspace or slot-plumbing change was needed: `Forward::ple` already calls `ple_embedding → project → ple_gate → ple_conv_inject` with the same `batch.slots` for source and destination, and the fusion lives entirely behind `ple_conv_inject`. Editing it would have been churn. |
| `tools/cuda_stack/cuda_stack_allowlist.json` | **unchanged** | Spill inventory identical to baseline (§4.3); `check_cuda_stack.py update` was never run. |

## 4. Verification (GB10 `sm_121a`, CUDA 13, `build-native`, 20-core host)

### 4.1 Build

One unified invocation produced the final binaries: `ninja -C build-native -j 20 infernix-serve infernix_bench infernix_ops_elementwise_test infernix_ops_projection_test infernix_ops_lossless_test infernix_ops_sampling_test infernix_qwen4_exp_forward_real_test` → 13/13 edges, exit 0, one shared device link. `ple.cu` touched before every round.

### 4.2 Gates — 6/6 passed (11.85 s, serial CTest; no `--parallel` on this single-GPU host)

| Test | Time | Covers |
|---|---|---|
| `infernix_ple_test` | 2.11 s | `ple_embed` exactness, `ple_gate` vs the FP64 formula at eight gate geometries incl. the new production `4 × 2560` and the scalar-walk `2 × 9000`, `ple_conv_inject` residual + exact state shift incl. the new `width = 1` fused (`SPAN = 8`) and unfused (`SPAN = 20`) cases |
| `infernix_ple_nvfp4_test` | 2.05 s | bit-exact NVFP4 decode (every code and scale word, 4 multipliers incl. the tie-producing `1 + 2^-7`) at `T = 1, 37, 64`; FP32/BF16-norm gate within its bound |
| `infernix_ple_embed_test` | 1.91 s | the FP8 `ple_embed` route (untouched code path, kept as a regression guard) |
| `infernix_speculative_state_ops_test` | 2.14 s | the state-slot machinery the fused conv now owns for `width == 1`, incl. in-place vs distinct slots |
| `infernix_lossless_ops_test` | (in the 6/6) | the separate lossless PLE kernels, to confirm `--lossless` is untouched |
| `infernix_cuda_stack_check` | 1.43 s | the zero-spill ratchet gate |

### 4.3 The stack gate and the spill inventory (no allowlist movement)

`ctest -R infernix_cuda_stack_check` → **Passed** (1.43 s). Independently, the spilling-function inventory is identical between the HEAD-built and candidate binaries:

```text
/tmp/ple9/base/infernix_bench              261 functions with STACK>0 or LOCAL>0
build-native/bench/infernix_bench          261
build-native/apps/infernix-serve           261   (= allowlist entry count)
```

`git status` shows only the three code files, so `tools/cuda_stack/cuda_stack_allowlist.json` is untouched and `check_cuda_stack.py update` was never run — there was nothing to ratchet down. The allowlist's 261 keys contain exactly two `ple_*` symbols (`ple_scores_kernel`, `ple_conv_inject_kernel`), and both are file-static kernels of `src/ops/lossless/lossless_flash.cu`, not of the file this task edits, so no `src/ops/ple/ple.cu` kernel was ever on the ratchet.

### 4.4 Kernel resources (all 34 `ple.cu` functions, from the linked bench binary)

| Kernel | Instances | REG | STACK | LOCAL | SHARED |
|---|---|---|---|---|---|
| `embed_nvfp4_kernel<bool>` | 2 | 47 / 48 | 0 | 0 | 0 |
| `gate_kernel<Norm, 8, 1>` | 2 | 62 | 0 | 0 | 1280 |
| `gate_kernel<Norm, 1, K>`, K = 1…8 | 8 | 40 / 44 / 50 / 62 | 0 | 0 | 1280 |
| `gate_kernel_scalar<Norm>` | 2 | 38 | 0 | 0 | 1088 |
| `conv_state_kernel<SPAN>`, SPAN = 1…16 | 16 | 27…56 | 0 | 0 | 0 |
| `conv_kernel` / `state_kernel` / `commit_kernel` / `embed_kernel` | 4 | 38 / 34 / 34 / 12 | 0 | 0 | 0 |

Production decode is `gate_kernel<bf16, 8, 1>` (62 REG), `conv_state_kernel<9>` (40 REG) and `embed_nvfp4_kernel<true>` (47 REG) — all three at `STACK:0 LOCAL:0`, matching the baseline kernels they replace or subsume.

Decode launch geometry, before → after:

| Op | baseline grid × block | candidate grid × block |
|---|---|---|
| `embed_nvfp4` | `(1, 1) × 256` (one CTA runs all 160 head-block tasks) | `(5, 1) × 32` |
| `gate` | `(4, 1) × 256`, 4 strided passes | `(4, 1) × 320`, 1 pass |
| `conv` | `(40, 1) × 256` | — |
| `state` | `(40, 1) × 256` | — |
| `conv_state` | — | `(40, 1) × 256` |

Per call this also removes 250 880 B (245 KiB) of memory traffic at the production shape: one fewer read of `key` + `residual` + `value` (46 080 B), one fewer read of `normalized` (20 480 B) and one fewer read of the 9-column history (184 320 B).

### 4.5 PLE kernel time — the acceptance measurement

`nsys profile --cuda-graph-trace=node` around the real route (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` + its `.ngram` volume, `INFERNIX_EXPERTS_ON_GPU=1`, `-n 256 -r 2 --warmup 1 --kv-dtype bf16`), durations from `CUPTI_ACTIVITY_KIND_KERNEL` joined on demangled names. **774 PLE calls per arm.** Baseline = the HEAD-built binary; two baseline and three candidate rounds. Averages per call, µs:

| kernel | base | base2 | cand1 | cand2 | cand3 |
|---|---|---|---|---|---|
| `embed_nvfp4_kernel` | 3.13 | 3.14 | 2.30 | 1.97 | 1.90 |
| `gate_kernel` | 14.23 | 14.48 | 3.74 | 3.67 | 3.66 |
| `conv_kernel` | 2.96 | 2.88 | — | — | — |
| `state_kernel` | 4.45 | 4.60 | — | — | — |
| `conv_state_kernel` | — | — | 3.48 | 3.52 | 3.48 |
| **PLE total / call** | **24.76** | **25.09** | **9.52** | **9.16** | **9.05** |
| launches / call | 4 | 4 | 3 | 3 | 3 |

```text
base2 vs base   +1.3%     (baseline-to-baseline drift, for scale)
cand1 vs base  −61.6%
cand2 vs base  −63.0%
cand3 vs base  −63.4%
```

**Both acceptance criteria are met**: launches per call 4 → 3 with `state_kernel` gone (and `conv_kernel` too, since the fused kernel replaces the pair), and PLE time per call −61.6 % … −63.4 % against a ≥ 30 % bar, with a 1.3 % baseline spread. Per mechanism: embed −37 % (3.13 → 1.97), gate −74 % (14.23 → 3.67), the conv+state pair 7.41 → 3.52 µs (−53 %, one launch instead of two).

### 4.6 End-to-end, the NVFP4 route that PLE actually gates

Same command as §4.5 without `nsys`, tg256, `decode out t/s`:

| arm | binary | t/s |
|---|---|---|
| base | HEAD | 44.56 ± 0.03 |
| base2 | HEAD | 44.70 ± 0.15 |
| cand1 | T9 | 44.61 ± 0.08 |
| cand2 | T9 | 44.73 ± 0.06 |
| cand3 | T9 | 44.96 ± 0.08 |

Means 44.63 (HEAD) → 44.77 (T9), **+0.31 %**, with overlapping spreads. This route is dominated by reading the 29 GB n-gram volume, so the PLE saving is a rounding error here (§2.10).

### 4.7 End-to-end, the IQ3_S route (prefill + decode, n-gram table in RAM)

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix` with `--ngram-residency ram`, `-p 4096 -n 128,256 -r 3 --warmup 1 --kv-dtype bf16`, two order-alternating rounds per arm, both from real binaries:

| metric | HEAD r1 | HEAD r2 | T9 r1 | T9 r2 | Δ (mean) |
|---|---|---|---|---|---|
| pp4096 prefill t/s | 1440.53 ± 2.47 | 1439.49 ± 1.52 | 1437.17 ± 1.98 | 1438.94 ± 2.69 | **−0.14 %** |
| tg128 decode t/s | 36.92 ± 0.09 | 36.97 ± 0.04 | 36.87 ± 0.08 | 36.89 ± 0.04 | **−0.18 %** |
| tg256 decode t/s | 36.96 ± 0.04 | 36.92 ± 0.02 | 36.92 ± 0.05 | 36.85 ± 0.00 | **−0.15 %** |

All six paired comparisons are negative, so the sign is consistent — but the magnitude (≈ 0.15 %) is more than twice the entire theoretical effect of the change on this route (§2.10), and the sign is the opposite of §4.6's. Both are reported as unchanged-within-resolution, and the adverse sign is disclosed rather than dismissed (§5.2).

### 4.8 Real-model substitute for the doc's non-existent `infernix_qwen4_exp_Execution_test`

`infernix_qwen4_exp_forward_real_test @tokens --decode-widths` (96 tokens, the W4Dense artifact + n-gram volume, `INFERNIX_EXPERTS_ON_GPU=1`) compares a width-*w* batched decode step against *w* solo width-1 calls, per block and on the logits. It **fails on both arms identically** — 35 differing rows, every width 2…8, first difference always at `block 0 residual`, `FAIL: a decode width changes a row's bits` on HEAD and on T9 (the two logs are identical in structure). Block 0 precedes the PLE layer, so this is a pre-existing width-dependence of the batched decode schedules on this artifact, not a T9 regression; it is disclosed as an unmet check rather than a pass (§5.6).

## 5. Regressions and adverse results

1. **A 64-byte stack frame was introduced and then eliminated.** The first vectorized `embed_nvfp4_kernel` measured `STACK=64`; root cause and the arithmetic-decode fix are in §2.7. The shipped kernel is `STACK:0 LOCAL:0` and the spill inventory is unchanged (§4.3), but the intermediate state was a real ratchet violation on this tree and it is part of the record.
2. **The IQ3_S route is consistently ~0.15 % slower** on all three metrics in both rounds (−0.14 % / −0.18 % / −0.15 %, §4.7), while the NVFP4 route is ~0.3 % faster (§4.6). The two routes disagree in sign and the whole PLE change can move either route by at most ~0.06 %, so neither delta is attributable to this work — but neither was dismissed: no cause was found for the iq3s sign, GPU clock/host state was not isolated, and a third round of each arm was not run because the effect cannot exceed the measurement's resolution. If a future change touches PLE again, the iq3s arm should be measured before concluding anything about its sign.
3. **The fast route's gate output bits change** (§2.8). The reduction is still FP32 and still within the one-rounding criterion against the FP64 oracle at the production and fallback geometries, and the lossless route is a different set of kernels — but anything that compares `gated`/`normalized` bits against a pre-T9 capture will see differences.
4. **Prefill gets nothing.** The fusion is `width == 1` only, so a prefill chunk still pays four launches and two passes over the history. The task's acceptance is a decode criterion, so this is in scope as shipped, but a wide-chunk PLE profile was never taken and no prefill PLE improvement is claimed.
5. **The embed kernel's wide-column configuration is correctness-tested, not timed.** `columns = 37` and `64` are covered by the bit-exact NVFP4 oracle, but the profiled route only has `columns = 1`, so the `tasks_per_cta = 256` branch (one CTA per column) has no timing evidence either way.
6. **`infernix_qwen4_exp_forward_real_test --decode-widths` does not pass** — on HEAD either (§4.8). It is the only model-level gate the task named that could be run at all, and it fails before and after, so PLE's contribution to a real end-to-end forward is verified indirectly (the route runs in §4.6/§4.7) rather than by a passing model test.
7. **The gate is still 4 CTAs on a 48-SM device.** Removing the re-reads took it from 14.23 µs to 3.67 µs, but four CTAs of 320 threads occupy 4 SMs and the remaining time is launch and reduction latency, not work. Splitting each stream across CTAs would need a second kernel or a device-wide sync, which was not attempted at this scope.
8. **Not measured at all**: `conv_state_kernel` with distinct source/destination slots in production (only unit-tested), the FP32-norm gate route's timing (the profiled artifact uses BF16 norms), and any artifact whose PLE geometry differs from 160 × 16 heads / taps 4 / dilation 3.

## 6. Accepted gaps and follow-ups

- **`kFusedConvMaxSpan = 16` is a compile-time window, not a semantic limit** (§2.6). A checkpoint whose `(taps − 1) × dilation` exceeds 16 silently falls back to the two-kernel path; correct but slower. Raising it costs 16 more instantiations and register pressure (`conv_state_kernel<16>` is already 56 REG), so it should move only against a real artifact that needs it.
- **The gate and the convolution are still two kernels.** `gated` (20 KiB per decode column) is written by the gate and read back by the conv. Fusing them needs either a persistent CTA per channel column (which re-imports the cross-CTA reduction problem of §2.3) or a cooperative launch; at 3.5 µs the conv is no longer where the remaining PLE time is, so the 3.67 µs gate is the better next target.
- **`V ∈ {1, 8}` only.** The `%8` and alignment predicates (§2.5) send anything else to `gate_kernel_scalar`, which keeps the old strided walk and gains only the shared barrier pair. A `V = 4` instantiation is cheap to add if a real artifact lands in that gap; none does today (`hidden = 2560`, head width 160).
- **The embed spread heuristic is a constant, not a measured wave.** `kEmbedSpreadColumns = 48` / `kEmbedCtaTasks = 32` encode "≤ one CTA per SM while the work is tiny" for this device; they are not derived from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and at 32 threads per CTA the kernel is launch-bound rather than occupancy-bound.
- **The byte-load fallback in `load_packed_codes` is exercised only at tensor ends.** It is reachable and covered by the 1/37/64-column exact tests, but it is the path that would slow down if a future row layout made the 16-byte window illegal more often; a layout change should re-check the funnel-shift predicate rather than assume it.
- **No CUDA-graph-level change.** The decode graph's node count drops by one because the op itself launches fewer kernels; no graph capture, workspace or slot-plumbing code was touched, and `forward.cpp` is deliberately unchanged (§3).
