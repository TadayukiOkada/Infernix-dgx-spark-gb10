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
