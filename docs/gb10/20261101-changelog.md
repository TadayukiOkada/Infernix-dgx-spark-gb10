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
