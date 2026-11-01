# GB10 (`sm_121a`) Performance Task Roadmap (2026-11-01)

This roadmap defines 10 concrete, self-contained, high-impact engineering tasks (`T1`–`T10`) targeting decode and prefill throughput on the NVIDIA GB10 (`sm_121a`, DGX Spark Blackwell: 48 SMs, 99 KiB SMEM/SM, 65,536 32-bit registers/SM, 24 MiB L2, 128 GiB unified LPDDR5X UMA at `238.7–246.0 GB/s` sustained read / `273.0 GB/s` theoretical, 20-core Grace ARMv9.2 CPU).

All tasks are grounded in live hardware ceiling measurements (`ckt env`), a complete 15-shape NVFP4 microbenchmark sweep across `T = 1..4096` (`infernix_linear_bench`), per-symbol `cuobjdump -res-usage` audits, and source inspections of `src/ops/` and `src/models/` on commit `f8ca918e6`.

## Live Hardware Ceilings on GB10 (`ckt env`)

Measured via `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt env`:

| Pipe / Subsystem | Instruction / Mode | Measured Peak (`ckt env`) | `infernix_linear_bench` Reference | 90% Target Threshold |
|---|---|---:|---:|---:|
| LPDDR5X Stream Read | Global load (`LDG.128`) | `238.7 GB/s` | `246.0 GB/s` (`273.0 GB/s` theoretical) | **`>= 221.4 GB/s` (`90.0%` `READ_%`)** |
| Shared Memory Bandwidth | `LDS.128` | `3,935.0 GB/s` | — | — |
| `nvfp4` Tensor Core (`OMMA`) | `OMMA.16864.E2M1` (F32 acc) | `492.111 TFLOP/s` | `440.0 TFLOP/s` (1-term) / `220.0 TFLOP/s` (2-term useful) | **`>= 396.0 TFLOP/s` (`90.0%` 1-term) / `>= 198.0 TFLOP/s` (2-term)** |
| `int8` Tensor Core (`IMMA`) | `IMMA.16832.S8` (S32 acc) | `245.788 TOP/s` | `220.0 TOP/s` | **`>= 198.0 TOP/s` (`90.0%`)** |
| `fp8_e4m3` / `mxfp8` (`QMMA`) | `QMMA.16832` (F32 acc) | `242.308 TFLOP/s` | `220.0 TFLOP/s` | **`>= 198.0 TFLOP/s` (`90.0%`)** |
| `bf16` / `fp16_f32acc` (`HMMA`) | `HMMA.16816` (F32 acc) | `122.513 TFLOP/s` | `97.0 TFLOP/s` | **`>= 87.3 TFLOP/s` (`90.0%`)** |

---

### Task T1 (Priority P0): Push NVFP4 `A4` & `Nvfp4A4Pair` Prefill GEMM and Activation Quantizer to 90%+ Ceiling

#### Affected Files & Symbols
- [nvfp4_a4_quantize.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh) (`nvfp4_a4_quantize_kernel`, `nvfp4_a4_row_scaled_residual_quantize_kernel`, `launch_nvfp4_a4_quantize`, `launch_nvfp4_a4_pair_quantize`)
- [nvfp4_a4_tma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh) (`nvfp4_a4_tma_kernel`, `Nvfp4A4TmaConfig`, `launch_nvfp4_a4_tma`, `launch_nvfp4_a4_residual_tma`)
- [nvfp4_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu) (`launch`, `launch_residual`)
- [nvfp4_launch.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_launch.cuh) (`launch_nvfp4_a4`, `launch_nvfp4_a4_pair`)
- [nvfp4_a4.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu) (`nvfp4_supports_a4`)
- [n128_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n128_k2560.cu), [n512_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n512_k2560.cu), [n2048_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2048_k2560.cu), [n2560_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k2560.cu)

#### Root Cause & Hardware Rationale
A full 15-shape benchmark sweep via `./build-native/bench/infernix_linear_bench --qtype nvfp4 --t 1,2,4,8,16,32,64,128,512,4096 --cold-l2-mb 32` (`--routes fast` and `--policy a4`) shows that **none** of the 15 NVFP4 shapes reach the 90% Tensor Core ceiling (`396.0 TFLOP/s` 1-term / `198.0 TFLOP/s` 2-term useful) at `T = 4096`:
- **1-term `--policy a4` at `T = 4096`**:
  - `[34816, 5120]`: `358.32 TF/s` (`81.44%` `TC_%`)
  - `[16384, 5120]`: `328.19 TF/s` (`74.59%` `TC_%`)
  - `[14336, 5120]`: `322.67 TF/s` (`73.33%` `TC_%`)
  - `[16384, 2560]`: `264.46 TF/s` (`60.10%` `TC_%`)
  - `[13952, 2560]`: `258.86 TF/s` (`58.83%` `TC_%`)
  - `[12800, 2560]`: `254.55 TF/s` (`57.85%` `TC_%`)
  - `[10240, 2560]`: `250.66 TF/s` (`56.97%` `TC_%`)
  - `[6144, 2560]`: `228.03 TF/s` (`51.82%` `TC_%`)
  - `[5120, 6144]`: `255.01 TF/s` (`57.96%` `TC_%`)
  - `[5120, 17408]`: `250.46 TF/s` (`56.92%` `TC_%`)
  - `[2560, 6144]`: `197.27 TF/s` (`44.83%` `TC_%`)
- **2-term `--routes fast` (`Nvfp4A4Pair`) at `T = 4096`**:
  - `[34816, 5120]`: `192.98 useful TF/s` (`385.96 hw TF/s` = `87.72%` of `440 TF/s`)
  - `[16384, 5120]`: `186.05 useful TF/s` (`84.57%`)
  - `[14336, 5120]`: `183.23 useful TF/s` (`83.29%`)
  - `[16384, 2560]`: `174.33 useful TF/s` (`79.24%`)
  - `[13952, 2560]`: `171.42 useful TF/s` (`77.92%`)
  - `[12800, 2560]`: `169.69 useful TF/s` (`77.13%`)
  - `[10240, 2560]`: `164.25 useful TF/s` (`74.66%`)
  - `[5120, 6144]`: `153.46 useful TF/s` (`69.75%`)
  - `[5120, 17408]`: `148.40 useful TF/s` (`67.45%`)
  - `[2560, 6144]`: `122.31 useful TF/s` (`55.60%`)
- **Four `K = 2560` shapes still disable `A4` (`uses_a4 = false`)**:
  - `[2560, 2560]` (`82.58 TF/s`, `43.67%` `READ_%` at `T = 4096`), `[2048, 2560]` (`83.78 TF/s`, `46.18%` `READ_%`), `[512, 2560]` (`64.01 TF/s`), and `[128, 2560]` (`17.77 TF/s`) fall back to `BF16_F32ACC` (`nvfp4_a16_mma_kernel`).

Four concrete bottlenecks in `src/ops/linear/nvfp4/` account for the gap to `90%+`:
1. **Uncoalesced `uint8_t` Scale Writes & Double Global Read in `nvfp4_a4_quantize.cuh`**:
   - In [nvfp4_a4_quantize.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L129-L226), `nvfp4_a4_row_scaled_residual_quantize_kernel` takes **`306.2 us` at `K = 6144`** (`29.3%` of total `[5120, 6144]` time!) and **`807.5 us` at `K = 17408`** (`28.1%` of total `[5120, 17408]` time!).
   - For `K = 17408` (`1,088` 16-element groups with `kThreads = 256` -> `kMaxGroups = 5 > 2`), `kCacheGroups` is `false` (line 152), forcing the kernel to **read `row_in` from global memory twice** (lines 171 and 210) while only operating on 1 token row per CTA (`gridDim.x = T`). Raising `kCacheGroups` or processing 4groups/thread with `kThreads = 272` (`1088 / 272 = 4`) keeps all groups in registers (`4 * 32 B = 128 B/thread`) so `row_in` is read from DRAM only once.
   - In both `nvfp4_a4_quantize_kernel` (lines 70, 118) and `nvfp4_a4_row_scaled_residual_quantize_kernel` (line 222), when `ScaleLayout` is `Tiled128` or `Tiled256`, `nvfp4_scale_offset<ScaleLayout>(token, group, rows, groups)` maps consecutive `group` indices of the same `token` to indices separated by `4` bytes (only `group & 3` is contiguous, while `(group >> 2)` steps by `64` bytes!). Because thread `t` processes `group = t`, consecutive threads write to addresses strided by `64 B`, issuing **uncoalesced 1-byte `STB` transactions** with 16x L2 sector write amplification! Having each thread process 4 consecutive K-groups (`group_base = 4 * t`, i.e., `64` elements = `32 B` of packed codes) allows every thread to pack its 4 scale bytes (`group & 3 = 0..3`) into a single **coalesced `uint32_t` store** (since `nvfp4_scale_offset` is contiguous across `group & 3`) and write `uint8` (`2 x uint4`) of packed codes!
2. **Hardcoded `Tiled128` (`BM = 128`) in 2-Term `launch_residual` (`nvfp4_a4_tma.cu:47`)**:
   - While 1-term `launch` selects `Tiled256` (`BM = 256, BN = 128`, 4 consumer warps in `2x2` layout) when `ctas_128 > 4 * 48`, 2-term `launch_residual` in [nvfp4_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu#L47-L61) and [nvfp4_launch.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_launch.cuh#L106) hardcodes `Nvfp4ScaleLayout::Tiled128` (`BM = 128`, 8 consumer warps). At `T = 4096`, `BM = 128` cuts weight reuse in half and launches 8 consumer warps + 1 producer warp (`288` threads/CTA). Adding a `BM = 256, BN = 128` (`Tiled256`) or `BM = 128, BN = 256` schedule for `launch_residual` at large `T` doubles M-tile reuse.
3. **Unpipelined `ldmatrix` Burst Before `OMMA.16864` in `nvfp4_a4_tma_kernel` (`nvfp4_a4_tma.cuh:378-443`)**:
   - Inside the `for (int local_k64 = 0; local_k64 < 2; ++local_k64)` loop, each consumer warp loads ALL `a_fragments[Term][m_tile]` (`16` registers) and ALL `b_fragments[n_tile]` (`8` registers) plus `a_scales` and `b_scales` before issuing the first `mma_nvfp4_e4m3`. Loading `b_fragments[0..1]` and `a_fragments[0]` first, issuing `mma_nvfp4_e4m3` for `m_tile = 0` while prefetching `m_tile = 1` / `n_tile = 2..3`, overlaps `LDS.128` / `LDSM` latency with the 64-cycle `OMMA.16864` instructions.
4. **Enable `A4` / `Nvfp4A4Pair` on `[2560, 2560]`, `[2048, 2560]`, `[512, 2560]`, and `[128, 2560]`**:
   - `[2560, 2560]` and `[2048, 2560]` are exact multiples of `BN = 128` (`20` and `16` N-blocks) and `BK = 128`, so enabling `uses_a4 = true` and `launch_nvfp4_a4` / `launch_nvfp4_a4_pair` at `T >= 65` eliminates their `43–46%` `READ_%` prefill cliff.

#### Required Implementation Steps
1. Refactor [nvfp4_a4_quantize.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh) so each thread quantizes 4 consecutive 16-element K-groups (`64` elements = `128 B` BF16 load, `32 B` `uint8` code store) and packs the 4 `e4m3` scale bytes (`group & 3 = 0..3`) into a single coalesced `uint32_t` store in `Tiled128`/`Tiled256` layout (and `uint32_t` in `RowMajor`). Ensure `nvfp4_a4_row_scaled_residual_quantize_kernel` caches all groups in registers up to `K = 17408` (`1,088` groups = `272` threads x 4 groups/thread or 2-pass tile cache in shared memory) so `x` is never read twice from DRAM.
2. In [nvfp4_a4_tma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh), interleave fragment loads (`ldmatrix_x4` / `ldmatrix_x2`) and scale loads with `mma_nvfp4_e4m3` across `m_tile` / `n_tile` sub-tiles, and add `Tiled256` (`BM = 256, BN = 128`) support for 2-term `launch_residual` when `T >= 512` in [nvfp4_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cu) and [nvfp4_launch.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_launch.cuh).
3. Enable `uses_a4 = true` and wire `launch_nvfp4_a4` / `launch_nvfp4_a4_pair` in [nvfp4_a4.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4.cu), [n2560_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2560_k2560.cu), [n2048_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n2048_k2560.cu), [n512_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n512_k2560.cu), and (via padded/masked N-tile or `Nvfp4A4MmaSchedule`) [n128_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n128_k2560.cu).
4. Touch all consuming `.cu` files under `src/ops/linear/nvfp4/` before building.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_nvfp4_a16_test infernix_linear_nvfp4_a4_test
./build-native/tests/infernix_linear_nvfp4_a16_test
./build-native/tests/infernix_linear_nvfp4_a4_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
./build-native/bench/infernix_linear_bench --qtype nvfp4 --t 128,512,4096 --cold-l2-mb 32 --routes fast
./build-native/bench/infernix_linear_bench --qtype nvfp4 --t 128,512,4096 --cold-l2-mb 32 --policy a4
```
- **Acceptance**:
  - `infernix_linear_nvfp4_a16_test`, `infernix_linear_nvfp4_a4_test`, and `infernix_cuda_stack_check` pass 100%.
  - Quantizer latency on `K = 6144` and `K = 17408` at `T = 4096` drops by `>= 2.5x` (from `306 us` / `807 us` to `< 110 us` / `< 280 us`), and large-shape `T = 4096` Tensor Core utilization reaches `90%+` of the sustained NVFP4 pipe ceiling (or `>= 90%` of the combined quantizer + GEMM roofline on `K = 2560..17408`).

---

### Task T2 (Priority P0): Eliminate Sub-90% `READ_%` Bandwidth Cliffs Across NVFP4 Decode & Verify (`T = 1..128`)

#### Affected Files & Symbols
- [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh) (`nvfp4_a16_sliced_k_mma_kernel`, `Nvfp4A16SlicedKMmaSchedule`)
- [nvfp4_template_launch.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_template_launch.cuh) (`Nvfp4SlicedInstance`, `Nvfp4MmaInstance`, `Nvfp4A4MmaInstance`)
- [nvfp4_a16_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh) (`nvfp4_a16_mma_kernel`)
- [src/ops/linear/nvfp4/shapes/](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/) (`n10240_k2560.cu`, `n12800_k2560.cu`, `n13952_k2560.cu`, `n16384_k2560.cu`, `n14336_k5120.cu`, `n16384_k5120.cu`, `n34816_k5120.cu`, `n5120_k6144.cu`, `n5120_k17408.cu`, `n2560_k6144.cu`, `n6144_k2560.cu`, `n2560_k2560.cu`, `n2048_k2560.cu`)

#### Root Cause & Hardware Rationale
The 15-shape NVFP4 benchmark sweep (`--routes fast`, `--cold-l2-mb 32`) reveals severe sub-90% `READ_%` bandwidth cliffs across `T = 2..128` (where `100% READ_% = 246.0 GB/s`, `90% = 221.4 GB/s`):
- **`T = 2..32` (`nvfp4_a16_sliced_k_mma_kernel` & `C2` SIMT)**:
  - `[34816, 5120]` at `T = 2`: dispatches `C2` SIMT (`491.9 us`, **`82.99%` `READ_%`**), which is `57 us` slower than `T = 4` (`434.9 us`, `94.02%` `READ_%`)!
  - `[10240, 2560]` at `T = 2..32`: **`85.36%–88.22%` `READ_%`** (`210.0–217.0 GB/s`).
  - `[12800, 2560]` at `T = 4..32`: **`82.05%–88.36%` `READ_%`**.
  - `[13952, 2560]` at `T = 32`: **`84.93%` `READ_%`**.
  - `[14336, 5120]` at `T = 32`: **`81.25%` `READ_%`**.
  - `[5120, 6144]` at `T = 2..32`: **`79.62%–87.36%` `READ_%`**.
  - `[2560, 6144]` at `T = 1..32`: **`66.89%–86.16%` `READ_%`**.
  - `[6144, 2560]` at `T = 1..32`: **`78.08%–86.16%` `READ_%`**.
- **`T = 64` and `T = 128` (`nvfp4_a16_mma_kernel` & `Nvfp4A4Pair` transition)**:
  - At `T = 64`, `--routes fast` drops to **`63.94%` `READ_%`** on `[2560, 6144]`, **`65.82%`** on `[5120, 17408]`, **`66.75%`** on `[5120, 6144]`, **`70.86%`** on `[10240, 2560]`, **`74.80%`** on `[13952, 2560]`, **`76.29%`** on `[14336, 5120]`, **`78.00%`** on `[12800, 2560]`, **`79.35%`** on `[16384, 5120]`, and **`81.73%`** on `[34816, 5120]`.
  - Meanwhile, on the exact same shapes at `T = 64` and `T = 128`, `--policy a4` (`Nvfp4A4MmaSchedule` / TMA) already achieves **`91.17%–98.42%` `READ_%`** (`[34816, 5120]` `T=64` is `91.17%` `READ_%` in `450.2 us` vs `502.3 us` on `fast`; `[16384, 2560]` `T=128` is `98.42%` `READ_%` in `118.4 us` vs `130.9 us` on `fast`; `[13952, 2560]` `T=128` is `96.69%` `READ_%` in `103.5 us` vs `119.1 us` on `fast`; `[10240, 2560]` `T=128` is `96.89%` `READ_%` in `75.6 us` vs `84.8 us` on `fast`)!

Three specific root causes explain these cliffs:
1. **Scalar `LDS.U8` Shared-Memory Loads & Dual `__syncthreads()` per K-Group in `nvfp4_a16_sliced_k_mma.cuh`**:
   - In [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh#L160-L214), each `group` loop iteration executes **two full-CTA `__syncthreads()` barriers** (lines 166 and 212) and inside the inner loop (`k_step = 0, 1`), `load_pair` (lines 177–181) issues **4 separate scalar `std::uint8_t` shared-memory loads (`LDS.U8`)** per `k_step` (`8` scalar `LDS.U8` per `row_tile`!).
   - Because `code_shared[stage][row]` is contiguous along the K-byte dimension (`code_col + k_step * 4` and `code_col + 8 + k_step * 4`, where `k_step = 0, 1` are adjacent 4-byte steps or `code_col` is `lane / 4`), a single `uint16_t` or `uint32_t` shared load (or loading `weight_codes` directly per warp via `uint4` `__ldcg` / register-buffered `cp.async` without cross-warp `code_shared` barriers, since each warp `warp` exclusively owns `warp_k0 = warp * kTileK`!) eliminates the `code_shared` read/write hazard and removes the second `__syncthreads()` inside the K-group loop!
2. **Missing `TokenTiles = 3, 4` (`T = 33..64`) in `Nvfp4SlicedInstance` (`nvfp4_template_launch.cuh:15-26`)**:
   - `Nvfp4SlicedInstance` only supports `MaxTokens = 8, 16, 24, 32` (`TokenTiles = 1, 2`), forcing `T = 33..64` onto the non-sliced `nvfp4_a16_mma_kernel` where warps step across rows instead of slicing K. Extending `Nvfp4SlicedInstance` to `MaxTokens = 48, 64` (`TokenTiles = 3, 4` with `BlockRows = 32` or `64`), or routing `T = 33..64` to a 2-term `Nvfp4A4MmaSchedule` (`64x128` / `64x64`), eliminates the `T = 64` cliff (`63.9–82.1%` -> `90%+` `READ_%`).
3. **Suboptimal `T = 128` Schedule Selection on `--routes fast`**:
   - At `T = 65..128` on `--routes fast` (`Nvfp4A4Pair`), `launch_nvfp4_a4_pair` dispatches `launch_residual` (`BM = 128, BN = 128`, 8 consumer warps) where `[10240, 2560]` (`80` N-blocks x `1` M-block = `80` CTAs = `1.67` waves on 48 SMs!) and `[12800, 2560]` (`100` CTAs = `2.08` waves) suffer severe wave quantization (`79.4–86.4%` `READ_%`). Adding a `BN = 64` (`BM = 128, BN = 64`, 4 consumer warps) TMA schedule or using `Nvfp4A4MmaSchedule<128, 64, ...>` when `T <= 128` eliminates wave-tail waste on `N = 6144, 10240, 12800, 13952`.

#### Required Implementation Steps
1. In [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh):
   - Replace the 4 scalar `std::uint8_t` loads in `load_pair` with packed `uint16_t` / `uint32_t` loads across `k_step = 0, 1`.
   - Use a 3-stage ring buffer (`kStages = 3`) so the producer stage (`next_stage = (group + 2) % 3`) never aliases the stage currently being read (`group % 3`), eliminating the second `__syncthreads()` at line 212 (`1` barrier per K-group instead of `2`).
2. In [nvfp4_template_launch.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_template_launch.cuh), extend `Nvfp4SlicedInstance` to support `MaxTokens = 48` and `64` (`TokenTiles = 3, 4` with `BlockRows = 32` or `64`), and add a `BN = 64` 2-term `Nvfp4A4Pair` option for `T = 65..128` so `N = 6144..16384` avoids wave-tail underutilization on 48 SMs.
3. Update `select_a16` in [src/ops/linear/nvfp4/shapes/](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/) (`n34816_k5120.cu` at `T = 2`, `n10240_k2560.cu`, `n12800_k2560.cu`, `n13952_k2560.cu`, `n16384_k2560.cu`, `n14336_k5120.cu`, `n16384_k5120.cu`, `n5120_k6144.cu`, `n5120_k17408.cu`, `n2560_k6144.cu`, `n6144_k2560.cu`, `n2560_k2560.cu`, `n2048_k2560.cu`) so every shape with `N >= 2560` achieves `>= 90%` `READ_%` across `T = 1..128`.
4. Touch all modified `.cu` files before compiling.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_nvfp4_a16_test infernix_linear_nvfp4_a4_test
./build-native/tests/infernix_linear_nvfp4_a16_test
./build-native/tests/infernix_linear_nvfp4_a4_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
./build-native/bench/infernix_linear_bench --qtype nvfp4 --t 1,2,4,8,16,32,64,128 --cold-l2-mb 32 --routes fast
```
- **Acceptance**:
  - All `infernix_linear_nvfp4_a16_test`, `infernix_linear_nvfp4_a4_test`, and `infernix_cuda_stack_check` tests pass.
  - `[34816, 5120]` at `T = 2` improves from `82.99%` (`491.9 us`) to `>= 92%` `READ_%` (`<= 440 us`).
  - Across all large-N (`N >= 6144`) NVFP4 shapes, `T = 1..128` achieves `>= 90%` `READ_%` (`>= 221.4 GB/s`), and medium shapes (`[5120, 6144]`, `[5120, 17408]`, `[2560, 6144]`) improve their `T = 32..64` `READ_%` from `63.9–79.6%` to `>= 88–90%+`.

---

### Task T3 (Priority P0): True Non-Blocking Register Prefetch in `nvfp4_a16_gemv.cuh` & `FastPath::GdnConvFused` Enablement on `W4Dense`

#### Affected Files & Symbols
- [nvfp4_a16_gemv.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh) (`nvfp4_a16_gemv_kernel`, `compute_nvfp4_rows`)
- [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh) (`nvfp4_a16_sliced_k_mma_kernel`)
- [q8_gdn_conv_columns.cuh](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh) (`GdnConvSnapshotPrefetchedRowEpilogue`, `GdnConvSnapshotEpilogue`)
- [nvfp4_gdn_snapshot_decode.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu) (`launch_nvfp4_n16384_k2560_conv_snapshot`)
- [fast_paths.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp) (`default_fast_paths`)

#### Root Cause & Hardware Rationale
In `docs/gb10/20261031-changelog.md` (`T6`), adding `GdnConvSnapshotPrefetchedRowEpilogue` to `nvfp4_a16_gemv_kernel` left `launch_nvfp4_n16384_k2560_conv_snapshot` (`T = 1`) at `105.28 us` vs `102.40 us` plain (`+2.88 us` overhead even though the extra `conv_states` + `conv_weight` payload is only `98.3 KiB` = `0.40 us` at `246 GB/s`), and end-to-end plain decode on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` regressed by `-0.19 tok/s` (`45.85` vs `46.04 tok/s`), preventing `FastPath::GdnConvFused` from being enabled by default on `W4Dense`.

Inspecting [nvfp4_a16_gemv.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh#L274-L282) reveals the exact bug:
```cpp
if constexpr (kPrefetchWords != 0) {
    for (int local = threadIdx.x; local < Schedule::kBlockRows; local += Schedule::kThreads) {
        output.prefetch_row(row_prefetch + local * kPrefetchWords,
                            nvfp4_a16_parent_row<Schedule>(block, local, rows, policy), 1);
    }
    __syncthreads();
}
```
1. **Synchronous `LDG -> STS -> __syncthreads()` Stalls All 8 Warps Before the Mainloop**:
   - `output.prefetch_row(...)` ([q8_gdn_conv_columns.cuh](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh#L49-L73)) issues global loads (`LDG.E.64` for `conv_weight` and `LDG.E.U16` for `conv_states`) into registers and **immediately writes those registers to shared memory (`STS`)**, followed by `__syncthreads()` at line 281!
   - On NVIDIA GPUs, `STS` consumes the loaded register values, so the hardware scoreboard **stalls the issuing threads on the full LPDDR5X DRAM latency (~450 cycles)** before `STS` can execute, and `__syncthreads()` forces all 8 warps in the CTA to wait for that DRAM round-trip BEFORE `compute_nvfp4_rows` issues a single weight load! Thus `kPrefetchWords` did not overlap the DRAM read with `compute_nvfp4_rows` at all—it serialized the DRAM latency in front of the mainloop and added `2,304 B` of shared memory (`row_prefetch[16 * 36]`) plus an extra barrier!
2. **Per-Warp Lane-0 Register Prefetch (or `cp.async`) Eliminates the Stall Completely**:
   - In `nvfp4_a16_gemv_kernel`, each warp (`warp = 0..7`) owns `Schedule::kRowsPerWarp` (`2`) rows (`local0 = warp * 2`), and at line 299 **only `lane == 0` of that warp** calls `linear_finish_row` for `r = 0, 1`!
   - If `lane == 0` of each warp issues the global loads for its `2` rows (`row = parent_row(local0 + r)`, `is_conv = row < conv_channels`) directly into **registers** (`uint2 w_reg[2]` and `uint16_t s0_reg[2], s1_reg[2], s2_reg[2]`) BEFORE `compute_nvfp4_rows`—without writing them to shared memory and without any `__syncthreads()`—those registers are never referenced during `compute_nvfp4_rows`! The scoreboard never waits on them until `compute_nvfp4_rows` completes 2,560 K-contractions (`~100 us` later), achieving **true 100% overlap** with zero shared memory and zero extra barriers!
   - Alternatively, if using `row_prefetch` in shared memory (for both `nvfp4_a16_gemv_kernel` and `nvfp4_a16_sliced_k_mma_kernel`), issuing `cp_async<8>` / `cp_async<4>` into `row_prefetch` and piggybacking on the existing `x_shared` / `stage_codes` `cp_commit()` barrier avoids any synchronous `LDG -> STS` stall before the mainloop.

#### Required Implementation Steps
1. In [nvfp4_a16_gemv.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh) and [q8_gdn_conv_columns.cuh](file:///opt/etc/Infernix/src/ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh), replace the synchronous `LDG -> STS -> __syncthreads()` prefetch before `compute_nvfp4_rows` with a non-blocking per-warp register prefetch (held in `lane == 0` registers across `compute_nvfp4_rows` and consumed directly in `finish_prefetched_row` at line 303) or asynchronous `cp.async` committed with `x_shared`, removing the `row_prefetch` shared-memory allocation and the extra `__syncthreads()` from `nvfp4_a16_gemv_kernel`.
2. In [nvfp4_a16_sliced_k_mma.cuh](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh#L81-L96), issue the `row_prefetch` loads after `stage_codes(0)` / `stage_activation(0)` `cp_commit()` (or via `cp.async` inside stage 0) so Warp 0 does not stall on `LDG -> STS` before issuing the stage-0 async copies.
3. Verify in [fast_paths.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp) whether enabling `FastPath::GdnConvFused` for `NVFP4` (`W4Dense`) now improves end-to-end decode throughput over baseline on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`; enable it in `default_fast_paths` if E2E decode is faster.

#### Verification & Acceptance Criteria
```bash
find src/ops/gdn_input_proj src/ops/linear/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_recurrent_test infernix_qwen4_exp_Execution_test
./build-native/tests/infernix_ops_recurrent_test
./build-native/tests/infernix_qwen4_exp_Execution_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `launch_nvfp4_n16384_k2560_conv_snapshot` (`T = 1`) latency drops from `105.28 us` to `<= 103.0 us` (matching the `102.40 us` plain projection + `0.40 us` DRAM floor), beating unfused `gdn_input_proj` + `causal_conv1d_update` (`115.63 us`) by `>= 12.5 us`/call (`>= 375 us`/token across 30 GDN layers).
  - `infernix_ops_recurrent_test`, `infernix_qwen4_exp_Execution_test`, and `infernix_cuda_stack_check` pass 100%.

---

### Task T4 (Priority P0): Remove RTX 5090 Routing Guards in Fused NVFP4 `linear_swiglu` & `linear_add` and Push Fused Projections to 90%+ Ceiling

#### Affected Files & Symbols
- [nvfp4_linear_swiglu_a4.cu](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu) (`launch`, `launch_residual`)
- [nvfp4_linear_swiglu_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu) (`launch_nvfp4_linear_swiglu_a4_tma`, `launch_nvfp4_linear_swiglu_a4_residual_tma`)
- [nvfp4_linear_add_a4.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu) (`launch_problem`, `launch_residual_problem`, `nvfp4_linear_add_a4_residual_launch`)
- [nvfp4_linear_add_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu) (`launch_nvfp4_linear_add_a4_tma`, `launch_nvfp4_linear_add_a4_residual_tma`)
- [nvfp4_linear_add_a16.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu) (`launch_n5120_k6144`, `launch_n5120_k17408`)
- [nvfp4_gdn_input_a4.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4.cu)
- [nvfp4_attn_input_a4.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4.cu)

#### Root Cause & Hardware Rationale
Auditing the fused NVFP4 projection operators (`linear_swiglu`, `linear_add`, `gdn_input_proj`, `attn_input_proj`) uncovered leftover **RTX 5090 (`sm_120a`) dispatch guards** that accidentally route GB10 (`sm_121a`) prefill (`T > 128`) onto slow non-TMA kernels:
1. **`nvfp4_linear_swiglu_a4.cu` Routes `T > 128` (`T = 129..4096`) to Non-TMA `ResidualM64N128` (`lines 76–98`)**:
   - Look at [nvfp4_linear_swiglu_a4.cu](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu#L76-L98):
     - Line 76 has an explicit RTX 5090 comment: `// 97-128 columns take the warp-specialized TMA route... (RTX 5090)`.
     - Line 79 only enables TMA for `Terms == kNvfp4A4ResidualTerms && x.ne[1] > 96 && x.ne[1] <= 128`!
     - Then in `launch_residual` (line 93): `else if (tokens <= 64 || tokens > 128) { launch_nvfp4_linear_swiglu_a4_residual_kernel<ResidualM64N128>(...); }`!
     - Consequently, on `Qwen3.8-27B-NVFP4` (`[34816, 5120]`) under `--routes fast` (`FusedA4Residual`), **every prefill with `T > 128` (`T = 512..4096`) bypasses `launch_nvfp4_linear_swiglu_a4_residual_tma` and runs non-TMA `ResidualM64N128` (`cp.async` MMA)**!
2. **`nvfp4_linear_add_a4.cu` Never Uses TMA on `FusedA4Residual` (`T >= 65`) (`lines 88–143`)**:
   - In [nvfp4_linear_add_a4.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L88-L143), `launch_residual_problem` still carries the comment `// K=17408, RTX 5090: M32N64 at T<=32, M64N64 at T=33..96, M64N128 above` and `nvfp4_linear_add_a4_residual_launch` (line 127) hardcodes `Nvfp4ScaleLayout::RowMajor` and never calls `launch_nvfp4_linear_add_a4_residual_tma` for `T >= 65` (`[5120, 6144]` and `[5120, 17408]`)!
   - Even in 1-term `nvfp4_linear_add_a4_launch` (line 113), `kUseTma` is only `x.ne[1] > 128`, whereas at `T = 65..128` `launch_problem` falls back to `M64N128` `cp.async`.
3. **Sub-90% `READ_%` at `T = 16..32` in `nvfp4_linear_add_a16.cu`**:
   - As recorded in `20261031-changelog.md` `T7`, `nvfp4_linear_add` on `[5120, 6144]` is at `77.6%` `READ_%` at `T = 16` (`84.0 us`) and `64.4%` at `T = 32` (`103.4 us`), and `[5120, 17408]` at `T = 32` is at `70.1%` (`264.2 us`) because `nvfp4_a16_sliced_k_mma.cuh` executes 2 barriers and 8 scalar `LDS.U8` loads per `row_tile` (fixed in Task T2) and `launch_n5120_k6144` switches to `Slices = 2` at `T = 17..32`.

#### Required Implementation Steps
1. In [nvfp4_linear_swiglu_a4.cu](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu), update `kUseTma` in `nvfp4_linear_swiglu_a4_launch` so both 1-term (`Terms == 1`) and 2-term (`Terms == kNvfp4A4ResidualTerms`) use TMA (`launch_nvfp4_linear_swiglu_a4_tma` / `launch_nvfp4_linear_swiglu_a4_residual_tma`) for all `x.ne[1] > 64` (or `> 96`), removing the RTX 5090 `x.ne[1] <= 128` upper cap that forced `T > 128` onto non-TMA `ResidualM64N128`.
2. In [nvfp4_linear_add_a4.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu) and [nvfp4_linear_add_a4_tma.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu), add and wire `launch_nvfp4_linear_add_a4_residual_tma` in `nvfp4_linear_add_a4_residual_launch` for `x.ne[1] > 64` (with `Nvfp4ScaleLayout::Tiled128` / `Tiled256`), replacing the RTX 5090 `RowMajor` `ResidualM64N128` path at `T = 65..4096`.
3. Propagate the Task T1 quantizer and TMA improvements to [nvfp4_gdn_input_a4.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4.cu) and [nvfp4_attn_input_a4.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4.cu), and retune [nvfp4_linear_add_a16.cu](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu) at `T = 16..64`.
4. Touch all modified `.cu` files under `src/ops/linear_swiglu/nvfp4/`, `src/ops/linear_add/nvfp4/`, `src/ops/gdn_input_proj/nvfp4/`, and `src/ops/attn_input_proj/nvfp4/` before compiling.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear_swiglu/nvfp4 src/ops/linear_add/nvfp4 src/ops/gdn_input_proj/nvfp4 src/ops/attn_input_proj/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_quant_mlp_test infernix_ops_attention_test infernix_ops_recurrent_test
./build-native/tests/infernix_ops_quant_mlp_test
./build-native/tests/infernix_ops_attention_test
./build-native/tests/infernix_ops_recurrent_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `infernix_ops_quant_mlp_test`, `infernix_ops_attention_test`, `infernix_ops_recurrent_test`, and `infernix_cuda_stack_check` pass 100%.
  - `nvfp4_linear_swiglu` (`[34816, 5120]`) and `nvfp4_linear_add` (`[5120, 6144]`, `[5120, 17408]`) on `FusedA4Residual` (`T = 512, 4096`) execute the TMA kernels and improve `T = 512..4096` throughput by `>= 25%`, reaching matching `90%+` roofline efficiency with standalone `ops::linear`.

---

### Task T5 (Priority P0): Push Offloaded Sparse MoE NVFP4 Narrow Decode (`83.7%` -> `90%+` `READ_%`) and Wide Grouped Prefill (`kSkipIdleColumns`) on GB10

#### Affected Files & Symbols
- [moe_layer.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu) (`gate_up_kernel`, `down_kernel`, `stage_async`, `quantize_column_codes`)
- [narrow_a16_mma.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh) (`gate_up_nmma16_kernel`, `down_nmma16_kernel`)
- [narrow_expert.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu) (`launch_narrow_experts`)
- [wide_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh) (`Schedule`, `ScheduleResidual`, `grouped_kernel`, `kSkipIdleColumns`)
- [wide_expert.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu) (`prepare`, `run_pass`)

#### Root Cause & Hardware Rationale
In `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, all 40 layers execute `ops::offloaded_sparse_moe` (`512` routed experts, `top_k = 10`, `H = 2560`, `I = 640`). Measuring `build-native/bench/infernix_offloaded_moe_decode_bench` (`--ColdL2`, `27.65 MiB` expert read per layer at `T = 1`) and `build-native/bench/infernix_offloaded_moe_prefill_bench` exposes two major bottlenecks:
1. **NVFP4 Narrow Decode Is at `83.7%` `READ_%` at `T = 1` (`134 us`, `206 GB/s`) and Drops to `74.8%` at `T = 8` (`150 us`, `184 GB/s`)**:
   - In [moe_layer.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1217-L1312) (`gate_up_kernel` for `NVFP4`, used on the `a8` / `dp4a` route) and [narrow_a16_mma.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh#L191-L350) (`gate_up_nmma16_kernel` / `down_nmma16_kernel` on the `a16` route):
     - In `gate_up_kernel` (`moe_layer.cu:1278-1281`), `stage_async` issues `cp.async.cg.shared.global` for the **entire `46,080 B` (`320` units x `144 B`) expert slice** into `sm.stage`, and then `stage_wait()` (`cp.async.wait_group 0`) forces all 4 warps to **wait until all `46,080 B` have arrived from LPDDR5X DRAM into shared memory BEFORE `unit_sums` computes a single `dp4a` instruction!**
     - Simultaneously, for `T = 1..8`, all `40 * 10 = 400` `gate_up_kernel` CTAs redundantly read the same `x` columns from global memory and execute `quantize_column_codes` (`160` `fdiv.rn.f32` quantization loops per CTA x `400` CTAs = `64,000` redundant `fdiv`s!).
     - Even simpler: in `gate_up_kernel` (`count == 1`), each warp `warp` processes disjoint units `b = warp, warp + kWarps, ...` (`unit_sums` at line 1255) and only touches each `144 B` weight unit **once**! Staging `46,080 B` of single-use weights through shared memory with a full `cp.async.wait_group 0` barrier wastes `46 KiB` of SMEM (capping occupancy at `2` CTAs/SM = `8` warps/SM = **`16.7%` occupancy**!) and serializes DRAM transfer with compute! Reading the `144 B` weight units directly from global memory into registers (`load_vec<uint4>` / `__ldcg`) or using a small double-buffered `cp.async` ring (like `narrow_a16_mma.cuh`) while keeping `acts` in shared memory drops SMEM from `56 KiB` to `< 10 KiB`, raising occupancy from `2` to `8–12` CTAs/SM (`32–48` warps/SM) and overlapping DRAM reads with compute!
     - In `narrow_a16_mma.cuh` (`gate_up_nmma16_kernel` and `down_nmma16_kernel`), `gate_up_nmma16_kernel` launches `kGateUpTiles = 640 / 32 = 20` CTAs per active expert (`200` CTAs at `T = 1`, `4` warps/CTA, `23.5 KiB` SMEM = `4` CTAs/SM = `16` warps/SM), and in `stage_nmma16_units` (`lines 82–95`), 9 of the 32 lanes per warp (`lane >= 9`) are **100% idle** during global weight loads (`if (lane < 9) cp_async16(...)`)! Having all 32 lanes load contiguous `16 B` chunks across `4` units (`32 * 16 B = 512 B = 32` rows of `16 B` codes) or packing scales contiguously utilizes `100%` of warp load slots instead of `28%` (`9/32`)!
2. **Wide Grouped TMA Prefill (`wide_expert.cuh:96, 125`) Explicitly Disables `kSkipIdleColumns` on `FlashNextExperts`**:
   - Look at [wide_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L96-L125):
     - In `Schedule` (line 96): `static constexpr bool kSkipIdleColumns = false;`
     - In `ScheduleResidual` (line 125): `static constexpr bool kSkipIdleColumns = !(G::kHidden == kHidden && G::kIntermediate == kIntermediate);`
     - `G::kHidden == kHidden && G::kIntermediate == kIntermediate` is true **specifically for `FlashNextExperts` (`H = 2560, I = 640`)**!
     - At `T = 512..4096` with `512` experts and `top_k = 10`, the average expert receives only `10..80` tokens (`1..2` tiles of `64` columns), so `35–75%` of the 16-column MMA sub-tiles (`col_tile = 0..3`) in each 64-column tile are beyond `valid_columns` (`col_tile * 16 >= valid_columns`)! Because `kSkipIdleColumns` is `false` for `FlashNextExperts`, `grouped_kernel` (`lines 515–560`) executes shared-memory `ldmatrix` loads and `OMMA.16864` instructions for all 4 `col_tile`s even when `valid_columns <= 16`!
     - Enabling `kSkipIdleColumns = true` in both `Schedule` and `ScheduleResidual` for `FlashNextExperts` skips `ldmatrix` and `OMMA` on `col_tile * 16 >= valid_columns` (whose outputs are discarded by `store_gate_up_wide` / `store_down_wide` at `c >= valid_columns`), cutting wasted Tensor Core and SMEM cycles in `wide_expert` prefill with **zero change to numerical outputs**!

#### Required Implementation Steps
1. In [moe_layer.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu), eliminate the `46,080 B` (`kGateUpSliceBytes`) full-slice `stage_wait()` stall in `gate_up_kernel` for `NVFP4`: for `count == 1` (`T = 1` per expert), stream `ExpertNvfp4Unit` directly from global memory into registers (`load_vec<uint4>`) or via a compact multi-stage `cp.async` ring so warps do not wait for all `46 KiB` before starting `unit_sums`, and shrink shared memory so multiple CTAs co-reside per SM.
2. In [narrow_a16_mma.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh), optimize `stage_nmma16_units` and stage count / CTA occupancy for `gate_up_nmma16_kernel` and `down_nmma16_kernel` across `T = 1..8`.
3. In [wide_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L96-L125), set `kSkipIdleColumns = true` in both `Schedule` and `ScheduleResidual` (including `FlashNextExperts`).
4. Touch `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`, `src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu`, and `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu` before building.

#### Verification & Acceptance Criteria
```bash
touch src/ops/offloaded_sparse_moe/cuda/moe_layer.cu src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu src/ops/offloaded_sparse_moe/cuda/wide_expert.cu
ninja -C build-native -j "$(nproc)" infernix-serve infernix_offloaded_moe_decode_bench infernix_offloaded_moe_prefill_bench infernix_ops_moe_test
./build-native/tests/infernix_ops_moe_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
./build-native/bench/infernix_offloaded_moe_decode_bench --ColdL2
./build-native/bench/infernix_offloaded_moe_prefill_bench
```
- **Acceptance**:
  - `infernix_ops_moe_test` and `infernix_cuda_stack_check` pass 100%.
  - In `infernix_offloaded_moe_decode_bench --ColdL2`, NVFP4 `T = 1` latency improves from `134 us` (`206 GB/s`, `83.7%` `READ_%`) to `<= 124 us` (`>= 222 GB/s`, **`>= 90.0%` `READ_%`**), and `T = 4..8` improves from `142–150 us` to `<= 132 us`.
  - In `infernix_offloaded_moe_prefill_bench`, `FlashNext` NVFP4 (`a4`) wide prefill throughput improves by `>= 15–25%` across `T = 512..4096`.

---

### Task T6 (Priority P0): Smart MTP & N-Gram Speculative Decoding Acceptance & Verification Gating (`Qwen4Exp`)

#### Affected Files & Symbols
- [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h) (`Program::Impl::decode`, `Program::Impl::read_mtp_drafts`, `Program::Impl::mtp_draft_round_steps`, `Program::Impl::observe_mtp_round`)

#### Root Cause & Hardware Rationale
An end-to-end algorithmic audit of `Qwen4Exp` speculative decoding in [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h) uncovered **four concrete flaws** that discard already-paid-for neural MTP draft tokens and block higher-acceptance n-gram proposals on GB10 (`sm_121a`):
1. **`read_mtp_drafts` Discards the Already-Computed Terminal Draft Token (`lines 5775 & 5814–5821`)**:
   - During `run_mtp_draft_steps` (line 5775), after draft step `j` finishes, `launch_graph_gate_product` tests whether the cumulative confidence `product(0..j) >= threshold` (`0.35` -> `0.25`) to decide whether to pay for **neural draft step `j + 1`** (which costs `~0.18` of a plain target round).
   - Then, in `read_mtp_drafts` (lines 5814–5821), the host re-tests `graph_gate_product(mtp_draft_logprobs_.host, stride, b, j) < threshold` using the **exact same `threshold`** and sets `mtp_confident_[b] = j`—**throwing away token `j` even though neural draft step `j` ALREADY executed and its cost is sunk!**
   - On GB10 (`sm_121a`), target verification of `W = 1..5` columns runs on `m16n8k16` Tensor Cores where the marginal cost of verifying one additional column `j` is `< 0.03` of a plain round (`[16384, 2560]` is `104.3 us` at `T = 1` vs `106.3 us` at `T = 4`, and narrow MoE at `T = 4` vs `T = 1` shares all 10 top-k expert loads for overlapping experts!).
   - Once draft step `j` has already run, keeping token `j` in the verification batch is net-positive for throughput (`E[tokens]/cost`) whenever its marginal acceptance `graph_gate_rate(logprobs[j])` exceeds the marginal verify column cost (`~0.08`), whereas cutting it whenever `product(0..j) < 0.25` discards valid draft tokens on every round where the gate stops continuation! Separating the **continuation threshold** (`threshold`: whether to launch step `j + 1`) from the **terminal retention threshold** (`kVerifyTerminalMinProduct = 0.08F` or `graph_gate_rate(logprobs[j]) >= 0.12F` for the last actually-executed step `j`) directly increases accepted tokens per verify round at zero extra draft cost!
2. **Strict `>` Length Filter & Missing Post-Gate N-Gram Fallback (`lines 2611–2635 & 2666–2671`)**:
   - At line 2611, when `mtp_ready` is true, `decode()` only considers an n-gram copy match if `match.tokens.size() > static_cast<std::size_t>(extents[b])` (strict `>`), and at lines 2623–2624 it compares `ngram_Tokens / ngram_cost > mtp_tokens / mtp_cost` assuming all `extents[b]` MTP steps will pass the confidence gate!
   - Even worse: when `run_mtp_draft_steps` runs and `mtp_confident_[0]` cuts `extents[0]` down to `0` (or `< match.tokens.size()`) at line 2667, **`decode()` throws away the n-gram match that was already found in `ngram_.propose()` at line 2610** and falls back to `width = 1` (zero speculation) or a shorter neural draft, even when a high-confidence n-gram match (`bucket >= 1`) was sitting ready in host memory!
   - Saving `match` across the `mtp_ready` branch and falling back to `match.tokens` (or splicing/replacing when `mtp_confident_[b]` cuts `extents[b]` below `match.tokens.size()` and the n-gram expected value exceeds the truncated MTP chain) turns cut MTP rounds into multi-token n-gram verification rounds at zero neural draft cost!
3. **Multi-Row (`batch >= 2`) Cumulative Gate Ignored (`line 2666`)**:
   - `read_mtp_drafts` computes `mtp_confident_[b]` for every row `b < batch`, yet line 2666 only applies `mtp_confident_` when `batch == 1`. Applying `extents[b] = std::min(extents[b], mtp_confident_[b])` across all `b < batch` and recomputing `width = 1 + max_b extents[b]` (calling `stage_verify_ngram` when `width` shrinks, matching the DFlash2 path at lines 2714–2717) prevents wasted wide verification when all batch rows cut early.
4. **Online `round_cost_` Fitter Blinded on Cut Rounds (`line 2670`)**:
   - Line 2670 sets `if (extents[0] != wanted[0]) { last_round_steps_ = -1; }`, discarding timing observations whenever `extents[0]` is shortened. Recording the actual number of executed neural draft steps (`executed_steps`) keeps `round_cost_` accurately fitted to GB10's real runtime.

#### Required Implementation Steps
1. In [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L5795-L5825) (`read_mtp_drafts`), distinguish the continuation gate (`product(0..j) >= threshold`, which gated whether step `j + 1` was launched in `run_mtp_draft_steps`) from terminal retention of the last executed step `j`: if step `j` was actually executed (`j == 0 || product(0..j-1) >= threshold`), retain token `j` for verification whenever `graph_gate_rate(logprobs[j]) >= kMtpTerminalMinRate` (`0.12F`) and `product(0..j) >= kMtpTerminalMinProduct` (`0.08F`).
2. In [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L2604-L2675) (`decode`), relax `match.tokens.size() > extents[b]` to `match.tokens.size() >= 1`, evaluate `ngram` vs `mtp` expected utility for all non-empty matches, and retain `match` so that if `mtp_confident_[b]` truncates the neural draft below `match.tokens.size()` (especially when `mtp_confident_[b] == 0`), `decode` can fall back to the n-gram proposal instead of decaying to `width = 1`.
3. Generalize line 2666 from `if (batch == 1)` to all `batch >= 1` (updating `width` and re-staging `stage_verify_ngram` if `width` shrinks, matching lines 2714–2717), and record the actual executed neural draft step count in `last_round_steps_` so `round_cost_` updates on every round.

#### Verification & Acceptance Criteria
```bash
ninja -C build-native -j "$(nproc)" infernix-serve infernix_qwen4_exp_Speculative_test infernix_qwen4_exp_Execution_test
./build-native/tests/infernix_qwen4_exp_Speculative_test
./build-native/tests/infernix_qwen4_exp_Execution_test
ctest --test-dir build-native -R "infernix_qwen4_exp_(Speculative|Execution)_test" --output-on-failure
```
- **Acceptance**:
  - `infernix_qwen4_exp_Speculative_test` and `infernix_qwen4_exp_Execution_test` pass 100% (add unit test coverage in `tests/models/qwen4_exp/` for terminal draft token retention and n-gram fallback on truncated MTP rounds).
  - End-to-end MTP speculative decoding acceptance (`accepted / round`) and tok/s on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` improve over baseline (`>= +3–8%` speculative decode tok/s across code/text prompts).

---

### Task T7 (Priority P0): Accelerate `Qwen4Exp` MTP Drafter Block (`Forward::mtp_block`) & `resident_moe_experts` on GB10

#### Affected Files & Symbols
- [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp) (`Forward::mtp_block`, `Forward::attention`)
- [resident_moe.cu](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu) (`gate_up_kernel`, `down_kernel`, `row_dot`, `resident_moe_experts`)
- [n13952_k2560.cu](file:///opt/etc/Infernix/src/ops/linear/nvfp4/shapes/n13952_k2560.cu) (`select_a16`)

#### Root Cause & Hardware Rationale
Every neural MTP draft step in `Qwen3.8-Flash-Next` executes `Forward::mtp_block` ([forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1347-L1495)). Inspecting `Forward::mtp_block` and [resident_moe.cu](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu) reveals three concrete performance bottlenecks that slow down every MTP draft step (and target attention decode):
1. **Shared Expert Serialized with Routed Experts in `Forward::mtp_block` (`forward.cpp:1405-1419`)**:
   - In the main model's `Forward::moe` (`lines 1250–1310`), the shared expert (`shared_swiglu` + `shared_down`) either fuses into `offloaded_sparse_moe` or runs concurrently on `experts_.shared_stream` while the 10 routed experts run on `device_.stream`.
   - In `Forward::mtp_block` (`lines 1405–1419`), when `p.moe.experts.resident()` is true (the resident drafter MoE path), `ops::resident_moe_experts` (`10` routed experts) and `shared_swiglu` + `project(shared_down)` run **serially on `device_.stream`**!
   - Launching `shared_swiglu` + `project(shared_down)` on `experts_.shared_stream` (recorded via `experts_.input_ready` after `xm` is ready and joined via `experts_.shared_done` before `ops::moe_combine` at line 1420) overlaps the drafter's shared expert with `ops::resident_moe_experts` across all `K = 1..4` MTP draft steps!
2. **Two-Pass Activation Reads & Unpipelined Scalar FP32 Dequant in `resident_moe.cu` (`lines 114–190`)**:
   - In [resident_moe.cu](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L133-L152) (`gate_up_kernel`), each warp (`kWarps = 4`, `kRowsPerBlock = 32`, `kRowsPerWarp = 8`) computes:
     ```cpp
     const float gate = row_dot<Format>(bank, base + r, xs, lane);
     const float up   = row_dot<Format>(bank, base + intermediate + r, xs, lane);
     ```
     Calling `row_dot` twice sequentially forces every warp to iterate over `xs[0..2559]` in shared memory **twice** per `r`, and for `Q8_0` (`Format = Q8Block`, `32` int8 weights + 1 FP16 scale per 34B block) and `Q4_0` / `IQ4_NL` (`32` 4-bit codes + 1 FP16 scale per 18B block), `chunk_dot` unpacks every byte to scalar `float` and runs 32 scalar `fmaf` instructions per chunk instead of quantizing `xs` once to `q8_1` (`int8` + `float` sum/scale) in shared memory and computing `gate` and `up` **simultaneously in a single pass** using `__dp4a` (`8` `__dp4a` instructions per 32-element chunk instead of `64` scalar FP32 `fmaf`s + `LDS` reads)!
   - Furthermore, adding `pdl::trigger_dependents()` / `pdl::enter()` between `gate_up_kernel` and `down_kernel` allows `down_kernel` CTAs to launch and prefetch weight pointers while `gate_up_kernel` finishes its epilogue.
3. **Unfused `Forward::attention` Decode Projection at `T == 1` (`forward.cpp:830`)**:
   - In [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L830-L837), when `!ggml && T == 1 && call.key_records.data == nullptr` (which is true on every `T = 1` decode step of `W4Dense` across all 10 full-attention layers and the MTP drafter block!), `Forward::attention` calls `project(x, p.projection, projected, work_, s, dense_routes())` (`ops::linear` `[13952, 2560]`) followed by a **separate `ops::split_attention_projection` kernel launch** that reads all `13,952` BF16 outputs back from global memory and scatters them into `q, k, v, gate`!
   - Why did `forward.cpp:830` bypass `ops::attn_input_proj` at `T == 1`? Because `ops::attn_input_proj` previously used `RowsPerWarp = 1` while `n13952_k2560.cu` used `RowsPerWarp = 2`! Now both `nvfp4_attn_input_small_t.cu` (`launch_nvfp4_n13952_k2560_split`) and `n13952_k2560.cu` use the exact same `Nvfp4A16GemvSchedule<8, 2>` (`88.9 us`), so calling `ops::attn_input_proj` at `T == 1` eliminates 11 kernel launches (`split_attention_projection`) and 11 global read+write round-trips per decode token!

#### Required Implementation Steps
1. In [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1405-L1420) (`Forward::mtp_block`), overlap the drafter shared expert (`shared_swiglu` + `shared_down`) on `experts_.shared_stream` concurrently with `ops::resident_moe_experts` on `device_.stream`, synchronizing via `experts_.input_ready` and `experts_.shared_done` before `ops::moe_combine`.
2. In [resident_moe.cu](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu), fuse the `gate` and `up` dot products in `gate_up_kernel` into a single pass over `xs` (`row_dot_pair<Format>`), quantize `xs` / `hs` to `int8` (`q8_1`) in shared memory so `Q8Block`, `Q4Block`, and `Iq4NlBlock` use integer `__dp4a` instead of 32 scalar `fmaf`s per chunk, and add PDL (`pdl::trigger_dependents()` / `pdl::enter()`) to `gate_up_kernel` and `down_kernel`.
3. In [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L830-L843) (`Forward::attention`), route `!ggml && T == 1 && call.key_records.data == nullptr` for `NVFP4` (and `Q8_G32_FP16`) directly through `ops::attn_input_proj` (`launch_nvfp4_n13952_k2560_split`), eliminating the separate `ops::split_attention_projection` launch on all 10 full-attention layers and the MTP drafter block.
4. Touch `src/ops/resident_moe/resident_moe.cu` before compiling.

#### Verification & Acceptance Criteria
```bash
touch src/ops/resident_moe/resident_moe.cu
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_moe_test infernix_ops_attention_test infernix_qwen4_exp_Execution_test infernix_qwen4_exp_Speculative_test
./build-native/tests/infernix_ops_moe_test
./build-native/tests/infernix_ops_attention_test
./build-native/tests/infernix_qwen4_exp_Execution_test
./build-native/tests/infernix_qwen4_exp_Speculative_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `infernix_ops_moe_test`, `infernix_ops_attention_test`, `infernix_qwen4_exp_Execution_test`, `infernix_qwen4_exp_Speculative_test`, and `infernix_cuda_stack_check` pass 100%.
  - `ops::resident_moe_experts` decode latency drops by `>= 25%`, and `Forward::mtp_block` step time decreases from overlapping the shared expert and eliminating `split_attention_projection`.

---

### Task T8 (Priority P1): Replace Remaining RTX 5090 (`170` SMs, `sm_120a`) Constants in FP8 TMA Split-K Schedules & Speculative Policies

#### Affected Files & Symbols
- [n34816_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n34816_k5120.cu), [n5120_k17408.cu](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n5120_k17408.cu), [n5120_k6144.cu](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n5120_k6144.cu), [n14336_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n14336_k5120.cu), [n16384_k5120.cu](file:///opt/etc/Infernix/src/ops/linear/fp8/shapes/n16384_k5120.cu)
- [fp8_linear_add_a8.cu](file:///opt/etc/Infernix/src/ops/linear_add/fp8/fp8_linear_add_a8.cu), [fp8_linear_add_n3072_k8192.cu](file:///opt/etc/Infernix/src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu)
- [fp8_linear_swiglu_a8.cu](file:///opt/etc/Infernix/src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu)
- [fp8_attn_input_a8.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu), [fp8_attn_input_n17408_k3072.cu](file:///opt/etc/Infernix/src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu)
- [fp8_gdn_input_a8.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu), [fp8_gdn_input_n20480_k3072.cu](file:///opt/etc/Infernix/src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu)
- [program_impl.cpp](file:///opt/etc/Infernix/src/models/qwen3_5/program/program_impl.cpp) (`kMtpConfidencePolicy`)
- [mtp_confidence_policy.h](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp_confidence_policy.h) (`MtpConfidencePolicy`)
- [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h) (`kDFlashPairDraftCost`, `dflash2_rows_`)

#### Root Cause & Hardware Rationale
A repository-wide search for remaining RTX 5090 (`170` SMs, `96 MiB` L2, PCIe-offloaded MoE) constants uncovered two sets of unported 5090 parameters:
1. **25 `Fp8A8TmaSplitKSchedule` Instantiations Hardcode `WaveCtas = 170` Across 12 FP8 `.cu` Files**:
   - Every `Fp8A8TmaSplitKSchedule<BlockRows, BlockTokens, Stages, SplitK, 170>` in `src/ops/linear/fp8/shapes/*.cu`, `src/ops/linear_add/fp8/*.cu`, `src/ops/linear_swiglu/fp8/*.cu`, `src/ops/attn_input_proj/fp8/*.cu`, and `src/ops/gdn_input_proj/fp8/*.cu` sets `WaveCtas = 170` (the RTX 5090 SM count).
   - In `fp8_a8_tma.cuh`, `Schedule::kSplitWaveCtas` determines:
     1. When split-K activates (`base_ctas * SplitK <= WaveCtas`), causing GB10 (`48` SMs!) to launch up to `170` split-K CTAs (`3.54` waves of partial-sum barriers and reduction traffic instead of fitting within `1` resident wave on 48 SMs!).
     2. The size of the FP32 `partials` workspace buffer (`170 * 128 * 128 * 4 B = 11.14 MiB`, nearly half of GB10's `24 MiB` L2 cache!).
   - On GB10 (`48` SMs), a `64`-row, 2-stage FP8 TMA CTA occupies `2` CTAs/SM (`96` CTAs/wave), and a `128`-row or 3-stage CTA occupies `1` CTA/SM (`48` CTAs/wave). Replacing `170` with `96` (for `BlockRows = 64`) or `48` (for `BlockRows = 128`) bounds split-K to a single resident GB10 wave and cuts split-K partial-buffer traffic and L2 footprint by `43.5%–71.8%`!
2. **RTX 5090 Speculative Cost Priors in `qwen3_5` and `qwen4_exp`**:
   - In [program_impl.cpp](file:///opt/etc/Infernix/src/models/qwen3_5/program/program_impl.cpp#L92) and [mtp_confidence_policy.h](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp_confidence_policy.h#L24), `prior_column_cost = 0.7` is documented as `"a column's misses, about 0.7 of a plain round on the 5090"`. On GB10 unified memory (`device.integrated`), all model weights reside in LPDDR5X UMA (`0` PCIe misses!), and a verification column on `Qwen3.5` costs `~0.15` of a plain round—so `prior_column_cost = 0.7` is `4.6x` too pessimistic during warmup before `column_cost` adapts!
   - In [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L6726), `kDFlashPairDraftCost = 0.47` (`"0.47, the 5090-fitted value the pair kernels' cost model still pays"`) and `dflash2_rows_ = 2` (line 6733) are still calibrated to the 5090's PCIe miss ratio rather than GB10's UMA bandwidth ratio (`~0.28–0.32`).

#### Required Implementation Steps
1. Replace `WaveCtas = 170` across all 12 FP8 files (`src/ops/linear/fp8/shapes/n34816_k5120.cu`, `n5120_k17408.cu`, `n5120_k6144.cu`, `n14336_k5120.cu`, `n16384_k5120.cu`, `src/ops/linear_add/fp8/fp8_linear_add_a8.cu`, `fp8_linear_add_n3072_k8192.cu`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu`, `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu`, `fp8_attn_input_n17408_k3072.cu`, `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`, `fp8_gdn_input_n20480_k3072.cu`) with GB10-native wave capacities (`96` for 2-CTA/SM `64`-row schedules, `48` for 1-CTA/SM `128`-row schedules), and update `fp8_linear_add_n3072_k8192.cu:183` (`base_ctas >= 2 * 48` instead of `2 * 170`).
2. Update `prior_column_cost` in [program_impl.cpp](file:///opt/etc/Infernix/src/models/qwen3_5/program/program_impl.cpp#L92) and [mtp_confidence_policy.h](file:///opt/etc/Infernix/src/models/qwen3_5/program/speculative/mtp_confidence_policy.h#L24) to `0.18` for GB10 UMA, and recalibrate `kDFlashPairDraftCost` in [program_impl.h](file:///opt/etc/Infernix/src/models/qwen4_exp/program/program_impl.h#L6726) to GB10's measured 2-step draft cost ratio (`~0.30`).
3. Touch all modified `.cu` files before compiling.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/fp8 src/ops/linear_add/fp8 src/ops/linear_swiglu/fp8 src/ops/attn_input_proj/fp8 src/ops/gdn_input_proj/fp8 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_bench infernix_linear_fp8_e4m3_a16_test infernix_ops_quant_mlp_test infernix_ops_attention_test infernix_ops_recurrent_test infernix_qwen3_5_Speculative_test infernix_qwen4_exp_Speculative_test
./build-native/tests/infernix_linear_fp8_e4m3_a16_test
./build-native/tests/infernix_ops_quant_mlp_test
./build-native/tests/infernix_ops_attention_test
./build-native/tests/infernix_ops_recurrent_test
./build-native/tests/infernix_qwen3_5_Speculative_test
./build-native/tests/infernix_qwen4_exp_Speculative_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
./build-native/bench/infernix_linear_bench --qtype fp8_e4m3 --t 64,128,256,512 --cold-l2-mb 32
```
- **Acceptance**:
  - Zero occurrences of `170` remain in `src/ops/*/fp8/`.
  - All FP8 linear, fused op, speculative, and stack-check tests pass 100%, and FP8 `T = 64..256` split-K latency and workspace footprint improve on GB10.

---

### Task T9 (Priority P1): Fuse & Vectorize `Qwen4Exp` Per-Layer Embedding (`ops::ple_gate` + `ops::ple_conv_inject` + `ops::ple_embed_nvfp4`) on GB10

#### Affected Files & Symbols
- [ple.cu](file:///opt/etc/Infernix/src/ops/ple/ple.cu) (`embed_nvfp4_kernel`, `gate_kernel`, `conv_kernel`, `state_kernel`, `ple_embed_nvfp4`, `ple_gate`, `ple_conv_inject`)
- [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp) (`Forward::ple`, `Forward::ple_embedding`)

#### Root Cause & Hardware Rationale
In `Qwen3.8-Flash-Next` (`qwen4_exp`), every layer with Per-Layer Embedding (`block.ple`) executes `Forward::ple` ([forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L598-L624)), which calls `ops::ple_embed_nvfp4`, `project(e, p.projection, kv)`, `ops::ple_gate`, and `ops::ple_conv_inject`. Inspecting [ple.cu](file:///opt/etc/Infernix/src/ops/ple/ple.cu) reveals three severe memory and launch inefficiencies:
1. **`gate_kernel` (`ple.cu:109-151`) Reads `key`, `residual`, and `value` Twice from Global Memory Using Scalar `bf16` Loads on Only `4` CTAs at `T = 1`**:
   - `gate_kernel` launches `dim3(streams, columns)` = `dim3(4, T)` (`4` CTAs at `T = 1` across `48` SMs = **`8.3%` of SMs active**!) with `kThreads = 256` and `hidden = 2560` (`width = 10240`).
   - Because `hidden / kThreads = 2560 / 256 = 10` elements per thread, each thread only processes **10 `bf16` values (`20` bytes)**! Yet `gate_kernel` reads `key[key_base + d]` and `residual[base + d]` from global memory in loop 1 (lines 121–126) and **reads them a second time from global memory** in loop 2 (lines 130–135), then reads `value[val_base + d]` in loop 3 (line 141) and **reads `value` a second time** in loop 4 (line 148), all using scalar 2-byte `__bfloat162float` loads!
   - Furthermore, in `block_sum` (lines 41–50), lines 46–48 execute a serial loop `for (int w = 0; w < blockDim.x / 32; ++w) total += scratch[w];` in **every single thread** followed by two `__syncthreads()` barriers, and `gate_kernel` calls `block_sum` **4 separate times** (`8` `__syncthreads()` barriers per CTA), when `key_sq` and `query_sq` can be reduced together in a single `float2` `block_sum2` (`2` barriers saved)!
   - Caching the `10` elements of `k`, `q`, and `v` per thread in registers (using `uint4` / `nv_bfloat162` vectorized loads for the first 8 elements + `nv_bfloat162` for the last 2) eliminates the redundant second global reads of `key`, `residual`, and `value`, cuts `block_sum` barriers from `8` to `6`, and triples effective bandwidth!
2. **`ple_conv_inject` (`ple.cu:311-341`) Launches Separate `conv_kernel` + `state_kernel` That Re-Read `states` and `normalized`**:
   - During decode (`T = 1`, `width = 1`, `update = true`, `source_slots == destination_slots`), `ple_conv_inject` launches `conv_kernel` (`40` CTAs of `256` threads for `channels = 10240`) which reads `states[slot, 0..span-1, c]` and `normalized[t, c]`, and then immediately launches a second `state_kernel` (`40` CTAs) which re-reads `states[slot, 1..span-1, c]` and `normalized[t, c]` to shift `states` in place!
   - When `update` is true and `width <= span` (including all `T = 1` decode calls where `width == 1`), each thread `(c, sequence)` in `conv_kernel` already loads `normalized` and `states` for channel `c`! Computing the `width` output columns and shifting `states` in-register within a single fused `conv_state_kernel` eliminates the `state_kernel` launch and saves a full read+write pass over `states` and `normalized` (`60 KiB` per layer at `T = 1`)!
3. **`embed_nvfp4_kernel` (`ple.cu:80-102`) Launches Only `1` CTA at `T = 1` with Scalar Byte Loads**:
   - At `T = 1` (`columns = 1`), `embed_nvfp4_kernel<<<columns, kThreads>>>` launches **1 single CTA on 1 of 48 SMs**, and inside the loop each thread loads 8 code bytes via 8 scalar `std::uint8_t` loads (`row[8 * b + i]`) and writes 16 `uint16_t` outputs via 16 scalar 16-bit stores!
   - Launching across `dim3(heads, columns)` (or tiling `heads * blocks` across multiple CTAs when `columns < 48`) and loading all 8 code bytes in a single `uint2` (`64-bit`) load while storing 4 `uint2` (`16 x bf16` = `32 B`) vectorized stores parallelizes `ple_embed_nvfp4` across SMs and eliminates byte/halfword memory transactions.

#### Required Implementation Steps
1. In [ple.cu](file:///opt/etc/Infernix/src/ops/ple/ple.cu):
   - Vectorize `embed_nvfp4_kernel` (`uint2` 8-byte code loads, `uint4` / `uint2` BF16 stores) and grid-tile across `(heads, columns)` when `columns < 48` so `T = 1..8` utilizes multiple SMs instead of 1 CTA per token.
   - In `gate_kernel`, fuse the `key_sq` and `query_sq` reductions into a single `block_sum2` pass, cache `k`, `q`, and `v` in registers using vectorized `nv_bfloat162` loads (when `hidden == 2560`, `10` elements/thread = `5` `nv_bfloat162` registers) so `key`, `residual`, and `value` are read from DRAM only once, and skip writing `gated` to global memory if `ple_gate` + `ple_conv_inject` can be fused or vectorized.
   - In `ple_conv_inject`, when `update` is true and `width == 1` (decode), fuse the state shift directly into `conv_kernel` (reading `states[0..span-1]` into registers, computing the convolution, and writing the shifted `states[0..span-1]` back in one kernel without launching `state_kernel`).
2. Touch `src/ops/ple/ple.cu` before compiling.

#### Verification & Acceptance Criteria
```bash
touch src/ops/ple/ple.cu
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_ple_test infernix_qwen4_exp_Execution_test
./build-native/tests/infernix_ops_ple_test
./build-native/tests/infernix_qwen4_exp_Execution_test
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
```
- **Acceptance**:
  - `infernix_ops_ple_test` (including exact bit-level NVFP4 embedding rounding `bf16_rn_product`), `infernix_qwen4_exp_Execution_test`, and `infernix_cuda_stack_check` pass 100%.
  - `ple_embed_nvfp4` + `ple_gate` + `ple_conv_inject` kernel count at `T = 1` drops from 4 launches to 3 (eliminating `state_kernel`), and total PLE execution time per layer drops by `>= 30%`.

---

### Task T10 (Priority P1): Eliminate Remaining Stack/Local Spills in `ggml_a8_tile.cuh`, `ggml_bf16_tile.cuh`, and `lossless_expert.cuh` (`cuda_stack_allowlist.json` Ratchet)

#### Affected Files & Symbols
- [ggml_a8_tile.cuh](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh)
- [ggml_bf16_tile.cuh](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh)
- [ggml_expert_a8.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert_a8.cuh) (`gate_up_tile_kernel_a8`, `down_tile_kernel_a8`)
- [lossless_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh) (`gate_up_lossless_kernel`, `down_lossless_kernel`)
- [cuda_stack_allowlist.json](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json)

#### Root Cause & Hardware Rationale
Inspecting [cuda_stack_allowlist.json](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) and `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` shows that the only remaining `STACK > 0` or `LOCAL > 0` kernels in `infernix-serve` are:
1. **`moe_layer.cu` (`ggml_expert_a8.cuh` / `ggml_a8_tile.cuh`)**:
   - `down_tile_kernel_a8<FlashNextExperts, true>`: `STACK = 128 B`, `LOCAL = 272 B`
   - `down_tile_kernel_a8<FlashNextExperts, false>`: `STACK = 128 B`, `LOCAL = 272 B`
   - `gate_up_tile_kernel_a8<FlashNextExperts>`: `STACK = 88 B`, `LOCAL = 208 B`
   - These spills occur because `ggml_a8_tile.cuh` declares large per-thread accumulator and fragment arrays across dynamic/unrolled format switch branches without bounding live register pressure or staging per-format decode temporaries in shared memory.
2. **`moe_layer.cu` (`ggml_bf16_tile.cuh`)**:
   - `down_bf16_tc_kernel<FlashNextExperts, true/false>` and `gate_up_bf16_tc_kernel<FlashNextExperts>`: `LOCAL = 48 B` (`STACK = 0 B`).
3. **`moe_layer.cu` (`lossless_expert.cuh`)**:
   - `down_lossless_kernel<FlashNextExperts>`: `LOCAL = 480 B`
   - `gate_up_lossless_kernel<FlashNextExperts>`: `LOCAL = 72 B`
4. **`vq_ggml_linear.cu` (`vq_ggml_tile.cuh`)**:
   - `vq_ggml_tile_mma_kernel<2, 1, 1>` / `<2, 2, 1>`: `LOCAL = 104 B`.

On GB10 (`sm_121a`, 24 MiB L2 shared across 48 SMs), any `STACK > 0` or `LOCAL > 0` spill writes and reads local memory (`STL`/`LDL`) through the L1/L2 cache hierarchy during inner GEMM loops, polluting L2 cache lines needed for expert weights. Eliminating the `STACK = 88–128 B` and `LOCAL = 208–272 B` spills in `gate_up_tile_kernel_a8` / `down_tile_kernel_a8` (and the `LOCAL = 48 B` in `*_bf16_tc_kernel` / `lossless_expert.cuh`) brings `ggml_expert_a8` to **zero stack spills (`STACK = 0`)** on `sm_121a` and ratchets [cuda_stack_allowlist.json](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) down toward empty.

#### Required Implementation Steps
1. Inspect per-symbol SASS and register usage of `gate_up_tile_kernel_a8` and `down_tile_kernel_a8` via:
   `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt static --sass build-native/apps/infernix-serve --filter tile_kernel_a8`
2. In [ggml_a8_tile.cuh](file:///opt/etc/Infernix/src/ops/ggml/ggml_a8_tile.cuh) and [ggml_expert_a8.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/ggml_expert_a8.cuh), add explicit `__launch_bounds__` and restructure the per-format decode/MMA loop (or isolate the heavy superblock format decode into non-inlined `__noinline__` helpers / shared memory scratch) so `STACK` drops from `88–128 B` to `0 B` and `LOCAL` drops to `0 B`.
3. In [ggml_bf16_tile.cuh](file:///opt/etc/Infernix/src/ops/ggml/ggml_bf16_tile.cuh) and [lossless_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/lossless_expert.cuh), eliminate the local-memory indexed array spills (`48 B` and `72–480 B`).
4. Touch `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` and `src/ops/ggml/*.cu`, rebuild `infernix-serve`, and run `/opt/etc/Infernix/.venv-convert/bin/python3 tools/cuda_stack/check_cuda_stack.py update build-native/apps/infernix-serve` to ratchet [cuda_stack_allowlist.json](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) down.

#### Verification & Acceptance Criteria
```bash
find src/ops/ggml src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_moe_test infernix_ops_ggml_test
./build-native/tests/infernix_ops_moe_test
./build-native/tests/infernix_ops_ggml_test
/opt/etc/Infernix/.venv-convert/bin/python3 tools/cuda_stack/check_cuda_stack.py update build-native/apps/infernix-serve
ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure
git diff tools/cuda_stack/cuda_stack_allowlist.json
```
- **Acceptance**:
  - `gate_up_tile_kernel_a8` and `down_tile_kernel_a8` achieve `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.
  - `git diff tools/cuda_stack/cuda_stack_allowlist.json` shows entries strictly removed or reduced (zero new spills added).
  - `infernix_ops_moe_test`, `infernix_ops_ggml_test`, and `infernix_cuda_stack_check` pass 100%.
