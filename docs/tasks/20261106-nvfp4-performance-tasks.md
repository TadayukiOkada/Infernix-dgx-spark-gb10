# GB10 (`sm_121a`) NVFP4 Performance Optimization Tasks — 2026-11-06

> **Hardware Target**: NVIDIA GB10 (`sm_121a`, 48 SMs, 99 KiB SMEM/SM, 65,536 32-bit registers/SM, 24 MiB L2, 128 GiB unified LPDDR5X UMA at ~273 GB/s, 20-core Grace ARMv9.2 CPU, CUDA 13.0 `V13.0.88`).
>
> **Mandatory Workflow Rules (`AGENTS.md`)**:
> 1. **Dead CUDA Depfiles in `build-native`**: Modifying any `.cuh` or `.h` header does **not** automatically recompile `.cu.o` objects. Always touch the consuming `.cu` files (`find src/ops/<subdir> -name '*.cu' -exec touch {} +`) **before** invoking `ninja`, and verify `.cu.o` timestamps (`stat`).
> 2. **Single-Invocation Ninja Build**: Never run concurrent `ninja` processes. Build `infernix-serve`, tests, and benchmark binaries in a single `ninja -C build-native -j "$(nproc)" ...` command to share the ~55.8s `cmake_device_link.o` (`nvlink`) step.
> 3. **1:1 FP32 Accumulation & Numerical Qualification**: Preserve FP32 Tensor Core accumulation (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` and `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`) on production routes. Qualify all modified kernels against their independent FP32/FP64 oracles.
> 4. **One-Way CUDA Stack-Spill Ratchet Gate**: Run `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` after every build. Never run `check_cuda_stack.py update` to mask newly introduced or increased spills; only run `update` to ratchet [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json) **downward** when a task reduces or eliminates spills.
> 5. **Mandatory Changelog Entry**: Every completed task must append its section to [`docs/gb10/20261106-changelog.md`](file:///opt/etc/Infernix/docs/gb10/20261106-changelog.md) with sections `1. Accounting`, `2. Task-premise corrections / Architectural rationale`, `3. Subsystem breakdown`, `4. Verification`, `5. Regressions & adverse results`, and `6. Accepted gaps / follow-ups`.

---

### Task T1 (Priority P0): Eliminate Redundant 2-Pass `split_input` Weight Scan and Enable 2-Column Pair Loop in Narrow W4A4 `gate_up_kernel`

#### Affected Files & Symbols
- [`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu)
  - [`ActCodes`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L911-L916)
  - [`quantize_activation_slice_to_shared`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L918-L961)
  - [`quantize_activation_pair_to_shared`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L963-L1001)
  - [`unit_sums`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1003-L1037)
  - [`unit_sums_pair`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1039-L1079)
  - [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1251-L1328)

#### Root Cause & Hardware Rationale
Running `./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 10` on GB10 (`sm_121a`) reveals a severe latency cliff in the narrow W4A4 MoE route whenever `split_input` (`scales.input_up != scales.input_gate`, which holds on calibrated ModelOpt NVFP4 checkpoints where gate and up input scales differ) is active:

| Width `T` | `split=0` Warm / Cold (`us`) | `split=1` Warm / Cold (`us`) | `split=1` vs `split=0` Slowdown |
|---:|---:|---:|---:|
| `T=1` | `92.608` / `101.664` | `103.264` / `113.984` | `+11.5%` / `+12.1%` |
| `T=4` | `148.928` / `164.224` | `274.944` / `291.552` | **`+84.6%` (`1.85x`)** |
| `T=8` | `276.672` / `289.600` | `531.904` / `541.376` | **`+92.3%` (`1.92x`)** |

Inspection of [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1251-L1328) establishes two compounded root causes:
1. **Disabled 2-Column Pair Reuse (`T >= 2`)**: Line 1265 gates the 2-column pair loop behind `if (!split_input)`, forcing `split=1` at `T=2..16` (MTP, DFlash2, and batched decode) to stream every expert's `2048 × 5120` weight tile twice per column pair instead of amortizing weight reads across two columns that share an expert.
2. **Redundant 2-Pass Shared-Memory Weight Scan & `e2m1 -> f16x2` Conversion (`T >= 1`)**: Inside the single-column fallback (`lines 1296–1316`), `split=1` executes `for (int parity = 0; parity < (split_input ? 2 : 1); ++parity)`, re-quantizing activations and running `unit_sums` **twice** over the entire `46,080 B` staged weight tile while discarding half of the computed rows on each pass (`if (!split_input || gate == (parity == 0))`).
3. **Key Invariant**: Because gate rows are even (`(r & 1) == 0`) and up rows are odd (`(r & 1) == 1`), and [`unit_sums`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1003-L1037) assigns lane `l = 8*q + i` (`i = lane & 7`) exclusively to rows `i` and `i + 8`—both of which have the **exact same parity** (`(i & 1) == ((i + 8) & 1)`)—and reduces across `q` via `__shfl_xor_sync(..., 8)` and `__shfl_xor_sync(..., 16)` (which preserve `i = lane & 7`), **even lanes (`(lane & 1) == 0`) only ever compute gate rows and odd lanes (`(lane & 1) == 1`) only ever compute up rows!**

#### Required Implementation Steps
1. **Single-Pass Split-Scale Single-Column Path in [`moe_layer.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu)**:
   - Extend [`quantize_activation_slice_to_shared`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L918-L961) (or add `quantize_activation_split_to_shared`) so that when `input_gate != input_up`, threads quantize `x` under `input_gate` into the low nibbles (`0x0f0f0f0fU`, `act.scale[g]`) and under `input_up` into the high nibbles (`<< 4`, `act.scale1[g]`) of [`ActCodes`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L911-L916) in one pass over `x`.
   - In [`unit_sums`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1003-L1037), add a `bool split_input` parameter: when `split_input` is true, odd lanes (`(i & 1) != 0`) shift `act.packed[g * 2 + half] >> 4` and read `act.scale1[g]`, while even lanes read low nibbles and `act.scale[g]`.
   - Remove the `for (int parity = 0; parity < (split_input ? 2 : 1); ++parity)` loop from [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1298-L1316) so all 16 interleaved gate/up rows are computed and written in a **single pass** over `sm.stage`.
2. **Enable 2-Column Pair Loop for `split_input` in [`gate_up_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1265-L1294)**:
   - Mirror the two-half K-staging technique used by [`gate_up_pairs_a4r`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L2258-L2300) (`kPairBlocks = kBlocksPerRow / 2 = 80`): when `split_input` is true and two columns `(col0, col1)` are paired (`col1 != col0`), stage `80` blocks (`1,280` K elements = `23,040 B` weights) per half, quantize `(col0_gate, col0_up)` into `sm.act[0]` and `(col1_gate, col1_up)` into `sm.act[1]` (reusing the second half of the `46,080 B` staging buffer for `sm.act[1]` or splitting `ActCodes` into `80`-block halves), and accumulate `acc0` and `acc1` in registers via [`unit_sums`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1003-L1037) across both K-halves before applying [`write_swiglu_8`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu#L1081-L1097).
3. **Preserve Stack & Register Footprint**:
   - Confirm via `cuobjdump --dump-resource-usage` that `gate_up_kernel` remains at `STACK:0 LOCAL:0`.

#### Verification & Acceptance Criteria
```bash
find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_offloaded_moe_test infernix_offloaded_moe_decode_bench
ctest --test-dir build-native -R "infernix_offloaded_sparse_moe|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_offloaded_moe_decode_bench --repeat 15
```
- **Acceptance**:
  - All `infernix_offloaded_sparse_moe*` and `infernix_cuda_stack_check` CTests pass (`0` new stack spills).
  - `infernix_offloaded_moe_decode_bench` `split=1` warm latency improves by **$\ge 8\%$ at `T=1`** (`103.3 us -> <= 95.0 us`) and **$\ge 40\%$ at `T=4` and `T=8`** (`274.9 us -> <= 165.0 us` at `T=4`, `531.9 us -> <= 310.0 us` at `T=8`).

---

### Task T2 (Priority P0): Double-Buffer `ldmatrix_x4` A-Fragments and Parallelize `GateUpEpilogue` Warps in Wide NVFP4 MoE `grouped_kernel`

#### Affected Files & Symbols
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh)
  - [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L316-L561)
  - [`GateUpEpilogue`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L469-L530)

#### Root Cause & Hardware Rationale
Running `./build-native/bench/infernix_offloaded_moe_wide_bench` on GB10 measures `grouped_kernel` at `8,199.6 us` (`11.3 TFLOP/s`) for `T=1024` and `8,282.2 us` (`48.6 TFLOP/s`) for `T=4096`. In end-to-end `pp8192` prefill profiling (`47` MoE layers), `grouped_kernel` consumes `457 ms` (`9.7 ms/layer`), sitting at only `59%` of the GB10 LPDDR5X DRAM bandwidth floor (`5.75 ms/layer`).

Inspecting [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L316-L561) and its resource usage (`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`: `REG:174 STACK:0 SHARED:43008` for `GateUpEpilogue`, `REG:168 STACK:0 SHARED:50176` for `DownEpilogue`) identifies two stalls inside the consumer warps:
1. **Unpipelined `ldmatrix_x4` Shared-Memory Reads in the K-Tile Loop (`lines 417–447`)**:
   - Inside `for (int local_k64 = 0; local_k64 < 2; ++local_k64)`, each consumer warp issues `ldmatrix_x4(a_fragments[mma_m], ...)` immediately before `mma_nvfp4_e4m3(...)`, exposing the ~25-cycle `LDS.128` shared-memory latency on every `local_k64` step.
   - By contrast, [`nvfp4_a4_tma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_tma.cuh#L384-L435) double-buffers `a_fragments[2][MmaM][4]`, preloading `local_k64 = 0` before the loop and prefetching `local_k64 = 1` before issuing `local_k64 = 0`'s MMAs. Because `grouped_kernel` runs at `1 CTA/SM` (`SHARED: 43–50 KiB`, `384` threads = `12` warps/CTA) and uses `168–174` registers, each thread has headroom up to `208` registers (`12 warps * 32 * 208 = 79,872` vs `65,536` registers/SM gives `168` max at 12 warps—wait: `65536 / 384 = 170.6`, so `168` is the exact 8-register granularity ceiling for `384` threads!).
   - Because `384` threads/CTA caps registers at `168` (`384 * 168 = 64,512 <= 65,536`), note that `grouped_kernel<GateUpEpilogue>` is currently at **`REG:174`** (`384 * 176 = 67,584 > 65,536`), which only launches because `--maxrregcount` or runtime adjustment caps it or because `__launch_bounds__(384, 1)` forces 168! Wait: `grouped_kernel` has **NO `__launch_bounds__`** at line 317 (`__global__ void grouped_kernel(...)`), so at `REG:174` (`176` allocated per thread), `384 * 176 = 67,584 > 65,536`! Why didn't it fail to launch? Let's check: `kThreads = (kConsumerWarps + 1) * 32 = (8 + 1) * 32 = 288` threads (`9` warps/CTA)! At `288` threads/CTA, `65536 / 288 = 227.5` (`224` registers/thread max at 1 CTA/SM, or `112` at 2 CTAs/SM)! Thus at `174` registers/thread, `grouped_kernel` has **`50` free registers per thread** (`224 - 174 = 50`) at `1 CTA/SM` without any occupancy drop or spill!
2. **Serial Single-Warp Execution in `GateUpEpilogue` When `n <= 64` (`lines 469–530`)**:
   - At `T = 64..1024`, most routed experts receive `n <= 64` tokens (`live_mma == 0`, only `mma_m == 0` active).
   - In [`GateUpEpilogue`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L469-L530), line 480 gates the entire SwiGLU + NVFP4 re-quantization epilogue behind `if (mma_warp_m <= live_mma)`. When `live_mma == 0`, warps with `mma_warp_m == 1` (`4` of the `8` consumer warps, i.e., `128` threads) sit completely idle while each `mma_warp_m == 0` warp serially loops `4` times (`for (int col_block = 0; col_block < kColsPerMmaWarp; col_block += 16)`) over 64 columns!

#### Required Implementation Steps
1. **Double-Buffer `a_fragments` Across `local_k64 = 0..1` in [`wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L417-L447)**:
   - Declare `std::uint32_t a_fragments[2][kMmaM][4];` and preload `local_k64 = 0` (`for (int mma_m = 0; mma_m <= live_mma; ++mma_m)`) immediately after `mbarrier_wait_parity` (`line 415`).
   - Inside `for (int local_k64 = 0; local_k64 < 2; ++local_k64)`, when `local_k64 == 0`, prefetch `a_fragments[1][mma_m]` before issuing `mma_nvfp4_e4m3` on `a_fragments[0][mma_m]`, overlapping `LDS.128` with the `OMMA.16864` Tensor Core pipeline.
   - Add `__launch_bounds__(kThreads, 1)` to [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L317).
2. **Parallelize `GateUpEpilogue` Across Both `mma_warp_m` Warp Groups When `live_mma == 0` (`lines 469–530`)**:
   - After writing `sm.epilogue` (`line 478`) and `__syncthreads()` (`line 479`), when `live_mma == 0`, split the `kColsPerMmaWarp = 64` columns (`4` blocks of 16 columns) across `mma_warp_m = 0` (`col_block = 0, 16`) and `mma_warp_m = 1` (`col_block = 32, 48`) so all 8 consumer warps participate in the cross-split reduction, SwiGLU, warp max-abs reduction, and NVFP4 `h_codes`/`h_scales` output stores, cutting epilogue latency in half for `n <= 64` experts.

#### Verification & Acceptance Criteria
```bash
find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_offloaded_moe_test infernix_offloaded_moe_wide_bench
ctest --test-dir build-native -R "infernix_offloaded_sparse_moe|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_offloaded_moe_wide_bench
```
- **Acceptance**:
  - All `infernix_offloaded_sparse_moe*` and `infernix_cuda_stack_check` CTests pass (`STACK:0 LOCAL:0`).
  - `infernix_offloaded_moe_wide_bench` latency improves by **$\ge 5\%$ at `T=1024` and `T=4096`**.

---

### Task T3 (Priority P0): Eliminate NVFP4 `linear_add` A4/A4R Schedule Cliffs (`T=33..64` and `T=129..287`) and Adopt 48-SM Narrow TMA Tile Selection in `linear_add` and `linear_swiglu`

#### Affected Files & Symbols
- [`src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu)
  - [`uses_tma`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L34)
  - [`launch_problem`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L71-L92)
  - [`launch_residual_problem`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L95-L108)
- [`src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu)
  - [`launch_nvfp4_linear_add_a4_tma`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu#L79-L89)
- [`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu)
  - [`launch_problem`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu#L76-L92)
- [`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu)
  - [`launch_nvfp4_linear_swiglu_a4_tma`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu#L139-L150)

#### Root Cause & Hardware Rationale
Benchmarking `infernix_nvfp4_linear_add_bench` and `infernix_nvfp4_linear_swiglu_bench` on GB10 exposes three major schedule cliffs across the dense attention/GDN output (`[5120, 6144]`), MLP down (`[5120, 17408]`), and MLP gate-up (`[34816, 5120]`) projections:
1. **`1.86x` Latency Cliff at `T=33..64` in `nvfp4_linear_add` (`--policy a4r`, Default Production Route)**:
   - Running `./build-native/bench/infernix_nvfp4_linear_add_bench --policy a4r --tokens 16,32,33,48,64,65,128` at `N=5120, K=17408` shows latency jumping from `226.1 us` at `T=32` to **`442.4 us` (`121.3 GB/s`) at `T=64`**, then dropping back to `282.6 us` at `T=65`!
   - Running `--variants` at `N=5120, K=17408, T=33,48,64` proves the cause: [`launch_residual_problem`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L95-L108) dispatches `ResidualM32N64<Terms>` (`kBlockTokens = 32`) for the entire `17 <= T <= 64` range. For `33 <= T <= 64`, `grid.y = 2` launches two token tiles that stream the `49.0 MiB` weight matrix (exceeding GB10's `24 MiB` L2) **twice from LPDDR5X DRAM**, costing `424.7–441.2 us`.
   - Meanwhile, `a4r_m64n128` (`ResidualM64N128<Terms>`, which [`nvfp4_linear_swiglu_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu#L105-L107) already uses for `33 <= T <= 64`!) runs in a single token tile in **`226.3 us` (`T=33`), `228.2 us` (`T=48`), and `236.9 us` (`T=64`) — a `1.86x` (`-46.4%`) speedup**!
2. **`T=256` Cliff in `nvfp4_linear_add` (`uses_tma` Set to `288` Instead of `256`)**:
   - In [`nvfp4_linear_add_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L34), `uses_tma(tokens)` requires `tokens >= 288`, leaving `T=256..287` on `M128N128Resident`, where `T=256` takes `528.7 us` (`121.6 GB/s`) at `K=17408` and `166.2 us` (`157.0 GB/s`) at `K=6144` vs `384.0 us` and `139.3 us` at `T=288`.
3. **Missing 48-SM Narrow TMA Token-Tile Selection (`128`-Token vs `256`-Token TMA Tile) in `linear_add` and `linear_swiglu`**:
   - Standalone `nvfp4_linear` ([`nvfp4_launch.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_launch.cuh#L113-L138)) uses `nvfp4_a4_prefer_narrow_token_tile(rows, k, tokens)` to dispatch a 128-token TMA tile (`Nvfp4A4TmaMmaSchedule<128, 128, 128, 2>`, `Nvfp4ScaleLayout::Tiled128`) whenever `ceil(rows/128) * ceil(T/256)` suffers tail-wave under-utilization on GB10's 48 SMs and extra weight traffic fits within `52 MiB` (yielding `+5%..+16%` at `T=257..384`, `640..896`, `1152..1536`).
   - Neither [`nvfp4_linear_add_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu) nor [`nvfp4_linear_swiglu_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu) wires this narrow 128-token TMA tile; both hardcode `Nvfp4A4TmaMmaSchedule<256, 128, 128, 2>` (`Tiled256`) for all `T >= 256`/`288`, leaving `N=5120, K=6144` (`40` row tiles) and `N=5120, K=17408` (`40` row tiles) with a `40/48` single-wave or `80/48` (`1.67` waves) tail wave at `T=256..512`!

#### Required Implementation Steps
1. **Fix `T=33..64` Residual Schedule in [`nvfp4_linear_add_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu#L95-L108)**:
   - Update `launch_residual_problem` to split `tokens <= 32` (`ResidualM32N64<Terms>`) from `33 <= tokens <= 64` (`ResidualM64N128<Terms>`), matching `nvfp4_linear_swiglu_a4.cu`.
2. **Add Narrow 128-Token TMA Tile (`M128N128K128`, `Tiled128`) and 48-SM Wave-Fill Selector to `linear_add` and `linear_swiglu`**:
   - In [`nvfp4_linear_add_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu) and [`nvfp4_linear_swiglu_a4_tma.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu), instantiate narrow TMA launchers using `Nvfp4A4TmaMmaSchedule<128, 128, 128, 2>` (`Nvfp4ScaleLayout::Tiled128`).
   - In [`nvfp4_linear_add_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_a4.cu) and [`nvfp4_linear_swiglu_a4.cu`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu), lower `uses_tma` to `tokens >= 129` (where narrow 128-token TMA or `M128N128Resident` is selected by `nvfp4_a4_prefer_narrow_token_tile(N, K, tokens)`) and ensure the quantizer writes `Tiled128` when the narrow TMA tile is chosen and `Tiled256` when the 256-token TMA tile is chosen.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear_add/nvfp4 src/ops/linear_swiglu/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_add_nvfp4_test infernix_linear_swiglu_nvfp4_test infernix_nvfp4_linear_add_bench infernix_nvfp4_linear_swiglu_bench
ctest --test-dir build-native -R "infernix_linear_add_nvfp4_test|infernix_linear_swiglu_nvfp4_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_nvfp4_linear_add_bench --policy a4r --tokens 16,32,33,48,64,65,128,256,512 --warmup 3 --repeat 10
./build-native/bench/infernix_nvfp4_linear_add_bench --policy a4 --tokens 64,128,256,288,384,512,640,768 --warmup 3 --repeat 10
./build-native/bench/infernix_nvfp4_linear_swiglu_bench --policy a4 --tokens 128,256,384,512,640,768 --warmup 3 --repeat 10
```
- **Acceptance**:
  - `infernix_linear_add_nvfp4_test`, `infernix_linear_swiglu_nvfp4_test`, and `infernix_cuda_stack_check` pass (`0` new stack spills).
  - `infernix_nvfp4_linear_add_bench --policy a4r` at `N=5120, K=17408, T=33..64` drops from `424–442 us` to **`<= 245 us` ($\ge 44\%$ latency reduction / `1.8x` speedup)**.
  - `infernix_nvfp4_linear_add_bench --policy a4` at `T=256` (`K=6144` and `K=17408`) improves by **$\ge 20\%$** (`528.7 us -> <= 390 us` at `K=17408`, `166.2 us -> <= 135 us` at `K=6144`).

---

### Task T4 (Priority P0): Deduplicate Redundant `e4m3 -> f16x2` Scale Conversions Across `nvfp4_codec.cuh` and All NVFP4 A16 MMA Kernels

#### Affected Files & Symbols
- [`src/ops/linear/nvfp4/nvfp4_codec.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh)
  - [`nvfp4_scaled_two_pairs_bf16`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L91-L108)
  - [`nvfp4_scaled_oct_bf16`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L127-L154)
- [`src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh)
  - [`nvfp4_a16_sliced_k_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh#L171-L195)
- [`src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh)
  - [`nvfp4_a16_slice_ring_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh#L178-L202)
- [`src/ops/linear/nvfp4/nvfp4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh)
  - [`load_nvfp4_a_frag_direct`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh#L201-L222)
- [`src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh)
  - [`load_b_scaled_pairs`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh#L78-L101)
- [`src/ops/resident_moe/resident_moe.cu`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu)
  - [`load_nvfp4_oct_bf16`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L82-L97)

#### Root Cause & Hardware Rationale
On GB10's CUDA 13.0 toolchain (`nvcc V13.0.88`), `ptxas -arch=sm_121a` does not support the CUDA 13.2+ `cvt.rn.bf16x2.e2m1x2` PTX instruction, so all NVFP4-to-BF16 weight dequantization compiles the `#else` branch of [`nvfp4_codec.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L60-L154) (`cvt.rn.f16x2.e2m1x2` + `cvt.rn.f16x2.e4m3x2` + `HFMA2` + `HADD2.F32` + `F2FP.BF16.F32.PACK_AB`).

Inspecting the generated SASS (`cuobjdump -sass`) and source across [`nvfp4_codec.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh) and its callers reveals massive redundant `F2FP.F16.E4M3.UNPACK_B` (`cvt_e4m3x2_to_f16x2_bits`) and `PRMT` (`pack_same_low16_u32`) instructions inside the inner K-loops of every A16 MMA kernel:
1. **[`nvfp4_scaled_oct_bf16`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L127-L154)** (called by [`nvfp4_a16_mma.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh#L212-L219)):
   - Calls `nvfp4_scaled_pair_bf16` **4 separate times** for the same `uint8_t scale` byte, executing `pack_same_e4m3x2(scale)` + `cvt_e4m3x2_to_f16x2_bits` **4 times** and unpacking `packed` 2 nibbles at a time (`cvt_e2m1x2_to_f16x2_bits` × 4) instead of converting `scale` once and using `cvt_e2m1x4_to_f16x2_pair_bits` × 2.
2. **[`nvfp4_a16_sliced_k_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh#L171-L195), [`nvfp4_a16_slice_ring_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh#L178-L202), [`load_b_scaled_pairs`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh#L78-L101), and [`load_nvfp4_oct_bf16`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L82-L97)**:
   - Each 8-weight group (`uint32_t w0`, `uint32_t w1`) has one `uint8_t` scale (`s0`, `s1`).
   - Current code calls `pack_same_low16_u32(s0)` and then calls `nvfp4_scaled_two_pairs_bf16` **twice** for `w0` and **twice** for `w1`. Inside `nvfp4_scaled_two_pairs_bf16`, `cvt_e4m3x2_to_f16x2_bits(packed_scale)` is called on every invocation (`4` `F2FP.F16.E4M3` instructions per `(w0, w1)` pair!).
   - Because `cvt.rn.f16x2.e4m3x2` converts **two** `e4m3` bytes in one instruction, packing `s0 | (s1 << 8)` converts **both `s0` and `s1` in a single `F2FP.F16.E4M3.UNPACK_B` instruction**, and broadcasting the low and high `half` via `__low2half2` / `__high2half2` (`PRMT` / `.H0_H0` / `.H1_H1` operand reuse in `HFMA2`) eliminates **75% (3 out of 4) of all `F2FP.F16.E4M3` scale conversion instructions** and **100% of `pack_same_low16_u32` instructions** while producing **bit-identical** FP16/BF16 results!

#### Required Implementation Steps
1. **Add Pre-Converted Scale Helpers in [`nvfp4_codec.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh)**:
   - Add `nvfp4_scaled_two_pairs_bf16_f16scale(std::uint16_t packed4, std::uint32_t s_f16x2, __nv_bfloat162& low, __nv_bfloat162& high)` and `nvfp4_scaled_oct_bf16_f16scale(std::uint32_t packed8, std::uint32_t s_f16x2, __nv_bfloat162& p0, __nv_bfloat162& p1, __nv_bfloat162& p2, __nv_bfloat162& p3)` that take an already-converted `f16x2` scale register (`s_f16x2`).
   - Add `nvfp4_unpack_two_scales_f16x2(std::uint8_t s0, std::uint8_t s1, std::uint32_t& s0_f16x2, std::uint32_t& s1_f16x2)` that executes **one** `cvt_e4m3x2_to_f16x2_bits(static_cast<std::uint16_t>(s0) | (static_cast<std::uint16_t>(s1) << 8))` and duplicates the low and high `f16` halves into `s0_f16x2` and `s1_f16x2`.
   - Rewrite [`nvfp4_scaled_oct_bf16`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L127-L154) on the `#else` path to convert `scale` once (`const std::uint32_t s = cvt_e4m3x2_to_f16x2_bits(pack_same_e4m3x2(scale))`) and decode both 16-bit halves via `nvfp4_scaled_two_pairs_bf16_f16scale`.
2. **Use `nvfp4_unpack_two_scales_f16x2` + `nvfp4_scaled_oct_bf16_f16scale` Across All A16 MMA Callers**:
   - Update [`nvfp4_a16_sliced_k_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh#L171-L195), [`nvfp4_a16_slice_ring_mma_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh#L178-L202), [`load_nvfp4_a_frag_direct`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a16_mma.cuh#L201-L222), [`load_b_scaled_pairs`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/narrow_a16_mma.cuh#L78-L101), and [`load_nvfp4_oct_bf16`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L82-L97) to convert pairs of scales `(s0, s1)` in a single `cvt.rn.f16x2.e4m3x2` instruction and pass the resulting `f16x2` scale registers to `nvfp4_scaled_oct_bf16_f16scale`.

#### Verification & Acceptance Criteria
```bash
find src/ops/linear/nvfp4 src/ops/linear_add/nvfp4 src/ops/linear_swiglu/nvfp4 src/ops/attn_input_proj/nvfp4 src/ops/gdn_input_proj/nvfp4 src/ops/offloaded_sparse_moe src/ops/resident_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_nvfp4_a16_test infernix_linear_add_nvfp4_test infernix_linear_swiglu_nvfp4_test infernix_gdn_input_proj_test infernix_ops_offloaded_moe_test infernix_resident_moe_test infernix_linear_bench infernix_nvfp4_linear_add_bench
ctest --test-dir build-native -R "infernix_linear_nvfp4_a16_test|infernix_linear_add_nvfp4_test|infernix_linear_swiglu_nvfp4_test|infernix_gdn_input_proj_test|infernix_offloaded_sparse_moe|infernix_resident_moe_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_nvfp4_linear_add_bench --policy a16 --tokens 2,4,8,16,32 --warmup 3 --repeat 10
```
- **Acceptance**:
  - All NVFP4 A16 oracle tests pass with bit-identical results and zero new stack spills.
  - Per-symbol SASS inspection (`cuobjdump -sass -fun <symbol>`) confirms a **4x reduction in `F2FP.F16.E4M3` instructions** inside `nvfp4_a16_sliced_k_mma_kernel` and `nvfp4_a16_mma_kernel`.
  - Small-batch A16 MMA throughput (`T=2..32`, MTP/DFlash2 verification widths) improves across `infernix_nvfp4_linear_add_bench --policy a16` and `infernix_linear_bench`.

---

### Task T5 (Priority P1): Cache Input Vectors Across RMSNorm Passes, Coalesce 4-Byte Scale Stores (`store_nvfp4_scale_quad`), and Enable PDL in `rmsnorm_nvfp4_quantize_kernel`

#### Affected Files & Symbols
- [`src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu)
  - [`rmsnorm_nvfp4_quantize_kernel`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu#L30-L90)
  - [`nvfp4_attn_input_fused_a4_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu#L131-L157)

#### Root Cause & Hardware Rationale
Every full-attention layer on NVFP4 (`W4Dense`) executes [`nvfp4_attn_input_fused_a4_launch`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu#L131-L157) for `T >= 5`. Inspecting [`rmsnorm_nvfp4_quantize_kernel`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu#L30-L90) against [`nvfp4_a4_quantize_kernel`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L35-L78) reveals four inefficiencies:
1. **Uncoalesced 1-Byte Global Scale Stores (`lines 40–44` and `88`)**:
   - Each thread writes `scales[nvfp4_tiled_scale_offset<Geometry, Layout>(token, group)] = quantized.scale;` as an uncoalesced 1-byte `STG.E.U8` transaction (`320` 1-byte stores per token for `K=5120`, `160` for `K=2560`).
   - In [`nvfp4_a4_quantize.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L35-L44), `store_nvfp4_scale_quad` packs 4 consecutive threads' `uint8_t` scales via `__shfl_sync` (`group_base = group & ~3`, which are contiguous in `RowMajor`, `Tiled128`, and `Tiled256` layouts!) and issues a single coalesced 4-byte `uint32_t` store from `(lane & 3) == 0`, reducing scale store transactions by **4x**.
2. **Redundant Second Global Read of `x` (`lines 50–53` vs `77–81`)**:
   - With `256` threads/CTA and `16` `bf16` values (`uint4 lo, hi`) per group:
     - For `K = 2560` (`kGroups = 160 < 256`), each thread handles **at most 1 group** (`group = threadIdx.x < 160`, i.e., `2 × uint4` = `8` registers)!
     - For `K = 5120` (`kGroups = 320`), each thread handles **at most 2 groups** (`threadIdx.x` and `threadIdx.x + 256 < 320`, i.e., `4 × uint4` = `16` registers)!
   - Yet `rmsnorm_nvfp4_quantize_kernel` loads `x_vec4[group * 2]` and `x_vec4[group * 2 + 1]` from global memory during the sum-of-squares pass (`lines 50–53`), discards them, and re-loads them from global/L2 memory in the quantize pass (`lines 77–81`)!
   - Even better: during the first pass, each thread can multiply `bf16` input elements by `weight` (`w_lo`, `w_hi`), accumulate `sum_sq` on the unscaled `x`, and keep the `weight`-multiplied `float` or `__nv_bfloat162` values (or `lo, hi` + `w_lo, w_hi`) in registers so the second pass performs **zero global memory loads**!
3. **Missing `MinBlocksPerSm` and Missing Programmatic Dependent Launch (`pdl::enter()`)**:
   - `rmsnorm_nvfp4_quantize_kernel` uses `__launch_bounds__(256)` without `MinBlocksPerSm` (`__launch_bounds__(256, 2)`) and never calls `pdl::enter()` (`cudaTriggerProgrammaticLaunchCompletion`), delaying the downstream `qkv_gemm` (`pdl::Dependency::Programmatic`) until the entire quantizer CTA exits instead of overlapping preamble/TMA setup.

#### Required Implementation Steps
1. **Cache `lo, hi` and `w_lo, w_hi` in Registers Across the RMSNorm Reduction in [`nvfp4_attn_input_fused_rmsnorm.cu`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm.cu#L30-L90)**:
   - Because `kIters = (kGroups + 255) / 256` is a compile-time constant (`1` for `K=2560`, `2` for `K=5120`), store `uint4 cached_lo[kIters], cached_hi[kIters], cached_w_lo[kIters], cached_w_hi[kIters]` in registers during pass 1 so pass 2 performs zero global memory loads.
2. **Use `store_nvfp4_scale_quad` for Coalesced 4-Byte Scale Writes**:
   - Replace the 1-byte `scales[...] = ...` writes in both the `rms_inv == 0.0f` zero path and the normal quantize path with [`store_nvfp4_scale_quad<Geometry, Layout>`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh#L35-L44).
3. **Add `__launch_bounds__(256, 2)` and `pdl::enter()` / `pdl::launch_with`**:
   - Annotate `rmsnorm_nvfp4_quantize_kernel` with `__launch_bounds__(256, 2)`, call `pdl::enter()` at the start of `rmsnorm_nvfp4_quantize_kernel`, and launch `rmsnorm_nvfp4_quantize_kernel` via `pdl::launch_with(pdl::Dependency::None, ...)` while launching the downstream GEMM (`nvfp4_linear_a4_mma` / `launch_nvfp4_a4_tma_linear`) with `pdl::Dependency::Programmatic`.

#### Verification & Acceptance Criteria
```bash
find src/ops/attn_input_proj/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_linear_fused_test
ctest --test-dir build-native -R "infernix_attn_input_proj|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance**:
  - `infernix_attn_input_proj*` and `infernix_cuda_stack_check` CTests pass (`STACK:0 LOCAL:0`).
  - `cuobjdump -sass` on `rmsnorm_nvfp4_quantize_kernel` verifies zero global `LDG` instructions after the `BlockReduce` barrier and 4-byte `STG.E` scale writes instead of `STG.E.U8`.

---

### Task T6 (Priority P1): Eliminate Stack Spills (`REG:255 STACK:8 -> 0`) in `mxfp8_kv_tiled_mma_kernel` (`Nvfp4KvArenaTiledSchedule`) and Remove Redundant Shared-Memory Zeroing/Barriers in `nvfp4_kv_grouped_mma_kernel`

#### Affected Files & Symbols
- [`src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh)
  - [`mxfp8_kv_tiled_mma_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh#L96-L469)
- [`src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh)
  - [`nvfp4_kv_grouped_mma_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh#L121-L265)
- [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json)

#### Root Cause & Hardware Rationale
1. **8-Byte Local-Memory Stack Spills in All 3 `Nvfp4KvArenaTiledSchedule` Instantiations**:
   - Running `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` and inspecting [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json#L20-L36) shows that all 3 instantiations of `mxfp8_kv_tiled_mma_kernel<..., Nvfp4KvArenaTiledSchedule, Nvfp4TiledKeys, Nvfp4G16TiledValues, ..., false>` hit the 255-register ceiling and spill `8 B` of stack (`STL`/`LDL` local memory traffic):
     - `CausalGeometry<256, 16, 2>, ..., PagedKVDirectMetadata, false` (`REG:255 STACK:8 SHARED:38016`)
     - `CausalGeometry<256, 32, 2>, ..., PagedKVDirectMetadata, false` (`REG:255 STACK:8 SHARED:38016`)
     - `CausalGeometry<256, 24, 4>, ..., PagedKVBatchMetadata<false>, false` (`REG:254 STACK:8 SHARED:38016`)
   - Inspecting [`mxfp8_kv_tiled_mma_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh#L156-L176) shows why live register count reaches 255 when `Keys::kNvfp4` is true:
     - `q_scale_r0` and `q_scale_r1` (`2` FP32 registers, `lines 161–162`) are hoisted before the `for (int step = 0; ...)` loop and kept live across the entire `slab` QK MMA loop (`lines 259–370`), even though they are only used **once per KV tile** after the `slab` loop at `lines 373–377` (`score[nt] *= q_scale[warp_row0 + gid]`)!
     - `v_lane_base`, `v8_lane_base`, `q_lane_base`, `k_lane_base` (`lines 163–175`) and `warp_row0`, `warp_row1` (`lines 157–158`) are precomputed and kept live across the `step` and `slab` loops instead of being computed on demand in their respective `if constexpr (Keys::kNvfp4)` / `Pv8` scopes.
2. **Redundant 8 KiB `q_s` Zeroing and Back-to-Back `__syncthreads()` in [`nvfp4_kv_grouped_mma_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh#L180-L215)**:
   - At lines 181–183, every CTA zeroes all `Br * D = 16 * 256 = 4,096` `bf16` elements (`8 KiB`) of `q_s`, runs `__syncthreads()`, and then immediately overwrites the first `RowCount * D` elements in lines 184–202! When `RowCount == Br` (`G == 16`), 100% of the zeroing is overwritten; when `RowCount < Br`, only `index >= RowCount * D` needs zeroing, and it can be done in the **same pass** without the first `__syncthreads()`!
   - Furthermore, line 203 executes `__syncthreads()`, lines 205–214 perform purely register-local integer math (`best_split`, `split_len`), and line 215 executes a **second back-to-back `__syncthreads()`** with zero shared-memory reads or writes in between!

#### Required Implementation Steps
1. **Eliminate the 8-Byte Stack Spill in [`mxfp8_tiled_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh)**:
   - Remove the hoisted `q_scale_r0` and `q_scale_r1` registers (`lines 161–162`) and read `q_scale[warp * kRowsPerWarp + gid]` directly from shared memory at lines 374–377 (after the `slab` QK MMA loop).
   - Scope `q_lane_base`, `q_as`, `q_r`, `k_lane_base`, `k_as`, `k_r` strictly inside the `else` (`!Keys::kNvfp4`) branch, and compute `v_lane_base` / `v8_lane_base` inside the PV phase (`lines 422–455`) after the QK registers are dead.
2. **Remove Redundant Zeroing and Merge Barriers in [`grouped_mma.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/causal_cache/nvfp4/grouped_mma.cuh#L180-L215)**:
   - Replace the separate `q_s` zeroing loop (`lines 181–183`) + `__syncthreads()` with tail-only zeroing (`for (int index = RowCount * D + tid; index < Br * D; index += kThreads)`) alongside the `RowCount` Q-quantization loop, and remove the duplicate `__syncthreads()` at line 203 (keeping the single `__syncthreads()` at line 215 before the split-K loop).
3. **Ratchet Down [`tools/cuda_stack/cuda_stack_allowlist.json`](file:///opt/etc/Infernix/tools/cuda_stack/cuda_stack_allowlist.json)**:
   - Run `/opt/etc/Infernix/.venv-convert/bin/python3 tools/cuda_stack/check_cuda_stack.py update build-native/apps/infernix-serve` and verify via `git diff tools/cuda_stack/cuda_stack_allowlist.json` that the 3 `mxfp8_kv_tiled_mma_kernel<..., Nvfp4KvArenaTiledSchedule, ...>` entries are removed (`24 B` total stack eliminated, zero new entries).

#### Verification & Acceptance Criteria
```bash
find src/ops/softmax_attention -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_softmax_attention_nvfp4_test infernix_qsa_nvfp4_test infernix_causal_softmax_attention_bench infernix_qsa_attention_bench
/opt/etc/Infernix/.venv-convert/bin/python3 tools/cuda_stack/check_cuda_stack.py update build-native/apps/infernix-serve
ctest --test-dir build-native -R "infernix_softmax_attention_nvfp4_test|infernix_qsa_nvfp4_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_qsa_attention_bench --storage nvfp4 --tokens 1,64,512 --context 4096,16384 --warmup 2 --repeat 5
```
- **Acceptance**:
  - `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` shows `STACK:0 LOCAL:0` on all 3 `mxfp8_kv_tiled_mma_kernel<..., Nvfp4KvArenaTiledSchedule, ...>` symbols (`tools/cuda_stack/cuda_stack_allowlist.json` ratcheted down by 3 symbols).
  - All `infernix_softmax_attention_nvfp4_test`, `infernix_qsa_nvfp4_test`, and `infernix_cuda_stack_check` CTests pass.

---

### Task T7 (Priority P1): Increase NVFP4 MTP & DFlash2 Speculative Acceptance by Eliminating Draft-vs-Verify Activation Quantization Mismatch at `T=5..16` and Fixing `AllowA4Residual` Fallbacks in `attn_input_proj` and `gdn_input_proj`

#### Affected Files & Symbols
- [`src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp)
  - [`select_nvfp4_attn_input_route`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L40-L56)
- [`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp)
  - [`select_nvfp4_gdn_input_route`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L20-L37)
- [`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp)
  - [`select_nvfp4_linear_swiglu_route`](file:///opt/etc/Infernix/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp#L20-L43)
- [`src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp)
  - [`select_nvfp4_linear_add_route`](file:///opt/etc/Infernix/src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp#L15-L37)

#### Root Cause & Hardware Rationale
Deep analysis of the NVFP4 route planners across `attn_input_proj`, `gdn_input_proj`, `linear_swiglu`, and `linear_add` reveals why speculative decoding (MTP-4 at `T=5` and DFlash2 at `T=5..16`) suffers reduced acceptance on NVFP4 (`W4Dense`) models:
1. **Draft (`T=1`) vs. Verify (`T=5..16`) Activation Quantization Mismatch in `attn_input_proj` (`[13952, 2560]` / `[14336, 5120]`)**:
   - During autoregressive drafting (`T=1`, or narrow `T=2..4`), [`select_nvfp4_attn_input_route`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L47-L56) runs `DecodeA16` / `SmallTA16`, keeping input activations in **16-bit BF16** (`A16`).
   - During speculative verification (`T=5` for 4 draft tokens, or `T=8..16` for DFlash2 / multi-lane verify), `tokens <= 4` is false, so `select_nvfp4_attn_input_route` abruptly switches to **1-term 4-bit `FusedA4`** (`E2M1` 4-bit activation quantization without residual second term)!
   - By contrast, [`select_nvfp4_gdn_input_route`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L33-L35) stays on `A16` through `tokens <= 16`, and `linear_swiglu` / `linear_add` switch to **2-term `A4Residual`** (`Nvfp4A4Pair`) at `tokens >= 5`!
   - Thus, at `T=5..16`, **`attn_input_proj` is the ONLY dense projection in the model that drops from 16-bit BF16 activations (during `T=1` draft) to 1-term 4-bit E2M1 activations (during `T=5..16` verification)**, injecting 1-term 4-bit activation quantization noise directly into Q, K, V, attention gate, and QSA index keys during verification!
   - Furthermore, our live `infernix_linear_bench` measurement on GB10 at `[13952, 2560]` (`Qwen3.8-Flash-Next` `attn_input_proj`) shows that at `T=5` (`80.896 us`, `250.4 GB/s`, `91.72% DRAM`) and `T=16` (`82.944 us`, `248.6 GB/s`, `91.06% DRAM`), the `BF16_F32ACC` (`SmallTA16` sliced-K MMA) route ALREADY runs at **91–92% of LPDDR5X DRAM ceiling** (`80.9–82.9 us`, faster than `T=64` `NVFP4_F32ACC` at `97.1 us`)!
2. **Broken `allows_residual(policy)` Handling in `attn_input_proj` and `gdn_input_proj`**:
   - In [`nvfp4_attn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L42-L45) and [`nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L23-L26):
     ```cpp
     if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
         allows_residual(policy)) {
         return Nvfp4GdnInputRoute::A16;
     }
     ```
   - Whenever a caller passes `LinearPolicy::AllowA4Residual` (which means "allow A4 and 2-term A4 residual"), `attn_input_proj` and `gdn_input_proj` treat it as `A16Only` even at `T=4096` prefill instead of falling back to `FusedA4` / `A4` at prefill widths (`T >= 17`)!

#### Required Implementation Steps
1. **Extend `SmallTA16` Through `T <= 16` in [`nvfp4_attn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L40-L56)**:
   - Raise the `SmallTA16` boundary in [`select_nvfp4_attn_input_route`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L54) from `tokens <= 4` to `tokens <= 16` (matching `select_nvfp4_gdn_input_route`'s `tokens <= 16` boundary and covering all MTP-4 `T=5` and DFlash2 `T=5..16` verification widths with 16-bit BF16 activations at 91%+ DRAM efficiency).
   - Verify in `src/ops/attn_input_proj/nvfp4/` that `SmallTA16` supports `tokens <= 16` (expanding any `kMaxTokens` template cap in `nvfp4_attn_input_decode.cu` from `4` to `16` or dispatching the fused RMSNorm + `select_a16` sliced-K MMA for `5 <= tokens <= 16`).
2. **Fix `allows_residual(policy)` in [`nvfp4_attn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp#L42-L45) and [`nvfp4_gdn_input_plan.cpp`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp#L23-L26)**:
   - When `allows_residual(policy)` is true, use `SmallTA16` / `A16` for `tokens <= 16` (verification widths) and `FusedA4` / `A4` for `tokens >= 17` (prefill widths), rather than forcing slow `A16` at `T=512..8192` prefill.
3. **Update Route Plan Unit Tests**:
   - Update affected route plan expectations in `tests/ops/attn_input_proj_test.cu` and `tests/ops/gdn_input_proj_test.cu`.

#### Verification & Acceptance Criteria
```bash
find src/ops/attn_input_proj src/ops/gdn_input_proj -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_linear_fused_test infernix_gdn_input_proj_test infernix_gdn_input_proj_bench
ctest --test-dir build-native -R "infernix_attn_input_proj|infernix_gdn_input_proj|infernix_cuda_stack_check" --output-on-failure
```
- **Acceptance**:
  - At `T=5..16` (MTP-4 and DFlash2 verification widths), `attn_input_proj` and `gdn_input_proj` both preserve 16-bit BF16 activations (`SmallTA16` / `A16`), eliminating the 1-term 4-bit `E2M1` draft-vs-verify mismatch on Q/K/V/index-queries while running at $\ge 90\%$ DRAM efficiency.
  - All `infernix_attn_input_proj*`, `infernix_gdn_input_proj*`, and `infernix_cuda_stack_check` tests pass.

---

### Task T8 (Priority P1): Enable Multi-Row Fused `gdn_input_proj_conv_snapshot` (`T=2..16`) in `Forward::gdn` and Qualify `ExpertA4Residual` / `ExpertA4ResidualDecode` on `DenseKernels::Nvfp4`

#### Affected Files & Symbols
- [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp)
  - [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L688-L775)
- [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp)
  - [`kQualified`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L618-L633)
  - [`fast_path_applies`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L746-L769)
- [`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu)
  - [`nvfp4_gdn_snapshot_small_t_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu#L89-L97)

#### Root Cause & Hardware Rationale
1. **Unfused Global `qkv` Write + Re-Read at `T=2..16` Multi-Lane Decode in [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L690-L716)**:
   - In [`Forward::gdn`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L690), `const bool decode = batch.verify == nullptr && batch.width <= 1;` restricts `ops::gdn_input_proj_conv_snapshot` strictly to `batch.width == 1` (`T=1`).
   - Whenever `batch.verify == nullptr && batch.width > 1` (`T=2..8` batched decode across resident lanes), `Forward::gdn` falls back to the unfused `ops::gdn_input_proj` + `ops::causal_conv1d_silu` pair (`lines 752–773`), writing the `[8192, T]` `qkv` tensor out to global memory and reading it back in a separate `causal_conv1d_silu` kernel across all 36 GDN layers (`36` extra kernel launches + `36 × 16 KiB × T` global round-trips per step).
   - However, [`nvfp4_gdn_snapshot_small_t_launch`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu#L89-L97) is **already implemented and tested for `T = 2..16`** (`select_nvfp4_gdn_input_route` returns `SmallTA16` for `2 <= T <= 16`!). Wait: note that in multi-lane decode (`batch.width > 1, batch.verify == nullptr`), each row belongs to a *different* sequence lane, whereas `gdn_input_proj_conv_snapshot` advances a single sequence's `conv_state` across `T` contiguous tokens! Let's check what `gdn_input_proj_conv_record` does when `batch.verify != nullptr` (`T = 2..16` MTP/DFlash2 verification on a single sequence):
     - At lines 697–708 of [`forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L697-L708): when `batch.verify != nullptr && fast_.gdn_conv_fused && ops::gdn_input_proj_conv_record_supported(...)`, it calls `ops::gdn_input_proj_conv_record`!
     - Let's check `ops::gdn_input_proj_conv_record_supported` in `src/ops/gdn_input_proj/nvfp4/`: at `T = 2..16`, `nvfp4_gdn_record_small_t_launch` runs `nvfp4_linear_a16_sliced_k` (`conv_pre`) + `nvfp4_gdn_conv_post_kernel` (`conv_post`).
2. **Missing `DenseKernels::Nvfp4` Qualification for `ExpertA4Residual` and `ExpertA4ResidualDecode` in [`fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L618-L633)**:
   - In [`fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L764-L768), `FastPath::ExpertA4Residual` and `FastPath::ExpertA4ResidualDecode` only check `configuration.dense == DenseKernels::Fp8Row || configuration.dense == DenseKernels::Mxfp8`, excluding `configuration.dense == DenseKernels::Nvfp4` when `!configuration.a4_experts` (`W4Dense` with `--a16-experts`).
   - Adding `configuration.dense == DenseKernels::Nvfp4` to `fast_path_applies` and `kQualified` (`DenseKernels::Nvfp4, false`) enables the 2-term W4A4 residual MoE fast path (`gate_up_pairs_a4r` / `down_a4r` at `T=1..16` and `grouped_kernel_a4r` at `T >= 17`) on NVFP4 models when `--a16-experts` is used.

#### Required Implementation Steps
1. **Enable `ExpertA4Residual` and `ExpertA4ResidualDecode` for `DenseKernels::Nvfp4` (`a4_experts == false`) in [`src/models/qwen4_exp/program/fast_paths.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp)**:
   - In [`fast_path_applies`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L764-L768), allow `configuration.dense == DenseKernels::Nvfp4` alongside `Fp8Row` and `Mxfp8` when `!configuration.a4_experts`.
   - In [`kQualified`](file:///opt/etc/Infernix/src/models/qwen4_exp/program/fast_paths.cpp#L628-L633) (`{DenseKernels::Nvfp4, false}`), include `FastPath::ExpertA4Residual` and `FastPath::ExpertA4ResidualDecode` and update the comment at line 573 to match the table.
2. **Optimize `nvfp4_gdn_conv_post_kernel` (`T=2..16` MTP/DFlash2 `gdn_input_proj_conv_record` & `snapshot` Post-Kernel) in [`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu)**:
   - Currently [`nvfp4_gdn_conv_post_kernel`](file:///opt/etc/Infernix/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu#L18-L130) launches `256` CTAs of **`1,024` threads (`kWarpsPerCta = 32`, 1 CTA/SM)** even at `T = 2..16` (`token_tiles = 1` or `2`), leaving **30 to 31 out of 32 warps (`93.75%–96.88%` of threads!) completely idle** (`for (int tile = warp; tile < token_tiles; tile += kWarpsPerCta)`) and serializing all shared-memory loads of `conv_states` (`3 × 32` floats) and `conv_weight` (`4 × 32` bf16) onto `warp == 0` (`lines 51–67`)!
   - Template `nvfp4_gdn_conv_post_kernel<int WarpsPerCta, bool Record>` with `WarpsPerCta = 4` (`128` threads/CTA, `__launch_bounds__(128, 4)`) for `T <= 32` and `WarpsPerCta = 8` (`256` threads/CTA, `__launch_bounds__(256, 2)`) for `T > 32`, parallelize the `conv_states` + `conv_weight` shared-memory loads across all warps in the CTA, and add `pdl::enter()` + `pdl::launch_with(pdl::Dependency::Programmatic, ...)` so `conv_post` overlaps with `conv_pre` (`nvfp4_linear_a16_sliced_k` / `nvfp4_a4_mma`).

#### Verification & Acceptance Criteria
```bash
find src/ops/gdn_input_proj/nvfp4 -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_gdn_input_proj_test infernix_qwen4_exp_fast_paths_test infernix_gdn_input_proj_bench
ctest --test-dir build-native -R "infernix_gdn_input_proj_test|infernix_qwen4_exp_fast_paths_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_gdn_input_proj_bench --format nvfp4 --nvfp4-policy a4
```
- **Acceptance**:
  - `infernix_gdn_input_proj_test`, `infernix_qwen4_exp_fast_paths_test`, and `infernix_cuda_stack_check` pass (`0` new stack spills).
  - `infernix_gdn_input_proj_bench --format nvfp4 --nvfp4-policy a4` `fused` latency at `T=2, 4, 8, 16, 32` improves over the baseline (`113.0 us` at `T=2`, `115.6 us` at `T=4`, `118.4 us` at `T=8`, `122.4 us` at `T=16`, `178.7 us` at `T=32`) so `fused` beats `composed` across `T=2..32`.

---

### Task T9 (Priority P2): Parallelize Across Columns, Vectorize `float4` Loads, and Enable PDL in `nvfp4_logits_topk_kernel` (`linear_topk/nvfp4.cu`)

#### Affected Files & Symbols
- [`src/ops/linear_topk/nvfp4.cu`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu)
  - [`nvfp4_logits_topk_kernel`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu#L26-L67)
  - [`launch_nvfp4_linear_topk`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu#L69-L90)

#### Root Cause & Hardware Rationale
When serving NVFP4 models with NVFP4 LM head or router `linear_topk` (`[248320, 2560]` or `[512, 2560]`), [`launch_nvfp4_linear_topk`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu#L69-L90) runs `rg16_linear` followed by [`nvfp4_logits_topk_kernel`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu#L26-L67). Inspecting `nvfp4.cu` reveals three bottlenecks:
1. **Serial Loop Over `columns` Inside Every CTA (`lines 42–66`)**:
   - `nvfp4_logits_topk_kernel` launches a 1D grid `kBlocks = (rows + 127) / 128` (`1,940` blocks for `rows = 248,320`, or `4` blocks for `rows = 512`) and loops serially `for (int column = 0; column < columns; ++column)` inside each CTA with **two `__syncthreads()` per column**!
   - When `rows = 512` (MoE router `512` experts) and `columns = 8..4096`, launching only `4` CTAs on a 48-SM GPU leaves **44 of 48 SMs (`91.7%` of the GPU) completely idle** while those 4 CTAs loop serially up to `4,096` times!
2. **Scalar `float` Loads Instead of Vectorized `float4` Loads (`lines 44–45`)**:
   - Each thread loads `4` strided scalar floats (`row = blockIdx.x * 128 + lane + item * 32`) instead of loading a contiguous `float4` (`row_base = blockIdx.x * 128 + lane * 4`) in a single `LDG.E.128` instruction.
3. **Missing `__launch_bounds__` and Programmatic Dependent Launch (`pdl`)**:
   - `nvfp4_logits_topk_kernel` has no `__launch_bounds__` and is launched without `pdl::enter()` or `pdl::launch_with(pdl::Dependency::Programmatic, ...)`.

#### Required Implementation Steps
1. **Use a 2D Grid `dim3(kBlocks, columns)` and Remove the Serial `column` Loop in [`src/ops/linear_topk/nvfp4.cu`](file:///opt/etc/Infernix/src/ops/linear_topk/nvfp4.cu)**:
   - Set `const int column = static_cast<int>(blockIdx.y);` and launch `dim3(kBlocks, static_cast<unsigned int>(columns))` so all `(row_tile, column)` tiles execute in parallel across all 48 SMs without inter-column `__syncthreads()` barriers.
2. **Vectorize Contiguous Rows with `float4` (`LDG.E.128`)**:
   - Have each lane `lane = threadIdx.x` load `4` contiguous rows starting at `row_base = static_cast<int>(blockIdx.x) * kRowsPerBlock + lane * kItemsPerThread` via `reinterpret_cast<const float4*>(col_logits + row_base)[0]` when `row_base + 3 < rows` (with scalar fallback on the tail block).
3. **Add `__launch_bounds__(kThreadsPerBlock, 8)` and Enable `pdl`**:
   - Annotate `nvfp4_logits_topk_kernel` with `__launch_bounds__(kThreadsPerBlock, 8)`, call `pdl::enter()` at kernel entry, and launch via `pdl::launch_with(pdl::Dependency::Programmatic, ...)`.

#### Verification & Acceptance Criteria
```bash
touch src/ops/linear_topk/nvfp4.cu
ninja -C build-native -j "$(nproc)" infernix-serve infernix_linear_topk_test infernix_linear_topk_bench
ctest --test-dir build-native -R "infernix_linear_topk_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_linear_topk_bench
```
- **Acceptance**:
  - `infernix_linear_topk_test` and `infernix_cuda_stack_check` pass (`STACK:0 LOCAL:0`).
  - `cuobjdump -sass` on `nvfp4_logits_topk_kernel` confirms `LDG.E.128` vectorized loads and `CS2R` / PDL instructions.

---

### Task T10 (Priority P2): Shrink `resident_moe_nvfp4_kernel` Shared Memory (`24.6 KiB -> 16.6 KiB` for 5 CTAs/SM) and Vectorize Down-Projection Accumulation with `__nv_bfloat162`

#### Affected Files & Symbols
- [`src/ops/resident_moe/resident_moe.cu`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu)
  - [`resident_moe_nvfp4_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L318-L420)

#### Root Cause & Hardware Rationale
Every MTP draft step on `Qwen3.8-Flash-Next` (`forward.cpp` line 1487) executes [`resident_moe_nvfp4_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L318-L420) (`kHidden = 2560, kIntermediate = 4096` or `kHidden = 2048, kIntermediate = 512`). Inspecting `resident_moe_nvfp4_kernel` and its resource usage on GB10 (`REG:55 STACK:0 SHARED:25152` for `<2560, 4096>`, `4` CTAs/SM max on 99 KiB SMEM) reveals two bottlenecks:
1. **Oversized `float intermediate_s[4096]` (`16 KiB`) Caps Occupancy at 4 CTAs/SM and Doubles Shared-Memory Reads in the Down Phase (`lines 324–328, 390–414`)**:
   - `SharedStorage` allocates `float intermediate_s[IntermediateSize]` (`4096 * 4 = 16,384 B`) plus `x_s` (`5,120 B`), `x_pad_s` (`2,560 B`), and `out_s` (`1,024 B`) = `25,152 B` (`24.56 KiB`), which limits occupancy to `floor(101376 / 25152) = 4` CTAs/SM (`16` warps/SM = `33%` warp occupancy) even though `REG:55` (`56` allocated) can support **`9` CTAs/SM** (`36` warps/SM, `65536 / (128 * 56) = 9.14`)!
   - Furthermore, in the Down phase (`lines 393–414`), `intermediate_s` is read via two `float4` loads (`32 B` per 8 elements) and accumulated with scalar FP32 `fmaf` against `nvfp4_scaled_oct_f32` (which executes 4 separate `nvfp4_scaled_pair_f32` calls per 8 weights!).
2. **Even Better — Exact FP32 Tensor Core / `HFMA2` Accumulation or Compact Shared Storage**:
   - Notice that in the GateUp phase (`lines 352–385`), `resident_moe_nvfp4_kernel` uses `load_nvfp4_oct_bf16` + `mma_sync_m16n8k16_row_col_f32_bf16` (1:1 FP32 accumulation on Tensor Cores!), where `load_nvfp4_oct_bf16` (`lines 82–97`) converts `gate_scale0`, `up_scale0`, `gate_scale1`, `up_scale1` via **8 separate `cvt_e4m3x2_to_f16x2_bits` calls** per K-step.
   - And in `SharedStorage`, `x_s` (`5,120 B`) and `x_pad_s` (`2,560 B`) = `7,680 B` are **dead after the GateUp phase (`line 391` `__syncthreads()`)**! Overlapping `intermediate_s` with `(x_s + x_pad_s)` via a `union` inside `SharedStorage` immediately shrinks `sizeof(SharedStorage)` from **`25,152 B` (`24.56 KiB`) to `17,472 B` (`17.06 KiB`)**, raising SMEM occupancy from **4 CTAs/SM to 5 CTAs/SM (`+25%` occupancy)** without changing a single bit of `float intermediate_s[IntermediateSize]`!
   - In addition, replacing the 4-call `nvfp4_scaled_pair_f32` chain in `nvfp4_scaled_oct_f32` (`nvfp4_codec.cuh` lines 110–125) so `scale` is converted **once** per 8 weights instead of 4 times eliminates 3 redundant `F2FP` scale conversions per 8 weights in the Down loop!

#### Required Implementation Steps
1. **Overlap Dead `(x_s, x_pad_s)` with `intermediate_s` in [`resident_moe_nvfp4_kernel`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L324-L330)**:
   - Place `struct { __nv_bfloat16 x_s[HiddenSize]; __nv_bfloat162 x_pad_s[HiddenSize / 16][8]; }` and `float intermediate_s[IntermediateSize];` inside an anonymous `union` in `SharedStorage`, and raise `__launch_bounds__(kThreads, 4)` to `__launch_bounds__(kThreads, 5)` so `sizeof(SharedStorage)` drops from `25,152 B` to `17,472 B` (`5 CTAs/SM` on GB10's `99 KiB` SMEM).
2. **Deduplicate `e4m3` Scale Conversions in `nvfp4_scaled_oct_f32` ([`nvfp4_codec.cuh`](file:///opt/etc/Infernix/src/ops/linear/nvfp4/nvfp4_codec.cuh#L110-L125)) and `load_nvfp4_oct_bf16` ([`resident_moe.cu`](file:///opt/etc/Infernix/src/ops/resident_moe/resident_moe.cu#L82-L97))**:
   - In `nvfp4_scaled_oct_f32`, convert `scale` once (`const std::uint32_t s = cvt_e4m3x2_to_f16x2_bits(pack_same_e4m3x2(scale));`) and use `cvt_e2m1x4_to_f16x2_pair_bits` twice instead of calling `nvfp4_scaled_pair_f32` 4 times.
   - In `resident_moe_nvfp4_kernel`, unpack `(gate_scale0, up_scale0)` and `(gate_scale1, up_scale1)` with `nvfp4_unpack_two_scales_f16x2` (from Task T4).

#### Verification & Acceptance Criteria
```bash
find src/ops/resident_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_resident_moe_test infernix_resident_moe_bench
ctest --test-dir build-native -R "infernix_resident_moe_test|infernix_cuda_stack_check" --output-on-failure
./build-native/bench/infernix_resident_moe_bench
```
- **Acceptance**:
  - `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` confirms `resident_moe_nvfp4_kernel<2560, 4096>` shared memory drops from `25152 B` to **`<= 17472 B`** (`5 CTAs/SM` on GB10) with `STACK:0 LOCAL:0`.
  - `infernix_resident_moe_test` and `infernix_cuda_stack_check` pass.
  - `infernix_resident_moe_bench` NVFP4 latency improves across `T=1..8`.
