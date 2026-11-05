# GB10 (`sm_121a`) Performance Optimization Tasks — 2026-11-05

This roadmap defines 10 self-contained, high-impact engineering tasks targeting a **20–30% end-to-end throughput increase** across prefill, long-context QSA attention, and multi-row speculative decode (MTP + DFlash2 + N-gram) on the NVIDIA GB10 (`sm_121a`, 48 SMs, 99 KiB L1/SMEM per SM, 65,536 32-bit registers per SM, 24 MiB L2 cache, 128 GiB unified LPDDR5X memory at ~273 GB/s, 20-core Grace ARMv9.2 CPU).

All tasks are grounded in live hardware benchmarks (`infernix_offloaded_moe_wide_bench`, `infernix_qsa_attention_bench`, `infernix_gated_delta_net_bench`, `infernix_hyper_connection_bench`, `infernix_dynamic_grouped_conv_prepare_bench`, `infernix_linear_dynamic_grouped_conv_add_bench`, `infernix_linear_topk_bench`) and static binary resource inspections (`cuobjdump --dump-resource-usage build-native/apps/infernix-serve`) collected on this host at commit `98aff48b0`.

---

## Mandatory GB10 (`sm_121a`) Workflow & Architectural Invariants

1. **Dead CUDA Depfiles in `build-native` (CRITICAL)**:
   - Editing `.cuh` or `.h` headers does **not** automatically recompile `.cu.o` objects in `build-native`.
   - Whenever you edit any header under `src/ops/`, **always** `touch` the consuming `.cu` files before running Ninja (e.g., `find src/ops/<subdir> -name '*.cu' -exec touch {} +`) and verify `.cu.o` modification timestamps with `stat`.
2. **Single-Invocation Ninja Build & `nvlink` Amortization**:
   - Never run concurrent `ninja` builds against `build-native` (causes `cmake_device_link.o` fatbin hash mismatches `__fatbinwrap_*`).
   - Always build `infernix-serve`, benchmarks, and test binaries in a **single** Ninja invocation to share the ~55.8s device link:
     ```bash
     ninja -C build-native -j "$(nproc)" infernix-serve <test_and_bench_targets...>
     ```
3. **One-Way CUDA Stack-Spill Ratchet Gate (`infernix_cuda_stack_check`)**:
   - Every task must pass `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` (currently `7577 kernels checked, 0 over their allowed stack or local memory`, `259` allowlist entries).
   - Never run `check_cuda_stack.py update` to mask newly introduced or increased stack spills. Only run `update` when ratcheting `tools/cuda_stack/cuda_stack_allowlist.json` down.
4. **1:1 FP32 Tensor Core Accumulation & Oracle Verification**:
   - Preserve 1:1 FP32 Tensor Core accumulation on production routes unless a qualified fast path is explicitly measured faster and passes model/oracle gates.
5. **Mandatory GB10 Changelog Entry (`docs/gb10/20261105-changelog.md`)**:
   - Every completed task must record its accounting, task-premise corrections / architectural rationale, subsystem breakdown, verification numbers, regressions/adverse results, and accepted gaps in `docs/gb10/20261105-changelog.md`.

---

### Task T1 (Priority P0): Reorder Wide Routed-Expert Work-Tile Traversal to `(Job, RowBlock)`-Outer `ColTile`-Inner and Deduplicate Even/Odd Scale TMA Loads in `grouped_kernel`

#### Affected Files & Symbols
- [wide_expert.cuh](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh)
  - [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L219-L491) (`REG:95, SHARED:18816` for `Schedule`; `REG:135..144, SHARED:36224` for `WarpTile64<Schedule>`)
  - [`WidePlan`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L139-L152)
- [wide_expert.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu)
  - [`plan_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L91-L162) (`REG:32, SHARED:5136`)
  - [`launch_wide_expert`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L345-L440)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Wide routed-expert W4A4 NVFP4 grouped GEMM (`grouped_kernel`) is the single largest prefill hotspot on Qwen3.8-Flash-Next (`471 ms` at `T=8192`, `20.8%` of prefill; `1,920 ms` at `T=32768`).
  - Live `./build-native/bench/infernix_offloaded_moe_wide_bench --profile flash-next --qtype nvfp4 --activation a4 --tokens 64,256,1024,4096` measurements on GB10:
    - `T=64`: `9,280.8 us/layer` (`0.445 s` across 48 layers)
    - `T=256`: `13,747.0 us/layer` (`0.660 s` across 48 layers)
    - `T=1024`: `17,574.0 us/layer` (`0.844 s` across 48 layers, `5.3 TFLOP/s`)
    - `T=4096`: `17,898.4 us/layer` (`0.859 s` across 48 layers, `22.5 TFLOP/s`, only `59%` of the weight-stream floor in-model)
- **Root Causes**:
  1. **L2 Weight Eviction Across Multi-Tile Experts (`col_tile`-Outer Traversal)**:
     In [`plan_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L130-L156), an expert job with `count > 64` routed tokens (`tiles = ceil(count / 64) >= 2`, which occurs heavily at `T >= 1024` where average `count` per expert is `64..128` tokens = `2..4` 64-column tiles) emits `tiles` separate entries in `tile_job[tile]` and `tile_col_offset[tile]`. Then in [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L247-L255):
     ```cpp
     const int tile      = tile_begin + w / kRowBlocks;
     const int row_block = w % kRowBlocks;
     ```
     As `w` increments linearly across the persistent `96`-CTA grid, CTAs sweep all `kRowBlocks` (`10` row-blocks for `GateUp`, `20` row-blocks for `Down`, reading `1.77 MiB` of expert weights) for `col_tile = 0` before any CTA reaches `col_tile = 1` of that same expert. With `96` persistent CTAs active across `~10` concurrent experts (`17.7 MiB` working set), the `88.5 KiB` weight slice of `(job, row_block)` is evicted from GB10's 24 MiB L2 cache before `col_tile = 1..M-1` of the same expert re-reads it, multiplying DRAM weight traffic by `tiles` (`2–3×` at `T=1024..8192`).
  2. **Duplicate Even/Odd `a_scales` TMA Loads in Producer Warp**:
     In [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L286-L301), every `k_tile` (`kBlockK = 128`, `8` K-groups of `16`) issues a `tma::load_2d` on `a_scales` with `x = (k_tile / 2) * 16`. Because CUtensorMap box X is rounded up to 16 bytes (`16` scale bytes = `256` K elements = `2` K-tiles of `128`), `k_tile = 2*m` (even) and `k_tile = 2*m + 1` (odd) issue **identical** 16-byte TMA loads of the exact same `a_scales` cache lines on every single K-tile (`20` K-tiles for `GateUp`, `10` for `Down`), doubling `a_scales` MBAR transaction bytes and TMA unit issue pressure.

#### Required Implementation Steps
1. **Group Multi-Column Tiles per `(job, row_block)` for L2 Weight Reuse**:
   - In [`WidePlan`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L139-L152) and [`plan_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cu#L91-L162) (or directly in [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L244-L270)), reorder the work-unit mapping `w` so consecutive work units iterate over all column tiles `t_local = 0..job_tiles - 1` of the **same** `(job, row_block)` before advancing `row_block`.
   - Specifically, in `plan_kernel`, record for each `tile` (or per-job span) the job's `first_tile` and `job_tiles` (or emit work units indexed by `job_tile_span * kRowBlocks` where `row_block` is outer to `t_local`), so that within each expert `job`, `w` enumerates `(row_block = 0..kRowBlocks-1, t_local = 0..job_tiles-1)`. That way, `job_tiles` adjacent CTAs in the persistent grid process `t_local = 0..job_tiles-1` of the exact same `(job, row_block)` simultaneously, hitting the `88.5 KiB` weight block at 100% L2 residency.
2. **Hoist/Deduplicate Odd-`k_tile` `a_scales` TMA Loads**:
   - In [`grouped_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L286-L315), allocate a persistent per-CTA shared buffer `__shared__ alignas(128) std::uint8_t s_a_scales_pair[Schedule::kTileCols * 16]` (`1,024 B` for `kTileCols = 64`), or retain the even `k_tile`'s `s_a_scales` across the `(2*m, 2*m+1)` pair:
     - On even `k_tile` (`(k_tile & 1) == 0`), issue `tma::load_2d` for `kTileCols * 16` bytes (`expected_bytes = kStageWeightBytes + kStageScaleBytes + kStageActBytes + kStageActScaleBytes`).
     - On odd `k_tile` (`(k_tile & 1) == 1`), skip the `a_scales` `tma::load_2d`, reduce `expected_bytes` by `kStageActScaleBytes`, and have the consumer warps read the high 8 bytes (`+ 8`) from the even stage's `s_a_scales` buffer (with `kStages = 3`, stage `(stage + 2) % 3` still holds `k_tile - 1`'s `s_a_scales` untouched during `k_tile`'s consumer phase because stage `(stage + 2) % 3` is not overwritten until `k_tile + 2`!).
3. Verify that `cuobjdump --dump-resource-usage` shows zero new stack/local memory and register count stays within occupancy bounds.

#### Verification & Acceptance Criteria
```bash
# 1. Touch modified header consumers and build targets in a single invocation
find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_offloaded_sparse_moe_test infernix_offloaded_moe_wide_bench

# 2. Run stack-spill ratchet gate and oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_offloaded_sparse_moe_test" --output-on-failure

# 3. Benchmark wide expert GEMM across T=64..4096
./build-native/bench/infernix_offloaded_moe_wide_bench --profile flash-next --qtype nvfp4 --activation a4 --tokens 64,256,1024,4096
```
- **Acceptance Criteria**:
  - `infernix_offloaded_sparse_moe_test` and `infernix_cuda_stack_check` pass 100%.
  - `infernix_offloaded_moe_wide_bench` (`placement=frames`) latency at `T=1024` and `T=4096` improves by **>= 10%** over the `17,574.0 us` / `17,898.4 us` baseline without regressing `T=64` or `T=256`.

---

### Task T2 (Priority P0): Eliminate Empty-Split Work/Writes and Duplicate Value Decoding in `qsa::attention_kernel` & Vectorize `qsa::merge_kernel` for `W <= 8` Decode/Verification

#### Affected Files & Symbols
- [qsa.cu](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu)
  - [`attention_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L714-L927) (`REG:128, SHARED:0` with `49,824 B` dynamic SMEM for `Int8Group64`)
  - [`merge_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L929-L962) (`REG:28, SHARED:0`)
  - [`attention_splits`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L964-L976)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Live `./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 8192 --width 1,8,16,64` measurements on GB10 reveal a severe latency inversion at `width = 8` (the exact column width of 2-row 4-token MTP verification and 8-token DFlash2 verification):
    - `int8, context=8192, width=1`: `total = 49.02 us` (`select = 18.11 us`, `attention = 30.91 us`)
    - `int8, context=8192, width=8`: `total = 147.23 us` (`select = 22.37 us`, **`attention = 124.86 us`**) — **`1.61×` slower attention than `width = 16`!**
    - `int8, context=8192, width=16`: `total = 104.42 us` (`select = 27.04 us`, **`attention = 77.38 us`**)
    - `bf16, context=8192, width=8`: `total = 146.11 us` (`select = 22.21 us`, **`attention = 123.90 us`**) vs `width=16`: **`100.32 us`**.
- **Root Causes**:
  1. **Zero-Tile Splits Execute Full Query Load, Hadamard Transform, and 12 KiB Global Zero-Fill**:
     For `columns <= kSmallCallColumns` (`8`), [`attention_splits`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L968-L971) returns fixed `splits = 33` (`grid(33, 2, columns)` = `528` CTAs at `W = 8`) to preserve width-invariant split reduction boundaries. When `tiles == 0` (`begin >= end`, which occurs for every split index past `ceil(total / per_split)` — e.g., at `total = 64`, `per_split = 2`, splits `32` is empty; at `context < 2048` where `total < 128`, up to **31 of 33 splits** have `tiles == 0`), [`attention_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L728-L745) **never early-exits**! Every empty split CTA still loads `12 * 256` query BF16s from global memory, runs 2 `__syncthreads()` barriers, executes `normalized_hadamard_d256_inplace` across 12 heads, and at [lines 909–924](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L909-L924) writes `12 * 256 * 4 = 12,288 B` of useless zeros to `partial` in global memory, even though [`merge_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L940) skips any split with `slot[kHeadDim] == -INFINITY`!
  2. **Duplicate Value Cell Decoding in `attention_kernel`**:
     In the value accumulation loop ([`qsa.cu:865-879`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L865-L879)), threads `d_own = 0..255` (`h_base = 0`, heads `0..5`) and `d_own + 256 = 256..511` (`h_base = 6`, heads `6..11`) both call `Tile::value(stage, j0 + u, d_own)` independently, reading and dequantizing every INT8/FP8/NVFP4 value element + scale from shared memory **twice** per token.
  3. **Redundant Uncoalesced Broadcast Reads and 2-Pass Global Scans in `merge_kernel`**:
     In [`merge_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L929-L962), all 256 threads in the CTA loop twice over `s = 0..splits-1` (`33` splits), each reading `slot[kHeadDim]` (`m_s`) and `slot[kHeadDim + 1]` (`l_s`) at non-128B-aligned stride `258` floats and computing `expf(m_s - m)` 256 times redundantly per split.

#### Required Implementation Steps
1. **Immediate Early Exit for Empty Splits (`tiles == 0`) in `attention_kernel`**:
   - Right after computing `begin`, `end`, and `tiles = max(0, end - begin)` in [`attention_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L761-L765), if `tiles == 0`:
     ```cpp
     if (tiles == 0) {
         if (tid < group) {
             float* slot = partial + (((static_cast<std::size_t>(t) * kv_heads + kv_head) * group + tid) * splits + split) * kPartialStride;
             slot[kHeadDim]     = -INFINITY;
             slot[kHeadDim + 1] = 0.0f;
         }
         return;
     }
     ```
     This skips query global loads, shared memory initialization, Hadamard transform, and the `12 KiB` global zero-fill on `partial`, while producing the exact same `-INFINITY` sentinel consumed by `merge_kernel`.
2. **Precompute Split Weights in Shared Memory in `merge_kernel`**:
   - In [`merge_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L929-L962), have threads `d < splits` (`<= 48`, 2 warps) load `float2(slot[kHeadDim], slot[kHeadDim + 1])` once per split, reduce `m = max_s(m_s)` and `l = sum_s(expf(m_s - m) * l_s)` in shared memory, and store `s_weight[live_idx] = expf(m_s - m) * inv_l` and `s_split[live_idx] = s` for non-empty splits (`m_s > -INFINITY`).
   - After a single `__syncthreads()`, all 256 threads iterate only over `0..live_count - 1` reading `slot[d]` coalesced and multiplying by `s_weight[idx]`.
3. **Deduplicate `Tile::value` Decoding or Reduce `attention_kernel` Dynamic SMEM for 2 CTAs/SM**:
   - Notice `attention_kernel<Int8Group64>` currently uses `49,824 B` dynamic SMEM (`qs[12 * 256]` FP32 = `12,288 B` + `2 * 18,432 B` tiles + `768 B` stats + `336 B` row stats), which exceeds `99 KiB / 2 = 49.5 KiB` by just **`324 bytes`** (`49,824 B > 49,664 B`), forcing **1 CTA/SM**!
   - Stage `qs` as `__nv_bfloat16` (or `__half` / `uint2` after Hadamard transform, or overlap `qs` initialization with tile stage 1 before the double-buffered tile loop starts) to save `6,144 B` (`49,824 B -> 43,680 B <= 49.5 KiB`), unlocking **2 CTAs/SM** for `Int8Group64`, `Fp8E4M3Row256`, `Nvfp4Group16`, and `Fp8KeyNvfp4Value`! (Verify against the FP32 oracle that staging `qs` or shrinking SMEM preserves exact numerical tolerances.)

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_qsa_attention_bench

# 2. Run stack-spill check and QSA oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa_test" --output-on-failure

# 3. Benchmark QSA attention at W=1,8,16,64 across int8 and bf16
./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 2048,8192,32768 --width 1,8,16,64
```
- **Acceptance Criteria**:
  - `infernix_qsa_test` and `infernix_cuda_stack_check` pass 100%.
  - `infernix_qsa_attention_bench` `attention_us` at `width=8, context=8192` (`int8` and `bf16`) improves by **>= 20%** over the `124.86 us` / `123.90 us` baseline, and `width=1` does not regress.

---

### Task T3 (Priority P0): Overlap `cp.async` Key/Value Tile Gathers with Softmax/PV MMA in `qsa_prompt_kernel` (`Stages == 1`)

#### Affected Files & Symbols
- [qsa_prompt.cu](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu)
  - [`qsa_prompt_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L341-L815) (`REG:121..128, SHARED:46592` for `Int8Group64`/`Fp8E4M3Row256`/`Nvfp4Group16`, `SHARED:47616` for `BFloat16`)
  - [`PromptShape`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L44-L53)
  - [`Stage`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L349-L361)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - `qsa_prompt_kernel` is the #2 non-GEMM prefill hotspot (`183 ms` at `T=8192` = `8.3%` of prefill; `805 ms` at `T=32768`).
  - Live `./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 8192,32768 --width 512,2048` measurements on GB10:
    - `int8, context=8192, width=512`: `attention = 1,089.63 us`; `width=2048`: `attention = 9,335.84 us`
    - `int8, context=32768, width=512`: `attention = 1,178.66 us`; `width=2048`: `attention = 11,786.14 us`
    - `bf16, context=8192, width=512`: `attention = 2,257.82 us`; `width=2048`: `attention = 14,111.94 us`
    - `bf16, context=32768, width=512`: `attention = 4,476.29 us`; `width=2048`: `attention = 17,126.40 us`
- **Root Cause**:
  - On GB10 (`99 KiB` SMEM/SM), `qsa_prompt_kernel<Storage, 1>` uses `Stages = 1` (`46.5 KiB` SMEM/CTA -> `2 CTAs/SM` = `8 warps/SM`), because `Stages = 2` increases SMEM to `67–76 KiB/CTA` and drops occupancy to `1 CTA/SM` (`26–41%` slower).
  - However, in `Stages == 1` ([`qsa_prompt.cu:779-788`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L779-L788)), each warp executes a strictly unpipelined loop:
    ```cpp
    for (int i = warp; i < tiles; i += Shape::Warps) {
        issue(i, 0);
        cp_wait<0>();
        __syncwarp();
        expand(0);
        compute(i, 0);
        __syncwarp();
    }
    ```
  - Look at the lifetime of `k_s` (`K` rows + `KeyScales`) and `v_s` (`V` rows + `ValueScales`) inside [`compute(i, 0)`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L473-L751):
    - `k_s` is ONLY read during the QK MMA loop ([lines 499–587](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L499-L587)). Once QK MMA finishes at line 587, **`k_s` is completely dead** during the online softmax reduction/exp2f/rescale phase ([lines 589–633](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L589-L633)) AND during the entire PV MMA loop ([lines 635–736](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L635-L736))!
    - Conversely, `v_s` is NOT read at all during QK MMA or online softmax ([lines 499–633](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L499-L633)) — `v_s` is only needed starting at line 635 (PV MMA)!
  - Because `issue(i, 0)` currently bundles both K and V into a single `cp_commit()` and immediately calls `cp_wait<0>()`, every warp stalls for the full scattered L2/DRAM gather latency before starting QK MMA, leaving Tensor Cores idle for 66% of cycles!

#### Required Implementation Steps
1. **Split `issue(i, 0)` into `issue_k(i)` and `issue_v(i)` for Non-VQ Storages (`!Storage::kVectorQuantized`)**:
   - In [`qsa_prompt_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L422-L471), factor the `Stage` loads for `Stages == 1 && !Storage::kVectorQuantized` (`BFloat16`, `Int8Group64`, `Fp8E4M3Row256`, `Nvfp4Group16`, `Fp8KeyNvfp4Value`) into two separate `cp.async` commit groups using **zero additional shared memory**:
     - `issue_k(tile_idx)`: resolves valid row indices `j = 0..TileN-1`, stores valid positions/indices (or computes `block/offset` directly), issues `cp_async_16` for `k_s` and `ks_s`, and calls `cp_commit()`.
     - `issue_v(tile_idx)`: issues `cp_async_16` for `v_s` and `vs_s`, and calls `cp_commit()`.
2. **Pipeline K and V `cp.async` Groups Across QK+Softmax and PV MMA Within `Stages == 1`**:
   - Before the tile loop (if `warp < tiles`), issue `issue_k(warp)` (Group A in flight).
   - Inside `for (int i = warp; i < tiles; i += Shape::Warps)`:
     1. Issue `issue_v(i)` (Group B in flight; now 2 groups in flight: `K(i)` and `V(i)`).
     2. Call `cp_wait<1>()` and `__syncwarp()` — this waits for `K(i)` to land in `k_s`/`ks_s` while **`V(i)` continues streaming in the background**!
     3. Execute the QK MMA loop ([lines 499–587](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L499-L587)) and online softmax + accumulator rescaling ([lines 589–633](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L589-L633)). At the end of QK MMA, `k_s`/`ks_s` is completely consumed.
     4. Call `__syncwarp()`. If `i + Shape::Warps < tiles`, issue `issue_k(i + Shape::Warps)` into `k_s`/`ks_s` (Group C in flight; now 2 groups in flight: `V(i)` and `K(i + Warps)`), and call `cp_wait<1>()` (which waits for `V(i)` to complete while **`K(i + Warps)` streams in the background during PV MMA**!). Otherwise (`i + Shape::Warps >= tiles`), call `cp_wait<0>()`.
     5. Call `__syncwarp()` and execute the PV MMA loop ([lines 635–736](file:///opt/etc/Infernix/src/ops/qsa/qsa_prompt.cu#L635-L736)) from `v_s`/`vs_s`.
3. Verify that shared memory stays at `<= 47,616 B` (`2 CTAs/SM` preserved across all storages) and bitwise output matches the existing `Stages == 1` kernel (`0` ulp difference).

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_qsa_attention_bench

# 2. Run stack-spill check and QSA oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa_test" --output-on-failure

# 3. Benchmark QSA prompt attention at W=64, 512, 2048 across int8 and bf16
./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 8192,32768 --width 64,512,2048
```
- **Acceptance Criteria**:
  - `infernix_qsa_test` and `infernix_cuda_stack_check` pass 100%, and `cuobjdump --dump-resource-usage` confirms `SHARED <= 48,128 B` (`2 CTAs/SM`) and `STACK:0` for all `qsa_prompt_kernel` instantiations.
  - `infernix_qsa_attention_bench` `attention_us` at `width=512` and `width=2048` (`context=8192` and `32768`, `int8` and `bf16`) improves by **>= 10%** over baseline (`1,089.6 us` / `9,335.8 us` for `int8` and `2,257.8 us` / `14,111.9 us` for `bf16` at `context=8192`).

---

### Task T4 (Priority P1): Stage Packed BF16 Queries in `qsa_score_wide_kernel` to Unlock 3 CTAs/SM and Replace `cub::BlockScan` in `qsa_select_global_kernel`

#### Affected Files & Symbols
- [qsa_select.cu](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu)
  - [`qsa_score_wide_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L334-L388) (`REG:49, SHARED:42240`)
  - [`wide_leaves`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L266-L322)
  - [`qsa_select_global_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L649-L712) (`REG:40, SHARED:25488`)
  - [`find_score_bin`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L517-L537)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - In `./build-native/bench/infernix_qsa_attention_bench`, `qsa_select` (`select_us`) takes `661.5 us/layer` at `context=8192, width=2048` and `1,794.1 us/layer` at `context=32768, width=2048` (`21.5 ms` across 12 layers at `T=32768` per chunk, `484 ms` across `pp32768`).
  - `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` reports for `qsa_score_wide_kernel`:
    - `REG:49, STACK:0, SHARED:42240` (`41.25 KiB` static SMEM, `256` threads/CTA).
- **Root Causes**:
  1. **`qsa_score_wide_kernel` Still Stages Unpacked FP32 `float4` Queries (`33,792 B`), Capping Occupancy at 2 CTAs/SM**:
     In commit `b28ccccee`, `qsa_score_kernel` had its query shared memory halved from `float4` (`FP32`) to packed BF16 `uint2`, **but `qsa_score_wide_kernel` ([`qsa_select.cu:246-256, 342-366`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L246-L256)) was left staging `float4 qs[2112]` (`33,792 B`)**:
     - `kWideQueryFloat4 = 4 * 16 * 33 = 2,112 float4 = 33,792 B`
     - `kWideKeyPairs = 16 * 33 = 528 uint4 = 8,448 B`
     - `kWideSharedBytes = 42,240 B` (`41.25 KiB`), which limits `qsa_score_wide_kernel` to `kWideCtasPerSm = 2` (`2 * 41.25 KiB = 82.5 KiB`; 3 CTAs would require `123.75 KiB > 99 KiB`).
     - Crucially, `qsa_score_wide_kernel` already loads `queries` as BF16 mapped through `bf16_to_f32(queries[...]) * w` ([line 356](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L356)) OR notice that in [`qsa_score_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L180-L202), `qs` is staged as `uint2` (`4` packed BF16s per lane, `16,896 B`) and `ws` (`float ws[kIndexHeads][kWideColumns]`, `4 * 16 * 4 = 256 B`) stores the scalar `weights` per `(head, column)`, multiplying the dot product by `ws[h][c]` after summing over the `32` `uint2` lanes (`sum_d (q_d * w) * k_d == w * sum_d (q_d * k_d)` up to FP32 scaling, or if `qs` stores `uint2` raw BF16s and `widen_lane(q_u2)` is scaled by `w` before `fma`, i.e., `q0.x * w`, it is **100% bitwise-identical** to `bf16_to_f32(q) * w`!).
     - Let's verify that! In `qsa_score_wide_kernel` ([line 356](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L356)), `qs` currently stores `bf16_to_f32(queries[...]) * w`. If `qs` instead stores raw packed BF16 `uint2 qs[kWideQueryFloat4]` (`16,896 B`) and `float ws[kIndexHeads * kWideColumns]` (`256 B`), and inside `wide_leaves` ([line 291](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L291)) each lane widens `const float4 q0 = widen_lane(qs[...])` and multiplies `q0` by `w = ws[...]` (or scales inside `dot4(q0, k0)`), the IEEE-754 operations `fmaf(bf16_to_f32(q) * w, bf16_to_f32(k), sum)` are **100% bitwise identical**!
     - And total shared memory per CTA drops from `42,240 B` to `16,896 + 8,448 + 256 = 25,600 B` (`25.0 KiB`)!
     - At `25.0 KiB` per CTA (`3 * 25.0 KiB = 75.0 KiB <= 99 KiB`) and `49` registers (`3 * 256 * 49 = 37,632 <= 65,536` registers/SM), updating `kWideCtasPerSm = 3` (`__launch_bounds__(256, 3)`) raises occupancy from **2 CTAs/SM (16 warps) to 3 CTAs/SM (24 warps, +50% occupancy)**!
  2. **Heavyweight `cub::BlockScan<int, 512>` in `find_score_bin` (`qsa_select_global_kernel`)**:
     In [`find_score_bin`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L517-L537), a 512-thread `cub::BlockScan` is run over 4 items/thread (`2,048` histogram bins). Replacing `cub::BlockScan<int, 512>` with a lightweight 16-warp `__shfl_up_sync` prefix sum + 16-element warp-0 scan in shared memory removes `SelectShared::Scan` temp storage and cuts barrier overhead in `qsa_select_global_kernel`.

#### Required Implementation Steps
1. **Stage Packed BF16 `uint2` Queries + `float ws[64]` Weights in `qsa_score_wide_kernel`**:
   - In [`qsa_select.cu`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L246-L388):
     - Change `qs` in `qsa_score_wide_kernel` from `float4 qs[kWideQueryFloat4]` (`33,792 B`) to `uint2 qs[kWideQueryFloat4]` (`16,896 B`) plus `float ws[kIndexHeads * kWideColumns]` (`256 B`).
     - Load `queries` via `uint2` (4 BF16s per lane) into `qs[q_index]` and `weights` into `ws[h * kWideColumns + column_offset]`.
     - In [`wide_leaves`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L266-L322), widen `qs` with `widen_lane(...)` and multiply by `w = ws[h * kWideColumns + column_offset]` before `dot4` so the exact same FP32 product `bf16_to_f32(q) * w` is fed into `dot4(q0, k0)` (preserving 100% bitwise parity with existing `wide_leaves`).
     - Update `kWideCtasPerSm` from `2` to `3` (`__launch_bounds__(kWideThreads, 3)`).
2. **Replace `cub::BlockScan` in `find_score_bin` with a 2-Level Warp Prefix Scan**:
   - In [`find_score_bin`](file:///opt/etc/Infernix/src/ops/qsa/qsa_select.cu#L517-L537), compute the 4-bin local prefix sum per thread, run a 32-lane warp inclusive scan via `__shfl_up_sync`, write the 16 warp totals to a 16-int shared array, scan the 16 warp totals in warp 0, and add the warp prefix back after `__syncthreads()`. Remove `cub::BlockScan` and `TempStorage` from `SelectShared`.

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_qsa_attention_bench

# 2. Verify resource usage and run stack check + QSA oracle tests
cuobjdump --dump-resource-usage build-native/apps/infernix-serve | grep -E "qsa_score_wide_kernel|qsa_select_global_kernel"
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa_test" --output-on-failure

# 3. Benchmark QSA select latency across widths and contexts
./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 8192,32768 --width 16,64,512,2048
```
- **Acceptance Criteria**:
  - `cuobjdump --dump-resource-usage` shows `qsa_score_wide_kernel` `SHARED <= 26,000 B` (down from `42,240 B`), `REG <= 64`, `STACK:0`, achieving 3 CTAs/SM.
  - `infernix_qsa_test` and `infernix_cuda_stack_check` pass 100%.
  - `select_us` in `infernix_qsa_attention_bench` at `width=512` and `width=2048` (`context=8192, 32768`) improves by **>= 12%** over baseline (`180.2 us` / `661.5 us` at `C=8192`, `487.8 us` / `1,794.1 us` at `C=32768`).

---

### Task T5 (Priority P1): Share GQA Query/Key Normalization and `QK^T`/`KK^T` Tensor Core MMAs Across Value-Head Triplets in GDN `prepare_kernel` and Store FP16 State Snapshots in `chunk_recurrence_kernel`

#### Affected Files & Symbols
- [prepare.cu](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu)
  - [`prepare_kernel`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L25-L120) (`REG:48, SHARED:12800`)
  - [`launch_prepare`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L122-L129)
- [recurrence.cu](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu)
  - [`RecurrenceShared`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L204-L214)
  - [`chunk_recurrence_kernel`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L217-L345) (`REG:78..128, SHARED:44032`)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Gated DeltaNet (`gated_delta_net`) runs `36×` per forward pass (`75%` of layers in Qwen3.8-Flash-Next and Qwen3.5).
  - Live `./build-native/bench/infernix_gated_delta_net_bench --tokens 2048` (`state_dim=128, qk_heads=16, value_heads=48, batch=1`) on GB10:
    - `T=2048`: `median = 639.968 us/layer` (`23.04 ms` across 36 layers; `~92 ms` at `T=8192`), `logical_gbs = 115.9 GB/s`, `traffic_gbs = 418.2 GB/s` (`CTA traffic = 255.25 MiB` vs `logical = 70.75 MiB` -> **`3.61×` memory traffic amplification**).
- **Root Causes**:
  1. **3× Redundant GQA Query/Key Loads, L2 Norms, and Tensor Core MMAs in `prepare_kernel`**:
     In [`prepare.cu:25-128`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L25-L128), `launch_prepare` launches `grid(chunks, value_heads = 48)`. Because `qk_heads = 16` and `value_heads = 48` (`group = value_heads / qk_heads = 3`), the 3 value heads `head = qh * group + {0, 1, 2}` map to the **exact same** `qh = head / group` ([line 30](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L30)).
     Consequently, 3 separate CTAs load the exact same `args.q` and `args.k` tensors from global memory, compute the exact same L2 normalizations (`q_inv`, `k_inv`) into `sm.q` and `sm.k`, and execute the exact same Tensor Core MMAs for `K * K^T` (`kk[m]`) and `Q * K^T` (`qk[m]`) **3 times redundantly**!
  2. **Unnecessary 32 KiB FP32 `sm.snapshot` and Software FP32->FP16 Re-Packing on the FP16 Fast Path in `chunk_recurrence_kernel`**:
     In [`recurrence.cu:204-291`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L204-L291), `RecurrenceShared<64>` allocates `float snapshot[128 * 64]` (`32,768 B`), bringing total SMEM to `44,032 B`. When `FastPath == true && std::is_same_v<StateHalf, __half>`, `state[idx]` is converted to FP32 `sm.snapshot` before the K-loop ([line 253](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L253)), and then inside every K-step ([lines 274–285](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L274-L285)), `load_f16_b(sm.snapshot, ...)` issues 8 scalar shared memory FP32 loads and 4 `pack_f16x2_sat` conversions per lane! Since `state[idx]` on the FP16 fast path is already quantized to `__half` at chunk boundaries (`state_from_f32<__half>(s)`), storing `sm.snapshot` as `__half` on the fast path cuts `RecurrenceShared<64>` by **`16 KiB` (`44,032 B -> 27,648 B`, enabling 3 CTAs/SM by SMEM)** and eliminates the software FP32->FP16 re-conversion inside the inner loop.

#### Required Implementation Steps
1. **Amortize `prepare_kernel` Across Each GQA Group (`qh = 0..qk_heads - 1`)**:
   - Change [`launch_prepare`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L122-L129) to launch `grid(chunks, args.qk_heads)` (`16` QK heads instead of `48` value heads).
   - Inside [`prepare_kernel`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/prepare.cu#L25-L120):
     - Load `args.q` and `args.k` for `qh = blockIdx.y`, compute L2 normalization (`sm.q`, `sm.k`), and compute the unscaled `K * K^T` (`kk_raw[m]`) and `Q * K^T` (`qk[m]`) Tensor Core MMAs **once** per `qh`.
     - Loop `for (int g_idx = 0; g_idx < group; ++g_idx)` (`head = qh * group + g_idx`):
       - Write `sm.q`, `sm.k`, and `qk[m]` to `args.qk_chunks[(head * chunks + chunk) * kSubchunks + sub]` (from registers/shared memory, without re-reading `args.q`/`args.k` or re-running `mma_sync`),
       - Load `args.g` and `args.beta` for `head`, scale `kk_raw` by `beta_row * expf(g_row - g_col)`, solve the `16×16` forward substitution in warp 0, and write `args.g_cum`, `args.k_beta_g`, and `args.t` for `head`.
   - Note: Because the Q/K normalization and `mma_sync` operations are identical, this produces **100% bitwise-identical** outputs while eliminating `66.7%` of all Q/K global reads, L2 norms, and Tensor Core MMAs in `prepare_kernel`!
2. **Store `sm.snapshot` in `StateHalf` When `FastPath == true` in `chunk_recurrence_kernel`**:
   - In [`RecurrenceShared`](file:///opt/etc/Infernix/src/ops/linear_attention/gated_delta_net/chunked/recurrence.cu#L204-L214), parameterize `RecurrenceShared<TileV, SnapshotT>` with `using SnapshotT = std::conditional_t<FastPath, StateHalf, float>;` (`16 KiB` instead of `32 KiB` for `TileV = 64, FastPath = true`).
   - When `FastPath == true`, store `state[idx]` (`StateHalf`) directly into `sm.snapshot[d * TileV + v]` and load `b0, b1` directly from `sm.snapshot` without redundant `pack_f16x2_sat` conversions on every K-step.

#### Verification & Acceptance Criteria
```bash
# 1. Touch consuming .cu files and build test/bench targets in one invocation
find src/ops/linear_attention/gated_delta_net -name '*.cu' -exec touch {} +
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_state_recurrent_test infernix_gated_delta_net_bench

# 2. Run stack-spill check and Gated DeltaNet oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_gated_delta_net_test" --output-on-failure

# 3. Benchmark Gated DeltaNet across T=16, 64, 512, 2048
./build-native/bench/infernix_gated_delta_net_bench --tokens 16
./build-native/bench/infernix_gated_delta_net_bench --tokens 64
./build-native/bench/infernix_gated_delta_net_bench --tokens 512
./build-native/bench/infernix_gated_delta_net_bench --tokens 2048
```
- **Acceptance Criteria**:
  - `infernix_gated_delta_net_test` and `infernix_cuda_stack_check` pass 100%.
  - `infernix_gated_delta_net_bench` latency at `T=512` and `T=2048` improves by **>= 8%** over the `159.97 us` / `639.97 us` baseline.

---

### Task T6 (Priority P1): Extend Fused `hc_up_collapse` Down to `T = 5..16` with a 16-Column MMA Schedule and Vectorize `gates_kernel` / `collapse_kernel` (`uint4` + PDL)

#### Affected Files & Symbols
- [hyper_connection_up_collapse.cu](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu)
  - [`hc_up_collapse_supported`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L172-L185)
  - [`hyper_connection_up_collapse`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L187-L245)
  - [`hc_up_collapse_mma_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L55-L169)
- [hyper_connection.cu](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu)
  - [`gates_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L502-L521) (`REG:24, SHARED:0`)
  - [`collapse_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L469-L498) (`REG:32, SHARED:0`)
  - [`hyper_connection_mix`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L583-L695)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - `hyper_connection_mix` executes **`97×` per forward pass** (`2×` per layer × `48` layers + final norm).
  - Live `./build-native/bench/infernix_hyper_connection_bench --columns 1,2,3,4,5,6,8,12,16,32,64,128,512,2048` measurements on GB10:
    - `T=1..4` (`hc_mix_fused`): `30.75 us` (`250 GB/s`), `34.78 us` (`221 GB/s`), `38.32 us` (`201 GB/s`), `46.09 us` (`167 GB/s`)
    - `T=5..16` (5-kernel fallback gap: `norm_kernel` -> `linear(down)` -> `gates_kernel` -> `linear(up)` -> `collapse_kernel`): `47.35 us`, `48.05 us`, `49.05 us`, `46.87 us`, `47.12 us` (`485` kernel launches per target verification pass!)
    - `T=128, 512, 2048`: `85.98 us` (`90 GB/s`), `246.54 us` (`31 GB/s`), `971.16 us` (`8 GB/s`)
- **Root Causes**:
  1. **`T = 5..16` Speculative Verification Falls Into the Unfused 5-Kernel Route**:
     Multi-row MTP verification (`B=2..4, W=3..4 -> T=6..16`), DFlash2 verification (`W=5..16`), and N-gram copy verification (`W=6..16`) all hit `columns = 5..16`. Because `hc_mix_fused` caps Q8 at `columns <= 4` (`kHcMixFusedMaxColumnsQ8 = 4`) while [`hc_up_collapse_supported`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L177) requires `columns >= kMinColumns` (`17`), `columns = 5..16` falls into a 5-kernel sequence that materializes `y` (`[10240, T]` BF16) to global memory in `linear(up)` and immediately re-reads it in `collapse_kernel` (`485` launches per forward pass instead of `291`).
  2. **Scalar 16-bit Global Memory Accesses and Missing PDL in `gates_kernel` and `collapse_kernel`**:
     In [`gates_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L502-L521) (which runs on **every** `T >= 5` call, `97×` per prefill chunk or verification pass!), 128 threads read `z[j]` (`[320, T]` BF16) using scalar 16-bit loads in a loop to compute the sum of squares, then **re-read `z[j]` a second time** from memory to write `p[j]` with scalar 16-bit stores, and `gates_kernel` is launched with raw `<<<...>>>` instead of `pdl::launch_consumer`!
     Similarly, [`collapse_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L469-L498) reads `x[s]` and `y[s]` with scalar BF16 loads and lacks `pdl::launch_consumer`.

#### Required Implementation Steps
1. **Add a 16-Column MMA Tile Schedule (`MmaR64x16T16`) to `hc_up_collapse` for `columns = 5..16`**:
   - In [`hyper_connection_up_collapse.cu`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L47-L53), define:
     ```cpp
     using MmaR64x16T16 = linear::bf16_detail::Bf16MmaSchedule<64, 16, 16, 64, 2, 2>;
     ```
     (`RowsPerBlock = 64` per stream -> `2560 / 64 = 40` row blocks = `40` CTAs for `columns = 5..16`, `TileColumns = 16`, `Warps = 2`).
   - Lower `kMinColumns` in [`hc_up_collapse_supported`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection_up_collapse.cu#L172-L185) from `17` to `5` (`kHcMixFusedMaxColumnsQ8 + 1`), and dispatch `columns <= 16` to `hc_up_collapse_mma_kernel<MmaR64x16T16>`.
   - This eliminates `linear(up)` + `collapse_kernel` at `T = 5..16`, removing `194` kernel launches per verification pass and eliminating the `[10240, T]` `y` global round-trip!
2. **Vectorize `gates_kernel` with Register-Cached `uint4` Loads/Stores and `pdl::launch_consumer`**:
   - In [`gates_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L502-L521), since `r` is a multiple of 32 (`r = 320` on Qwen3.8-Flash-Next, `320 / 8 = 40` `uint4` vectors per column), use a 64-thread (2-warp) or 32-thread (1-warp) CTA when `r % 8 == 0`:
     - Have `pdl::enter()` at kernel entry and launch with `pdl::launch_consumer`.
     - Each thread loads its `uint4` chunks (`8` BF16s) from `z + t * (r + s)` **once into registers**, accumulates `local_sq`, reduces `local_sq` across the CTA, converts the register-resident values `zf * inv` to `uint4` BF16, and writes `p + t * r` via 128-bit `uint4` stores (`pdl::exit()`).
3. **Vectorize `collapse_kernel` (`uint4` + `pdl::launch_consumer`)**:
   - In [`collapse_kernel`](file:///opt/etc/Infernix/src/ops/hyper_connection/hyper_connection.cu#L469-L498), when `d % 8 == 0`, have each thread process 8 consecutive channels (`uint4`) across `i = 0..s-1` (`s = 4`), loading `h_pre[i * columns + t]` once per 8 channels and issuing `uint4` loads from `x`/`y` and `uint4` stores to `branch`, launched via `pdl::launch_consumer`.

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_norm_elementwise_test infernix_hyper_connection_bench

# 2. Run stack-spill check and hyper-connection oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_hyper_connection_test" --output-on-failure

# 3. Benchmark hyper_connection_mix across T=1..2048
./build-native/bench/infernix_hyper_connection_bench --columns 1,2,3,4,5,6,8,12,16,32,64,128,512,2048
```
- **Acceptance Criteria**:
  - `infernix_hyper_connection_test` and `infernix_cuda_stack_check` pass 100%.
  - `infernix_hyper_connection_bench` latency at `T = 5, 6, 8, 12, 16` improves by **>= 15%** over the `46.87–49.05 us` baseline, and `T = 32..2048` improves or does not regress.

---

### Task T7 (Priority P1): Fuse Decode `ple_gate` + `ple_conv_inject` into a Single Kernel Launch (`width == 1`) and Add PDL Overlap Across All PLE Kernels

#### Affected Files & Symbols
- [ple.cu](file:///opt/etc/Infernix/src/ops/ple/ple.cu)
  - [`gate_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L257-L328) (`REG:46, SHARED:32`)
  - [`conv_state_vec8_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L438-L523) (`REG:40, SHARED:0`)
  - [`conv_vec8_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L387-L436) (`REG:32, SHARED:0`)
  - [`state_vec8_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L557-L586) (`REG:27, SHARED:0`)
  - [`embed_nvfp4_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L155-L216) (`REG:32, SHARED:0`)
  - [`ple_gate`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L332-L385), [`ple_conv_inject`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L588-L680)
- [ple.h](file:///opt/etc/Infernix/include/infernix/ops/ple.h)
- [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp)
  - [`Forward::ple`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L569-L629)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - `Forward::ple` executes **`48×` per forward pass** (every layer of `Qwen4ExpForCausalLM`).
  - At every layer ([`forward.cpp:616-625`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L616-L625)), `Forward::ple` launches `ops::ple_gate` followed immediately by `ops::ple_conv_inject` (`96` kernel launches per forward pass, plus `48` `ple_embed` + `48` `linear(ple_proj)` = `192` launches).
- **Root Causes**:
  1. **Unnecessary 2-Kernel Split and Global Round-Trip at Decode (`width == 1, update == true`)**:
     Look at what [`gate_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L257-L328) and [`conv_state_vec8_kernel`](file:///opt/etc/Infernix/src/ops/ple/ple.cu#L438-L523) do when `width == 1`:
     - `gate_kernel` (`grid(streams=4, sequences)`, `block(128)`) loads `residual` (`[10240]` BF16) into registers `qf[i][0..7]`, computes `scaled[u] = gate * vf[i][u]` and `out[u] = f32_to_bf16((scaled[u] * inv_rms) * bf16_to_f32(scale[u]))`, and writes `gated` (`20 KiB` per token) and `normalized` (`20 KiB` per token) to global memory (`work_.ple_gated` and `work_.ple_normalized`).
     - Immediately afterward, `conv_state_vec8_kernel` (`grid(5, sequences)`, `block(256)`) reads `gated` (`20 KiB`), `normalized` (`20 KiB`), and **re-reads `residual` (`20 KiB`)** from global memory just to compute a strictly channel-independent 1D depthwise convolution over `[past_state(taps-1), normalized]` and write `residual = f32_to_bf16(residual + gated + acc * sigmoid(acc))` and the shifted `conv_state`!
     - Because the depthwise convolution is 100% channel-independent (`c = s * hidden + d`, where `s = 0..3` and `d = 0..2559`), at `width == 1` each thread in `gate_kernel` (`s = blockIdx.x`, `t = blockIdx.y`, processing `i = 0..2` slices of 8 channels `d = tid * 8 + i * 1024`) **already holds `qf[i][u]` (`residual`), `f32_to_bf16(scaled[u])` (`gated`), and `out[u]` (`normalized`) in registers**!
     - Fusing `conv_state_vec8_kernel` directly into `gate_kernel` for `width == 1` eliminates **48 kernel launches per decode step**, eliminates the `work_.ple_gated` and `work_.ple_normalized` global memory writes and reads (`80 KiB/layer = 3.84 MiB` per decode step), and eliminates the duplicate `residual` global memory read (`40 KiB/layer = 1.92 MiB`)!
  2. **Zero Programmatic Dependent Launch (`pdl`) Usage Across All PLE Kernels**:
     None of the kernels in [`ple.cu`](file:///opt/etc/Infernix/src/ops/ple/ple.cu) (`embed_nvfp4_kernel`, `gate_kernel`, `conv_vec8_kernel`, `state_vec8_kernel`, `conv_state_vec8_kernel`) use `pdl::launch_consumer` / `pdl::enter()` / `pdl::exit()`, forcing full GPU pipeline drains between `ple_embed -> linear(ple_proj) -> ple_gate -> ple_conv_inject` across all 48 layers (`192` launches per prefill/verification step).

#### Required Implementation Steps
1. **Fuse `ple_gate` + `ple_conv_inject` for `width == 1` (or Provide `ops::ple_gate_conv_inject` / Fused Path)**:
   - Add a fused kernel `gate_conv_state_vec8_kernel<SPAN>` (`grid(streams, sequences)`, `block(128)`) in [`ple.cu`](file:///opt/etc/Infernix/src/ops/ple/ple.cu) (invoked when `width == 1 && update == true && hidden % 8 == 0`, either via a new `ops::ple_gate_conv_inject` entrypoint called from [`Forward::ple`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L616-L625) and [`ple_test.cpp`](file:///opt/etc/Infernix/tests/ops/ple_test.cpp) or dispatched automatically):
     - Compute `gate`, `scaled[u]`, `gated_bf16 = f32_to_bf16(scaled[u])`, and `norm_bf16 = f32_to_bf16((scaled[u] * inv_rms) * bf16_to_f32(w[u]))` in registers using the exact same rounding boundaries as `gate_kernel`.
     - For each slice `c = s * hidden + d`, load `past[0..SPAN-1]` (`uint4` from `state`) and `weight[0..SPAN]` (`uint4` from `conv_weight`), accumulate `acc[u]` over `past` and `norm_bf16`, write the shifted `conv_state` (`past[1..SPAN-1]` + `norm_bf16`), and write `residual` (`f32_to_bf16(qf[i][u] + bf16_to_f32(gated_bf16) + acc[u] * (1.0f / (1.0f + expf(-acc[u]))))`) via `uint4` store!
     - Because every intermediate `f32_to_bf16` rounding boundary (`gated_bf16` and `norm_bf16`) is preserved identically in registers, the fused kernel is **100% bitwise-identical** to `ple_gate` + `ple_conv_inject`!
2. **Enable PDL (`pdl::launch_consumer`, `pdl::enter()`, `pdl::exit()`) Across All PLE Kernels**:
   - Include `"ops/common/pdl.cuh"` in [`ple.cu`](file:///opt/etc/Infernix/src/ops/ple/ple.cu) and wrap `embed_nvfp4_kernel`, `gate_kernel`, `gate_conv_state_vec8_kernel`, `conv_vec8_kernel`, `state_vec8_kernel`, and `conv_state_vec8_kernel` with `pdl::enter()` / `pdl::exit()` and `pdl::launch_consumer`.

#### Verification & Acceptance Criteria
```bash
# 1. Build test and server binaries in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_norm_elementwise_test

# 2. Run stack-spill check and PLE oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_ple_test" --output-on-failure
```
- **Acceptance Criteria**:
  - `infernix_ple_test` (including both `width == 1` decode and `width > 1` prefill/chunked equivalence) and `infernix_cuda_stack_check` pass 100%.
  - At `width == 1, update == true`, `Forward::ple` launches 1 fused kernel instead of 2 (`48` fewer kernel launches and `5.76 MiB` fewer global memory transfers per decode step).

---

### Task T8 (Priority P1): Vectorize `q8_dynamic_grouped_conv_add` `finish_kernel` (`uint4`), Fix the `W=16, B=3..4` Schedule Cliff, and Vectorize `dynamic_grouped_conv_prepare_reduce_kernel`

#### Affected Files & Symbols
- [q8_dynamic_grouped_conv_add_materialized.cu](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu)
  - [`finish_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu#L63-L82) (`REG:20, SHARED:0`)
  - [`tiled_projection`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu#L91-L132)
- [q8_dynamic_grouped_conv_add_plan.cpp](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.cpp)
  - [`resolve_plan`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.cpp#L28-L59)
- [bf16_dynamic_grouped_conv_prepare_reduce.cu](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu)
  - [`dynamic_grouped_conv_prepare_reduce_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L15-L47) (`REG:38, SHARED:2304`)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Every DFlash2 speculative draft round executes `5` or `6` layers × `2` = **`10–12` calls** to `rmsnorm_dynamic_grouped_conv_prepare` and **`10–12` calls** to `linear_dynamic_grouped_conv_add`.
  - Live `./build-native/bench/infernix_linear_dynamic_grouped_conv_add_bench` measurements on GB10 reveal a severe latency and tail-latency cliff at `C=4096, W=16, B=3..4` (`T=48..64`):
    - `C=4096, W=16, B=1 (T=16)`: `118.43 us` (`205.8 GB/s`)
    - `C=4096, W=16, B=2 (T=32)`: `131.78 us` (`182.9 GB/s`)
    - `C=4096, W=16, B=3 (T=48)`: **`139.17 us` (`p95 = 232.35 us`, `170.7 GB/s`)**
    - `C=4096, W=16, B=4 (T=64)`: **`166.18 us` (`p95 = 283.55 us`, `145.9 GB/s`)**
- **Root Causes**:
  1. **Scalar 1-BF16-per-Thread `finish_kernel<Rows>` in `q8_dynamic_grouped_conv_add_materialized.cu`**:
     Look at [`finish_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu#L63-L82):
     ```cpp
     const int row = static_cast<int>(blockIdx.x) * kFinishThreads + static_cast<int>(threadIdx.x);
     ```
     Each thread processes **1 scalar `__nv_bfloat16` (`2 bytes`)**, issuing up to 6 scalar 16-bit loads (`projected[index]`, `projected[index - Rows]`, `base[2*Rows + row]`, `base[3*Rows + row]`, plus `delta[di]` and `delta[di + kGroups]`) and 1 scalar 16-bit RMW on `residual[index]`!
     Because `Rows` (`2560`, `3072`, `5120`) is always a multiple of 16 (`kGroupChannels = 16`), 8 consecutive channels (`row0 = 8 * tid`) lie entirely within the **same** 16-channel group `di = (col * kGroups) + (row0 / 16)`. Vectorizing `finish_kernel` to `uint4` (`8` BF16s per thread) cuts CTA count by `8×` (`10 -> 2` CTAs per column), loads `delta[di]` and `delta[di + kGroups]` once per 8 channels, and converts all `projected`, `base`, and `residual` memory accesses into 128-bit `uint4` transactions!
  2. **Suboptimal `S::TiledMma` Plan Selection at `T = 41..64` for `C <= 12288`**:
     In [`resolve_plan`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.cpp#L41-L45), `columns <= 64` for `C = 4096, 8192, 9216, 12288` selects `S::TiledMma`, which at `T = 48..64` dispatches `launch_q8_a16_mma_r64x32_t48_w4_b2_a1` / `t64_w4_b3_a1` (`40` CTAs = `< 1` wave on 48 SMs with high register pressure `REG:96`, throttling occupancy). Meanwhile, `S::MmaK128` (`launch_q8_a16_mma_r64x32_t64_k128_a1`, already used for `C = 17408` at `32 < columns <= 64`) or `S::Mma` is already compiled for all `(H, C)` in [`tiled_projection`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu#L91-L132).
  3. **Scalar 16-bit Channel Accesses in `dynamic_grouped_conv_prepare_reduce_kernel`**:
     In [`bf16_dynamic_grouped_conv_prepare_reduce.cu:30-46`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L30-L46), each thread processes 1 scalar BF16 channel (`channel = tid % 16`). Processing 2 (`__nv_bfloat162`) or 4 (`uint2`) channels per thread halves CTA threads (`256 -> 128`) and coalesces `base` reads and `prepared` writes.

#### Required Implementation Steps
1. **Vectorize `finish_kernel<Rows>` to `uint4` (8 BF16s per Thread)**:
   - In [`q8_dynamic_grouped_conv_add_materialized.cu:63-89`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu#L63-L89), have each thread handle `row0 = (blockIdx.x * kFinishThreads + threadIdx.x) * 8` (`8` BF16 channels = `1` `uint4` within a single 16-channel group `row0 / 16`), reading `delta[di]` and `delta[di + kGroups]` once, loading `projected`, `base`, and `residual` as `uint4`, and storing `residual` as `uint4`.
2. **Sweep and Retune `resolve_plan` for `32 < columns <= 64` in `q8_dynamic_grouped_conv_add_plan.cpp`**:
   - Benchmark `S::Mma`, `S::MmaK128`, and `S::TiledMma` across `C = 4096..17408` at `T = 33..64` using `infernix_linear_dynamic_grouped_conv_add_bench`, and update [`resolve_plan`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.cpp#L41-L45) so `T = 48..64` uses the fastest schedule without the `166.2 us` (`p95 = 283.6 us`) cliff.
3. **Vectorize Channel Writes in `dynamic_grouped_conv_prepare_reduce_kernel`**:
   - In [`bf16_dynamic_grouped_conv_prepare_reduce.cu`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L15-L60), vectorize the `channel` dimension by 2 (`__nv_bfloat162`, `128` threads/CTA = `8` channels × `16` groups) or 4 (`uint2`), preserving exact FP32 accumulation of `delta`.

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_conv_vision_test infernix_dynamic_grouped_conv_prepare_bench infernix_linear_dynamic_grouped_conv_add_bench

# 2. Run stack-spill check and dynamic grouped conv oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_rmsnorm_dynamic_grouped_conv_prepare_test|infernix_linear_dynamic_grouped_conv_add_test" --output-on-failure

# 3. Run dynamic grouped conv benchmarks
./build-native/bench/infernix_dynamic_grouped_conv_prepare_bench --width 8,16 --batch 1,2,3,4
./build-native/bench/infernix_linear_dynamic_grouped_conv_add_bench --in-features 4096 --width 8,16 --batch 1,2,3,4
```
- **Acceptance Criteria**:
  - All `dynamic_grouped_conv` oracle tests and `infernix_cuda_stack_check` pass 100%.
  - `infernix_linear_dynamic_grouped_conv_add_bench` at `C=4096, W=16, B=3..4` (`T=48, 64`) improves by **>= 10%** over the `139.2 us` / `166.2 us` baseline, and `infernix_dynamic_grouped_conv_prepare_bench` improves or does not regress.

---

### Task T9 (Priority P2): Eliminate the `U = 33..40` and `U = 97..120` Schedule Cliffs in `linear_topk` and Fuse Target-Layer `hyper_connection_stream_mean` into `moe_combine_inject` for DFlash2

#### Affected Files & Symbols
- [linear_topk.cpp](file:///opt/etc/Infernix/src/ops/linear_topk/linear_topk.cpp)
  - [`plan_for`](file:///opt/etc/Infernix/src/ops/linear_topk/linear_topk.cpp#L87-L128)
- [moe_combine_inject.cu](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_combine_inject.cu)
  - [`moe_combine_inject_vec8_kernel`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_combine_inject.cu#L50-L95) (`REG:40, SHARED:0`)
- [forward.cpp](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp)
  - [`Forward::layer`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L325-L355)
  - [`Forward::capture_draft_features`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L540-L567)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Live `./build-native/bench/infernix_linear_topk_bench` measurements on GB10 expose two severe schedule cliffs in [`plan_for`](file:///opt/etc/Infernix/src/ops/linear_topk/linear_topk.cpp#L87-L128):
    1. **`q8-full` at `U = 33..40` (`{64, 40}`)**:
       - `U = 32` (`{128, 32}`): `6,401.5 us` (`211.1 GB/s`)
       - `U = 38, 39` (`{64, 40}`): **`14,118.9 us` (`95.7 GB/s`) — `2.23×` slower than `U = 41`!**
       - `U = 41..48` (`{64, 48}`): `6,287.4–6,317.1 us` (`214.0–215.0 GB/s`)
    2. **`q4-optimized` at `U = 97..120` (`{64, 112, 64}` and `{64, 120, 64}` with `block_k = 64`)**:
       - `U = 96` (`{64, 96, 128}`): `2,705.1 us` (`131.8 GB/s`)
       - `U = 97..112` (`{64, 112, 64}`): **`3,841.9 us` (`92.8 GB/s`) — `42%` slower!**
       - `U = 113..120` (`{64, 120, 64}`): **`4,498.1 us` (`79.3 GB/s`) — `66%` slower!**
  - Furthermore, in [`Forward::layer`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L344-L348), whenever DFlash2 speculation is active (`batch.dflash_features.data != nullptr`), the target model calls `capture_draft_features(layer_index, x, batch, s)` at 5 tapped layers (`layers 1, 12, 23, 34, 45`) immediately after `moe_combine_inject`. Each call launches a standalone `ops::hyper_connection_stream_mean` kernel (`5` extra kernel launches per target verification pass) that re-reads all 4 hyper-connection streams `x` (`[2560, 4, T]` BF16 = `5 × 20 KiB × T`) that `moe_combine_inject_vec8_kernel` just wrote to global memory!

#### Required Implementation Steps
1. **Eliminate the `{64, 40}` and `block_k = 64` Cliffs in `linear_topk.cpp` (`plan_for`)**:
   - In [`plan_for`](file:///opt/etc/Infernix/src/ops/linear_topk/linear_topk.cpp#L99-L128):
     - For `HeadProfile::Q8Full` and `HeadProfile::Fp8Full`, route `columns = 33..48` directly to `{64, 48}` (removing the `{64, 40}` bucket that drops `Q8Full` to `95.7 GB/s` at `U = 38..39`, whereas `{64, 48}` runs at `214.0 GB/s`).
     - For `HeadProfile::Q4Optimized`, test `block_k = 128` (`{64, 112, 128}`, `{64, 120, 128}`, `{64, 128, 128}`) vs `block_k = 64` at `U = 97..128` in `infernix_linear_topk_bench` and select the `block_k` that eliminates the `2.70 ms -> 3.84–4.50 ms` jump.
2. **Fuse Optional `stream_mean` Output into `moe_combine_inject`**:
   - Add an optional `Tensor* stream_mean_out = nullptr` parameter (or overload) to `ops::moe_combine_inject` in [offloaded_sparse_moe.h](file:///opt/etc/Infernix/include/infernix/ops/offloaded_sparse_moe.h) and [`moe_combine_inject.cu`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_combine_inject.cu#L50-L95):
     - Inside `moe_combine_inject_vec8_kernel`, each thread already computes the updated BF16 values `x_new[s]` for all `s = 0..3` streams (`8` channels `uint4` per thread) in registers before writing `x`!
     - When `stream_mean_out != nullptr`, accumulate the FP32 sum of the 4 rounded BF16 stream values `0.25f * (bf16_to_f32(x_new[0]) + bf16_to_f32(x_new[1]) + bf16_to_f32(x_new[2]) + bf16_to_f32(x_new[3]))` in registers and write the `uint4` BF16 mean directly to `stream_mean_out`!
     - Update [`Forward::layer`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L332-L348) to pass the tapped layer's `dflash_features` slice directly into `moe_combine_inject`, eliminating all 5 `hyper_connection_stream_mean` kernel launches and `5 × [2560, 4, T]` global memory reads per verification pass!

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_sampling_Penalty_test infernix_offloaded_sparse_moe_test infernix_linear_topk_bench

# 2. Run stack-spill check and oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_linear_topk_test|infernix_offloaded_sparse_moe_test" --output-on-failure

# 3. Benchmark linear_topk across the cliff regions (U=32..48 and U=96..120)
./build-native/bench/infernix_linear_topk_bench --profile q8-full --columns 32,36,38,39,40,41,48
./build-native/bench/infernix_linear_topk_bench --profile q4-optimized --columns 96,97,104,112,113,120
```
- **Acceptance Criteria**:
  - `infernix_linear_topk_test`, `infernix_offloaded_sparse_moe_test`, and `infernix_cuda_stack_check` pass 100%.
  - `infernix_linear_topk_bench` `q8-full` at `U = 38, 39` drops from `14,118.9 us` (`95.7 GB/s`) to **<= 6,500 us (`>= 205 GB/s`, > 2.1× speedup)**.

---

### Task T10 (Priority P2): Fuse `qsa_pool_keys` `pool_kernel` + `tail_vec8_kernel` into a Single Kernel Launch and Eliminate Shared Memory / Barriers in `index_query_kernel`

#### Affected Files & Symbols
- [qsa.cu](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu)
  - [`index_query_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L105-L132) (`REG:32, SHARED:16`)
  - [`pool_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L140-L183) (`REG:32, SHARED:16`)
  - [`tail_vec8_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L204-L234) (`REG:26, SHARED:0`)
  - [`qsa_index_query`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L1058-L1094)
  - [`qsa_pool_keys`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L1096-L1141)

#### Root Cause & Hardware Rationale
- **Measured Evidence**:
  - Every QSA layer (`12×` per forward pass in Qwen3.8-Flash-Next + `1×` in MTP drafter = `13×` per round) calls `qsa_index_query` and `qsa_pool_keys`.
  - Look at [`qsa_pool_keys`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L1096-L1141): when `update_tails` is true (which is true on every normal prefill, decode, and verification pass), `qsa_pool_keys` launches **two separate kernels** per QSA layer: `pool_kernel` (`grid(columns), block(128)`) followed by `tail_vec8_kernel` (`grid(batch), block(32)`).
- **Root Causes**:
  1. **`pool_kernel` and `tail_vec8_kernel` Operate on 100% Disjoint Columns of the Same Input Tensor**:
     Look at [`pool_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L140-L183):
     - It launches `grid(columns)` (`1` CTA per column `t = 0..columns - 1`), reads `p = positions[t]`, and at [line 154](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L154) checks:
       ```cpp
       if ((p + 1) % ratio != 0) { return; }
       ```
       With `ratio = 4`, **75% of all CTAs launched by `pool_kernel` immediately exit without doing any work** (and at plain decode `width = 1`, on 3 out of every 4 tokens, **100% of `pool_kernel` CTAs exit without doing any work**!).
     - Now look at [`tail_vec8_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L204-L234): for each `sequence`, `last = positions[sequence * width + width - 1]`, `base = last - last % ratio`, and `end = min(last, base + ratio - 2)`. Which columns `t` of `sequence` get copied into `tails`? Only those columns `t` in `sequence` whose position `p = positions[t]` satisfies `p >= base && p <= end`!
     - And since `end <= base + ratio - 2`, for every such column `t`, `p % ratio <= ratio - 2 < ratio - 1`, which mathematically guarantees **`(p + 1) % ratio != 0`**!
     - That means the CTAs `t` in `pool_kernel` that satisfy `p >= base && p <= end` are **always** among the idle CTAs that currently hit `if ((p + 1) % ratio != 0) { return; }`!
     - Having those otherwise-idle CTAs in `pool_kernel` write `tails[(tail_slots[sequence] * (ratio - 1) + (p - base)) * kIndexDim + d] = raw[t * kIndexDim + d]` when `update_tails` is true **completely eliminates `tail_vec8_kernel` / `tail_kernel` (`12–13` kernel launches per forward pass = `100%` of tail launches)** with zero extra CTAs and zero extra threads!
  2. **Scalar 16-bit Loads/Stores and Multi-Warp `__syncthreads()` in `index_query_kernel` and `pool_kernel`**:
     In [`index_query_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L105-L132), 128 threads (`4` warps) are launched per `(column, head)` (`kIndexDim = 128`), requiring shared memory `block_sum` and `__syncthreads()` just to normalize and apply RoPE to 128 BF16 elements (`256 B`). A single warp (`32` threads, `4` elements = `uint2` per thread: `2` low-half elements `d = 2*lane + {0, 1}` and their `2` corresponding high-half RoPE partners `d + 64`) can load `raw_queries`, `norm_weight`, and `rope_cos_sin`, compute the L2 norm via a single warp reduction `warp_sum` (**zero shared memory, zero `__syncthreads()`!**), apply RoPE in registers, and write `queries` via coalesced vector stores!

#### Required Implementation Steps
1. **Fuse `tail_vec8_kernel` into `pool_kernel` in `qsa_pool_keys`**:
   - Pass `update_tails` (`bool`) into [`pool_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L140-L183).
   - Inside `pool_kernel`, when `(p + 1) % ratio != 0`:
     ```cpp
     if ((p + 1) % ratio != 0) {
         if (update_tails) {
             const int last = positions[sequence * width + width - 1];
             if (last >= 0) {
                 const int base = last - (last % ratio);
                 const int end  = min(last, base + ratio - 2);
                 if (p >= base && p <= end) {
                     const std::size_t dst = (static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (p - base)) * kIndexDim;
                     tails[dst + d] = raw[static_cast<std::size_t>(t) * kIndexDim + d];
                 }
             }
         }
         return;
     }
     ```
   - Remove the separate `tail_vec8_kernel` / `tail_kernel` launch from [`qsa_pool_keys`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L1096-L1141).
2. **Restructure `index_query_kernel` to 1 Warp (32 Threads, 4 Channels/Thread, Zero Shared Memory/Barriers)**:
   - In [`index_query_kernel`](file:///opt/etc/Infernix/src/ops/qsa/qsa.cu#L105-L132), launch `block(32)` (or `block(32, kIndexHeads)` = 128 threads covering all 4 index heads in 1 CTA per column, reducing grid size by `4×`!).
   - Each warp of 32 threads handles one index head (`kIndexDim = 128`): lane `l = 0..31` loads `d0 = 2*l, d1 = 2*l + 1` (`__nv_bfloat162`) and `d0 + 64, d1 + 64` (`__nv_bfloat162`), computes `local_sq` over its 4 elements, reduces via `warp_sum(local_sq)` (no `__syncthreads()`, no shared memory), applies RMSNorm + RoPE to its `(d, d + 64)` pairs directly in registers, and stores `__nv_bfloat162` to `queries`.

#### Verification & Acceptance Criteria
```bash
# 1. Build test and benchmark targets in one invocation
ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_qsa_attention_bench

# 2. Run stack-spill check and QSA oracle tests
ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa_test" --output-on-failure

# 3. Benchmark QSA across decode and prefill widths
./build-native/bench/infernix_qsa_attention_bench --kv-dtype int8,bf16 --context 8192 --width 1,8,16,64,512
```
- **Acceptance Criteria**:
  - `infernix_qsa_test` (including `pool_keys` tail state updates across partial and full blocks) and `infernix_cuda_stack_check` pass 100%.
  - `qsa_pool_keys` launches 1 kernel instead of 2 (`12–13` fewer kernel launches per forward pass), and `index_query_kernel` uses `0` shared memory bytes and `0` CTA barriers.
