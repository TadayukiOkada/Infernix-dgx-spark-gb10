# GB10 (`sm_121a`) Performance Optimization Task Roadmap (2026-11-06)

**Target Architecture:** NVIDIA GB10 (`sm_121a`, 48 SMs, 99 KiB SMEM/SM, 65,536 32-bit registers/SM, 24 MiB L2, 128 GiB Unified LPDDR5X UMA at ~273 GB/s, 20-core Grace ARMv9.2 CPU)
**Baseline Commit:** `09f38a502` (`dev-rebased-20261028`, post-T1..T10 campaign)
**Focus Areas:** Wrong memory usage (uncoalesced strided accesses, oversized shared-memory planes, L2-thrashing CTA schedules), redundant operations (dead position loops, redundant split-KV `expf` passes, 15-barrier reductions), redundant memory transfers (`FP32` global scratch round-trips), and double dequantization / double precision casts (`int8 -> BF16` dual SMEM buffers, full-vocab `FP32 -> BF16 -> FP32` casts).

---

### Task T1 (Priority P0): Eliminate `key_scratch` Global-Memory Round-Trip, Dual `code_values`/`codes` SMEM Buffering, and `context_kv_key_post_kernel` Launch in `context_kv_materialize`

#### Affected Files & Symbols
- [`src/ops/context_kv_materialize/materialize.cu`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu)
  - [`MaterializeStorage`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L79-L92)
  - [`decode_signed_codes`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L186-L197)
  - [`MaterializeProjectionEpilogue`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L289-L309)
  - [`context_kv_mma_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L312-L355)
  - [`context_kv_grouped_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L358-L418)
  - [`context_kv_key_post_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L420-L437)
  - [`launch_mma`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L468-L491)
  - [`launch_projection`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L494-L523)
  - [`context_kv_materialize_launch`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L533-L579)
- [`src/ops/context_kv_materialize/launch.h`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/launch.h)
  - [`ContextKVMaterializeRoute`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/launch.h#L16-L25)
  - [`context_kv_materialize_route`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/launch.h#L27-L49)
  - [`context_kv_materialize_uses_scratch`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/launch.h#L51-L55)
- [`src/ops/context_kv_materialize/context_kv_materialize.cpp`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/context_kv_materialize.cpp)
  - [`context_kv_materialize_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/context_kv_materialize.cpp#L47-L56)

#### Root Cause & Hardware Rationale
1. **Redundant `FP32` `key_scratch` global memory materialization and second kernel launch (`graph_nodes = 2`):**
   - In [`MaterializeProjectionEpilogue::finish`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L294-L308), only `Rows == kHeadDim` (`128`) fuses per-head Key RMSNorm + RoPE + `BF16` cache write (`store_key_head`) inside the GEMM epilogue (`Route::Fused64`, active only for `columns = 97..256`).
   - Every other route (`KSplit16` for `17..20`, `KSplit24` for `21..32`, `Mma32` for `33..64`, `Mma80` for `65..80`, `Mma96` for `81..96`, and `Mma64` for `> 256`) uses `Rows = 64` or `16`. As a result, all Key CTAs (`50%` of the grid) write uncompressed `FP32` intermediate keys to global memory `key_scratch` (`[1024, max_count * batch, 5]`), and [`context_kv_materialize_launch`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L557-L567) launches a second kernel [`context_kv_key_post_kernel`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L420-L437) that re-reads the entire `FP32` `key_scratch` buffer from L2/LPDDR5X just to apply RMSNorm + RoPE and write `BF16` to `layer.cache_k`.
2. **Redundant dual `codes` + `code_values` shared-memory allocation preventing `Rows = 128` across MMA routes:**
   - In [`MaterializeStorage`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L81-L86), `mainloop` allocates both `__nv_bfloat16 code_values[Rows][BlockK]` (`2 * Rows * BlockK` bytes) and `std::uint8_t codes[Rows][BlockK]` (`Rows * BlockK` bytes) separately, plus `float scales[Rows][BlockK / kGroupSize]`.
   - For `Rows = 128, BlockK = 128`, `codes[128][128]` wastes `16,384` bytes of SMEM because `cp.async` loads raw `uint8_t` codes before [`decode_signed_codes`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L186-L197) converts them to `__nv_bfloat16` after `cp_async_wait_all()` + `__syncthreads()`. Staging `codes` into the upper half of `code_values` (or unpacking in-place per 16-byte chunk in registers) saves `16 KiB` of SMEM, allowing `Rows = 128` (`kHeadDim`) to fit well inside GB10's `99 KiB` SMEM limit for **all** MMA routes (`Mma32`, `Mma64`, `Mma80`, `Mma96`, `Fused64`, and a `Rows = 128, Columns = 16/24` MMA route for `17..32`).
3. **Live GB10 Evidence Collected in This Session:**
   - Command: `cuobjdump --dump-resource-usage build-native/apps/infernix-serve`
     - `context_kv_mma_kernel<5120, 128, 64, 128, 2>`: `REG:128 STACK:0 SHARED:4` (dynamic SMEM `66,560` B)
     - `context_kv_mma_kernel<5120, 64, 96, 128, 3>`: `REG:123 STACK:0 SHARED:4` (dynamic SMEM `49,920` B)
     - `context_kv_mma_kernel<5120, 64, 32, 128, 1>`: `REG:80 STACK:0 SHARED:4` (dynamic SMEM `33,280` B)
     - `context_kv_key_post_kernel`: `REG:32 STACK:0 SHARED:544`
   - Command: `./build-native/bench/infernix_context_kv_materialize_bench`
     - `W=16, B=2 (T=32, KSplit16, 2 graph_nodes, 655 KB scratch)`: `median = 339.680 us` (`min = 268.096 us`) — **28.5 µs slower** than `W=16, B=3 (T=48, Mma32, 2 graph_nodes, 983 KB scratch)` at `311.136 us`!
     - `W=8, B=8 (T=64, Mma32, 2 graph_nodes, 1.31 MB scratch)`: `median = 388.576 us`
     - `W=12, B=8 (T=96, Mma96, 2 graph_nodes, 1.97 MB scratch)`: `median = 425.664 us` (`min = 420.128 us`)
     - `W=13, B=8 (T=104, Fused64, 1 graph_node, 0 B scratch)`: `median = 422.080 us` (`min = 417.504 us`) — faster than `T=96` despite 8.3% more tokens because `Fused64` eliminates `1.97 MiB` of `key_scratch` traffic and the second kernel launch!
     - `W=128, B=8 (T=1024, Mma64, 2 graph_nodes, 20.97 MB scratch)`: `median = 1974.640 us`
     - `W=256, B=8 (T=2048, Mma64, 2 graph_nodes, 41.94 MB scratch)`: `median = 3981.328 us`

#### Required Implementation Steps
1. **Eliminate dual `codes` + `code_values` SMEM plane in [`MaterializeStorage`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L79-L92):**
   - In `MaterializeStorage<Rows, Columns, BlockK>`, remove the separate `std::uint8_t codes[Rows][BlockK]` array by staging raw `uint8_t` codes via `cp_async_ca_shared_global_16` into the second half of `code_values` (offset `Rows * BlockK` bytes) and having `decode_signed_codes` read `uint8_t` from that upper half (into a 16-byte register buffer per thread before writing `nv_bfloat162` to `code_values`), saving `Rows * BlockK` bytes (`16 KiB` at `Rows=128, BlockK=128`; `8 KiB` at `Rows=128, BlockK=64`).
2. **Promote MMA routes (`Mma32`, `Mma64`, `Mma80`, `Mma96`) and `T = 17..32` to fused `Rows = 128` (`kHeadDim`) epilogue:**
   - Update [`launch_mma`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/materialize.cu#L468-L491) so `Mma32` (`Rows=128, Columns=32, BlockK=128, ColumnWarps=1`), `Mma80` (`Rows=128, Columns=80, BlockK=128, ColumnWarps=3`), `Mma96` (`Rows=128, Columns=96, BlockK=128, ColumnWarps=3`), and `Mma64` (`Rows=128, Columns=64, BlockK=64 or 128, ColumnWarps=2`) all instantiate `Rows = 128` and execute the fused `store_key_head` epilogue directly inside `MaterializeProjectionEpilogue::finish` (`key_scratch = nullptr`, `graph_nodes = 1`, `workspace_bytes = 0`).
   - Benchmark replacing or fusing `KSplit16`/`KSplit24` (`T = 17..32`) with a `Rows = 128, Columns = 16/24/32` fused MMA route (or fusing key post-processing into `context_kv_grouped_kernel` when `Columns <= 16` by processing a full 128-row head or eliminated scratch) so `context_kv_materialize_uses_scratch` returns `false` across all MMA routes (and optionally all routes).
3. **Update [`context_kv_materialize_uses_scratch`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/launch.h#L51-L55) and [`context_kv_materialize_workspace_capacity_bytes`](file:///opt/etc/Infernix/src/ops/context_kv_materialize/context_kv_materialize.cpp#L47-L56):**
   - Return `false` (and `0` workspace bytes) for all routes that now fuse `store_key_head` in `Rows = 128`, reducing CUDA Graph node count from `2` to `1` and eliminating up to `41.94 MiB` of global memory workspace.

#### Verification & Acceptance Criteria
- **Build & Numerical Oracle Verification:**
  ```bash
  find src/ops/context_kv_materialize -name '*.cu' -exec touch {} +
  ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_context_kv_materialize_bench
  ctest --test-dir build-native -R "context_kv_materialize|infernix_cuda_stack_check" --output-on-failure
  ```
- **Performance Benchmark:**
  ```bash
  ./build-native/bench/infernix_context_kv_materialize_bench
  ```
- **Acceptance Criteria:**
  - All `context_kv_materialize` oracle tests and `infernix_cuda_stack_check` pass with 0 new stack/local spills.
  - `graph_nodes` drops from `2` to `1` and `workspace_bytes` drops to `0` across all `T > 32` routes (and `T = 17..32` if promoted to fused `Rows = 128`).
  - Eliminate the `T=32` (`339.7 us`) vs `T=48` (`311.1 us`) inversion and achieve a measurable latency reduction (`>= 10–25%` on formerly unfused `T = 17..96` and `T >= 257` shapes).

---

### Task T2 (Priority P0): Transpose `partial` Scratch Layout to Eliminate 32× Uncoalesced Strided Global Reads/Writes in `bf16_dynamic_grouped_conv_prepare`

#### Affected Files & Symbols
- [`src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu)
  - [`dynamic_grouped_conv_prepare_partial_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu#L46-L109)
- [`src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu)
  - [`dynamic_grouped_conv_prepare_reduce_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L15-L53)
- [`src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp)
  - [`resolve_route`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp#L20-L50)

#### Root Cause & Hardware Rationale
1. **Wrong memory layout causing 32× uncoalesced `2.5–5.0 KiB`-strided global stores and loads:**
   - In [`dynamic_grouped_conv_prepare_partial_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu#L96-L108), each warp's `m16n8k16` MMA accumulator holds row `r0 = row0 + warp_row * 16 + gid` (`gid = lane / 4`, identical across the 4 consecutive lanes `lid = 0..3`) and two adjacent token columns `c` and `c + 1` (`c = col0 + warp_col * kWarpCols + n * 8 + 2 * lid`).
   - However, `put(r, c, value)` writes `partial` in `[split][token][coefficient_row]` layout:
     ```cpp
     partial[(static_cast<std::size_t>(split) * tokens + c) * kCoefficientRows<Hidden> + r] = value;
     ```
     Because `kCoefficientRows<Hidden> = Hidden / 4` (`640` floats = `2,560` B for `H=2560`; `1,280` floats = `5,120` B for `H=5120`), each thread's two stores (`c` and `c + 1`) are separated by `2.5–5.0 KiB`, and the 4 consecutive lanes `lid = 0..3` sharing row `r` write to addresses `5.0–10.0 KiB` apart!
   - Even worse, in [`dynamic_grouped_conv_prepare_reduce_kernel`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L27-L36), each CTA `(group, batch)` maps `tid = 0..4*Capacity-1` (`Capacity = 8` or `16`) to `coefficient = tid / Capacity` (`0..3`) and `position = tid % Capacity` (`0..width-1`), and reads:
     ```cpp
     partial[(static_cast<std::size_t>(split) * tokens + batch * width + position) * kCoefficientRows<Hidden> +
             coefficient * kGroups<Hidden> + group];
     ```
     Adjacent lanes (`position = 0, 1, ..., width-1`) read memory addresses separated by **`kCoefficientRows` floats (`2,560` to `5,120` bytes)** on every `split` iteration (`splits = 4` or `8`)! Every active lane in the warp fetches a different 128-byte L2 cache line (`3.1%` cache-line utilization / `32×` L2 bandwidth amplification).
2. **Transposing `partial` to `[split][coefficient_row][token]` makes both writes and reads 100% coalesced and bit-identical:**
   - With index `((static_cast<std::size_t>(split) * kCoefficientRows<Hidden> + r) * tokens) + c`:
     - In `dynamic_grouped_conv_prepare_partial_kernel`: each thread's `(c, c + 1)` pair is contiguous in memory (enabling a single `float2` vectorized store when `c + 1 < tokens` and `tokens` is even, or contiguous scalar stores), and the 4 consecutive lanes `lid = 0..3` sharing row `r` write **8 contiguous floats (`32` bytes = 1 L2 sector)**!
     - In `dynamic_grouped_conv_prepare_reduce_kernel`: for each `(split, coefficient)`, adjacent lanes `position = 0..width-1` read **contiguous `float` elements (`+4` bytes per lane)** within a single cache line!
     - Every `(coefficient, position)` sums the exact same `split = 0..splits-1` values in the exact same order, preserving **100% bit-exactness**.
3. **Live GB10 Evidence Collected in This Session:**
   - Command: `cuobjdump --dump-resource-usage build-native/apps/infernix-serve`
     - `dynamic_grouped_conv_prepare_partial_kernel<5120, 64, 64, 64, 2, 2>`: `REG:103 STACK:0 SHARED:4` (dynamic SMEM `24,576` B)
     - `dynamic_grouped_conv_prepare_reduce_kernel<5120, 16>`: `REG:32 STACK:0 SHARED:256`
   - Command: `./build-native/bench/infernix_dynamic_grouped_conv_prepare_bench`
     - `H=5120, W=16, B=1 (T=16, 262 KB scratch)`: `median = 72.320 us` (`2.90 TFLOP/s`)
     - `H=5120, W=16, B=2 (T=32, 524 KB scratch)`: `median = 75.552 us` (`5.55 TFLOP/s`)
     - `H=5120, W=16, B=4 (T=64, 1.31 MB scratch)`: `median = 85.664 us` (`9.79 TFLOP/s`)
     - `H=5120, W=14, B=6 (T=84, 1.44 MB scratch)`: `median = 103.776 us` (`10.60 TFLOP/s`)
     - `H=5120, W=16, B=8 (T=128, 2.62 MB scratch)`: `median = 103.136 us` (`16.27 TFLOP/s`)

#### Required Implementation Steps
1. **Transpose `partial` addressing in [`bf16_dynamic_grouped_conv_prepare_partial.cu`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu#L96-L108):**
   - Change `partial` indexing from `(split * tokens + c) * kCoefficientRows<Hidden> + r` to `(split * kCoefficientRows<Hidden> + r) * tokens + c`.
   - When `c + 1 < tokens` and `(tokens & 1) == 0` (so row base is 8-byte aligned), emit a vectorized `float2` store (`make_float2(acc[m][0], acc[m][1])` and `make_float2(acc[m][2], acc[m][3])`); fall back to adjacent scalar `float` stores when `tokens` is odd or `c + 1 == tokens`.
2. **Update `partial` reads in [`bf16_dynamic_grouped_conv_prepare_reduce.cu`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu#L27-L36):**
   - Precompute `const std::size_t row = static_cast<std::size_t>(coefficient) * kGroups<Hidden> + group;` outside the `split` loop and read `partial[(static_cast<std::size_t>(split) * kCoefficientRows<Hidden> + row) * tokens + batch * width + position]`, so all lanes in a `Capacity` group issue coalesced contiguous 4-byte reads per `split`.
3. **Enable Programmatic Dependent Launch (PDL) overlap between `partial` and `reduce` kernels (if beneficial and supported on the stream):**
   - Check route transitions in [`bf16_dynamic_grouped_conv_prepare_plan.cpp`](file:///opt/etc/Infernix/src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp#L20-L50) across `T = 1..128` after the layout fix.

#### Verification & Acceptance Criteria
- **Build & Numerical Oracle Verification:**
  ```bash
  find src/ops/dynamic_grouped_conv -name '*.cu' -exec touch {} +
  ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_conv_state_test infernix_dynamic_grouped_conv_prepare_bench
  ctest --test-dir build-native -R "dynamic_grouped_conv|infernix_cuda_stack_check" --output-on-failure
  ```
- **Performance Benchmark:**
  ```bash
  ./build-native/bench/infernix_dynamic_grouped_conv_prepare_bench
  ```
- **Acceptance Criteria:**
  - 100% bit-exact match with the existing oracle tests (`dynamic_grouped_conv`) and `infernix_cuda_stack_check` passes.
  - Measurable latency speedup (`>= 15–30%` at `T = 32..128`) and elimination of the `W=14, B=6 (T=84)` `103.8 us` latency spike in `infernix_dynamic_grouped_conv_prepare_bench`.

---

### Task T3 (Priority P1): Unlock Multi-CTA Occupancy, Eliminate Dead Position Loops, and Vectorize Split-KV Reduction in `sliding_window_attention` & `context_softmax_attention`

#### Affected Files & Symbols
- [`src/ops/softmax_attention/sliding_window/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh)
  - [`SlidingWindowContextPolicy`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L23-L45)
  - [`sliding_window_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L120-L157)
  - [`sliding_window_attention_reduce_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L221-L320)
- [`src/ops/softmax_attention/sliding_window/launch.cu`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/launch.cu)
  - [`sliding_window_attention_resolve_plan`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/launch.cu#L53-L94)
- [`src/ops/softmax_attention/common/context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh)
  - [`context_query_split_partial_body`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L229-L410)
  - [`context_query_reduce_body`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L513-L580)
- [`src/ops/softmax_attention/dense/context/kernel.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh)
  - [`DenseContextPolicy`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L13-L33)
  - [`context_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L39-L73)
  - [`context_attention_reduce_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L79-L108)

#### Root Cause & Hardware Rationale
1. **Severe register bloat (`223–245` regs/thread) from `__launch_bounds__(..., 1)` capping occupancy at 1–2 CTAs/SM despite only 16 KiB SMEM:**
   - Both [`sliding_window_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L121) and [`context_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L40) hardcode `__launch_bounds__(WarpsPerCta * 32, 1)`.
   - Live `cuobjdump` shows `sliding_window_attention_split_partial_kernel<16, 4, 32, 128, 32, 8>` (the exact kernel executed 5 times per DFlash2 16-token verification round in `Qwen4Exp`) compiles to **`REG:233 STACK:0`**, while `context_attention_split_partial_kernel<16, 4, 32, ...>` compiles to **`REG:245 STACK:0`**.
   - At `233–245` registers/thread (`128` threads/CTA = `30,720–32,768` regs/CTA), GB10 (`65,536` regs/SM, `99 KiB` SMEM/SM) is capped at **2 CTAs/SM** (`8` warps/SM = `16.7%` warp occupancy), even though `KeyBlock = 32` uses only **`16,384` bytes of dynamic SMEM** (`6` CTAs/SM SMEM limit)!
   - Increasing `MinBlocksPerMultiprocessor` to `3` (for `WarpsPerCta == 4`, capping registers at `168` regs/thread = `21,504` regs/CTA with `0` spills) or `4` (for `WarpsPerCta <= 3`, `128` regs/thread) raises residency to **3–4 CTAs/SM (`12` warps/SM = +50% active warps)**. Crucially, for `T = 16, splits = 16`, the grid has `8 KV heads × 16 splits = 128 CTAs`: at 2 CTAs/SM (`96` CTAs/wave) it requires **2 waves** (the second wave running at `33%` SM utilization), whereas at **3 CTAs/SM (`144` CTAs/wave)** the entire `128`-CTA grid fits in **1 single wave** across GB10's 48 SMs!
2. **Dead register array and loop in `SlidingWindowContextPolicy` & `DenseContextPolicy`:**
   - In [`SlidingWindowContextPolicy`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L25-L37), every thread declares `int q_positions[Tokens]; int min_position = 0; int max_position = 0;` and `prime()` loops over all `0..Tokens-1` loading global `positions[token]` and computing `min_position`/`max_position`.
   - Inspection of [`context_query.cuh`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh) proves `min_position` and `max_position` are **100% dead** (never referenced anywhere in the codebase), and `policy.query_position(token)` is called only twice during setup ([`context_query.cuh:246-247`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L246-L247)). Removing `q_positions[Tokens]`, `min_position`, `max_position`, and the `prime()` loop eliminates redundant global loads and frees up to `18` registers per thread.
3. **Redundant 15-barrier tree reductions and `128 × active_splits` `expf()` evaluations in [`context_query_reduce_body`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L513-L580):**
   - `context_query_reduce_body` (used by `context_attention_reduce_kernel`) executes **15 `__syncthreads()` barriers** across two 128-thread shared-memory tree reductions (`stride = 64, 32, 16, 8, 4, 2, 1` twice), and then every one of the 128 threads (`tid = 0..127`, 1 dimension per thread) loops sequentially over all `split = 0..active_splits-1`, recomputing `expf(partial_m[stat] - global_m)` (`128 × 32 = 4,096` `expf` evaluations per CTA) and issuing unvectorized 2-byte `BF16` loads.
   - Replacing `context_query_reduce_body` with a warp-striped split reduction (4 warps × 32 lanes, 4 head dimensions per lane = `float4`/`__nv_bfloat162` pairs, computing `expf` and `local_l` in a single pass with `numerator0..3` like [`sliding_window_attention_reduce_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L241-L319)) cuts `__syncthreads()` barriers from `15` to `3`, cuts `expf()` calls by `4×`, eliminates the redundant middle pass over `partial_m`/`partial_l`, and vectorizes `partial_out` loads and `out` writes.
   - In [`sliding_window_attention_reduce_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L283-L318), also vectorize the `partial_out` global loads (`2 × __nv_bfloat162` = `uint2` per split) and final `out` write (`2 × __nv_bfloat162`).
4. **Live GB10 Evidence Collected in This Session:**
   - Command: `cuobjdump --dump-resource-usage build-native/apps/infernix-serve`
     - `sliding_window_attention_split_partial_kernel<16, 4, 32, 128, 32, 8>`: `REG:233 STACK:0 SHARED:0`
     - `context_attention_split_partial_kernel<16, 4, 32, 128, 32, 8>`: `REG:245 STACK:0 SHARED:0`
     - `context_attention_reduce_kernel<128, 32, 8>`: `REG:32 STACK:0 SHARED:1024`
   - Command: `./build-native/bench/infernix_sliding_window_attention_bench`
     - `T=16, L=512 (splits=8)`: `22.976 us` (`100.3 GB/s`, `40.8%` of `246 GB/s` target)
     - `T=16, L=1024 (splits=16)`: `31.680 us` (`141.2 GB/s`, `57.4%` of target)
     - `T=16, L=2048 (splits=16)`: `48.864 us` (`178.4 GB/s`, `72.5%` of target)
     - `T=16, L=4096 (splits=16)`: `80.704 us` (`211.9 GB/s`, `86.1%` of target)
   - Command: `./build-native/bench/infernix_context_softmax_attention_bench`
     - `T=16, L=128`: `18.240 us`; `T=16, L=512`: `26.272 us` (`87.7 GB/s`); `T=16, L=2048`: `51.392 us` (`169.6 GB/s`).

#### Required Implementation Steps
1. **Remove dead `q_positions[Tokens]`, `min_position`, `max_position`, and `prime()` from [`SlidingWindowContextPolicy`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L23-L45) and [`DenseContextPolicy`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L13-L33):**
   - Store `const int* positions` directly in the policy struct and read `positions[token]` on-demand in `query_position(int token)`.
2. **Tune `__launch_bounds__` on [`sliding_window_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L121) and [`context_attention_split_partial_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/dense/context/kernel.cuh#L40):**
   - For `KeyBlock == 32` (`16 KiB` dynamic SMEM), set `MinBlocksPerMultiprocessor = (WarpsPerCta <= 3 ? 4 : 3)` (verify via `cuobjdump --dump-resource-usage` that `STACK == 0` across all template instantiations `<1..16>`).
   - For `KeyBlock == 64` (`32 KiB` dynamic SMEM), set `MinBlocksPerMultiprocessor = 3` (verified `3 × 32 KiB = 96 KiB <= 99 KiB` SMEM on GB10 with `STACK == 0`).
3. **Unify and vectorize [`context_query_reduce_body`](file:///opt/etc/Infernix/src/ops/softmax_attention/common/context_query.cuh#L513-L580) and [`sliding_window_attention_reduce_kernel`](file:///opt/etc/Infernix/src/ops/softmax_attention/sliding_window/kernel.cuh#L221-L320):**
   - Replace the 15-barrier 2-pass scalar reduction in `context_query_reduce_body` with the 3-barrier single-pass 4-warp parallel reduction (`kReduceWarps = 4`, 4 head-dim elements per lane), using vectorized `uint2` (`2 × __nv_bfloat162`) loads from `partial_out` and stores to `out`.

#### Verification & Acceptance Criteria
- **Build & Numerical Oracle Verification:**
  ```bash
  find src/ops/softmax_attention -name '*.cu' -exec touch {} +
  ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_attention_test infernix_sliding_window_attention_bench infernix_context_softmax_attention_bench
  ctest --test-dir build-native -R "sliding_window_attention|context_softmax_attention|infernix_cuda_stack_check" --output-on-failure
  ```
- **Performance Benchmark:**
  ```bash
  ./build-native/bench/infernix_sliding_window_attention_bench
  ./build-native/bench/infernix_context_softmax_attention_bench
  ```
- **Acceptance Criteria:**
  - `cuobjdump --dump-resource-usage` confirms `REG <= 168` (`3+ CTAs/SM`) and `STACK:0` on `sliding_window_attention_split_partial_kernel` and `context_attention_split_partial_kernel`.
  - All `sliding_window_attention` and `context_softmax_attention` CTests and `infernix_cuda_stack_check` pass.
  - Measurable latency reduction (`>= 10–20%` across `T = 2..16, L = 128..2048`) on both benchmarks.

---

### Task T4 (Priority P1): Eliminate Unconditional `FP32 -> BF16 -> FP32` Full-Vocabulary Projection and Casts in `Forward::mtp_block` and Route Unscreened MTP Selection Through `ops::linear_topk` / Direct `BF16` Projection

#### Affected Files & Symbols
- [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp)
  - [`Forward::logits`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1433-L1459)
  - [`Forward::mtp_block`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1461-L1547)
- [`src/models/qwen4_exp/execution/dflash2.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/dflash2.cpp)
  - [`DFlash2::head_candidates`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/dflash2.cpp#L102-L135)

#### Root Cause & Hardware Rationale
1. **Redundant unconditional `FP32` projection + `FP32 -> BF16` cast + `BF16 -> FP32` conversion in `Forward::mtp_block`:**
   - Compare [`Forward::logits`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1441-L1458) with [`Forward::mtp_block`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1497-L1546):
     - In `Forward::logits`, `ops::projection_fp32` is ONLY called when `p_.lossless == true`; on standard production routes (`!p_.lossless`), it calls `project(normed, p_.output, out_bf16)` directly into `BF16` using the fast Tensor Core / sliced-K `ops::linear` kernels!
     - In `Forward::mtp_block` ([`lines 1498-1546`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1498-L1546)), however, the unscreened path (`!screened`) **unconditionally** executes:
       ```cpp
       Tensor head_logits = work_.alloc(DType::FP32, {rows_out, T});
       ops::projection_fp32(normed_out, *head_weight, head_logits, s);
       ...
       Tensor narrow = work_.alloc(DType::BF16, {rows_out, T});
       ops::cast_fp32_to_bf16(head_logits, narrow, s);
       if (sampled) {
         ops::top_logprobs(narrow, ..., s);
       } else if (greedy) {
         ops::argmax(narrow, *out.draft_token, s);
       }
       ```
   - This causes **three compounding inefficiencies** on every unscreened MTP step (`rows_out = 248,320` or `131,072`):
     1. **Slow fallback kernel**: `ops::projection_fp32` bypasses the tuned `ops::linear` / `ops::linear_topk` kernels even when `p_.lossless == false`.
     2. **Redundant `1.49 MiB/token` global memory transfer**: It writes `248,320 × 4B = 993.3 KiB/token` of `FP32` `head_logits` to global memory, launches `ops::cast_fp32_to_bf16` to read `993.3 KiB/token` and write `496.6 KiB/token` of `BF16` `narrow` to global memory, and then `ops::top_logprobs` / `ops::argmax` reads `496.6 KiB/token` of `BF16` and converts every element **back to `FP32`** (`__bfloat162float`)!
     3. **Missed fused `ops::linear_topk` fast path**: When `!p_.lossless && out.logits == nullptr`, and `head_weight` is `Q8_G32_FP16` or `Q4_G64_FP16` (`RowSplit`), [`ops::linear_topk`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/dflash2.cpp#L126-L134) (already used in `DFlash2::head_candidates`) fuses the LM-head projection and top-K / argmax (`k = 1` for greedy or `k = out.candidate_ids->size(0)` for sampled) in-register and in-SMEM **without materializing `[248320, T]` logits to global memory at all**! And when `out.logits != nullptr` (with `out.logits->dtype() == DType::BF16` and `!p_.lossless`), `project(normed_out, *head_weight, narrow)` writes `BF16` directly without `head_logits` (`FP32`) or `ops::cast_fp32_to_bf16`.
2. **Live GB10 Evidence Collected in This Session:**
   - Command: `./build-native/bench/infernix_candidate_selector_bench` and `./build-native/bench/infernix_linear_topk_bench` (verified in `docs/gb10/20261105-changelog.md` T9):
     - Fused `ops::linear_topk` on `[248320, 2560/5120]` runs in `467–1,488 us` across `T = 1..16` with `0 B` of full-vocabulary logit materialization, whereas `ops::projection_fp32` + `ops::cast_fp32_to_bf16` + `ops::top_logprobs`/`ops::argmax` launches 3 separate kernels and streams an extra `2.0 MiB/token` through global memory.

#### Required Implementation Steps
1. **Fast-path unscreened `Forward::mtp_block` when `!p_.lossless` in [`src/models/qwen4_exp/execution/forward.cpp`](file:///opt/etc/Infernix/src/models/qwen4_exp/execution/forward.cpp#L1497-L1546):**
   - Preserve the existing `ops::projection_fp32` path when `p_.lossless == true` or when `out.logits != nullptr && out.logits->dtype() == DType::FP32`.
   - When `!p_.lossless && out.logits == nullptr`:
     - If `(sampled || greedy)` and `head_weight` supports `ops::linear_topk` (`head_weight->packing == Packing::RowSplit` and codec is `Q8_G32_FP16` or `Q4_G64_FP16`):
       - For `sampled`: call `ops::linear_topk(normed_out, *head_weight, *out.candidate_ids, *out.candidate_logprobs, work_slice, s)` directly (zero full-vocab logit allocation, zero `cast_fp32_to_bf16`).
       - For `greedy`: either call `ops::linear_topk` with a `[1, T]` candidate view into `*out.draft_token` + scratch `[1, T]` logprobs, or project directly into `BF16` `narrow` via `project(normed_out, *head_weight, narrow)` followed by `ops::argmax(narrow, *out.draft_token, s)`.
     - Otherwise (or when `out.logits != nullptr && out.logits->dtype() == DType::BF16`): project directly into `BF16` via `project(normed_out, *head_weight, narrow)` without allocating `FP32` `head_logits` or launching `ops::cast_fp32_to_bf16`.

#### Verification & Acceptance Criteria
- **Build & Numerical/Model Verification:**
  ```bash
  ninja -C build-native -j "$(nproc)" infernix-serve infernix_qwen4_exp_mtp_test infernix_ops_linear_test
  ctest --test-dir build-native -R "mtp|linear_topk|infernix_cuda_stack_check" --output-on-failure
  ```
- **Acceptance Criteria:**
  - All MTP model/execution tests (`ctest -R mtp`), `linear_topk` tests, and `infernix_cuda_stack_check` pass.
  - On the `!p_.lossless` route, `Forward::mtp_block` no longer allocates `FP32 [rows_out, T]` `head_logits` or calls `ops::projection_fp32` + `ops::cast_fp32_to_bf16` when `out.logits == nullptr`.

---

### Task T5 (Priority P1): Port `(job, row_block)`-Outer `col_tile`-Inner Work-Unit Traversal to `wide_expert_a16.cuh` and `wide_expert_a8.cuh` to Eliminate L2 GateUp Weight Thrashing

#### Affected Files & Symbols
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh)
  - [`grouped_kernel_a16`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh#L121-L265)
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh)
  - [`grouped_kernel_a8`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh#L105-L265)
- [`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh)
  - [`map_work_unit`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L211-L228)

#### Root Cause & Hardware Rationale
1. **Wrong work-unit traversal order in `grouped_kernel_a16` and `grouped_kernel_a8` thrashing GB10's 24 MiB L2 cache:**
   - In commit `87652fc1d` (T1 of 2026-11-05), `grouped_kernel` in [`wide_expert.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh#L211-L228) introduced `map_work_unit<kColTileInner>` (`kColTileInner = (kKTiles * S::kRowGroups * S::kUnitRunBytes >= 64 * 1024)`), reordering multi-column-tile GateUp jobs from `(tile, row_block)` to `(job, row_block, col_tile)` so concurrent CTAs on GB10's 48 SMs share the same 128-row GateUp weight block (`90 KiB`) in L2 across all column tiles of the same expert.
   - However, [`grouped_kernel_a16`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh#L148) (lines `148` and `222`) and [`grouped_kernel_a8`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh#L130) (lines `130` and `197`) still hardcode the legacy L2-thrashing traversal:
     ```cpp
     const int tile = tile_begin + w / kRowBlocks, row_block = w % kRowBlocks;
     ```
   - For an expert receiving `> 64` routed tokens (`col_tiles >= 2`, common at `T >= 256` with 128 experts and `top_k = 10`), `w / kRowBlocks` forces SMs to stream all `kRowBlocks = 6` (`540 KiB`) or `12` (`1,080 KiB`) GateUp weight blocks for `col_tile = 0` before starting `col_tile = 1`, evicting the expert's weights from L2 between column tiles of the same expert!
2. **Duplicate scale TMA loads in `grouped_kernel_a8`:**
   - Just like `wide_expert.cuh` prior to `87652fc1d`, examine whether `grouped_kernel_a8` / `grouped_kernel_a16` producer warps issue redundant per-K-tile TMA transactions or whether `map_work_unit<kColTileInner>` can be shared cleanly across all three wide expert kernels (`wide_expert.cuh`, `wide_expert_a16.cuh`, `wide_expert_a8.cuh`).
3. **Live GB10 Evidence Collected in This Session:**
   - Command: `cuobjdump --dump-resource-usage build-native/apps/infernix-serve`
     - `grouped_kernel_a16<128, 384, 5120, 64, 3, true>`: `REG:116 STACK:0 SHARED:32`
     - `grouped_kernel_a16<128, 768, 2560, 64, 3, true>`: `REG:118 STACK:0 SHARED:32`
     - `grouped_kernel_a8<128, 384, 5120, 64, 3, true>`: `REG:108 STACK:0 SHARED:32`
     - `grouped_kernel_a8<128, 768, 2560, 64, 3, true>`: `REG:108 STACK:0 SHARED:32`
   - Command: `./build-native/bench/infernix_offloaded_moe_wide_bench --activation a16`
     - `T=256`: `frames_us = 10542.6 us` (`staged_us = 15221.3 us`, `2.4 TFLOP/s`)
     - `T=1024`: `frames_us = 14109.0 us` (`staged_us = 21671.8 us`, `6.6 TFLOP/s`)
     - `T=4096`: `frames_us = 20421.2 us` (`staged_us = 35870.7 us`, `19.7 TFLOP/s`)
   - Command: `./build-native/bench/infernix_offloaded_moe_wide_bench --activation a8`
     - `T=256`: `frames_us = 7084.9 us` (`staged_us = 11791.3 us`, `3.5 TFLOP/s`)
     - `T=1024`: `frames_us = 9577.9 us` (`staged_us = 17118.4 us`, `9.7 TFLOP/s`)
     - `T=4096`: `frames_us = 15527.9 us` (`staged_us = 31246.4 us`, `25.9 TFLOP/s`)

#### Required Implementation Steps
1. **Apply `map_work_unit<kColTileInner>` in [`wide_expert_a16.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh#L148) and [`wide_expert_a8.cuh`](file:///opt/etc/Infernix/src/ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh#L130):**
   - Define `constexpr bool kColTileInner = (kKTiles * S::kRowGroups * S::kUnitRunBytes >= 64 * 1024);` (so GateUp with `90 KiB` per 128-row block uses `(job, row_block)`-outer `col_tile`-inner order, while Down with `45 KiB` per 128-row block preserves `col_tile`-outer `row_block`-inner order to keep the `5 KiB` activation tile resident).
   - Replace both the producer (`warp_id == 0`) and consumer (`warp_id > 0`) `tile` / `row_block` computations in `grouped_kernel_a16` and `grouped_kernel_a8` with `map_work_unit<kColTileInner>(w, tile_begin, kRowBlocks, jobs, job_count, tile, row_block)`.
2. **Verify register pressure and stack spills:**
   - Confirm via `cuobjdump --dump-resource-usage` that `grouped_kernel_a16` and `grouped_kernel_a8` remain well below the `128`-register 2-CTA/SM ceiling with `STACK:0`.

#### Verification & Acceptance Criteria
- **Build & Numerical Oracle Verification:**
  ```bash
  find src/ops/offloaded_sparse_moe -name '*.cu' -exec touch {} +
  ninja -C build-native -j "$(nproc)" infernix-serve infernix_ops_moe_test infernix_offloaded_moe_wide_bench
  ctest --test-dir build-native -R "offloaded_sparse_moe|infernix_cuda_stack_check" --output-on-failure
  ```
- **Performance Benchmark:**
  ```bash
  ./build-native/bench/infernix_offloaded_moe_wide_bench --activation a16
  ./build-native/bench/infernix_offloaded_moe_wide_bench --activation a8
  ```
- **Acceptance Criteria:**
  - 100% bit-exact numerical equivalence (each CTA computes the exact same `(tile, row_block)` K-tile accumulation in the exact same order).
  - All `offloaded_sparse_moe` CTests and `infernix_cuda_stack_check` pass with `0` stack spills.
  - Measurable throughput gain (`+10–25%` at `T = 1024..4096`) on `infernix_offloaded_moe_wide_bench --activation a16` and `--activation a8`.
