# GB10 (`sm_121a`) Changelog — 2026-11-22 (Hardware Utilization Campaign: Waves 1 & 2 Complete — `T1`, `T2`, `T3.1`, `T4.1`, `T5.1`, `T6.1`/`T6.4`, `T8.1`/`T8.3`)

## 1. Commit Range & Accounting

- **Branch & Base Commit:** `backup-perf-changes`, base `db8611804` (`perf(qwen4_exp,ops): optimize hc4 speculative decode, narrow A4 OMMA MoE, UMA PLE, and UniverTrees`). (Repository `origin/main` ancestor `0c4abd02c` recorded in git history; working tree diffs against `db8611804`.)
- **Combined Working Tree Delta (`include/`, `src/`, `tests/`, `tools/run_server.sh`, excluding `docs/` and `cuda-kernel-toolkit/`):** `32` modified files, `3,120` insertions(+), `280` deletions(-).
  - **Core / Model / Runtime (`14` files + `tools/run_server.sh`):**
    - T1.2 (`10` files): `src/core/device.{h,cu}`, `src/ops/screened_head/screened_head.cu`, `src/models/qwen4_exp/program/{program_impl.h,graph_set.cpp,prefill_walk.cpp}`, `src/models/qwen3_5/program/{graph_execution.h,speculative/mtp.cpp,prefill.cpp}`, `tests/test_{device,decode_graph}.cpp`.
    - T2.1 (`8` additional + `4` shared): `src/core/host_memory.{h,cpp}`, `include/infernix/ops/ple.h`, `src/ops/ple/ple.cu`, `src/models/qwen4_exp/execution/forward.{h,cpp}`, `src/models/qwen4_exp/program/ngram_volume.{h,cpp}`, `tests/ops/{test_ple_nvfp4.cpp,test_ple_embed.cu}`.
    - T2.2 (`3` files retained core primitives; all 13 model bindings reverted): `src/core/{arena.h,arena.cu,device.h,device.cu}`, `tests/test_device.cpp`.
    - T8.1 (`1` file): `src/runtime/engine/context_cache/context_cost_defaults.cpp`.
    - T8.3 (`tools/run_server.sh` lines 37 & 195 `SPEC_BACKEND` default updated alongside pre-existing unstaged launcher edits).
  - **Ops / Kernels (`9` files):**
    - T1.1 (`5` files): `src/ops/common/memory.cuh`, `src/ops/linear/nvfp4/{nvfp4_instances.cuh,nvfp4_a16_sliced_k_mma.cuh}`, `src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh`, `src/ops/hyper_connection/hyper_connection_mix_fused.cu`.
    - Wave 1 T3.1 (`nvfp4_template_launch.cuh`, `nvfp4_a16_sliced_k_mma.cuh`), T4.1 (`narrow_a4_omma.cuh`), T6.4 (`ple.cu`); T3.2 tested & reverted.
    - Wave 2 T5.1 (`src/ops/qsa/qsa.cu` + `hyper_connection_mix_fused.cu`), T6.1 (`src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}`, `tests/ops/test_hyper_connection.cpp`).
- **Ratchet Gate Status:** `infernix_cuda_stack_check` passed (`STACK=0`, `LOCAL=0` on `infernix-serve`); SASS verified `0` odd `desc[UR...]` descriptors and `0` `STL/LDL` spills across Wave 1 (`narrow_a4_omma` `REG:56..94 STACK:0 ublkcp=16..44`; `ple.cu` `REG:46..56 STACK:0`) and Wave 2 (`qsa.cu` `cga=9`; `hc_down_partials_q4_mma_kernel` `REG:77..86 STACK:0`; `hc_up_collapse_q4_mma_kernel` `REG:44..59 STACK:0`).
- **Oracle & Unit Qualification:** Wave 1 passed `6/6` NVFP4/GDN/stack CTests, `infernix_offloaded_moe_{cuda,layer}_test`, `infernix_ple_{test,embed_test,nvfp4_test}`, plus earlier T1.1/T1.2/T2.1 suites. Wave 2 passed `9/9` QSA CTests (`index`, `select`, `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`), `2/2` context-cost CTests, `infernix_hyper_connection_test` (with `Codec::Q4` oracle, `T=5/9/16` column invariance, graph replay, and `inject_mix` bit identity), `infernix_hyper_connection_stream_mean_test`, `infernix_qwen4_exp_{speculative,fast_paths}_test`, and `infernix_cuda_stack_check`.

---

## 2. Architectural Rationale ("Why")

### T1.1 — Hoisted `L2::evict_first` Streaming Weight Hints
GB10 (`sm_121a`) provides only **`24 MiB`** of L2 cache (one-quarter of RTX 5090's `96 MiB`). One-shot streaming decode weight traffic across NVFP4 sliced-K dense linears, `narrow_a4_omma` routed experts, and `hyper_connection_mix` evicts reusable workspace buffers, KV cache lines, and GDN/QSA state unless marked with `L2::evict_first`:

1. **Hoisted `createpolicy.fractional.L2::evict_first.b64` (`src/ops/common/memory.cuh`):**
   - Issuing `createpolicy` inside per-load or per-copy loops prevents compiler predication and caused `+71` / `+16` extra branches and `+142` / `+32` `BSSY` sync barriers in `nvfp4_a16_sliced_k_mma_kernel` and `hc_up_collapse_v4_kernel`.
   - Adding a hoisted volatile `l2_evict_first_policy()` plus `cp_async_hinted` (with `asm("" : "+r"(dst))` output tie) and `ld_nc_na_evict_first_vec<V, T>(ptr, policy)` preserves exact register count and zero extra branches/barriers (`+6`..`+12` uniform ops only).
2. **One-Shot Weight Streams Marked `evict_first`; Activations & Scales Kept Reusable:**
   - `nvfp4_a16_sliced_k_mma.cuh` / `nvfp4_instances.cuh`: `Cache::cg_evict_first` for weight `cp.async`, keeping `Cache::ca` for activations.
   - `narrow_a4_omma.cuh`: `Cache::cg_evict_first` for routed expert weight stages (`gate_up` and `down`), leaving activation staging untouched.
   - `hyper_connection_mix_fused.cu`: `ld_nc_na_evict_first_vec` for `Q8Codec`, `Q4Codec`, and `Bf16Codec` weight loads across all 4 fused HC kernels, keeping scale loads untouched.
3. **CUDA 13.0 `ptxas -arch=sm_121a` Blocker on `screen_kernel` (`src/ops/screened_head/screened_head.cu`):**
   - Applying `createpolicy` + `cp.async.cg.shared.global.L2::cache_hint` inside `screen_kernel` (`extern __shared__` across prologue and mainloop) causes CUDA 13.0 `ptxas` to miscompile into `LDGSTS.E.BYPASS.128 [R+UR0+imm], desc[UR1]` (invalid odd `desc[UR1]`), crashing `infernix_screened_head_test` with `cudaErrorIllegalInstruction`.

### T1.2 — Phase-Switched `9.0 MiB` (`3/8` L2) Persisting L2 Window + `screen_kernel` Streaming Window
Hardware probes (`l2_limit_diag`, `l2_gemm_diag`, `l2_sweetspot`) revealed a critical GB10 L2 invariant that overrode the initial `18 MiB` roadmap assumption:

1. **GB10 `18 MiB` Persisting-L2 `2x` Throughput Cliff:**
   - Setting `cudaLimitPersistingL2CacheSize` to the hardware max (`18.0 MiB` of `24 MiB` L2) collapsed streaming DRAM read bandwidth by `-21.2%` (`242.0 -> 190.6 GB/s`), L2 `16M` bandwidth by `-40.9%` (`1578.0 -> 933.2 GB/s`), and BF16 `[16384,2048,2560]` GEMM throughput by **`-50.2%`** (`107.38 -> 53.38 TF`), regressing E2E prefill `-40%`..`-68%` and decode `-7%`..`-33%`.
   - At `9.0 MiB` (`3/8` of `24 MiB`), DRAM BW (`242.0 GB/s`), L2 `16M` BW (`1564.6 GB/s`), and BF16 GEMM (`105.00 TF`) stayed cliff-free while improving pinned-buffer L2 hit latency (`0.5492 vs 0.5765 ms`, `-4.74%`).
2. **Controlled A vs B vs C Ablation (`9.0 MiB` > `4.5 MiB` > `0 MiB` screen-only):**
   - `CandA (9.0 MiB)` monotonically outperformed both `CandB (4.5 MiB default)` and `CandC (0 MiB / screen_kernel-only)` across fixed-trace decode throughput/latency (`b4_conc pp+tg`: `CandA +1.05%` / `dec_ms -0.69%` > `CandB +0.90%` / `-0.52%` > `CandC -0.20%` / `+2.53%`; `b8_conc tg256`: `CandA +0.59%` / `dec_ms -0.57%` > `CandB +0.23%` / `-0.24%` > `CandC -0.25%` / `+0.26%`).
3. **Phase-Switched `0 MiB` Prefill / `9.0 MiB` Decode (`CandA+ v3`):**
   - Leaving a static `9.0 MiB` carveout active during long prefill (`pp8192`) softened large prefill GEMMs (`-0.61%`..`-1.28%` vs T1.1).
   - `enable_persisting_l2_cache(ctx)` / `disable_persisting_l2_cache(ctx)` cache `active_persisting_l2_bytes` on `DeviceContext` (no-op within a phase; `~8.6–11.2 us` only on actual prefill<->decode phase transitions; capture-safe `cudaStreamIsCapturing` guard) so prefill sees the full `24 MiB` L2 (`0 MiB` carveout) and decode pins up to `9.0 MiB` of `*work_` across `device.stream`, `shared_stream`, and `split_stream` (baked into captured CUDA Graph kernel nodes via `ScopedL2AccessPolicyWindow`).
4. **`screen_kernel` Launch-Level `Streaming` Window (`src/ops/screened_head/screened_head.cu`):**
   - Switched `launch_screen` to `cudaLaunchKernelEx` carrying optional `PDL` + `cudaLaunchAttributeAccessPolicyWindow` (`hitRatio = 1.0F`, `hitProp = cudaAccessPropertyNormal`, `missProp = cudaAccessPropertyStreaming`) spanning `[x, lists, partials]` without overlapping `codes` or `scales` (`total = 880/1240 +0%`, `regs 75/117 +0`, `0` odd `desc[UR1]`).

### T2.1 — GPU-Direct UMA Gather for Default `--ngram-residency volume` Cache (`NgramVolume::cache_`)
On GB10 (`integrated_uma = true`, `can_map_host_memory = 1`, `async_engine_count = 1`), `--ngram-residency volume` (`16 GiB` host cache) previously copied cache-hit rows from `NgramVolume::cache_` into `host_io() + io_layout_.ngram` on the CPU and uploaded them through `upload_pinned_when_kernel`. T2.1 eliminates that staging copy and uncovered four GB10 UMA performance invariants in `v1 -> v2` iteration:

1. **Per-CTA `MEMBAR.SC.SYS` Barrier Cost vs Single-Thread `wait_pinned_word_kernel`:**
   - Embedding `ple_gate_wait()` (`__threadfence_system()`) in every gather CTA executed up to `2,048` (NVFP4) or `81,920` (FP8 `pp8192`) system-wide barriers (`gather_embed_nvfp4_kernel` `33.20 ms` total, warm `pp2048` `2,167.6 us`, `b8_conc_vol16g pp2048 -1.05%`, `m_dense8 pp8192 -0.96%`).
   - Extracting the gate into a 1-CTA, 1-thread `wait_pinned_word_kernel` (`<<<1, 1>>>`, 1 `MEMBAR.SC.SYS`, followed by plain `<<<...>>>` launch so `wait_pinned_word` retires cleanly before gather begins) reduced `gather_embed_nvfp4_kernel` total time by **`-98.2%`** (`33.20 -> 0.59 ms`) and warm `pp2048` from `2,167.6 -> 82.24 us`.
2. **Hardware C2C Coherency & Vectorized Loads vs `__ldcv` (`STRONG.SYS`):**
   - On Grace-Blackwell coherent UMA (`DMB ISHST` host release before gate publication + separate stream-ordered kernel launch), `__ldcv` compiled to narrow `LDG.E.*.STRONG.SYS` system-scope loads. Switching to normal loads (`LDG.E`) plus vectorized `LDG.E.128`/`STG.E.128` (`gather_embed_fp8_vec16_kernel`) eliminated `STRONG.SYS` serialization and cut `m_dense8` FP8 `pp8192` GPU kernel time by `2.13×` (`473.12 -> 222.56 us`). 256-iteration CUDA Graph host-mutation stress tests mutating `slot 0` and `slots[]` behind the gate verified zero stale L1/L2 reads.
3. **`MADV_HUGEPAGE` on `RegisteredHostBuffer` (`16 GiB` UMA TLB Pressure) & Parallel `prefault`:**
   - Host THP defaults to `[madvise]`, leaving plain `mmap` allocations on `4 KiB` pages (`4,194,304` PTEs for `16 GiB`). Adding `madvise(..., MADV_HUGEPAGE)` (`>= 2 MiB`) alongside parallel page-aligned `prefault` via `pool_` cut process wall-clock by `-2.0s..-2.2s` across all `6/6` volume runs and resolved large-grid GPU TLB stalls.
4. **Zero-Pin Warm-Hit Fast Path in `NgramVolume::resolve_slots`:**
   - Warm hits (`misses_.empty() && !keep_pinned`) skip `kPinBit` (`0x02U`) marking and only unpin when `!pinned_slots_.empty()`, while `find_slot` checks `ref_bits_` before issuing `__atomic_fetch_or`.

---

## 3. Subsystem Breakdown

- `M src/core/device.h`, `M src/core/device.cu`: `DeviceContext::persisting_l2_bytes` (`3/8` of `l2CacheSize`, `9.0 MiB` on GB10), `mutable active_persisting_l2_bytes`, `ScopedL2AccessPolicyWindow` RAII (nested-scope restoring & capture-safe), `reset_persisting_l2_cache`, and idempotent `enable_persisting_l2_cache` / `disable_persisting_l2_cache`; 1-CTA 1-thread `wait_pinned_word` / `wait_pinned_word_kernel`.
- `M src/core/host_memory.h`, `M src/core/host_memory.cpp`: `RegisteredHostBuffer::try_pin_if_integrated_uma() noexcept` and `MADV_HUGEPAGE` (`>= 2 MiB`).
- `M include/infernix/ops/ple.h`, `M src/ops/ple/ple.cu`: `kPleCacheStagingBit`, `struct PleCacheGather`, `gather_embed_nvfp4_kernel<VectorStore>`, `decode_fp8x8_to_bf16x8`, `gather_embed_fp8_vec16_kernel`, scalar fallback `gather_embed_fp8_kernel`, `ple_gather_embed_nvfp4`, and `ple_gather_embed_fp8`.
- `M src/models/qwen4_exp/program/ngram_volume.h`, `M src/models/qwen4_exp/program/ngram_volume.cpp`: `cache_` migrated from `std::vector<std::byte>` to `std::unique_ptr<RegisteredHostBuffer>` with parallel `prefault` + `try_pin_if_integrated_uma()`, `kStagingSlotBit` (`0x80000000U`), `ref_bits_` `kPinBit` (`0x02U`), read-before-RMW `find_slot`, `resolve_slots(..., keep_pinned)`, and `unpin_slots()`.
- `M src/models/qwen4_exp/execution/forward.h`, `M src/models/qwen4_exp/execution/forward.cpp`: `ForwardBatch::ple_cache_gather` wiring into `ple_embedding()`.
- `M src/ops/common/memory.cuh`: `Cache::cg_evict_first`, hoisted volatile `l2_evict_first_policy()`, `ld_nc_na_evict_first_vec<V, T>`, and `cp_async_hinted<Bytes, Policy>`.
- `M src/ops/linear/nvfp4/{nvfp4_instances.cuh,nvfp4_a16_sliced_k_mma.cuh}`: Weight `Cache::cg_evict_first` + hoisted `weight_policy`.
- `M src/ops/offloaded_sparse_moe/cuda/narrow_a4_omma.cuh`: Routed-expert weight `Cache::cg_evict_first` + hoisted `weight_policy`.
- `M src/ops/hyper_connection/hyper_connection_mix_fused.cu`: Weight `ld_nc_na_evict_first_vec` across all 4 fused HC kernels.
- `M src/ops/screened_head/screened_head.cu`: `cudaLaunchKernelEx` per-launch `cudaLaunchAttributeAccessPolicyWindow` (`Normal` on `[x, lists, partials]`, `Streaming` on `codes`/`scales`).
- `M src/models/qwen4_exp/program/{program_impl.h,graph_set.cpp,prefill_walk.cpp}`: `DecodeL2PersistGuard` across eager decode, `capture_graph_set()`, and first-use `prepare_graph()`; `enable_persisting_l2_cache` before `replay()`/`replay_gated()`; `reset_l2_persist_for_prefill()` (`disable_persisting_l2_cache`) at `enter_wide()`, `walk_begin()`, prefill `run()`, and `run_rows_behind()`; `use_mapped_cache_ple()`, `ngram_slots_host_`, `ngram_slots_scratch_`, `ngram_rows_bytes()`, `resolve_ngram_to()`, `forward_call` `PleCacheGather` attachment, and split-stage `keep_pinned` / `GateRelease` cleanup (keeping `prefill_walk.cpp` on `allow_mapped_cache=false`).
- `M src/models/qwen3_5/program/{graph_execution.h,speculative/mtp.cpp,prefill.cpp}`: `make_decode_l2_persist_window()` on eager decode, `capture_graph()`, and `capture_mtp_decode_batch()`; `enable_persisting_l2_cache` in `run_prepared()`; `disable_persisting_l2_cache` in `prefill_text_chunk()`.
- `M tests/test_device.cpp`, `M tests/test_decode_graph.cpp`: Added `check_l2_persisting_policy` (nested restore + idempotent `disable`/`enable` limit checks) and `check_l2_policy_graph_capture` (captured kernel node L2 access-policy inspection + replay).
- `M tests/ops/{test_ple_nvfp4.cpp,test_ple_embed.cu}`: Added `gather_embed_nvfp4_exact()` and `test_gather_embed_fp8()` (`vec16` `width=160` and scalar fallback `width=24`, mixed `kPleCacheStagingBit`, background gated capture/replay + `wait_stats`, and 256-iteration `slot 0`/`slots[]` coherency stress tests against FP64/FP32 oracles).

---

## 4. Verification & Benchmark Summary

### T1.1 (`vs db8611804`)
- **Op Benchmarks:** `infernix_hyper_connection_bench` improved `-0.3%`..`-1.3%` (`q8`) and `-1.3%`..`-3.6%` (`q4`) with bit-identical FNV1a hashes; `NVFP4 A16 [16384, 2560]` `WITHIN_NOISE`; `infernix_offloaded_moe_decode_bench --omma` improved at `T=2/4/8` cold (`138.2->132.8 us` at `T=2 split=0`), while standalone `T=1` 30x repeat warm loop slowed (`76.96->91.62 us`) and interleaved `ckt ab --tokens 1` was `-3.05%` `WITHIN_NOISE`.
- **KV-Matrix:** Improved `9/9` metrics (`+0.25%`..`+2.84%`).

### T1.2 (`CandA+ v3` `0 MiB` prefill / `9.0 MiB` decode vs `T1.1` and `db8611804`)
- **KV-Matrix (`run_kv_matrix_bench.sh`, `/tmp/kvmatrix_t1_2_v3`):**
  - `m_dense8_int8_mtp`: `tg512` `83.56 tok/s` (`+0.97%` vs T1.1, `+1.28%` vs `db8611804`, `dec_ms -0.96%`, `rounds 207->207`); `pp8192+tg512` `176.61 tok/s` (`+1.73%` vs T1.1, `+2.21%` vs `db8611804`, `dec_ms -1.69%`, `rounds 68->68`).
  - `m_iq3s_int8_mtp`: `tg512` `82.62 tok/s` (`+1.24%` vs T1.1, `+1.56%` vs `db8611804`, `dec_ms -1.23%`, `rounds 233->233`); `pp8192+tg512` `162.46 tok/s` (`+2.07%` vs T1.1, `+4.97%` vs `db8611804`, `dec_ms -2.02%`, `rounds 68->68`).
  - `m_nvfp4_int8_mtp`: `pp8192+tg512` `201.35 tok/s` (`+0.18%` vs T1.1, `+1.60%` vs `db8611804`, `dec_ms -2.87%`, `rounds 71->73`); `tg512` shifted `--draft-tokens auto` length (`79.29 tok/s`, `+15.50%`, `acc 0.641->0.698`, `rounds 380->304`).
  - Paired back-to-back `pp8192` A/B vs `/tmp/gb10_hw_t1_1/infernix_bench`: `dense8` `3558.93 -> 3551.88 (-0.20%)`, `nvfp4` `4185.43 -> 4197.80 (+0.30%)` — zero prefill regression.
- **`hc4` 7-Case Paired E2E Suite vs `/tmp/gb10_hw_t1_1/infernix_bench`:**
  - All identical-rounds decode metrics improved on both tok/s and `dec_ms`: `b1_prose tg256` (`+0.30%`, `dec_ms -0.45%`), `b4_conc_vol16g tg256` (`+0.48%`, `-0.53%`), `b4_conc_ram tg256` (`+0.31%`, `-0.27%`), `b4_conc_ram pp2048+tg256` (`+1.37%`, `-0.98%`), `b8_conc_vol16g tg256` (`+0.59%`, `-0.65%`), `b8_conc_vol16g pp2048+tg256` (`+1.32%`, `-1.29%`), `b1_greedy_repro pp2048+tg256` (`+1.32%`, `-1.33%`).
  - Prefill stayed flat within `[-0.50%, +0.17%]` across all cases.
  - Adaptive `--draft-tokens auto` round-count variance appeared on `b1_copy` (`rounds 246->252`, `acc 0.632->0.611`, `med -3.88%`), `b1_prose_repro tg256` (`rounds 230->240`, `med +0.66%` / `mean -2.25%` / `dec_ms -2.25%`), `b1_prose pp+tg` (`rounds 45->69`, `acc 0.950->0.910`, `med -6.85%`), and `b4_conc_vol16g pp+tg` (`rounds 257->315`, `acc 0.935->0.910`, `med -3.28%`).

### T2.1 (`v2` GPU-Direct UMA Volume Cache Gather vs `T1.2`)
- **`nsys` Kernel Profile:**
  - `hc4 b1_prose --ngram-residency volume -pg 2048,64`: `upload_pinned_when_kernel` eliminated (`51 -> 0` calls); `gather_embed_nvfp4_kernel` dropped `33.20 ms` (v1) -> `0.59 ms` (v2, `-98.2%`), warm `pp2048` dropped `2,167.62 -> 82.24 us` (`wait 1.92 + gather 82.24 = 84.16 us` vs T1.2 `82.50 us`). Decode `T=1..32` GPU time (`wait + gather`) was `6.7..26.3 us` vs T1.2 `4.1..6.5 us`.
  - Paired `m_dense8` (`-pg 8192,128`, `11` identical rounds): `gather_embed_fp8_vec16_kernel` matched `embed_kernel` (`0.63 ms` total, `0.00 ms` delta), cut `pp8192` GPU time `2.13×` (`473.12 -> 222.56 us`), cut `T=12..16` by `12–14%` (`5.73..5.89 -> 4.86..5.28 us`), cut total `upload/wait` time `54.45 -> 44.10 ms`, and improved decode `+1.70%` (`150.28 -> 152.84 tok/s`, `dec_ms 77.72 -> 76.40`) / `pp8192 +0.21%` (`3535.25 -> 3542.71`).
- **KV-Matrix (`/tmp/kvmatrix_t2_1_v2` vs `/tmp/kvmatrix_t1_2_v3` & `db8611804`):**
  - `m_dense8_int8_mtp` (`volume` FP8): `pp8192` `3543.44 tok/s` (`+0.11%` vs T1.2, `-0.26%` vs `db8611804`), `tg512` `83.16 tok/s` (`-0.48%` vs T1.2, `+0.80%` vs `db8611804`, `rounds 207->207`), `pp8192+tg512` `175.26 tok/s` (`-0.76%` vs T1.2, `+1.43%` vs `db8611804`, `rounds 68->68`; paired `-pg 8192,512` showed intra-binary thermal ramp across reps `2.8884->2.9076s` vs `2.9066->2.9308s`).
  - `m_iq3s_int8_mtp` (`ram`): `pp8192 +0.03%`, `tg512 +0.17%`, `pp+tg +0.02%`.
  - `m_nvfp4_int8_mtp` (`ram`): `pp8192 +0.25%`, `pp+tg +0.53%` (`tg512` shifted `--draft-tokens auto` `304->380` rounds).
- **`hc4` 7-Case Paired E2E Suite vs `/tmp/gb10_hw_t1_2/infernix_bench`:**
  - Process wall-clock improved **`-2.0s` to `-2.2s`** across all `6/6` volume runs (`parallel prefault + MADV_HUGEPAGE`).
  - `b8_conc_vol16g pp2048` recovered `-1.05%` (v1) -> `-0.29%` (`3170.02 -> 3160.89`).
  - Fixed-round decode remained flat within `[-0.32%, -0.01%]` (`b4_conc_vol16g tg256 -0.21%` / `pp+tg -0.32%`, `b8_conc_vol16g tg256 -0.01%`, `b1_greedy_repro pp+tg -0.07%`), with `prefill_sig_eq=True` across all 7 cases.

---

## 5. Rebase Conflict Resolution Policy

- Preserve hoisted `l2_evict_first_policy()` placement (single policy creation per kernel outside inner loops) to avoid upstream `ptxas` branch/barrier inflation on `sm_121a`.
- Do **not** merge upstream `createpolicy` hints inside `screen_kernel` (`screened_head.cu`) on CUDA 13.0 `sm_121a` without re-verifying zero odd `desc[UR1]` `LDGSTS` descriptors via `ckt static --sass` and `infernix_screened_head_test`; preserve `cudaLaunchKernelEx` `cudaLaunchAttributeAccessPolicyWindow` (`Streaming` on `codes`/`scales`).
- Preserve GB10's `l2_bytes * 3 / 8` (`9.0 MiB`) persisting-L2 cap and prefill/decode phase switching (`enable_persisting_l2_cache` / `disable_persisting_l2_cache`) over upstream static `persistingL2CacheMaxSize` (`18 MiB` cliff on `sm_121a`) or RTX 5090 `96 MiB` L2 assumptions.
- Preserve `RegisteredHostBuffer::try_pin_if_integrated_uma()` + `MADV_HUGEPAGE`, `wait_pinned_word` single-CTA gating, `ple_gather_embed_{nvfp4,fp8}` normal-load/vectorized kernels, and `NgramVolume::resolve_slots()` warm-hit zero-pin path when `integrated_uma && can_map_host_memory`; do not reintroduce per-CTA `MEMBAR.SC.SYS` or `__ldcv` (`STRONG.SYS`) loads without re-qualifying on GB10 UMA.
- Keep 0 `STACK` / 0 `LOCAL` spills enforced via `infernix_cuda_stack_check`.

### T2.2 — Control / Metadata Buffer Zero-Copy Evaluation (Rejected for Model Bindings; Core UMA Infrastructure Retained)
- **Tested Hypothesis:** Replace per-step `upload_pinned_kernel` / `cudaMemcpyAsync` H2D/D2H control transfers (`io_host_`, `spec_host_`, `mtp_host_`, `dflash_host_`, `host_configs_`, `constraint_pinned_`, and `qwen3_5` `round_buffers`) with `cudaHostAllocMapped` + `DMB ISHST` release ordering when `integrated_uma == true`.
- **Correctness Verified (`verify5` + `16` CTest suites):** `infernix_device_test` (`check_mapped_pinned_zerocopy` 256-iter CUDA Graph coherency), `infernix_cuda_stack_check` (`STACK=0/LOCAL=0`), and all `8/8` `hc4` path checks (MTP repro greedy, MTP fast greedy, DFlash2 repro greedy, N-gram prose/copy repro, plain decode, sampled MTP & DFlash2 UniverTrees) matched `sig_eq=True` and exact token/speculative counts.
- **Why Rejected (`nsys` on `hc4_vol -pg 2048,64`):**
  - Transfer savings were only **`~0.75 ms`** (`upload_pinned_kernel` `317 -> 3` calls / `0.5825 -> 0.0108 ms`; post-load D2H `cudaMemcpyAsync` `138 -> 3` calls / `46,576 -> 48 B` / `0.1929 -> 0.0159 ms`).
  - Consumer kernels accessing `cudaHostAllocMapped` sysmem regressed by **`+12.68 ms`** (`16×–22×` net slowdown):
    * `argmax_tiled_atomic_kernel`: `0.5610 -> 7.1765 ms` (`+6.6155 ms`, `+1179.2%`) due to uncached cross-CTA atomic reductions over Grace Coherence Fabric instead of GPU L2.
    * 48-layer `io_device_` consumers (`qsa_score*`, `causal_conv1d_snapshot_decode`, `index_query`, `rope_generic`, `embed_gather_dense`, `pool`, `qsa_select_global`): `31.91 -> 44.01 ms` (`+12.10 ms` including `argmax`, `+5.48 ms` excluding `argmax`).
    * Vocab-wide `sampling*` / `speculative_sampling*` / `commit_tail_vec8`: `+0.58 ms`.
  - `upload_pinned_kernel` runs on the GPU compute queue (`0` Copy Engine contention) and pre-warms GPU L2-cached VRAM in `1.84 us/call`.
- **Disposition:**
  - Reverted all 13 model files (`qwen4_exp` & `qwen3_5` mapped control bindings), restoring `argmax_tiled_atomic_kernel` (`0.5968 ms`), `io_suspects` (`32.27 ms`), and `b1_greedy_repro` (`268.11 -> 269.30 tok/s`, `+0.44%`, `22 -> 22` rounds) with `18/18` CTests passing (`STACK=0/LOCAL=0`).
  - **Retained** verified Core UMA primitives in `src/core/{device.h,device.cu,arena.h,arena.cu}` and `tests/test_device.cpp` (`DeviceContext::integrated_uma()`, `publish_pinned_memory()`, `PinnedHostBuffer::device_data()`, `upload_pinned_when()` shortcut, and `check_mapped_pinned_zerocopy`).

---

## 6. Wave 1 (`T3.1`, `T3.2` Reverted, `T4.1`, `T6.4`) & Wave 2 (`T5.1`, `T5.2` Non-Goal, `T6.1`, `T6.2` Closure, `T8.1`, `T8.3`)

### Wave 1 Architecture & Findings
- **T3.1 (`nvfp4_template_launch.cuh`, `nvfp4_a16_sliced_k_mma.cuh`):** Enabled `pdl::Dependency::Programmatic` by default on `launch_nvfp4_a16_sliced_k_mma` and hoisted `weight_policy`, `scale_ptr_top/bot`, and coalesced `stage_codes(0, 0)` ahead of `pdl::enter_streaming()`, preserving coalesced loads, `__syncthreads()`, and `kPrefetchWords`.
- **T3.2 Rejected (`nvfp4_a16_sliced_k_mma.cuh`):** Warp-private `stage_codes` loading broke `128B/256B` intra-warp global coalescing into 16 strided 32-byte row loads per warp, causing `+5.10%..+12.08%` E2E decode regressions on `hc4`; `nsys --cuda-graph-trace=node` isolated `+205.17 ms` (`+52%`) in `nvfp4_a16_sliced_k_mma_kernel`. Reverted immediately.
- **T4.1 (`narrow_a4_omma.cuh`):** Converted weight staging to 1D bulk TMA (`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`) at `5 CTAs/SM` (`__launch_bounds__(128, 5)`, `persistent_omma_ctas() = sms * 5`), using scalar bit-packed `phase_bits` (avoiding 8B dynamic-array stack frame) and `#pragma unroll 2` on `kGateUpChunks` (`REG:56..94 STACK:0`, `ublkcp=16..44`, `stl_ldl=0`).
- **T6.4 (`ple.cu`):** Switched software E2M1 nibble unpack to hardware `cvt.rn.f16x2.e2m1x2` (`decode_e2m1_pair_f32`) in `embed_nvfp4_kernel`, `hash_embed_nvfp4_kernel`, and `gather_embed_nvfp4_kernel` (`REG:46..56 STACK:0`).
- **Wave 1 Single-Model `hc4` Paired Benchmark (vs `/tmp/gb10_hw_t2_1/infernix_bench`, `sig_eq=True`):**
  - `b4_conc_vol16g`: `pp2048` `3187.98 -> 3198.82 (+0.34%)`; `tg256` `dec 32.05 -> 32.28 (+0.71%)`, `dec_ms 51.952 -> 51.552 (-0.77%, 1168/847)`; `pp2048+tg256` `pp 3188.69 -> 3183.04 (-0.18%)`, `dec_ms 56.565 -> 56.413 (-0.27%, 315/1679)`.
  - `b1_prose_fixed4_repro`: `pp2048` `2688.28 -> 2686.56 (-0.06%)`; `tg256` `dec 72.44 -> 72.66 (+0.31%)`, `dec_ms 41.179 -> 41.056 (-0.30%, 172/340, drafted 708->709 via async volume read gate)`; `pp2048+tg256` `pp 2686.18 -> 2692.91 (+0.25%)`, `dec_ms 78.492 -> 76.355 (-2.72%, 32/480 vs base 32->30)`. `nsys` (`-n 64 --draft-tokens 4 --reproducible`): `dec_ms 40.039 -> 40.062 (+0.06%)`, `nvfp4_a16_sliced_k_mma 399.99 -> 400.95 ms`.

### Wave 2 Architecture & Findings
- **T5.1 (`qsa.cu` & `hyper_connection_mix_fused.cu`):**
  - Added `<cooperative_groups.h>` + `ClusterSplits = 1, 2, 4` to `sparse_gqa_columns_kernel` using existing SMEM (`12 * 1,040 B <= 16,384 B`) and dual `cluster.sync()` safe for `total <= 0` and `start >= end`; rank 0 reduces splits `0..ClusterSplits-1` via `cluster.map_shared_rank`, bypassing `sparse_gqa_merge_warp_kernel` at `mma_splits = 2, 4` (`9/9` QSA CTests passed, `cga=9`, `infernix_qsa_attention_bench --storage int8`: `w=1 8k: 26.75 us`, `w=5 8k: 51.17 us`, `w=1 131k: 30.72 us`, `w=5 131k: 59.23 us`).
  - Used `clusterDim = (1, 4, 1)` DSMEM reduction across the 4 stream CTAs inside `hc_down_partials_q4_mma_kernel`.
- **T5.2 Hardware Non-Goal:** Standalone `ptxas -arch=sm_121a` emits a performance warning and lowers `.multicast::cluster` into slow runtime call `CALL.ABS.NOINC` instead of `UTMALDG`; documented in guardrail §4 #7 of the task roadmap.
- **T6.1 (`hyper_connection_mix_fused.{h,cu}`, `test_hyper_connection.cpp`):**
  - Rejected SIMT `PassCols=8` `T=5..16` prototype after microbench regression (`45.09..114.91 us` vs `40.86..41.44 us` composed MMA).
  - Implemented native `HMMA.16816` (`mma_bf16`) + Cluster-DSMEM fused Q4 kernels (`hc_down_partials_q4_mma_kernel<Norm, Pending, TCap>` + `hc_up_collapse_q4_mma_kernel<Norm, Pending, TCap>`, `TCap in {8, 16}`, `REG:44..86 STACK:0`, `stl_ldl=0`, `odd_desc=[]`). `infernix_hyper_connection_bench --codec q4 --repeat 20`: **`T=5..16` dropped from `41.44..40.90 us` (composed) to `24.18 / 24.57 / 25.36 / 34.23 / 36.41 / 39.32 us`** (`-41.7%` at `T=5`), with `T=1..4` SIMT intact (`18.82..24.71 us`).
- **T8.1 (`context_cost_defaults.cpp`):** Calibrated `"nvidia-gb10-sm121"` `.prefill` preset `kGb10FlashNextW4DensePrefill` on `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`p95 = 12.76%` train, `0/5` validation ordering failures; held-out `text-b2048-s256` `16.93%` from hybrid QSA prefix scan overhead; previously `~59%` on RTX 5090 fallback), binding both W4Dense signatures (`b99760c8...`, `79428db0...`) and leaving foreign signatures unbound.
- **T8.3 (`tools/run_server.sh`):** Updated default `SPEC_BACKEND="${SPEC_BACKEND:-mtp}"` (lines 37 & 195), preserving pre-existing launcher edits; confirmed `kPairWidthCost = 0.16` / `kMultiRowWidthCost = 0.12` in `program_impl.h`.
- **T6.2 / T6.3 / T7.1 / T8.2 Disposition:**
  - `T6.2` closed (`ckt math roofline`: `23.59 MB` `[16384, 2560]` gives `56.9..170.7 FLOP/B` at `T=16..48` vs `510.3 FLOP/B` HMMA ridge; `96.53 us` BW bound).
  - `T6.3` Part 1 (`Q8` column-invariant `SlicedKMma` `T=2..8`) confirmed already in-tree in `q8_invariant_families.cu`; Part 2 (`Dense8`/`IQ3_S`) deferred.
  - `T7.1` deferred (`nsys4` kernel sum `2,571.24 ms` > active span `2,473.96 ms`, already >100% overlapped; `volume` mode out of scope).
  - `T8.2` closed (`nsys4` inter-kernel gaps `<50 ms` across entire run = `81.33 ms` / `3.16%`, dominated by prefill/warmup transitions; steady decode host gaps `<1%`).
- **Wave 2 Single-Model `hc4` Paired Benchmark (vs `/tmp/gb10_hw_wave1/infernix_bench`, `sig_eq=True`):**
  - `b1_prose_fixed4_repro`: `pp2048` `2675.65 -> 2694.31 (+0.70% | vsT2.1 +0.22%)`; **`tg256` `dec 72.42 -> 72.76 (+0.47%)`, `dec_ms 41.174 -> 39.451 (-4.18% | vsT2.1 -4.20%)`** (`rounds 172->178`, `acc 340->333`, `drafted 709->763` with async volume read + BF16/FP32 Q4 MMA rounding change); `pp2048+tg256` `pp 2667.00 -> 2673.33 (+0.24%)`, `dec 207.22 -> 219.54 (+5.95%)`, `dec_ms 74.970 -> 75.360 (+0.52% | vsT2.1 -3.99%)`, `rounds 33->31`, `acc 479->481`.
  - `b4_conc_vol16g`: `pp2048` `3178.20 -> 3174.33 (-0.12%)`; **`tg256` `dec 32.32 -> 34.06 (+5.39% | vsT2.1 +6.26%)`** (`rounds 1168->1075`, `acc 847->913`, `batches 324->300`, `dec_ms 51.520 -> 53.001 +2.87%` from `+12%` more drafted tokens/batch); `pp2048+tg256` `pp 3182.34 -> 3167.78 (-0.46%)`, **`dec 89.36 -> 90.68 (+1.47% | vsT2.1 +1.58%)`** (`rounds 315->287`, `acc 1679->1716`, `batches 199->169`, `dec_ms 56.395 -> 62.383 +10.62%` from `+17.7%` larger verify batches).

### Additional Rebase Rules
- Preserve `qsa.cu` `ClusterSplits = 2, 4` cluster merge and `hyper_connection_mix_fused.cu` `hc_down_partials_q4_mma_kernel` (`clusterDim = (1, 4, 1)`) + `hc_up_collapse_q4_mma_kernel` (`T = 5..16`) over upstream SIMT/composed fallbacks.
- Preserve 1D bulk TMA (`narrow_a4_omma.cuh`), scalar `phase_bits`, `#pragma unroll 2`, `5 CTAs/SM`, and coalesced `nvfp4_a16_sliced_k_mma.cuh` `stage_codes(0, 0)` prefetch + PDL wiring.
- Preserve `"nvidia-gb10-sm121"` `kGb10FlashNextW4DensePrefill` in `context_cost_defaults.cpp` and `SPEC_BACKEND="${SPEC_BACKEND:-mtp}"` in `tools/run_server.sh`.
- Do not reintroduce TMA `.multicast::cluster` on `sm_121a`.
