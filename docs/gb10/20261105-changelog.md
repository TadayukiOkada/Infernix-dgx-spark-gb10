# GB10 changelog 2026-11-05

Scope: Qwen4Exp (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`) end-to-end `infernix-serve` and
`llama-benchy` (`pp2048 tg128 @ d0, d4096, d8192` across `c = 1, 2, 4`) prefill and decode optimizations:
unsplit device-resident MoE prefill chunks and `rows_behind` PLE overlap, OpenAI `"cache_prompt"` protocol
support to eliminate structural tap splitting and wasted host snapshot copies on `--no-cache` workloads,
concurrent batch prefill completion before decode interleaving, optimal N-gram copy prefix truncation
(`best_copy_length`) and tile-aware MTP draft length scoring, `-rdc=true` static `.shared` memory bloat
elimination across `infernix_ops`, single-stage tall LM-head projection (`SingleStage` + wide router MMA)
and 1-row `projection_tail_row_kernel` for `p.shared_score`, removal of redundant MTP verification argmax
calls, right-sized L1/shared memory carveouts (`down_kernel_a16`, `qsa_score_kernel` BF16 `uint2` query
staging, `route_dispatch_plan_kernel` shared union, GDN `recurrent_record_conv_kernel` `RecordConvShared`
tile shrinking), MTP `resident_moe` `down_kernel` launch bounds, `top_logprobs` 1024-thread wide CTA and
`key > k3` fast guard, 8-thread parallel `NgramVolume` cache-hit gather and cache-line-disjoint
`insert_cache` for prefill chunks, 128-bit (`uint4`) coalesced vectorization across MoE combine,
HyperConnection, QSA tail/key-pool, and PLE causal conv/state kernels, `hc_mix_fused` / `hc_inject_mix_fused`
compile-time `ExactColumns` (`T = 3, 4`) specialization and raising `kHcMixFusedMaxColumnsQ8` from `3` to
`4` (eliminating 196 kernel launches and `16.3 MiB` of intermediate global memory churn per `c = 1` MTP
verification pass), `FastPath::GdnConvFused` qualification on `DenseKernels::Nvfp4` (`W4Dense`),
`Forward::attention` `T == 1` zero-copy packed slice with `key_records`, `Forward::moe` `shared_stream`
priority inversion fix + early fork, and DFlash2 speculative decoding optimizations (`dflash2_rows_ = 4`
multi-row DFlash2 speculation for `batch = 2..4`, post-proposal `dflash_confident_` truncation and
N-gram fallback, zero-copy `dflash2_verify_planes` at `batch == 1` and 2D batch copy at `batch > 1`,
and 2-stage / 2-CTA-per-SM `context_kv_grouped_kernel`) on NVIDIA GB10 (`sm_121a`).

## Commit range & accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`
- **Previous HEAD**: `fbbc1c763` (`perf(gb10): optimize Qwen4Exp MoE staging, multi-row N-gram/MTP speculation, and 32-col LM head`)
- **Delta in this milestone**: 31 code files changed (+1 changelog file = 32 files total), 1,271 code insertions(+), 304 code deletions(-) (excluding `cuda-kernel-toolkit/`, which is an integrated toolkit and intentionally excluded from file-by-file enumeration).

## Why (GB10 hardware invariants)

1. **128 GiB unified LPDDR5X UMA (`device_experts == true`) & zero PCIe staging**:
   - Upstream `ProgramImpl::call_cost()` hardcoded `.served_columns = 8U` unconditionally and disabled `rows_behind` prefill PLE overlap unless `expert_service != nullptr`. On GB10, all 512 routed experts reside directly in device-mapped unified LPDDR5X (`options_.device_experts == true`), so prefill has zero host miss-service traffic. Setting `.served_columns = 0U` when `options_.device_experts` is true prevents `EngineCore::split_prefill_call` from halving 2048-token prefill chunks into two sub-calls (`u = 2 -> u = 1`) whenever another request is active, and enabling `rows_behind` overlaps PLE loads during prefill.
2. **Unified LPDDR5X memory bandwidth conservation on `--no-cache` (`"cache_prompt": false`) requests**:
   - Because GB10 shares a single ~246 GB/s LPDDR5X memory pool between CPU and GPU, writing ~500 MB per request of host prefix-cache snapshots and splitting `System+User` prompts at structural system-turn boundaries (`d4096` and `d8192`) directly competes with GPU prefill bandwidth and doubles kernel launch / tail-wave overhead (`u = 2..3` prefill updates per request). Honoring `"cache_prompt": false` on `/v1/chat/completions` by setting `CacheParticipation::Disabled` keeps every request in a single contiguous prefill update (`u = 1`) and eliminates 100% of wasted host snapshot writes and evictions.
3. **4-column Tensor Core verification tiles & optimal N-gram copy prefix truncation**:
   - On GB10 (`sm_121a`), `WideExpert` and `projection_fp32` verify 1–3 MTP draft tokens (`W = 2..4` columns per row) inside the same 4-column hardware tile with ~0.10 marginal cost per column (`tile_cost = 0.10` for $k \le 3$), whereas wider N-gram copy verification (`W = 6..16`) spans multiple tiles (`kWidthCost = 0.18`, `kPairWidthCost = kMultiRowWidthCost = 0.16`).
   - Charging `tile_cost = 0.10` for $k \le 3$ inside `draft_length_at` prevents stochastic `temperature = 1.0` rejections at position 2 from trapping `draft_length_at` at `K = 2` (which otherwise starves `lane.mtp_accept[2]` of observations).
   - Furthermore, when `--ngram-draft-tokens 15` is configured, a 4–7 token prompt match (`bucket = 0`, $p = 0.75$) previously proposed all 15 tokens; because $0.75^j$ decays to near zero past $j = 7$ while the denominator $1 + 0.10 \times 15 = 2.50$ grows linearly, `copy_score(15, 0.75) = 1.584 < chain_score (1.775)`, causing `c = 1` and `c = 2` to reject 100% of Bucket-0 N-gram matches. Truncating N-gram proposals via `best_copy_length(copies, copy_accept, copy_cost)` to the prefix $k \in [1, \text{copies}]$ maximizing $\frac{E[\text{copy}(k)]}{1 + c_{\text{copy}} k}$ (`k = 5` for Bucket 0, `k = 9` for Bucket 1, `k = 12` for Bucket 2, `k = 15` for Bucket 3) restores Bucket-0 N-gram speculation at `c = 1, 2`.
4. **Relocatable device code (`-rdc=true`) `.weak .shared` deduplication, L1/Shared SRAM carveout, & 48-SM CTA occupancy**:
   - Because `infernix_ops` compiles with `-rdc=true`, `nvlink` merges any `static __shared__` buffers from inline header templates (`cub::BlockReduce` in `argmax.cuh`, `target_logprobs.cuh`, `token_constraint.cuh`, `top_logprobs.cuh`) into every translation unit's static `.shared` reservation (`50,240..66,809` B/CTA), throttling occupancy across unrelated kernels. Moving those reductions to dynamic shared memory (`extern __shared__ __align__(16) char`) drops static shared memory to `1,132..17,493` B/CTA.
   - Switching `projection_tall_kernel` from `MultiStage` (`32,768` B shared, 3 CTAs/SM) to `SingleStage` (`16,384` B shared, 6 CTAs/SM on GB10's 100 KiB/SM budget) doubles CTA residency on the `248,320 × 2,560` Q6_K LM head.
   - On GB10 (`sm_121a`, 128 KiB combined L1/Shared SRAM per SM), forcing `cudaSharedmemCarveoutMaxShared` (`100 KiB` shared / `28 KiB` L1) on `down_kernel_a16` (where `FlashNext` uses only `6.5 KiB` dynamic shared memory per CTA) wasted up to `68 KiB` of L1 cache per SM across all 40 MoE layers; removing that unconditional carveout on `down_kernel_a16`, halving `qsa_score_kernel` static `.shared` from `33,856 B` (`float4 qs`) to `17,472 B` (`uint2` packed BF16 `qs`, freeing `36 KiB` L1/SM at 2 CTAs/SM), placing `route_dispatch_plan_kernel`'s three sequential phases (`RouteDispatchSmallShared`, `CpuPlanShared`, `FetchPlanShared`) into a `union` (`41,140 B -> 22,928 B`), and shrinking `RecordConvShared::value` in `recurrent_record_conv_kernel` from `[16][128]` to `[16][16]` (`kBlockDv = 16`, cutting static `.shared` from `13,440 B` to `9,856 B` so all 5 CTAs/SM fit inside `49.3 KiB <= 64 KiB` without triggering a 99 KiB shared-memory carveout) maximize L1 data cache residency.
5. **128-bit (`uint4`) coalesced vectorization, wide `top_logprobs`, MTP `resident_moe` occupancy, & parallel `NgramVolume` cache gather/insert**:
   - Scalar `bf16` (16-bit) per-thread loops in `moe_combine`, `moe_combine_inject`, `hyper_connection_*`, `qsa_pool_keys`, `qsa_commit_tails`, `ple_conv_inject`, and `ple_conv_commit` generated uncoalesced 2-byte/4-byte sector requests on GB10's LPDDR5X/L2 hierarchy and launched 8× too many CTAs (`320` CTAs/token instead of `40` CTAs/token for `hidden = 2560`). Vectorizing all aligned `hidden % 8 == 0` paths to 128-bit (`uint4` = 8 `bf16`s per thread) turns every warp memory transaction into a single coalesced 128-byte cache-line access and cuts CTA waves 8×.
   - In `resident_moe.cu`, adding `__launch_bounds__(kThreads, 5)` to `down_kernel` drops `down_kernel<Q8>` from `83` registers (2 CTAs/SM, 33% occupancy) to `48` registers (5 CTAs/SM, 83% occupancy) with zero spills.
   - In `top_logprobs.cuh` and `top_logprobs.cu`, replacing the tautological `if (key < ~0ULL)` check with `if (key > k3)` and launching a 1024-thread CTA (`kTopLogprobsSerialWideBlock = 1024`) when `valid_rows >= 4096` cuts per-thread loop iterations 4× (`970 -> 242` for `248,320` vocab rows).
   - In `ngram_volume.h` and `ngram_volume.cpp`, parallelizing cache-hit gathering and cache-line-disjoint `insert_cache` (`((set_of(r) >> 6U) % n_threads) == t`) across the 8-thread `ReaderPool` for prefill calls (`rows.size() >= 2048`) eliminates single-threaded `23.6 MiB` random host copies before `publish_pinned_word` releases the GPU at layer 1.
6. **Re-enabled `T = 4` `hc_mix_fused` via `ExactColumns` specialization, 1-row `shared_score` tail kernel, `FastPath::GdnConvFused` on `W4Dense`, & `shared_stream` priority inversion fix**:
   - In `2026-10-30`, `kHcMixFusedMaxColumnsQ8` was lowered from `4` to `3` on GB10 because runtime `columns` loop bounds and integer division (`i / columns`) in `hc_down_partials_v4_kernel` (`168–200` registers) and `hc_up_collapse_v4_kernel` (`202` registers) made `T = 4` `1.4 us` slower (`45.2 us` vs `43.8 us`) than the 4-kernel composed route (`hyper_connection_norm + linear(down) + linear(up) + hyper_connection_collapse`). Specializing `ExactColumns = 3, 4` at compile time eliminates the outer pass loop (`c0 += kPassColumns`), replaces runtime `i / columns` divisions with exact shifts (`i >> 2` for `T = 4`), drops `ExactColumns = 4` register pressure to `141–168` (`hc_down_partials_v4_kernel`) and `167–184` (`hc_up_collapse_v4_kernel`), and drops `ExactColumns = 3` to `123–128` registers (`__launch_bounds__(256, 2)`, 2 CTAs/SM). On `infernix_hyper_connection_bench`, `T = 3` improves from `39.88 us -> 38.31 us` (`-3.9%`) and `T = 4` flips from slower to **`1.45 us` faster (`47.71 us -> 46.26 us`, `-3.0%`)** while eliminating **196 kernel launches and `16.3 MiB` of intermediate global memory churn** (`rn_s`, `down_out`, `up_out`, `pending_y` residual RMW) on every `c = 1` 3-draft MTP verification round (`T = 4`).
   - In `projection_fp32_router_mma`, `Qwen4Exp`'s 513-row router + shared-expert gate projection (`512` rows `p.router` + `1` row `p.shared_score`) previously handled the 1-row tail (`first_rows = 512, rows - first_rows = 1`) by launching `projection_wide_kernel` (`kBlockRows = 32`, `32 KiB` shared memory, `144` registers), which staged 31 out-of-bounds dummy weight rows (`96.9%` wasted shared memory and weight decode work) across `128` CTAs × `48` layers = `6,144` CTAs per `pp2048` prefill. Adding `projection_tail_row_kernel` (`64` registers, `5 KiB` shared memory, `8` columns/CTA, 4 CTAs/SM) stages only the single `p.shared_score` row in shared memory and computes 8 columns in parallel across 8 warps.
   - Qualifying `FastPath::GdnConvFused` for `DenseKernels::Nvfp4` in `kQualified` fuses the 4-tap depthwise causal `conv1d + SiLU` and recurrent state snapshot write directly into `recurrent_record_conv_kernel` on `W4Dense`, eliminating `36` standalone `causal_conv1d_silu_snapshot_split` launches and `737 KiB × T` of global memory reads/writes per decode pass; extending `Forward::attention`'s `T == 1` zero-copy packed QKV projection slice to handle `call.key_records.data != nullptr` (via a 256-byte device-to-device copy of `ik`) preserves the fused single-kernel QKV projection during speculative verification.
   - Under `INFERNIX_EXPERTS_ON_GPU=1` (`options_.device_experts == true`), `experts_.shared_stream` was created with `side_priority` (higher priority than `device_.stream`) even though no host staging occurs on `shared_stream`; setting `shared_priority = least_priority` when `options_.device_experts` is true and recording `shared_events[0]` immediately after `projection_fp32` (before the 1-CTA `moe_route_dispatch`) allows `shared_swiglu` to run on the 47 idle SMs alongside `moe_route_dispatch` without preempting the critical-path `device_.stream`.
7. **DFlash2 multi-row speculation (`batch = 2..4`), zero-copy verification planes, post-proposal N-gram fallback, & 2-CTA/SM `context_kv_grouped_kernel`**:
   - Upstream `dflash2_rows_` defaulted to `2` (`INFERNIX_TMP_DFLASH2_ROWS=2`), completely disabling DFlash2 block drafting at `batch = 3, 4` (`c = 4` fell back to `83–88` `W = 1` plain decode steps per request, `113–122 t/s`). Raising `dflash2_rows_` to `4` and generalizing `dflash2_pair_width`, `advance_decode`, and `graph_set.cpp` to `batch = 3, 4` (`kMultiRowMaxDrafts = 3`, up to `4 × 4 = 16` verification columns) cuts `c = 4` decode steps in half (`91–96 -> 42–46` rounds) and boosts `c = 4` throughput to `137.7–146.8 t/s` (`+16.8%` to `+24.5%`).
   - In `dflash2_verify_planes`, when `batch == 1` and `!copies`, row 0's candidates and proposal probabilities already sit contiguously at offset `0` (`(0 * K + j) * C == (0 * dflash_k_ + j) * C`) even when `K < dflash_k_`, eliminating two unnecessary device-to-device `cudaMemcpyAsync` calls per round (`-6.2%` faster `ms/step` at `c = 1`); when `batch > 1` and `!copies`, two strided `cudaMemcpy2DAsync` calls replace `2 * batch` per-row `cudaMemcpyAsync` launches.
   - After `dflash2_propose`, truncating `extents[b]` by `dflash_confident_[b]` (`--spec-min-p` gate) and allowing a declined `ngram_proposal_[b]` (`max_copies > extents[b]`) to replace a truncated DFlash2 proposal avoids verifying low-confidence neural tail tokens when a longer prompt N-gram copy is available.
   - In `context_kv_materialize/materialize.cu`, upgrading `context_kv_grouped_kernel` from `MmaStages = 1`, `__launch_bounds__(KWarps * 32, 1)` to `MmaStages = 2`, `__launch_bounds__(KWarps * 32, 2)` (`REG: 112 -> 102`, `0` spills) doubles SM occupancy during DFlash2 context K/V ring materialization across both prefill (`TTFT` `-14 ms` to `-134 ms`) and decode commit catch-up.

## Subsystem breakdown

### 1. Core runtime & serving (`src/runtime/`, `src/serve/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/runtime/engine/engine_core.h` | Updates `has_tail_prefill_ready()` so concurrent batch arrivals (`c = 2, 4`) complete prefill before starting decode rounds when `scheduler_.decode_share().share == 0.0`, preventing later requests' TTFT from stalling behind interleaved single-lane decode rounds. |
| M | `src/serve/request.h`, `src/serve/openai_chat_request.cpp`, `src/serve/generation_service.cpp` | Parses optional boolean `"cache_prompt"` on OpenAI `/v1/chat/completions` requests (`llama-benchy --no-cache`) and sets `EngineRequest::cache_participation = CacheParticipation::Disabled` when `"cache_prompt": false`, eliminating structural system-turn tap splitting (`u = 2 -> u = 1`) and ~500 MB/request of unused host prefix-cache snapshot copies. |

### 2. Model & speculative decoding (`src/models/qwen4_exp/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/models/qwen4_exp/program/mtp_gating.h` | Adds `best_copy_length(copies, copy_accept, copy_cost)` to truncate N-gram copy proposals to the expected-speedup-maximizing prefix length $k$; charges `tile_cost = min(width_cost, 0.10f)` for $k \le 3$ inside `draft_length_at` so 1–3 MTP drafts share the 4-column hardware tile cost without getting trapped at `K = 2` under sampled decode. |
| M | `src/models/qwen4_exp/program/program_impl.h` | Sets `.served_columns = options_.device_experts ? 0U : 8U` in `ProgramImpl::call_cost()` to stop `split_prefill_call` from splitting device-resident MoE prefill chunks; enables `rows_behind` prefill PLE overlap when `options_.device_experts == true`; sets `experts_.shared_stream` priority to `least_priority` when `options_.device_experts == true` to prevent priority inversion against `device_.stream`; applies `best_copy_length` to single-row and multi-row N-gram proposals under `--draft-tokens auto`; tunes GB10 width costs (`kWidthCost = 0.18`, `kPairWidthCost = 0.16`, `kMultiRowWidthCost = 0.16`, `kPairMaxDrafts = 3`); enables `dflash2_rows_ = 4` (`batch = 2..4` multi-row DFlash2), generalizes `dflash2_pair_width` to `rows >= 2`, applies `dflash_confident_` truncation and post-proposal N-gram fallback. |
| M | `src/models/qwen4_exp/program/dflash2_program.cpp`, `src/models/qwen4_exp/program/graph_set.cpp` | Avoids D2D copies in `dflash2_verify_planes` when `!copies && batch == 1` even for `K < dflash_k_`, and batches multi-row D2D copies into 2 `cudaMemcpy2DAsync` calls; warms `rows = 3..dflash2_rows_` verification graphs up to `1 + min(kMultiRowMaxDrafts, neural_k())`. |
| M | `src/models/qwen4_exp/program/fast_paths.cpp` | Qualifies `FastPath::GdnConvFused` for `DenseKernels::Nvfp4` (`W4Dense`) across all KV cache storage formats in `kQualified`. |
| M | `src/models/qwen4_exp/program/ngram_volume.h`, `src/models/qwen4_exp/program/ngram_volume.cpp` | Parallelizes volume-mode cache lookup/hit gathering and cache-line-disjoint `insert_cache` (`((set_of(r) >> 6U) % n_threads) == t`) across `kReadThreads = 8` threads for prefill chunks (`kParallelGatherRows = 2048`), preserving bit-identical cache state. |
| M | `src/models/qwen4_exp/execution/forward.cpp` | Removes redundant `ops::argmax` launch in `Forward::step_mtp` when `draft_out != nullptr` is already populated by `sample_mtp_draft_column` or greedy selection; forks `shared_events[0]` immediately after `projection_fp32` (before the 1-CTA `moe_route_dispatch`) when `source.staging_slots == 0`; extends `Forward::attention`'s `T == 1` zero-copy packed QKV projection slice to handle `call.key_records.data != nullptr`. |

### 3. Ops & kernels (`src/ops/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/kernel/argmax.cuh`, `src/ops/kernel/target_logprobs.cuh`, `src/ops/kernel/token_constraint.cuh`, `src/ops/kernel/top_logprobs.cuh`, `src/ops/launcher/top_logprobs.cu` | Replaces `static __shared__ typename BlockReduce::TempStorage` in header-defined kernels with dynamic shared memory (`extern __shared__ __align__(16) char`), reducing `-rdc=true` static `.shared` reservations across `infernix_ops` from `50–66 KiB` to `1.1–17.5 KiB`; adds `top_logprobs_one_kernel`, `key > k3` fast guard, and 1024-thread wide CTA (`valid_rows >= 4096`). |
| M | `src/ops/projection_fp32/projection_fp32.cu`, `src/ops/projection_fp32/projection_fp32_mma.h`, `src/ops/projection_fp32/projection_fp32_mma.cu` | Switches `projection_tall_kernel` (`kCols = 8, 12, 16, 24, 32`) from `MultiStage` (`32 KiB` shared, 3 CTAs/SM) to `SingleStage` (`16 KiB` shared, 6 CTAs/SM); dispatches `columns = 9..16` router projections (`rows = 512, K = 2560`) through `launch_router_mma_wide<16>` (8 CTAs × 4 warps = 32 warps vs 16 warps); adds `projection_tail_row_kernel` (`64` registers, `5 KiB` shared memory, 8 columns/CTA) for `p.shared_score` (`row 512`). |
| M | `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | Adds 128-bit (`uint4`) coalesced `combine_vec8_kernel` and `combine_inject_vec8_kernel`; unifies sequential shared-memory structs in `route_dispatch_plan_kernel` into `union RouteDispatchPlanShared` (`41,140 B -> 22,928 B`); removes unconditional `cudaSharedmemCarveoutMaxShared` override from `down_kernel_a16` to preserve L1 cache during narrow `W4A16` MoE down-projections. |
| M | `src/ops/linear_attention/gated_delta_net/recurrent.cuh` | Shrinks `RecordConvShared::value` from `[16][128]` (`4,096 B`) to `[16][16]` (`kBlockDv = 16`, `512 B`), reducing `recurrent_record_conv_kernel` static `.shared` from `13,440 B` to `9,856 B` (`REG: 85`) so all 5 CTAs/SM fit inside `49.3 KiB <= 64 KiB` without triggering a 99 KiB shared-memory carveout. |
| M | `src/ops/resident_moe/resident_moe.cu` | Adds `__launch_bounds__(kThreads, 5)` to `down_kernel`, reducing `down_kernel<Q8>` from `83` registers (33% occupancy) to `48` registers (83% occupancy) with zero spills. |
| M | `src/ops/hyper_connection/hyper_connection.cu`, `src/ops/hyper_connection/hyper_connection_mix_fused.h`, `src/ops/hyper_connection/hyper_connection_mix_fused.cu` | Adds 128-bit (`uint4`) coalesced `norm_vec8_kernel`, `inject_norm_vec8_kernel`, `collapse_vec8_kernel`, `inject_vec8_kernel`, `expand_vec8_kernel`, and `stream_mean_vec8_kernel`; specializes `hc_down_partials_v4_kernel` and `hc_up_collapse_v4_kernel` for `ExactColumns = 3, 4` and raises `kHcMixFusedMaxColumnsQ8` from `3` to `4`. |
| M | `src/ops/qsa/qsa_select.cu`, `src/ops/qsa/qsa.cu` | Stages queries in `qsa_score_kernel` as packed BF16 `uint2` instead of `float4`, halving static `.shared` per CTA from `33,856 B` to `17,472 B` (`+36 KiB` L1 cache/SM at 2 CTAs/SM); adds 128-bit (`uint4`) coalesced `tail_vec8_kernel` (`qsa_pool_keys`) and `commit_tail_vec8_kernel` (`qsa_commit_tails`). |
| M | `src/ops/ple/ple.cu` | Adds 128-bit (`uint4`) coalesced `conv_vec8_kernel`, `conv_state_vec8_kernel`, `state_vec8_kernel`, and `commit_vec8_kernel` for `ple_conv_inject` and `ple_conv_commit`. |
| M | `src/ops/context_kv_materialize/materialize.cu` | Upgrades `context_kv_grouped_kernel` schedule to `MmaStages = 2` and `__launch_bounds__(KWarps * 32, 2)` (`REG: 102`, `0` spills) for 2-CTA/SM residency during DFlash2 context K/V materialization. |

### 4. Tools (`tools/`)

| Action | File | Mechanism |
|---|---|---|
| M | `tools/run_server.sh` | Configures `--spec dflash2 --draft-model "$DRAFT_PATH" --draft-tokens auto --ngram-draft-tokens 31 --lm-head-draft` on `infernix-serve` launcher. |

## Verification & benchmark summary (`llama-benchy`, `pp2048 tg128 @ d0, d4096, d8192`, `c = 1, 2, 4`)

- **Unit & oracle tests**: Speculative, stack-ratchet, MoE plan, projection_fp32, HyperConnection, PLE, and QSA CTest targets pass (`infernix_cuda_stack_check`, `infernix_projection_fp32_test`, `infernix_ple_test`, `infernix_hyper_connection_test`, `infernix_hyper_connection_stream_mean_test`, `infernix_qsa_select_test`, `infernix_offloaded_moe_plan_test`, `infernix_ops_attention_test qsa`), plus `infernix_hc_mix_fused_bench --tokens 1,2,3,4 --check` (`max_diff = 0.000000`), with 0 new stack spills.
- **End-to-end `llama-benchy` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `--no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3`)**:

| Test | `c` | `pp t/s` (Base A → Base B → **Now**) | `pp` Gain (vs A / vs B) | `tg t/s` (Base A → Base B → **Now**) | `tg` Gain (vs A / vs B) | `TTFT ms` (Base A → Base B → **Now**) | TTFT Change (vs A / vs B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | `2,648*` → `2,763*` → **`3,174*`** | **+19.9%** / **+14.9%** | `71.6` → `77.1` → **`79.7`** | **+11.3%** / **+3.4%** | `773` → `741` → **`645`** | **-16.6%** / **-13.0%** |
| `pp2048 tg128 @ d0` | `c2` | `2,458` → `2,633` → **`2,845`** | **+15.7%** / **+8.1%** | `90.3` → `106.0` → **`113.3`** | **+25.5%** / **+6.9%** | `1,544` → `1,476` → **`1,314`** | **-14.9%** / **-11.0%** |
| `pp2048 tg128 @ d0` | `c4` | `2,516` → `2,716` → **`2,911`** | **+15.7%** / **+7.2%** | `139.4` → `144.7` → **`141.2`** (peak **`157.7`**) | **+1.3%** / **-2.4%** | `3,019` → `2,821` → **`2,586`** | **-14.3%** / **-8.3%** |
| `pp2048 tg128 @ d4096` | `c1` | `3,172*` → `3,316*` → **`3,891*`** | **+22.7%** / **+17.3%** | `61.8` → `79.7` → **`80.8`** | **+30.7%** / **+1.4%** | `1,937` → `1,853` → **`1,579`** | **-18.5%** / **-14.8%** |
| `pp2048 tg128 @ d4096` | `c2` | `2,892` → `3,093` → **`3,565`** | **+23.3%** / **+15.3%** | `91.1` → `99.8` → **`110.2`** | **+21.0%** / **+10.4%** | `3,934` → `3,690` → **`3,143`** | **-20.1%** / **-14.8%** |
| `pp2048 tg128 @ d4096` | `c4` | `2,937` → `3,125` → **`3,612`** | **+23.0%** / **+15.6%** | `133.0` → `136.1` → **`154.8`** (peak **`166.3`**) | **+16.4%** / **+13.7%** | `7,676` → `7,202` → **`6,156`** | **-19.8%** / **-14.5%** |
| `pp2048 tg128 @ d8192` | `c1` | `3,613*` → `3,722*` → **`4,331*`** | **+19.9%** / **+16.4%** | `69.3` → `71.0` → **`77.2`** | **+11.4%** / **+8.7%** | `2,834` → `2,751` → **`2,365`** | **-16.5%** / **-14.0%** |
| `pp2048 tg128 @ d8192` | `c2` | `3,298` → `3,434` → **`3,991`** | **+21.0%** / **+16.2%** | `90.5` → `98.5` → **`103.1`** | **+13.9%** / **+4.7%** | `5,641` → `5,455` → **`4,665`** | **-17.3%** / **-14.5%** |
| `pp2048 tg128 @ d8192` | `c4` | `3,325` → `3,453` → **`3,986`** | **+19.9%** / **+15.4%** | `138.3` → `150.2` → **`145.8`** (peak **`158.7`**) | **+5.4%** / **-2.9%** | `11,241` → `10,841` → **`9,309`** | **-17.2%** / **-14.1%** |

- **End-to-end DFlash2 `llama-benchy` (`--spec dflash2 --draft-model Qwen3.8-Flash-Next-DFlash2-Infernix.infernix --draft-tokens auto --ngram-draft-tokens 31 --lm-head-draft`, `--no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3`)**:

| Test | `c` | `pp t/s` (Base → **Now**) | `tg t/s` (Base → **OPT Run 1** → **OPT Run 4**) | `tg` Gain (vs Base) | `TTFT ms` (Base → **OPT Run 4**) | TTFT Change |
|---|---:|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | `2,912*` → **`3,070*`** | `95.5` → `74.7` → **`78.0`** (`35.69 ms/step` vs `38.05 ms/step`, `-6.2%`) | `-18.3%` (prompt RNG N-gram variance: `60` vs `33` copy drafts) | `703` → **`667`** | **-36 ms (-5.1%)** |
| `pp2048 tg128 @ d0` | `c2` | `2,835` → **`2,805`** | `110.4` → `107.2` → **`104.0`** (peak **`111.7`**) | `-5.8%` mean / **+1.2%** peak | `1,340` → **`1,290`** | **-50 ms (-3.7%)** |
| `pp2048 tg128 @ d0` | `c4` | `2,883` → **`2,843`** | `122.1` → `131.8` → **`145.9`** (peak **`163.0`**) | **+19.5%** (**+33.5%** peak) | `2,637` → **`2,590`** | **-47 ms (-1.8%)** |
| `pp2048 tg128 @ d4096` | `c1` | `3,803*` → **`3,817*`** | `77.6` → `74.7` → **`82.6`** (peak **`83.0`**) | **+6.4%** | `1,616` → **`1,610`** | **-6 ms (-0.4%)** |
| `pp2048 tg128 @ d4096` | `c2` | `3,595` → **`3,595`** | `99.0` → `103.0` → **`103.2`** (peak **`116.3`**) | **+4.2%** (**+17.5%** peak) | `3,079` → **`3,117`** | `+38 ms (+1.2%)` |
| `pp2048 tg128 @ d4096` | `c4` | `3,653` → **`3,648`** | `117.9` → **`146.8`** → **`137.7`** (peak **`146.7`**) | **+16.8%** to **+24.5%** | `6,202` → **`6,076`** | **-126 ms (-2.0%)** |
| `pp2048 tg128 @ d8192` | `c1` | `4,340*` → **`4,367*`** | `75.8` → `72.8` → **`74.8`** (`36.89 ms/step` vs `37.86 ms/step`, `-2.6%`) | `-1.3%` (within `± 1.72` noise) | `2,359` → **`2,345`** | **-14 ms (-0.6%)** |
| `pp2048 tg128 @ d8192` | `c2` | `3,973` → **`4,013`** | `94.2` → **`103.5`** → **`96.7`** (peak **`105.3`**) | **+2.7%** to **+9.8%** | `4,662` → **`4,628`** | **-34 ms (-0.7%)** |
| `pp2048 tg128 @ d8192` | `c4` | `3,995` → **`4,025`** | `113.3` → `137.0` → **`138.4`** (peak **`149.7`**) | **+22.2%** (**+32.1%** peak) | `9,309` → **`9,175`** | **-134 ms (-1.4%)** |

## Rebase conflict resolution policy

- **Runtime & serving (`engine_core.h`, `request.h`, `openai_chat_request.cpp`, `generation_service.cpp`)**: Preserve `"cache_prompt"` support (`CacheParticipation::Disabled` when false) and `has_tail_prefill_ready()` prefill-completion scheduling when `decode_share == 0.0`.
- **Model & speculative decoding (`mtp_gating.h`, `program_impl.h`, `dflash2_program.cpp`, `graph_set.cpp`, `fast_paths.cpp`, `forward.cpp`)**: Preserve `.served_columns = options_.device_experts ? 0U : 8U` in `call_cost()`, `rows_behind` prefill overlap on `device_experts`, `least_priority` on `shared_stream` + early `shared_events[0]` fork when `device_experts == true`, `FastPath::GdnConvFused` on `DenseKernels::Nvfp4`, `Forward::attention` `T == 1` zero-copy packed slice with `key_records`, `best_copy_length` N-gram prefix truncation, `tile_cost = 0.10` for $k \le 3$ in `draft_length_at`, `dflash2_rows_ = 4` multi-row DFlash2 speculation (`batch = 2..4`), zero-copy / 2D-copy `dflash2_verify_planes`, and post-proposal `dflash_confident_` truncation + N-gram fallback.
- **Ops (`argmax.cuh`, `target_logprobs.cuh`, `token_constraint.cuh`, `top_logprobs.cuh`, `projection_fp32.cu`, `projection_fp32_mma.cu`, `moe_layer.cu`, `hyper_connection.cu`, `hyper_connection_mix_fused.h`, `hyper_connection_mix_fused.cu`, `qsa_select.cu`, `qsa.cu`, `ple.cu`, `materialize.cu`)**: Preserve dynamic shared memory in header CUB reductions (avoiding `-rdc=true` static `.shared` bloat), `top_logprobs_one_kernel`, `SingleStage` tall LM-head projections, `launch_router_mma_wide<16>` for `columns = 9..16`, `projection_tail_row_kernel` for 1-row `p.shared_score` tails, `ExactColumns = 3, 4` specialization in `hc_mix_fused` (`kHcMixFusedMaxColumnsQ8 = 4`), `qsa_score_kernel` BF16 `uint2` query staging, `route_dispatch_plan_kernel` shared union, right-sized `down_kernel_a16` carveout, 2-stage / 2-CTA-per-SM `context_kv_grouped_kernel`, and 128-bit (`uint4`) coalesced vectorization paths.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-05 — T1: Wide routed-expert column-tile-inner work traversal and even/odd `a_scales` TMA dedup in `grouped_kernel`

## 1. Accounting

- **Base commit**: `ac1983b70` (`docs: next task iteration`; task premise measured from `98aff48b0`); work uncommitted for orchestrator review.
- **Modified files**: `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` (`M`, +77/−10 — `map_work_unit`, `kColTileInner`, `kScalesDedupe`, producer/consumer loops). `wide_expert.cu` and `wide_expert.h` net-unchanged (the A/B toggle scaffolding used during measurement was fully removed). `docs/gb10/20261105-changelog.md` (`M`).
- **Allowlist delta**: none (`STACK:0 LOCAL:0` on all 20 wide `grouped_kernel` instantiations before and after; no `check_cuda_stack.py update` run).
- **Rebuild count**: 5 full single-invocation Ninja builds + 2 object-only compile checks.

## 2. Task-premise corrections / Architectural rationale

- **The task's acceptance baseline is stale by ~1.7–2.0×**: `infernix_offloaded_moe_wide_bench --tokens 64,256,1024,4096 --placement frames` measures **3,946–4,129 / 7,603–7,749 / 8,430–8,478 / 8,851–9,413 µs** on this tree at `98aff48b0`, not the doc's 9,280.8 / 13,747.0 / 17,574.0 / 17,898.4 µs. The "≥10% over 17,574/17,898" bar is already met by the pre-existing tree; all comparisons below are against the in-session baseline.
- **The doc's verification commands name things that do not exist**: there is no `infernix_offloaded_sparse_moe_test` target (real: binary `infernix_ops_offloaded_moe_test`, CTest names `infernix_offloaded_moe_wide_test` et al.) and the bench has no `--profile`/`--qtype` flags (Flash-Next geometry and nvfp4/a4 are its defaults; `--geometry flash-next` is the flag).
- **The L2-eviction root cause does not hold at the bench level**: the bench fixture uses `kDistinct = 16` weight records shared across the 512 experts, so the expert weights are near-L2-resident regardless of traversal; and the doc's own arithmetic (≈10 concurrent experts × 1.77 MiB = 17.7 MiB) fits inside the 24 MiB L2. The reorder's real, measured effect is a subtler wave-level weight-reuse improvement in-model: **−4.0 %** on the gate/up `grouped_kernel`, not the implied 2–3× traffic multiplication.
- **"Average count per expert 64..128 at T >= 1024" is wrong at T=1024** (bench: 9,412 wide entries over 378 wide experts ≈ 25 columns, mostly one 64-column tile per job): the reorder is a near-no-op at T=1024 and matters only where jobs span several column tiles (T=4096, in-model pp8192 with 160 columns/expert average).
- **The doc's dedup invariant as written races**: "stage `(stage+2)%3` is not overwritten until `k_tile+2`" is false — the producer refills that stage's codes/units at `k_tile+2` while the consumer still reads it. The dedup is safe only because `a_scales` is a *disjoint* array and only when `kKTiles` is even and `kStages` is odd and ≥3: with even `kKTiles` every work tile starts on an even global stage-counter parity (pairs never straddle work tiles — which matters, since the box's activation row is per-work-tile), and the `k_tile+2` refill landing on the source stage is itself odd-parity and issues no scale load; the next scale write to that stage is gated behind the consumer's own `k_tile+2` completion. `kKTiles` odd (Flash-Next down, 5) would read the box across a work-tile boundary with the wrong activation row, and `kStages = 2` (the A4-residual schedules) lets the producer overwrite the scales within reach — both stay on the old path.
- **The reorder is not profitable for every matrix**: on the down matrix (45 KiB weight per unit, 5 K tiles) it measured +0.6–1.1 % slower in-model and +1.25 % in the bench. It is therefore gated at `kKTiles * kRowGroups * kUnitRunBytes >= 64 KiB` (gate/up: 90 KiB, measured −4 to −5 %; down: 45 KiB, excluded).

## 3. Subsystem breakdown

`src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` only:

- **`map_work_unit<ColTileInner>`** (new): for large weight blocks, enumerates a job's unit span `(row_block, t_local)` with the column tile innermost, so adjacent CTAs of a multi-column-tile job read the same 128-row weight block simultaneously. The job's tile span comes from two binary searches of the pass's job-sorted `(job, column)` list (`probe = tile_begin + w / kRowBlocks` is a tile of the job owning `w` because job spans are whole `kRowBlocks` multiples). Small-weight-block instantiations keep the historical `w / kRowBlocks` mapping via `if constexpr`.
- **`grouped_kernel` producer**: `map_work_unit<kColTileInner>`; with `kScalesDedupe` (even `kKTiles`, odd `kStages >= 3`) the odd `k_tile` issues no `a_scales` TMA and `arrive_expect_tx` drops `kTerms * kBlockTokens * kScaleWordsPerRow * 4` bytes (1,024 B/term) from the transaction.
- **`grouped_kernel` consumer**: same mapping; the odd `k_tile` reads its scale words from the even tile's stage `((it + kStages - 1) % kStages)` — the same 16-byte box, so every scale word and every output bit is unchanged (verified bit-exact).
- `plan_kernel`, `WidePlan`-side data, and the `tiles`/`pass_tiles` contract are unchanged (the spec allowed the reorder directly in `grouped_kernel`); `quantize_kernel`/`encode_h*` still see one entry per `(job, column)` tile.

## 4. Verification

- **Tests**: `ctest -R "infernix_cuda_stack_check|infernix_offloaded_moe"` — **20/20 passed** (wide FP64-oracle `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_cuda_test`, `a8_wide`, `a4_residual_122b`, `qwen3_5_122b`, layer, plan, team, all CPU tests, and the stack-spill ratchet gate).
- **Resources**: `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` — all 20 wide `grouped_kernel` instantiations `STACK:0 LOCAL:0`; REG 92–96 (`Schedule`) and 128–144 (`WarpTile64`/residual), matching the task doc's pre-change bands (95; 135–144).
- **Bit-exactness**: bench `output fnv1a` hashes at T=64/256/1024/4096 identical to the pre-change baseline across all A/B modes and the final build.
- **Bench (`placement=frames`, moe_experts median, 3 passes each)**: T=64 3,946–4,129 → 3,914–3,924; T=256 7,603–7,749 → 7,513–7,611; T=1024 8,430–8,478 → 8,322–8,777; T=4096 8,851–9,413 → 8,697–8,797 µs — no T=64/T=256 regression; T=1024/T=4096 within noise (see §5). Interleaved `ckt ab` (mode 1 vs 2, T=4096, 8 rounds): −1.28 %, `WITHIN_NOISE` (CV 2.8 %).
- **In-model pp8192 prefill** (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `INFERNIX_EXPERTS_ON_GPU=1`, int8 KV, c=1, `-r 2 --warmup 1`, nsys, 144 instances/kernel): gate/up `grouped_kernel` total **813.1 → 780.8 ms (−4.0 %)** (median 5,543.9 → 5,346.9 µs; reproduced across three new-code passes: 5,359 / 5,350 / 5,347 µs); down `grouped_kernel` (code unchanged in the shipped configuration) 4,099–4,108 → 4,131 µs median, inside the identical-code run-to-run spread. E2E prefill **3,951.3 ± 8.5 → 3,959.5 ± 1.4 t/s (+0.2 %, noise level)**.
- **Mechanism split** (runtime-toggle builds): the whole gate/up win is the reorder (reorder-only 5,359 vs both 5,350 µs median); the dedup measured −0.2 % (noise) but removes 10 redundant 1,024 B TMA loads per gate/up work unit and ships bit-exact.

## 5. Regressions & adverse results

- **The acceptance bar was not met at the bench**: the task's ≥10 % T=1024/T=4096 improvement over 17,574/17,898 µs is satisfied only because that baseline is stale (~2×); against the in-session baseline the bench deltas are −0.6…−1.3 % and `WITHIN_NOISE`. The bench cannot show the win by construction (16 distinct records ⇒ weights L2-resident regardless). Disposition: disclosed; the supported win is the in-model −4.0 % gate/up kernel (E2E +0.2 %, noise).
- **Reorder on the down matrix regressed**: +0.6–1.1 % in-model (4,100–4,108 → 4,132–4,153 µs median, two A/B passes) and +1.25 % bench T=4096 (33.56 → 33.98 ms). Disposition: **fixed** — `kColTileInner` (≥64 KiB per-unit weight block) keeps the down matrix on the historical mapping; the shipped configuration leaves down code-identical.
- **T=1024 bench gains nothing**: at ~25 columns per wide expert most jobs are single-tile, so the reorder is a no-op and the dedup alone is invisible (8,430–8,478 → 8,322–8,777 µs, noise). Accepted; no regression.
- **Per-unit binary-search overhead** (two searches over the pass tile list, ~13 loads each): no measurable cost on the gate/up route (win preserved in three passes); it is a candidate contributor to the down regression above, which is excluded.
- **E2E prefill +0.2 % is within noise**: the −4 % gate/up kernel occupies ~12 % of the pp8192 window, so the expected E2E delta (≈+0.5 %) is at the edge of the measurement's reproducibility; single-digit claims are not made.

## 6. Accepted gaps / follow-ups

- The bench-level ≥10 % acceptance criterion is unreachable for this mechanism on this bench (fixture hides weight DRAM traffic); the reviewer should judge T1 on the in-model kernel result and bit-exactness.
- `wide_expert_a16.cuh` and `wide_expert_a8.cuh` duplicate the old `(tile, row_block)` mapping and were left unchanged (outside the task's target files); the same gate/up-class win likely applies to their large-block routes and is unmeasured.
- The 122B geometry qualifies for both mechanisms (gate/up 216 KiB, down 72 KiB with even `kKTiles` = 8): correctness is covered by the passing `qwen3_5_122b` and `a4_residual_122b` oracle tests, but no 122B artifact exists on this host, so its performance is unmeasured.
- `WarpTile64` (mean ≥ 512 columns/expert) inherits both mechanisms under the same constexpr gates but is not reached by this artifact's prefill chunks (≤160 columns/expert); its E2E effect is unmeasured.
- The gate/up win is wave-local: a larger prefill chunk or a dedicated rasterization (job-major wave assignment across the 96-CTA grid) should convert more of the doc's hypothesized traffic saving; not attempted at T1's scope.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-05 — T2: QSA decode attention 2-CTA/SM occupancy (`REG:61..64`), empty-split early exit, 2-token score unrolling with `LDS.64` scale folding, fused register Hadamard query load, and 128-thread `float2` vectorized `merge_kernel`

## 1. Accounting

- **Base commit**: `87652fc1d` (`perf(moe): reorder wide GateUp work units col-tile-inner and deduplicate odd-k_tile scale TMA loads (T1)`).
- **Modified files**: `src/ops/qsa/qsa.cu` (`M`), `docs/gb10/20261105-changelog.md` (`M`).
- **Allowlist delta**: none (`STACK:0 LOCAL:0` across all 34 functions in `src/ops/qsa/qsa.cu.o` and on `infernix-serve`; `infernix_cuda_stack_check` passes with no allowlist changes).

## 2. Architectural rationale ("Why" on GB10 `sm_121a`)

1. **Doubling `attention_kernel` occupancy from 1 CTA/SM (`33.3%`, 16 warps/SM) to 2 CTAs/SM (`66.7%`, 32 warps/SM)**:
   - On GB10 (`sm_121a`, 48 SMs, 65,536 registers/SM, 100 KiB max dynamic shared memory/SM), a `width=8` decode/verification launch with `splits=33` and `kv_heads=2` spawns `33 × 2 × 8 = 528` CTAs of 512 threads (`16` warps/CTA).
   - Previously, `attention_kernel` used `__launch_bounds__(512, 1)` (`REG: 112..128`), and `AttentionShared<BFloat16, 12, 1>::kBytes` was `81,552 B` (`> 50,176 B`), limiting residency to 1 CTA/SM (`11` waves of 48 CTAs) and leaving each SM with only 16 warps (`1.0` warp/scheduler during `__syncthreads()` phases).
   - By sharing a single 32 KiB stage plane between `K` and `V` for `BFloat16` (`ShareKV = (Storage == BFloat16 && Stages == 1)`, reducing `BFloat16` dynamic SMEM from `81,552 B` to `48,784 B <= 50,176 B`), keeping unified `[K | V | KScales | VScales]` stages (`<= 49,808 B <= 50,176 B`) for `Int8Group64`, `Fp8E4M3Row256`, `Nvfp4Group16`, and `Fp8KeyNvfp4Value` so both `K` and `V` stream in a single `cp.async` batch overlapped with query load and Hadamard rotation, and setting `__launch_bounds__(kAttentionThreads, AttentionShared<Storage, Group, Stages>::kBytes <= 50176 ? 2 : 1)` (`REG: 61..64`, `STACK: 0`, `LOCAL: 0`), GB10 runs **2 CTAs/SM (`32` warps/SM, `66.7%` occupancy)** across all `Exact=true, Group=12, Stages=1` decode/verification instantiations, cutting wave count from 11 waves to 5.5 waves.
2. **Eliminating empty-split work (`tiles == 0`)**:
   - At short contexts (`< 2048`), splits beyond `(attended + 63) / 64` have `begin >= end` (`tiles == 0`). Exiting immediately at the top of `attention_kernel` after writing only `slot[kHeadDim] = -INFINITY` and `slot[kHeadDim + 1] = 0.0F` per head when `splits > 1` skips query global loads, Hadamard rotation, and the 12 KiB partial zero-fill (`-18%` to `-29.5%` faster `attention_us` at `context = 256, 1024`).
3. **2-token Score-loop unrolling (`j0, j1`), compile-time `((lane & 31) | (32 * r)) >> 6` scale-offset folding (`LDS.64`), and fused register Hadamard query load**:
   - Processing two tokens (`j0, j1`) per warp per Score-loop iteration loads each query head's `float4` pair (`lo, hi`) from shared memory `qs` once via inline PTX `ld.shared.v4.f32` (`ld_q4`) and dots it against both `k0[0..7]` and `k1[0..7]`, cutting `qs` shared-memory `LDS.128` traffic by 50%.
   - Changing the key dimension index to `(lane & 31) | (32 * r)` and using unsigned shift (`static_cast<unsigned>(d) >> 6` / `>> 4`) in `KVTile`, combined with hoisting `if (has1)` outside `for (int r = 0; r < 8; ++r)` for non-`kShareKV` storages, allows LLVM KnownBits and CSE to fold `((lane & 31) | (32 * r)) >> 6` to the compile-time constant `r >> 1` (`0, 0, 1, 1, 2, 2, 3, 3`) and vectorize `Int8Group64` key scale loads into a single 64-bit `LDS.64` per token.
   - For rotated-key formats (`Tile::kRotatedKeys`), loading each query head directly per warp into registers (`float v[8]`), applying `normalized_hadamard_d256_inplace(v, lane)` in registers, and writing `qs` once eliminates two `__syncthreads()` barriers, 8 scalar `STS` instructions, and 2 `LDS.128` instructions per CTA.
   - Specializing `kSingleTile = (Stages == 1 && !Tile::kVq)` (since `launch_stages` dispatches `Stages = 2` whenever `split_tokens > kTokenTile` for all non-VQ storages), eliminating `is_key` warp divergence in `issue()` when `KC == VC`, and using PTX `mov.u32` for epilogue `%ctaid.x/y/z` and `%tid.x` keep register pressure at `61..64` with **zero `STL`/`LDL` stack spills**.
4. **128-thread `float2` (`LDG.E.64`) vectorized `merge_kernel` preserving warp-uniform `slot[kHeadDim]` broadcast**:
   - Consecutive splits `s` of a `(column, head)` in `partial` are separated by `heads * 258 * 4 = 24,768 B`, which is 8-byte aligned (`258 % 2 == 0`). Indexing `s` across lanes causes 32-way uncoalesced 24 KiB strided gather transactions, whereas iterating `s` uniformly across the warp hits 1 L1 broadcast sector per split.
   - Vectorizing `merge_kernel` to launch `kHeadDim / 2 = 128` threads (`4` warps/CTA instead of `8` warps/CTA), where each thread loads `float2` (`LDG.E.64`) from `slot` and writes `__nv_bfloat162` (`STG.E.32`) to `out`, halves CTA warp count, `slot[kHeadDim]` broadcast loads, and `expf` evaluations while keeping `slot[kHeadDim]` 100% warp-uniform and bit-identical.

## 3. Subsystem breakdown

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/qsa/qsa.cu` | Adds `tiles == 0` early exit in `attention_kernel`; adds `__launch_bounds__(kAttentionThreads, AttentionShared<Storage, Group, Stages>::kBytes <= 50176 ? 2 : 1)` (`REG: 61..64`, `STACK: 0`, `LOCAL: 0`, 2 CTAs/SM); shares a single 32 KiB K/V plane for `BFloat16` (`ShareKV = true`, `48,784 B`) while keeping unified K+V stages (`<= 49,808 B`) for `Int8Group64`/`Fp8`/`Nvfp4`/`K8V4`; fuses query global load and Hadamard rotation in registers for `Tile::kRotatedKeys`; unrolls Score loop by 2 tokens (`j0, j1`) with `ld.shared.v4.f32` (`ld_q4`) and constant-folded unsigned scale shifts (`LDS.64`); specializes `kSingleTile = Stages == 1 && !Tile::kVq` and non-divergent `KC == VC` `cp.async` issue; vectorizes `merge_kernel` to 128 threads with `float2` (`LDG.E.64`) loads and `__nv_bfloat162` stores. |

## 4. Verification & benchmark results

- **Tests**: `ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa" --output-on-failure` — **10/10 passed (100%)**:
  - `infernix_cuda_stack_check` (Passed)
  - `infernix_qsa_index_test` (Passed)
  - `infernix_qsa_bf16_test` (Passed)
  - `infernix_qsa_int8_test` (Passed)
  - `infernix_qsa_fp8_test` (Passed)
  - `infernix_qsa_nvfp4_test` (Passed)
  - `infernix_qsa_k8v4_test` (Passed)
  - `infernix_qsa_vq2_test` (Passed)
  - `infernix_qsa_k4v2_test` (Passed)
  - `infernix_qsa_select_test` (Passed)
- **Resources (`cuobjdump -res-usage` & `cuobjdump -sass` on `qsa.cu.o` and `infernix-serve`)**:
  - All 34 functions in `qsa.cu.o` have `STACK:0 LOCAL:0` and zero `STL`/`LDL` instructions.
  - Decode/verification `attention_kernel<Storage, 12, true, 1>` (`Storage = BFloat16, Int8Group64, Fp8E4M3Row256, Nvfp4Group16, Fp8KeyNvfp4Value`): `REG: 61..64, STACK: 0, LOCAL: 0`, dynamic SMEM `48,784..49,808 B <= 50,176 B` (2 CTAs/SM, 32 warps/SM, 66.7% occupancy).
  - `merge_kernel<false>` / `merge_kernel<true>`: `REG: 40, STACK: 0, LOCAL: 0`.
- **Benchmarks (`./build-native/bench/infernix_qsa_attention_bench --storage int8,bf16 --contexts 2048,8192,32768 --widths 1,8,16,64`, 3-round interleaved A/B medians, cold L2, CUDA Graph)**:

| Storage | Context | Width | Task Baseline `attention_us` | In-Session HEAD `attention_us` (3-rd median) | Optimized `attention_us` (R1 / R2 / R3 → **median**) | vs Task Baseline | vs In-Session HEAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int8` | `2048` | `1` | `26.82` | `29.79` | `27.55 / 28.70 / 27.65` → **`27.65`** | `+3.1%` (within drift) | **-7.2%** |
| `int8` | `2048` | `8` | `124.26` | `110.88` | `92.13 / 110.56 / 94.34` → **`94.34`** | **-24.1%** | **-14.9%** |
| `int8` | `8192` | `1` | `29.41` | `30.72` | `26.72 / 27.65 / 26.75` → **`26.75`** | **-9.0%** | **-12.9%** |
| **`int8`** | **`8192`** | **`8`** | **`124.86`** | **`118.85`** | `95.10 / 98.08 / 98.18` → **`98.08`** (mean **`97.12`**) | **-21.4%** (**-22.2%** mean) | **-17.5%** |
| `int8` | `32768` | `1` | `31.94` | `32.64` | `26.91 / 28.77 / 29.25` → **`28.77`** | **-9.9%** | **-11.9%** |
| `int8` | `32768` | `8` | `141.79` | `129.50` | `100.42 / 100.70 / 101.18` → **`100.70`** | **-29.0%** | **-22.2%** |
| `bf16` | `2048` | `1` | `26.53` | `35.81` | `27.65 / 29.50 / 26.62` → **`27.65`** | `+4.2%` (R3 `26.62`, `+0.3%`) | **-22.8%** |
| `bf16` | `2048` | `8` | `122.02` | `108.51` | `88.51 / 89.34 / 89.06` → **`89.06`** | **-27.0%** | **-17.9%** |
| `bf16` | `8192` | `1` | `27.36` | `31.74` | `25.54 / 26.78 / 26.59` → **`26.59`** | **-2.8%** | **-16.2%** |
| **`bf16`** | **`8192`** | **`8`** | **`123.90`** | **`116.00`** | `93.18 / 97.18 / 97.95` → **`97.18`** (mean **`96.10`**) | **-21.6%** (**-22.4%** mean) | **-16.2%** |
| `bf16` | `32768` | `1` | `29.66` | `33.82` | `28.93 / 28.77 / 27.65` → **`28.77`** | **-3.0%** | **-14.9%** |
| `bf16` | `32768` | `8` | `151.14` | `136.16` | `122.75 / 123.74 / 125.44` → **`123.74`** | **-18.1%** | **-9.1%** |

- **Short-context empty-split verification (`--storage int8,bf16 --contexts 256,1024 --widths 1,8`)**:
  - `int8` `context=256`: `w1` `14.30 -> 10.27 us` (**-28.2%**), `w8` `49.06 -> 34.59 us` (**-29.5%**).
  - `int8` `context=1024`: `w1` `20.67 -> 18.05 us` (**-12.7%**), `w8` `79.81 -> 60.38 us` (**-24.3%**).
  - `bf16` `context=256`: `w1` `12.42 -> 10.14 us` (**-18.4%**), `w8` `44.38 -> 32.32 us` (**-27.2%**).
  - `bf16` `context=1024`: `w1` `22.66 -> 18.43 us` (**-18.7%**), `w8` `73.73 -> 57.34 us` (**-22.2%**).

## 5. Rejected approaches & resolved regressions

- **Cross-lane strided stats gather in `merge_kernel` (`base[lane * stride + kHeadDim]`)**: Parallelizing the split stats read across `lane < splits` caused 32 lanes in warp 0 to gather across 32 distinct `24,768 B` pages simultaneously (32 uncoalesced L1 sector requests instead of 1 warp-uniform L1 broadcast sector), regressing `width=1` by `+3..+7 us`. Replaced with the 128-thread (`4`-warp) `float2` vectorized `merge_kernel` that keeps `s` warp-uniform while halving CTA warp count and vectorizing row loads/stores.
- **Sharing K/V SMEM plane on `Int8Group64` (`ShareKV = true` for `Int8Group64`)**: While `ShareKV = true` is required for `BFloat16` to fit below `50,176 B` (`81,552 B -> 48,784 B`), `Int8Group64`'s separate K and V planes already fit inside `49,808 B <= 50,176 B` at 2 CTAs/SM; serializing V `cp.async` behind the Score loop on `Int8Group64` added ~`6 us` at `width=8, context=8192`. Keeping `ShareKV` enabled only for `BFloat16` (`Stages == 1`) gives both `BFloat16` and `Int8Group64` 2 CTAs/SM while allowing `Int8Group64` to overlap both K and V `cp.async` with query load and Hadamard rotation.
- **Compiler register spills at 64 regs/thread (`__launch_bounds__(512, 2)`)**: Naive 2-CTA/SM bounds initially caused `4..16 B` scalar spills in `Exact=true` instantiations due to LLVM IndVarSimplify expanding `n` into `end - begin` across the kernel and keeping `blockIdx.x/y/z` live until the epilogue. Fixed to `0` spills across all 34 functions using `kSingleTile`, `asm("" : "+r"(n))`, and epilogue PTX `mov.u32` reads of `%ctaid.x/y/z` and `%tid.x`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-05 — T3: Overlap `cp.async` Key/Value tile gathers with QK/softmax and PV MMA in `qsa_prompt_kernel` (`Stages == 1`)

## 1. Accounting

- **Base commit**: `69bb2d5b6` (`perf(qsa): enable 2-CTA/SM decode occupancy, empty-split early exit, 2-token score loop, and float2 merge (T2)`).
- **Modified files**: `src/ops/qsa/qsa_prompt.cu` (`M`, `+149/−51`), `docs/gb10/20261105-changelog.md` (`M`).
- **Allowlist delta**: none. `tools/cuda_stack/cuda_stack_allowlist.json` is untouched — the ratchet neither admitted nor retired an entry, because `STACK:0 LOCAL:0` holds for all 14 `qsa_prompt_kernel` instantiations before and after.
- **Rebuilds**: 7 single-invocation Ninja builds of `infernix-serve infernix_ops_attention_test infernix_qsa_attention_bench` (HEAD baseline bench binary for the interleaved A/B, the doc's literal split, the value-scales-with-keys variant, the double-buffered fix, the per-storage `ScaleSlots` fix, the shipped build, and a comment-only rebuild).

## 2. Architectural rationale and task-premise corrections ("Why" on GB10 `sm_121a`)

1. **The mechanism itself holds, and its size is storage-dependent**:
   - `qsa_prompt_kernel<Storage, 1>` runs `2` CTAs/SM with one `Stage` per warp, and the grid is one CTA per `(selected column, kv_head)`, so the old loop `issue(i, 0); cp_wait<0>()` put each tile's whole scattered gather (`16` rows × `2` planes of up to `512 B`, in `16`-byte chunks pulled through `table[]`) directly in front of the only phases that could compute on it. Splitting the gather into a key commit group and a value commit group inside the same stage lets `V(i)` stream behind `QK` MMA + online softmax and `K(i + Warps)` behind the PV MMA, with `cp_wait<1>` leaving exactly the one group the other phase still needs in flight.
   - NCU (synthetic bench inputs) confirms the causal chain where the gather is the stall: `bf16, 8192/2048` `stalled_long_scoreboard` `2.85 → 1.31` warp-cycles per issued instruction and tensor-pipe utilization `16.9% → 20.3%`; `bf16, 32768/2048` (L2 hit `84.6%`) `3.37 → 1.89` and `15.8% → 18.2%`.
2. **Correction — the doc's K/V grouping is not what pays; the value *scales* must travel with the keys**:
   - The doc's step 1 groups the value rows with the value scales. That leaves the `EvenOddValues` decode-rescale (`vmax = warp_max(own)` over the tile's value scales, then `while (shift < 14 && ldexpf(vmax, -shift) > V::MaxScale)`) on the PV MMA's critical path immediately after the value wait, i.e. a serial `redux` + up-to-`14`-iteration loop between two dependent MMAs. Gathering the value scales into the key commit group moves that chain into the QK phase, which has the value rows still in flight, and is worth a further `2–4` points (`int8` `−0.6% → −4.2%`, `k8v4` `−6.5% → −9.4%`). `pack_f16x2(pr * up)` / `down` are an exact power-of-two pair, so this is bit-identical wherever it is computed.
3. **Correction — the split is not free for a small gather, so it is gated**:
   - With `128 B` key rows (NVFP4) the second per-tile addressing pass (`attended_token` + `table[]` load + `qsa_space_plane`), the second `cp.async.commit_group` and the second wait cost more than the overlap returns: the literal split measured `+3.5 … +5.1%` on `nvfp4`. `Stage::SplitGather = Shape::Vq == 0 && K::Bytes >= kD` (one byte per dimension = `16` scattered chunks per row) keeps one commit group for NVFP4 keys and every vector-quantized storage; after the gate `nvfp4` is `−0.6 … +0.7%`, i.e. unchanged.
4. **Correction — the doc's stated resources are stale**:
   - Live `cuobjdump -res-usage` on `infernix-serve` is `REG:238..255 STACK:0 LOCAL:0` (not `REG:121..128`), and the stage is *dynamic* shared memory, so `SHARED` reads `0` in `cuobjdump` (not `46,592`/`47,616`). The real per-CTA footprint is `41,984 B` (`int8`), `41,536 B` (`fp8`), `34,112 B` (`k8v4`), `40,960 B` (`bf16`), `26,624 B` (`nvfp4`), read from NCU `launch__shared_mem_per_block_dynamic`. `2` CTAs/SM is both the SMEM and the register limit here (`Block Limit Registers = 2` at `255` regs), which is why the stage growth had to stay small.
5. **Correction — a split stage needs double-buffered value scales, and the parity word must live in registers**:
   - Putting the value scales in the key commit group means the *next* tile's key gather refills them while the *current* tile's PV MMA is still reading them. `ScaleSlots = 2` slots indexed by `value_scale_at(i) = ((i / Warps) & 1) * (kTile * V::ScaleSlot)` makes the slot the PV phase reads the one the pending gather has not touched yet; the slot count is only allocated where the split route runs (`nvfp4`/VQ keep `1`), which also removed the `+256 B`/CTA that uniform `2` cost them. The FP8 row-scale parity word is likewise carried per tile in `QkState` registers rather than re-read from the stage, because the next tile's key gather overwrites it before PV runs.
6. **Correction — the doc's verification commands do not exist as written**: the bench takes `--storage/--contexts/--widths` (not `--kv-dtype/--context/--width`), and there is no `infernix_qsa_test` target; the gate is `ctest -R "infernix_cuda_stack_check|infernix_qsa"` (10 tests).

## 3. Subsystem breakdown

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/qsa/qsa_prompt.cu` | `issue(i, s, Part)` gains a compile-time gather-part (`kGatherKey`/`kGatherValue`/`kGatherBoth`) so one stage is filled by two commit groups; key group = key rows + key scales + value scales + parity word, value group = value rows. `Stage` adds `SplitGather` (`Vq == 0 && K::Bytes >= kD`) and `ScaleSlots` (`2` where the split runs, else `1`), with `value_scale_at(i) = ((i / Warps) & (ScaleSlots − 1)) * (kTile * V::ScaleSlot)` selecting the slot. The monolithic `compute` is split into `score_tile` (QK MMA + online softmax + the `EvenOddValues` decode-rescale folded into `QkState{pa[4], down, parity, scale_at}`) and `value_tile` (PV MMA reading `qk.scale_at`/`qk.parity`), so the rescale leaves the PV critical path. New `Stages == 1 && SplitGather` loop: prologue `issue(warp, 0, Key)`, then per tile `issue(i, 0, Value); cp_wait<1>; score_tile; __syncwarp; issue(i + Warps, 0, Key); cp_wait<1>/<0>; __syncwarp; value_tile; __syncwarp`, each half of the stage refilled only after the phase that reads it is done. NVFP4/VQ `Stages == 1` and the `Stages == 2` route keep the combined-gather form (behaviour unchanged). |

## 4. Verification & benchmark results

- **Tests** (`ctest --test-dir build-native -R "infernix_cuda_stack_check|infernix_qsa" --output-on-failure`, on the final shipped source): **10/10 passed (100%)** — `infernix_cuda_stack_check`, `infernix_qsa_index_test`, `infernix_qsa_bf16_test`, `infernix_qsa_int8_test`, `infernix_qsa_fp8_test`, `infernix_qsa_nvfp4_test`, `infernix_qsa_k8v4_test`, `infernix_qsa_vq2_test`, `infernix_qsa_k4v2_test`, `infernix_qsa_select_test`. These cover the prompt route at `W = 43, 86, 256, 257, 300, 4096` chunks against the independent FP32 oracle for every storage, plus the cross-width bit-exactness invariant.
- **Resources** (`cuobjdump -res-usage build-native/apps/infernix-serve`): all **14** `qsa_prompt_kernel` instantiations `STACK:0 SHARED:0 LOCAL:0`, `REG:238..255` — identical to HEAD, so no allowlist entry was added or retired.
- **Occupancy** (NCU `launch__shared_mem_per_block_dynamic`, `launch__occupancy_limit_shared_mem`):

| Storage | `Stage::Bytes` × warps + Q/merge, per CTA | HEAD | T3 | limit |
|---|---|---:|---:|---|
| `int8` | `4 × stage + 8,192` | `41,984 B` | `42,496 B` (`+512`) | `2` CTAs/SM |
| `fp8` | `4 × stage + 8,192` | `41,536 B` | `41,792 B` (`+256`) | `2` CTAs/SM |
| `k8v4` | `4 × stage + 8,192` | `34,112 B` | `35,136 B` (`+1,024`) | `2` CTAs/SM |
| `bf16` | `2 × stage + 8,192` | `40,960 B` | `40,960 B` (`+0`, no scales) | `2` CTAs/SM |
| `nvfp4` | `4 × stage + 8,192` | `26,624 B` | `26,624 B` (`+0`, gated out) | `2` CTAs/SM |

  Every value is `≤ 48,128 B` and the measured `launch__occupancy_limit_shared_mem` stays `2` for all five storages, so the `2` CTAs/SM the T2 work bought is preserved.
- **Benchmarks** (`./build-native/bench/infernix_qsa_attention_bench --storage int8,bf16,fp8,k8v4,nvfp4 --contexts 8192,32768 --widths 64,512,2048 --warmup 5 --repeat 50`, 3 interleaved rounds of a freshly HEAD-rebuilt baseline binary vs the shipped binary, cold L2, CUDA-graph medians; `attention_us = total − select`):

| Storage | Context | Width | Task-doc baseline | In-session HEAD (3-round mean) | T3 (3-round mean) | vs doc | vs HEAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int8` | `8192` | `64` | — | `189.0` | `183.6` | — | **−2.8%** |
| **`int8`** | **`8192`** | **`512`** | **`1,089.6`** | `1,075.3` | `1,041.0` | `−4.5%` | **−3.2%** |
| **`int8`** | **`8192`** | **`2048`** | **`9,335.8`** | `4,260.0` | `4,045.5` | `−56.7%` | **−5.0%** |
| `int8` | `32768` | `64` | — | `239.8` | `233.9` | — | **−2.5%** |
| `int8` | `32768` | `512` | `1,178.7` | `1,157.4` | `1,101.3` | `−6.6%` | **−4.8%** |
| `int8` | `32768` | `2048` | `11,786.1` | `4,469.7` | `4,282.4` | `−63.7%` | **−4.2%** |
| `bf16` | `8192` | `64` | — | `277.2` | `239.8` | — | **−13.5%** |
| **`bf16`** | **`8192`** | **`512`** | **`2,257.8`** | `1,645.2` | `1,353.8` | `−40.0%` | **−17.7%** |
| **`bf16`** | **`8192`** | **`2048`** | **`14,111.9`** | `6,513.4` | `5,413.9` | `−61.6%` | **−16.9%** |
| `bf16` | `32768` | `64` | — | `393.9` | `376.1` | — | **−4.5%** |
| `bf16` | `32768` | `512` | `4,476.3` | `1,872.7` | `1,712.8` | `−61.7%` | **−8.5%** |
| `bf16` | `32768` | `2048` | `17,126.4` | `7,199.6` | `6,435.5` | `−62.4%` | **−10.6%** |
| `fp8` | `8192` | `64 / 512 / 2048` | — | `231.0 / 1,336.7 / 5,136.2` | `202.0 / 1,171.5 / 4,305.9` | — | **−12.6 / −12.4 / −16.2%** |
| `fp8` | `32768` | `64 / 512 / 2048` | — | `283.1 / 1,375.3 / 5,298.2` | `250.4 / 1,198.7 / 4,502.1` | — | **−11.6 / −12.8 / −15.0%** |
| `k8v4` | `8192` | `64 / 512 / 2048` | — | `202.1 / 1,151.1 / 4,460.9` | `185.1 / 1,039.0 / 4,003.9` | — | **−8.4 / −9.7 / −10.2%** |
| `k8v4` | `32768` | `64 / 512 / 2048` | — | `233.9 / 1,187.3 / 4,627.9` | `233.0 / 1,076.6 / 4,198.0` | — | **−0.4 / −9.3 / −9.3%** |
| `nvfp4` | `8192` | `64 / 512 / 2048` | — | `172.0 / 943.5 / 3,734.5` | `172.1 / 944.8 / 3,712.9` | — | **+0.1 / +0.1 / −0.6%** |
| `nvfp4` | `32768` | `64 / 512 / 2048` | — | `184.4 / 967.4 / 3,838.1` | `185.8 / 963.2 / 3,834.3` | — | **+0.7 / −0.4 / −0.1%** |

- **Acceptance verdict against the doc's `≥ 10%` bar** (`width = 512` and `2048`, `context = 8192` and `32768`, `int8` and `bf16`), measured against the in-session HEAD binary (the doc's absolute bars are stale — e.g. it quotes `bf16 8192/2048 = 14,111.9 µs` where HEAD measures `6,513.4 µs` in this session):
  - `bf16, 8192`: `−17.7%` (`512`) and `−16.9%` (`2048`) — **met**.
  - `bf16, 32768`: `−10.6%` (`2048`) — **met**; `−8.5%` (`512`) — **missed**.
  - `int8`: `−2.5 … −5.0%` — **missed at all four points**.
  - Outside the bar: `fp8` `−11.6 … −16.2%`, `k8v4` `−8.4 … −10.2%` (except `32768/64` `−0.4%`), `nvfp4` unchanged by design.

## 5. Regressions and adverse results (all found, all disclosed)

- **A race in the first working variant**: the value-scales-with-keys grouping (correction 2) let `issue(i + Warps, Key)` refill the value-scale slots while `value_tile(i)` still read them. It was the *fastest* variant on the bench, and `ctest -R "infernix_cuda_stack_check|infernix_qsa"` failed 3 of 10 tests / 38 individual checks, e.g. `qsa_attention k8v4 prompt W=86 B=1 selected: reduction criterion failed at index 304891 actual=0.046875 reference=0.124421` and `FAIL: qsa_attention k8v4 prompt W=4096 from 0: the split stage gives the combined gather's bits (first difference at column 65, head 0)` in `infernix_qsa_int8_test`, `infernix_qsa_fp8_test`, `infernix_qsa_k8v4_test`. Fixed by the `ScaleSlots = 2` double buffer (§2.5); the shipped build is 10/10.
- **The doc's literal grouping regressed NVFP4 by `+3.5 … +5.1%`** (second addressing pass, commit group and wait per tile against a `128 B` key row). Fixed by the `Stage::SplitGather` gate; `nvfp4` is now `−0.6 … +0.7%`, i.e. inside run-to-run spread.
- **Uniform `ScaleSlots = 2` regressed storages that never take the split**: the `+256 B`/CTA stage growth cost `k8v4 32768/64` `221–223 → 231–236 µs` (`+4.7%`) and `nvfp4 32768/64` `178–180 → 185–188 µs` (`+4.5%`), plus `nvfp4 8192/64` `+1.5%`, consistently across 3 interleaved rounds. Fixed by tying `ScaleSlots` to `SplitGather`; those points are now `−0.4%` and `−0.4…+0.7%`.
- **`int8` is not gather-bound, so the bar is unreachable by reordering loads** (`−2.5 … −5.0%`, best case `−5.0%` vs the `≥ 10%` ask). At `int8, 8192/2048` the base kernel spends only `1.39` of `≈5.99` warp-cycles per issued instruction on `stalled_long_scoreboard`, against `wait 1.53`, `barrier 0.86`, `short_scoreboard 0.49`; after the overlap `long_scoreboard` falls to `1.18` while `wait` rises to `1.63` and `short_scoreboard` to `0.52` — the removed memory stall lands in the MMA/softmax dependency chain instead of leaving the SM, giving `issue_active 33.0% → 36.0%`, tensor `26.2% → 27.4%` and `−5%` duration. `4` warps × `2` CTAs already hide most of the gather; the `bf16` shape (`2` warps, `512 B` rows) is the one that is genuinely latency-bound.
- **`bf16, 32768/512` misses the bar at `−8.5%`** (vs `−17.7%`/`−16.9%` at `context = 8192` and `−10.6%` at `32768/2048`): the selected-column count is set by the budget, so the extra work at `32768` is that the same gathers come out of a `4×` larger page pool, which is a smaller share of the `24 MiB` L2 (NCU on the neighbouring `32768/2048` shape measured an `84.6%` L2 hit rate). `32768/512` itself was not profiled, so this is the plausible cause, not a measured one.
- **`k8v4, 32768/64` is flat (`−0.4%`)** and the `width = 64` points range `−2.5 … +0.7%`. The grid is `(columns, kv_heads)`, so `width = 64` launches `128` CTAs against the `96` resident ones (`2` CTAs/SM × `48` SMs) — a `1.33`-wave call whose tail and prologue dominate the per-tile gain the split buys; the per-warp tile count itself is unchanged. The only points above HEAD in the final build are `nvfp4 8192/64` (`+0.1%`), `nvfp4 8192/512` (`+0.1%`) and `nvfp4 32768/64` (`+0.7%`) — all three are gated out of the split and sit inside the baseline binary's own round-to-round spread (`nvfp4 32768/64` HEAD rounds: `187.8 / 185.5 / 179.9 µs`, `±4.3%`). No storage that takes the split got slower than HEAD at any measured point.

## 6. Accepted gaps

- The `int8`/`fp8`-class `4`-warp routes are dependency-bound, not gather-bound; removing the remaining `wait`/`short_scoreboard`/`barrier` chain needs the warp-specialized TMA producer/consumer redesign (task T5), not a load reorder. The `≥ 10%` `int8` bar is therefore not reachable inside T3's mechanism.
- No cross-binary bit-equality measurement was run. The split changes only *when* bytes are staged and *in which phase* the exact power-of-two decode rescale is folded, never the arithmetic order (per-warp tile sets, MMA order, online-softmax recurrence, warp-order merge are untouched), and correctness rests on the independent-FP32-oracle tests plus the cross-width bit-exact invariant. A `--prompt-v2 0,1,1,0`-style in-process bit comparison across the two groupings was not performed for the final build.
- `qsa_prompt_split()`, `qsa_prompt_force()` and the bench's `--prompt-v2` no longer route anything (the separate split stage was removed on GB10 before this task), so `test_prompt_variants` compares a variant against itself. Removing that dead API is outside T3's target file and was left alone.
- The FP8 parity word is still written into the stage by the key group and consumed from `QkState` registers; the stage slot is now redundant for the split route and could be reclaimed (≤`16 B`/warp) by a follow-up.
- `nvfp4` and the vector-quantized storages keep the combined gather. `Stage::SplitGather` would admit an NVFP4 route automatically if its key row ever reaches `256 B`, but that path is unmeasured.
- The `Stages == 2` route (1 CTA/SM) is unchanged; the doc's premise that a second stage costs occupancy is untouched, and no attempt was made to make the split fill that role.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-11-05 — T4: 2-level warp prefix scan replaces `cub::BlockScan` in `qsa_select_global_kernel`; packed-BF16 query staging in `qsa_score_wide_kernel` measured and rejected

## 1. Accounting

- Base commit: `d6e2d6d9f` (branch `dev-rebased-20261028`).
- Modified: `src/ops/qsa/qsa_select.cu` (`M`, final +34/−7) — scan replacement in `qsa_select_global_kernel` plus a source comment recording the measured rejection of the packed-BF16 query lever.
- `tools/cuda_stack/cuda_stack_allowlist.json`: unchanged (both affected kernels are `STACK:0`/`LOCAL:0` in the shipped build and have no allowlist entries; nothing to ratchet).
- Rebuilds: 3 full single-invocation device links (packed 3-CTA build, packed 2-CTA isolation build, final scan-only build) plus targeted object/bench builds. No header edits, so no `.cu` touch was needed.

## 2. Task-premise corrections / Architectural rationale

- **Stale doc baselines**: the task doc quotes `qsa_score_wide_kernel` `REG:49, STACK:0, SHARED:42240`. The linked `infernix-serve` at base measured `REG:107 STACK:0 SHARED:1088` (static) + `42,240 B` dynamic. `REG:107` already capped the kernel at 2 CTAs/SM by registers too (2 CTAs allow ≤128), so SMEM alone was never the only limiter. The doc's select_us baselines (180.2/661.5 at `C=8192`, `w=512/2048`; ~1,794 at `C=32768, w=2048`) were current.
- **No query weights exist here**: the doc's plan stages `bf16_to_f32(q) * w` and adds a `float ws[64]`; the kernel actually stages raw `bf16_to_f32(index_q)` with no per-(head, column) weight, so the packed variant needed no `ws` and its SMEM was `16,896 + 8,448 = 25,344 B` (not `25,600 B`).
- **"Bitwise identical ⇒ free" is falsified (the core correction)**: packed BF16 `uint2` query staging is indeed bit-exact (all QSA oracle tests passed on both packed builds) and it did reach the intended occupancy — `REG:80 STACK:0`, `25,408 B`/CTA, `ckt math` confirms 3 CTAs/SM (24/48 warps) vs baseline 2 CTAs/SM (16/48). But every leaf then needs one `LDS.64` plus four shift/mask ALU ops per head (16 ops) where the FP32 staging needed a single `LDS.128`: +512 leaf instructions per thread per key tile. `qsa_score_wide_kernel` is warp-instruction-issue-bound (the T2 finding for QSA kernels), so select_us regressed **+10.3…+18.2% at every wide point** in both packed variants: 3 CTAs/SM (`kWideCtasPerSm=3`, REG clamped to 85) and 2 CTAs/SM with the grid identical to baseline (isolating the mechanism from occupancy). The lever was reverted in full.
- **Bench flag corrections**: `infernix_qsa_attention_bench` takes `--storage/--contexts/--widths`, not the doc's `--kv-dtype/--context/--width`.
- **Scan replacement**: `cub::BlockScan<int, 512>::ExclusiveSum` over 4 bins/thread is replaced by `block_exclusive_sum`: a 32-lane `__shfl_up_sync` inclusive scan per warp, the 16 warp totals written to a 16-int shared array, warp 0 scans them, every thread adds its warp prefix. Integer sums are exact, so every `above`/`offset` is bit-for-bit cub's; both call sites (`find_score_bin`, `compact_words`) were converted because `SelectShared::TempStorage` was removed. All 512 threads reach both sites (the kernel's early exits precede them), and the call sites are separated by full barriers, so the reused `warp_totals` array has no WAR hazard.

## 3. Subsystem breakdown

- `src/ops/qsa/qsa_select.cu` (`M`):
  - Removed `#include <cub/block/block_scan.cuh>` (cub is no longer used in this file).
  - `SelectShared`: `typename Scan::TempStorage scan` → `int warp_totals[kSelectWarps]` (64 B); linked static SMEM 4,320 → 2,176 B.
  - New `block_exclusive_sum` (2-level warp scan, two barriers); `find_score_bin` and `compact_words` call it.
  - Wide-kernel section keeps the FP32 query staging and `kWideCtasPerSm = 2`, with a comment recording the measured rejection of packed staging so the lever is not retried blind.

## 4. Verification

- `cuobjdump --dump-resource-usage build-native/apps/infernix-serve` (final build):
  - `qsa_score_wide_kernel`: `REG:107 STACK:0 SHARED:1088 LOCAL:0` — byte-identical resources to base (lever reverted).
  - `qsa_select_global_kernel`: `REG:63 STACK:0 SHARED:2176 LOCAL:0` (base: `REG:63 STACK:0 SHARED:4320`).
- `ctest -R "infernix_cuda_stack_check|infernix_qsa"`: 10/10 passed on both the packed build (oracle proof of bit-exactness) and the final scan-only build (stack gate, `qsa_{index,bf16,int8,fp8,nvfp4,k8v4,vq2,k4v2}_test`, `qsa_select_test`).
- A/B (`/tmp/t4_base_qsa_bench` freshly HEAD-rebuilt vs candidate, 3 interleaved rounds, median of 50 cold-L2 graphs per point, `--storage int8,bf16 --contexts 8192,32768 --widths 16,64,512,2048`):

  | point | base select_us | scan select_us | Δ |
  |---|---|---|---|
  | bf16 C=8192 w=16/64/512/2048 | 28.4 / 30.4 / 177.9 / 657.1 | 28.4 / 30.4 / 177.9 / 659.3 | +0.0 / −0.1 / +0.0 / +0.3% |
  | bf16 C=32768 w=16/64/512/2048 | 44.9 / 67.3 / 456.4 / 1781.4 | 45.0 / 67.3 / 458.4 / 1799.8 | +0.2 / +0.0 / +0.4 / +1.0% |
  | int8 C=8192 w=16/64/512/2048 | 28.4 / 30.5 / 177.8 / 655.0 | 28.4 / 30.4 / 178.0 / 657.2 | +0.0 / −0.4 / +0.1 / +0.3% |
  | int8 C=32768 w=16/64/512/2048 | 45.2 / 67.3 / 456.4 / 1784.5 | 45.1 / 67.4 / 458.3 / 1800.0 | −0.3 / +0.1 / +0.4 / +0.9% |

  All points within ±1.0% (per-round spread comparable inside each arm) — the scan replacement is performance-neutral within the drift band.
- Packed-query variant A/B (3 rounds, bf16, select_us medians): `w=512` 173.8→196.6/198.3 at `C=8192` and 456.5→536.3 at `C=32768`; `w=2048` 655.1→735.1…739.0 at `C=8192` and 1771→2114…2120 at `C=32768`. The full 16-point ranges and worst cases are in §5.

## 5. Regressions & adverse results (all found, all disclosed)

- **Packed-BF16 query staging regressed every wide-route point** (both variants; per-round medians stable, not noise): 3-CTA build `select_us` +10.3…+17.3% at `w=512/2048`, +13.9…+18.2% at `w=64`, `total_us` +0.9…+5.8% (worst `int8 C=32768 w=2048`: 6,050.6→6,349.7 us); 2-CTA isolation build matched it (+13.0…+19.3%), proving the cost is the per-leaf widen instructions, not the occupancy or register clamp. Disposition: **reverted in full** — the shipped diff contains no wide-kernel change.
- **The packed 3-CTA build missed even its own resource sub-bar**: `REG:80`, not the doc's `REG ≤ 64` (80 is exactly the 3-CTA register boundary; `launch_bounds(256,3)` left zero headroom).
- **The ≥12% `select_us` acceptance bar is not met.** The scan replacement measured ±1.0% (neutral); the score kernel dominates wide-route `select_us` (T7 profile: score ≈ 2.8× select at `T=32768`), and its specified lever regresses. Both T4 levers are now exhausted; the bar would require a different score-kernel mechanism outside T4's spec.
- Narrow route (`w=16`, `qsa_score_kernel` + select kernel): unchanged (±0.3%), as expected — `qsa_score_kernel` was not touched.
- Regression check covered: all 16 bench points × 3 interleaved rounds vs the freshly HEAD-built baseline binary, STACK/LOCAL for all three select-pipeline kernels, static SMEM, oracle test results and tolerances (10/10), and the stack-spill ratchet (no allowlist change needed). No other regression found.

## 6. Accepted gaps / follow-ups

- `select_us` is unchanged (within noise) relative to base; the shipped value of T4 is the removed `cub::BlockScan` dependency, 2,144 B less static SMEM in `qsa_select_global_kernel`, and a measured falsification of the packed-query premise (now documented at the wide-kernel source).
- `qsa_score_wide_kernel` stays at 2 CTAs/SM with 42,240 B dynamic SMEM. Any future 3-CTA attempt must shrink SMEM without adding per-leaf instructions — e.g. keeping the FP32 query staging and halving the staged column count per pass — which re-reads pooled keys and needs its own A/B; not attempted here.
- The scan replacement's benefit is structural (dependency + SMEM); at widths where the select kernel is a larger share of a round (decode-side narrow batches with large contexts) it can at best be neutral — no point measured a real gain, none measured a real loss.
