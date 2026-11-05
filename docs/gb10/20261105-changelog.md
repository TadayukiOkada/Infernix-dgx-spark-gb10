# GB10 changelog 2026-11-05

Scope: Qwen4Exp (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`) end-to-end `infernix-serve` and
`llama-benchy` (`pp2048 tg128 @ d0, d4096, d8192` across `c = 1, 2, 4`) prefill and decode optimizations:
unsplit device-resident MoE prefill chunks and `rows_behind` PLE overlap, OpenAI `"cache_prompt"` protocol
support to eliminate structural tap splitting and wasted host snapshot copies on `--no-cache` workloads,
concurrent batch prefill completion before decode interleaving, optimal N-gram copy prefix truncation
(`best_copy_length`) and tile-aware MTP draft length scoring, `-rdc=true` static `.shared` memory bloat
elimination across `infernix_ops`, single-stage tall LM-head projection (`SingleStage` + wide router MMA),
removal of redundant MTP verification argmax calls, right-sized L1/shared memory carveouts (`down_kernel_a16`,
`qsa_score_kernel` BF16 `uint2` query staging, `route_dispatch_plan_kernel` shared union, GDN
`recurrent_record_conv_kernel` `RecordConvShared` tile shrinking), MTP `resident_moe` `down_kernel` launch
bounds, `top_logprobs` 1024-thread wide CTA and `key > k3` fast guard, 8-thread parallel `NgramVolume`
cache-hit gather and cache-line-disjoint `insert_cache` for prefill chunks, and 128-bit (`uint4`) coalesced
vectorization across MoE combine, HyperConnection, QSA tail/key-pool, and PLE causal conv/state kernels on
NVIDIA GB10 (`sm_121a`).

## Commit range & accounting

- **Base commit (`origin/main`)**: `68c54356fd490ab329bd1475d48957f886bb7dd1`
- **Previous HEAD**: `fbbc1c763` (`perf(gb10): optimize Qwen4Exp MoE staging, multi-row N-gram/MTP speculation, and 32-col LM head`)
- **Delta in this milestone**: 25 code files changed (+1 changelog file = 26 files total), 974 code insertions(+), 185 code deletions(-) (excluding `cuda-kernel-toolkit/`, which is an integrated toolkit and intentionally excluded from file-by-file enumeration).

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
| M | `src/models/qwen4_exp/program/program_impl.h` | Sets `.served_columns = options_.device_experts ? 0U : 8U` in `ProgramImpl::call_cost()` to stop `split_prefill_call` from splitting device-resident MoE prefill chunks; enables `rows_behind` prefill PLE overlap when `options_.device_experts == true`; applies `best_copy_length` to single-row and multi-row N-gram proposals under `--draft-tokens auto`; tunes GB10 width costs (`kWidthCost = 0.18`, `kPairWidthCost = 0.16`, `kMultiRowWidthCost = 0.16`, `kPairMaxDrafts = 3`). |
| M | `src/models/qwen4_exp/program/ngram_volume.h`, `src/models/qwen4_exp/program/ngram_volume.cpp` | Parallelizes volume-mode cache lookup/hit gathering and cache-line-disjoint `insert_cache` (`((set_of(r) >> 6U) % n_threads) == t`) across `kReadThreads = 8` threads for prefill chunks (`kParallelGatherRows = 2048`), preserving bit-identical cache state. |
| M | `src/models/qwen4_exp/execution/forward.cpp` | Removes redundant `ops::argmax` launch in `Forward::step_mtp` when `draft_out != nullptr` is already populated by `sample_mtp_draft_column` or greedy selection, and uses `Scratch::top_logprobs_bytes(1, 1)` for single-token argmax workspace. |

### 3. Ops & kernels (`src/ops/`)

| Action | File | Mechanism |
|---|---|---|
| M | `src/ops/kernel/argmax.cuh`, `src/ops/kernel/target_logprobs.cuh`, `src/ops/kernel/token_constraint.cuh`, `src/ops/kernel/top_logprobs.cuh`, `src/ops/launcher/top_logprobs.cu` | Replaces `static __shared__ typename BlockReduce::TempStorage` in header-defined kernels with dynamic shared memory (`extern __shared__ __align__(16) char`), reducing `-rdc=true` static `.shared` reservations across `infernix_ops` from `50–66 KiB` to `1.1–17.5 KiB`; adds `top_logprobs_one_kernel`, `key > k3` fast guard, and 1024-thread wide CTA (`valid_rows >= 4096`). |
| M | `src/ops/projection_fp32/projection_fp32.cu`, `src/ops/projection_fp32/projection_fp32_mma.h`, `src/ops/projection_fp32/projection_fp32_mma.cu` | Switches `projection_tall_kernel` (`kCols = 8, 12, 16, 24, 32`) from `MultiStage` (`32 KiB` shared, 3 CTAs/SM) to `SingleStage` (`16 KiB` shared, 6 CTAs/SM); dispatches `columns = 9..16` router projections (`rows = 512, K = 2560`) through `launch_router_mma_wide<16>` (8 CTAs × 4 warps = 32 warps vs 16 warps). |
| M | `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | Adds 128-bit (`uint4`) coalesced `combine_vec8_kernel` and `combine_inject_vec8_kernel`; unifies sequential shared-memory structs in `route_dispatch_plan_kernel` into `union RouteDispatchPlanShared` (`41,140 B -> 22,928 B`); removes unconditional `cudaSharedmemCarveoutMaxShared` override from `down_kernel_a16` to preserve L1 cache during narrow `W4A16` MoE down-projections. |
| M | `src/ops/linear_attention/gated_delta_net/recurrent.cuh` | Shrinks `RecordConvShared::value` from `[16][128]` (`4,096 B`) to `[16][16]` (`kBlockDv = 16`, `512 B`), reducing `recurrent_record_conv_kernel` static `.shared` from `13,440 B` to `9,856 B` (`REG: 85`) so all 5 CTAs/SM fit inside `49.3 KiB <= 64 KiB` without triggering a 99 KiB shared-memory carveout. |
| M | `src/ops/resident_moe/resident_moe.cu` | Adds `__launch_bounds__(kThreads, 5)` to `down_kernel`, reducing `down_kernel<Q8>` from `83` registers (33% occupancy) to `48` registers (83% occupancy) with zero spills. |
| M | `src/ops/hyper_connection/hyper_connection.cu` | Adds 128-bit (`uint4`) coalesced `norm_vec8_kernel`, `inject_norm_vec8_kernel`, `collapse_vec8_kernel`, `inject_vec8_kernel`, `expand_vec8_kernel`, and `stream_mean_vec8_kernel` across all 4-stream HyperConnection operations. |
| M | `src/ops/qsa/qsa_select.cu`, `src/ops/qsa/qsa.cu` | Stages queries in `qsa_score_kernel` as packed BF16 `uint2` instead of `float4`, halving static `.shared` per CTA from `33,856 B` to `17,472 B` (`+36 KiB` L1 cache/SM at 2 CTAs/SM); adds 128-bit (`uint4`) coalesced `tail_vec8_kernel` (`qsa_pool_keys`) and `commit_tail_vec8_kernel` (`qsa_commit_tails`). |
| M | `src/ops/ple/ple.cu` | Adds 128-bit (`uint4`) coalesced `conv_vec8_kernel`, `conv_state_vec8_kernel`, `state_vec8_kernel`, and `commit_vec8_kernel` for `ple_conv_inject` and `ple_conv_commit`. |

### 4. Tools (`tools/`)

| Action | File | Mechanism |
|---|---|---|
| M | `tools/run_server.sh` | Sets default `--ngram-draft-tokens 15` on `infernix-serve` launcher for GB10 single-pass 16-column verify tile alignment. |

## Verification & benchmark summary (`llama-benchy`, `pp2048 tg128 @ d0, d4096, d8192`, `c = 1, 2, 4`)

- **Unit & oracle tests**: Speculative, stack-ratchet, MoE plan, HyperConnection, PLE, and QSA CTest targets pass (`infernix_cuda_stack_check`, `infernix_ple_test`, `infernix_hyper_connection_test`, `infernix_hyper_connection_stream_mean_test`, `infernix_qsa_select_test`, `infernix_offloaded_moe_plan_test`, `infernix_ops_attention_test qsa`), with 0 new stack spills.
- **End-to-end `llama-benchy` (`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, `--no-cache --pp 2048 --tg 128 --depth 0 4096 8192 --concurrency 1 2 4 --runs 3`)**:

| Test | `c` | `pp t/s` (Base A → Base B → **Now**) | `pp` Gain (vs A / vs B) | `tg t/s` (Base A → Base B → **Now**) | `tg` Gain (vs A / vs B) | `TTFT ms` (Base A → Base B → **Now**) | TTFT Change (vs A / vs B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pp2048 tg128 @ d0` | `c1` | `2,648*` → `2,763*` → **`3,120*`** | **+17.8%** / **+12.9%** | `71.6` → `77.1` → **`76.1`** | **+6.3%** / **-1.3%** | `773` → `741` → **`657`** | **-15.0%** / **-11.4%** |
| `pp2048 tg128 @ d0` | `c2` | `2,458` → `2,633` → **`2,851`** | **+16.0%** / **+8.3%** | `90.3` → `106.0` → **`107.5`** | **+19.0%** / **+1.4%** | `1,544` → `1,476` → **`1,306`** | **-15.4%** / **-11.5%** |
| `pp2048 tg128 @ d0` | `c4` | `2,516` → `2,716` → **`2,890`** | **+14.9%** / **+6.4%** | `139.4` → `144.7` → **`148.2`** | **+6.3%** / **+2.4%** | `3,019` → `2,821` → **`2,625`** | **-13.1%** / **-6.9%** |
| `pp2048 tg128 @ d4096` | `c1` | `3,172*` → `3,316*` → **`3,885*`** | **+22.5%** / **+17.2%** | `61.8` → `79.7` → **`81.0`** | **+31.1%** / **+1.6%** | `1,937` → `1,853` → **`1,581`** | **-18.4%** / **-14.7%** |
| `pp2048 tg128 @ d4096` | `c2` | `2,892` → `3,093` → **`3,566`** | **+23.3%** / **+15.3%** | `91.1` → `99.8` → **`105.2`** | **+15.5%** / **+5.4%** | `3,934` → `3,690` → **`3,130`** | **-20.4%** / **-15.2%** |
| `pp2048 tg128 @ d4096` | `c4` | `2,937` → `3,125` → **`3,629`** | **+23.6%** / **+16.1%** | `133.0` → `136.1` → **`144.2`** | **+8.4%** / **+6.0%** | `7,676` → `7,202` → **`6,136`** | **-20.1%** / **-14.8%** |
| `pp2048 tg128 @ d8192` | `c1` | `3,613*` → `3,722*` → **`4,369*`** | **+20.9%** / **+17.4%** | `69.3` → `71.0` → **`79.3`** | **+14.4%** / **+11.7%** | `2,834` → `2,751` → **`2,344`** | **-17.3%** / **-14.8%** |
| `pp2048 tg128 @ d8192` | `c2` | `3,298` → `3,434` → **`3,993`** | **+21.1%** / **+16.3%** | `90.5` → `98.5` → **`102.5`** | **+13.3%** / **+4.1%** | `5,641` → `5,455` → **`4,626`** | **-18.0%** / **-15.2%** |
| `pp2048 tg128 @ d8192` | `c4` | `3,325` → `3,453` → **`3,991`** | **+20.0%** / **+15.6%** | `138.3` → `150.2` → **`146.7`** | **+6.1%** / **-2.3%** | `11,241` → `10,841` → **`9,357`** | **-16.8%** / **-13.7%** |

## Rebase conflict resolution policy

- **Runtime & serving (`engine_core.h`, `request.h`, `openai_chat_request.cpp`, `generation_service.cpp`)**: Preserve `"cache_prompt"` support (`CacheParticipation::Disabled` when false) and `has_tail_prefill_ready()` prefill-completion scheduling when `decode_share == 0.0`.
- **Model & speculative decoding (`mtp_gating.h`, `program_impl.h`, `forward.cpp`)**: Preserve `.served_columns = options_.device_experts ? 0U : 8U` in `call_cost()`, `rows_behind` prefill overlap on `device_experts`, `best_copy_length` N-gram prefix truncation, and `tile_cost = 0.10` for $k \le 3$ in `draft_length_at`.
- **Ops (`argmax.cuh`, `target_logprobs.cuh`, `token_constraint.cuh`, `top_logprobs.cuh`, `projection_fp32.cu`, `projection_fp32_mma.cu`, `moe_layer.cu`, `hyper_connection.cu`, `qsa_select.cu`, `qsa.cu`, `ple.cu`)**: Preserve dynamic shared memory in header CUB reductions (avoiding `-rdc=true` static `.shared` bloat), `top_logprobs_one_kernel`, `SingleStage` tall LM-head projections, `launch_router_mma_wide<16>` for `columns = 9..16`, `qsa_score_kernel` BF16 `uint2` query staging, `route_dispatch_plan_kernel` shared union, right-sized `down_kernel_a16` carveout, and 128-bit (`uint4`) coalesced vectorization paths.
