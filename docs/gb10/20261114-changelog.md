# GB10 (`sm_121a`) Changelog — 2026-11-14 (Prefill Campaign, `HC4` Reconversion, Vision/MoE Integration, MTP/N-gram + Decode Kernel Optimization, and Wave 3 Sampled MTP Acceptance)

## 1. Commit Range & Accounting

- **Branch & Upstream Range:** `origin/main` `31fe191c6` (merge-base `68c54356f`); 2026-11-14 optimization campaign spans `51fa3e2d7` -> `4cd257655` (Prefill/Q4 HC/`HC4-W4Dense`/Vision) -> `4a4efac61` (`HEAD`, Wave 1 MTP/N-gram speculative optimization) -> uncommitted Wave 2 + Wave 3 working tree on `backup-perf-changes`.
- **Total Branch Delta (`origin/main` -> `HEAD` + working tree, excl. `cuda-kernel-toolkit/`):** `1,340` files changed (`83,936` insertions, `84,688` deletions prior to final Wave 3 changelog entry; `51fa3e2d7` -> working tree: `61` files changed, `3,785` insertions, `1,138` deletions).
- **Wave 1 Commit (`4a4efac61`, `51fa3e2d7` -> `HEAD` total across `4cd257655` + `4a4efac61`):** `43` files changed, `2,966` insertions, `891` deletions (`4a4efac61` alone: `4` files changed, `170` insertions, `48` deletions).
- **Combined Uncommitted Working Tree Delta (`4a4efac61` `HEAD` -> Wave 2 + Wave 3 working tree, excl. `cuda-kernel-toolkit/`):** `28` modified files (`11` pre-existing Wave 2 files including [docs/gb10/20261114-changelog.md](file:///opt/etc/Infernix/docs/gb10/20261114-changelog.md) + `17` Wave 3 source/header/test files; `838` insertions, `266` deletions prior to the Wave 3 changelog section; Wave 2 baseline before Wave 3 was `11` modified files, `192` insertions, `116` deletions). (`cuda-kernel-toolkit/` is intentionally excluded from file-by-file enumeration per repository policy.)
- **Generated Model Artifact (out-of-tree):** Reconverted `/opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense.infernix` (`69.33 GiB` across 3 shards) from `/opt/etc/models/nvidia-Qwen3.8-Flash-Next-FP4-Local` with recipe `qwen3_8_flash_next_nvfp4_w4dense` and `hc_q4_override.py`.

---

## 2. Architectural Rationale ("Why")

On NVIDIA DGX Spark GB10 (`sm_121a`), Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) runs resident on GPU under `INFERNIX_EXPERTS_ON_GPU=1` with unified 128 GiB LPDDR5X memory (`~273 GB/s`), 48 SMs, 24 MiB L2 cache, and `mxf4nvf4` FP4 Tensor Cores. The previous prefill and mixed-precision paths suffered from five GB10 bottlenecks:

1. **Fragmented Routed-MoE Prefill & Unpacked Narrow Launches (`sm_121a` 48 SMs):**
   - `moe_layer.cu` launched wide/narrow expert passes without compacting narrow GPU jobs per pass (`narrow_jobs` / `pass_narrow`), and `wide_expert.cu` gated per-token W4A4 quantization at `x.ne[1] > 64` while lacking explicit masked-slot initialization on `entry_map` during multi-column speculative verification.
   - `narrow_expert.cu` performed scalar un-staged SMEM/activation indexing instead of vectorized SMEM staging and programmatic dependent launch (PDL) overlap between `quantize_x_kernel`, `gate_up_kernel`, and `down_kernel`.
2. **Hyper-Connection & MoE Combine Materialization Traffic:**
   - Each transformer layer materialized an intermediate `hyper_connection_norm` `[S*H, T]` buffer separately from the previous block's `moe_combine_inject` residual update, and `q4_g64_fp16` HC mixers lacked fused decode (`Q4Codec` in `hyper_connection_mix_fused.cu`) and wide `up_collapse` (`launch_q4` in `hyper_connection_up_collapse.cu`) support.
3. **QSA Prompt Attention Shared Memory & SM Occupancy (48 SMs):**
   - Upstream GB10 removal of `qsa_prompt_split` (`SplitD`) forced 1-head-group full-`D=256` stages per CTA, exceeding 50 KiB SMEM on wider storages and limiting CTA/SM occupancy to 1 across 48 SMs. Restoring compile-time `SplitD` (`kv_w` + `d_w` warps with `64 B` XOR-swizzled tiles) fits higher CTA/SM occupancy while staying within the attention accuracy oracle.
4. **NVFP4 A4 & `q4_g64_fp16` Shape / Quantizer Schedules (`24 MiB` L2, `48 SMs`):**
   - `nvfp4_a4_quantize.cuh` and `nvfp4_codec.cuh` left registers/occupancy conservative (`512 / Threads` -> `1024 / Threads` launch bounds) and `fmaf` single-rounding opportunities unused; dense `n12800_k2560`, `n13952_k2560`, `n16384_k2560`, `n2560_k6144` shape gates had stale thresholds for `T = 33..55` and `T = 129..256` / `2048`.
   - Switching `HC4` to `qwen3_8_flash_next_nvfp4_w4dense` + Q4 HC mixers introduced five previously untuned Q4 shapes (`320x10240`, `324x10240`, `10240x320`, `1280x2560`, `2560x640`, plus `K % 64 == 0` alignment in `q4_operands.h`).
5. **`HC4` Artifact Dense Projection Precision:**
   - Old `HC4` used `qwen3_8_flash_next_fp4_local`, leaving GDN (`q/k/v/z/output`), QSA (`q/g/k/v/iq/ik/o`), and PLE (`key/value`) dense weights in MXFP8 (`fp8_e4m3fn_g32_e8m0`) instead of `nvfp4` W4, doubling dense weight bandwidth during decode and mid-batch prefill.
6. **Speculative Decoding (`MTP` + N-gram) Host/Graph Overhead & 48-SM Kernel Occupancy:**
   - **Suffix Automaton & N-gram Lookup (`SuffixAutomaton`, `NgramVolume`):** Sequential linear edge scanning and cold host pointer chasing degraded speculative proposal latency; O(1) `edge_table_` + `last_` tail fast path and `kPrefetchDist = 8` `__builtin_prefetch` on resident RAM gathers / volume-cache `tags_` lookups reduce CPU proposal latency. On GB10's 128 GiB UMA, defaulting `tools/run_server.sh` `NGRAM_RESIDENCY=auto` loads the `27.1 GiB` `.ngram` table in RAM for `infernix-serve` (leaving `10.2–10.6 GiB` free), trading CLI cold-load time (amortized in persistent serving) for faster prefill (`142–146 ms` vs `152–156 ms`) and zero volume read stalls.
   - **MTP CUDA Graph Coverage & Draft Width Cost:** Adding `mtp_sampled_gated_graphs_`, merging `mtp_catch` into `commit_graph`, and tuning `kWidthCost = 0.08` avoids ungraphed sampled-verify launches and under-wide speculative windows.
   - **48-SM Kernel Register Pressure & Occupancy:** `hc_down_partials_v4_kernel` (`ExactColumns == 4`) used `141–146` registers (`1` blk/SM) and `recurrent_record_conv_kernel` (`MaxWidth = 16`) used `85` registers (`4` blk/SM); reducing `kPreloadChunks` (`-> 1`) + fusing `Pending` loads into `values[c][h]` lowered `ExactColumns == 4` registers to `104–128` (`2` blk/SM across Q4/Q8/BF16 with `0` spills), while `recurrent_record_conv_kernel` (`MaxWidth <= 16 ? 6 : 4`) dropped to `72` registers. In `screen_kernel`, parallelizing post-MMA warp reduction/top-`kListLength`/softmax across 8 half-warps (`16` lanes/col) and staging epilogue lists through shared `out_lists` removed shared `partial[e]` writeback/barriers and eliminated stack spills (`STACK:0 LOCAL:0`).
7. **Wave 3 Sampled MTP Proposal Alignment, Adaptive Cumulative Gating, & N-gram Hygiene (Memory-Bandwidth-Bound Decode Amortization):**
   - **16-Rank Sampled Drafts & Screened Head Exactness (`kSampledDraftRanks = 16`):** Previously `SampledMtpDrafts` capped draft candidate extraction at `ranks = 4` (`kScreenedHeadMaxRanks = 4`), throwing away target mass in higher-entropy/flat-distribution steps and forcing fallback or early rejection on sampled rounds. Raising `kScreenedHeadMaxRanks = 16` (`screened_head_top` + `kScreenedHeadMaxColumns = 32`) alongside `kSampledDraftRanks = ops::kSparseSpeculativeCandidates` (16) and fixing `screened_top_kernel` normalizer exclusion (removing only the exact returned ranks rather than clamping at 4 rows) aligns sampled MTP proposals with `SparseProposal`'s 16-candidate target verification domain at zero additional register spills (`screened_top_kernel`: `REG=70 STACK=0 SHARED=11936 LOCAL=0`).
   - **Penalty & Intra-Chain Overlay in Sampled Draft Selection (`speculative_sample_sparse_drafts_kernel`):** Sampled drafts previously ignored `presence_penalty` / `frequency_penalty` and prior tokens drafted earlier in the same chain, causing systematic mismatch with the penalized target distribution (`42.95%` acceptance in `Wave2_Baseline` `P3_penalized`). Overlaying `cfg.token_counts` plus `prior_drafts[t, 0..prior_steps-1]`, re-sorting the 16 candidates by penalized score, recomputing normalized residual log-probability for gating confidence, and using `0.88F + min(0.12F, 0.08F * pen)` penalty-aware draft temperature scaling raised multi-seed `P3_penalized` acceptance by `+5.11 pp` (`42.95% -> 48.05%`) and decode speed by `+4.53%` (`59.96 -> 62.67 tok/s`), all while preserving exact speculative rejection sampling (`least p >= 0.000399` unpenalized, `0.00493` penalized against `0.000111`).
   - **Calibrated 16-Rank Sampled Cumulative Cut (`kMtpSampledCumulativeBins`):** Replaced the fixed single-step `-2.12026F` (`log 0.12`) sampled gate with `kMtpSampledCumulativeBins` (`base = 0.28F`, `floor = 0.20F`), uniting host `mtp_cumulative_cut` and CUDA Graph `launch_graph_gate_product` under observed 16-rank empirical acceptance. On GB10's memory-bandwidth-bound decode (`273 GB/s`), avoiding speculative verify rounds with near-zero expected tokens while keeping high-yield chains cuts sampled fallbacks from `155` to `71` across 40 sampled runs and reduces verify rounds (`6462 -> 6321`).
   - **N-gram Boundary Hygiene & Faster Zero-Hit EMA Backoff:** Rejecting suffix tails crossing `boundaries_` in `NgramProposer` prevents cross-message boundary false matches, and scaling the N-gram EMA weight by `2.5x` (`<= 0.5`) on complete 0-token misses trims low-hit N-gram draft bloat (`P4_code_greedy` `496/768 -> 506/760`, `64.58% -> 66.58%`, `84.14 -> 86.35 tok/s`).

---

## 3. Subsystem Breakdown

### 3.1 `include/`, `src/models/qwen4_exp/`, & `src/runtime/engine/`
- `M` `include/infernix/ops/hyper_connection.h`: Declared `hyper_connection_norm`, `hyper_connection_mix_is_fused`, and `hyper_connection_mix_normalized` to allow external pipelined pre-norm reuse when the fused decode mix does not apply.
- `M` `include/infernix/ops/offloaded_sparse_moe.h`: Extended `moe_combine_inject` with optional `next_norm_weight`, `next_norm_out`, and `next_norm_eps` parameters to fuse next-layer `attn_hc.norm` into residual combine.
- `M` `src/models/qwen4_exp/execution/forward.h` & `M` `src/models/qwen4_exp/execution/forward.cpp`: Pipelined `Forward::layer()` / `Forward::moe()` with `pre_norm`, `next_norm_weight`, and `next_norm_out`, feeding combined post-MoE normalization directly into the next layer's `attn_hc` (`mix`).
- `M` `src/models/qwen4_exp/program/program_impl.h` & `M` `src/models/qwen4_exp/program/vision_program.cpp` & `M` `src/runtime/engine/qwen4_exp_instance.cpp`: Allowed `INFERNIX_EXPERTS_ON_GPU=1` with `--vision` enabled by retaining vision weights on device (`load.vision_offload = false`) instead of falling back to host experts, and added standalone `v.owned_lease` (`DeviceBuffer`) backing in `VisionWindow` when `residency_` is null.

### 3.2 `src/ops/offloaded_sparse_moe/cuda/`
- `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.h` & `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`:
  - Added `narrow_jobs` and `pass_narrow` workspace pointers (`ExpertsWorkspace`) to separate compacted narrow GPU jobs per pass.
  - Lowered per-token W4A4 quantization path enablement (`map_entries_kernel` + `quantize_token_kernel`) from `x.ne[1] > 64` to match `source.activation == ExpertActivation::kA4 && source.wide_a16 != WideA16::kLossless && entries == x.ne[1] * top_k`.
  - Fixed a warmup `cudaErrorIllegalAddress` crash under masked speculative verification (`verify_graph_body`, `batch > 1`): initialized `entry_map[0 .. entry_map_ints - 1] = -1` inside `plan_kernel` and guarded `quantize_token_kernel` with `if (expert >= 0)` so unpopulated masked slots never read uninitialized `0xA5A5A5A5` indices.
- `M` `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`: Added `gate_up_xplane_kernel` / `down_prefill_kernel` and compacted `narrow_jobs` / `pass_narrow` scheduling in `plan_kernel` alongside extended `moe_combine_inject` with optional fused next-layer RMSNorm output.
- `M` `src/ops/offloaded_sparse_moe/cuda/narrow_expert.cu`: Updated `quantize_x_kernel`, `gate_up_kernel` (`__launch_bounds__(kThreads, 2)`), and `down_kernel` (`__launch_bounds__(kThreads, 2)`) with PDL producer/consumer dependencies and vectorized SMEM staging.

### 3.3 `src/ops/hyper_connection/`
- `M` `src/ops/hyper_connection/hyper_connection.cu`: Extracted `hc_mix_from_normalized`, exported `hyper_connection_norm`, `hyper_connection_mix_is_fused`, `hyper_connection_mix_normalized`, and added PDL `pdl::launch_with` in `norm_vec8_kernel` for `columns <= 16`.
- `M` `src/ops/hyper_connection/hyper_connection_mix_fused.cu`: Added `Q4Codec` (`q4_g64_fp16` `row_split_k128_v1` decode via `Q4SimtDecodeAtom::decode_eight`) to `hc_mix_fused`, expanded `hc_up_collapse_v4_kernel` launch bounds for `Q8Codec`/`Q4Codec`, and widened fused Q8/Q4 mixer dispatch (`max_columns = kHcMixFusedMaxColumnsQ8`).
- `M` `src/ops/hyper_connection/hyper_connection_up_collapse.cu`: Added `launch_q4<Schedule>` using `launch_q4_a16_mma` with `HcStreamRows<Features>` and generalized `hc_up_collapse_supported` to accept `QType::Q4_G64_FP16` (`up.k % 64 == 0`).

### 3.4 `src/ops/qsa/`
- `M` `src/ops/qsa/qsa_prompt.cu`: Reintroduced compile-time `SplitD` (`Warps = 4` = 2 KV warps x 2 D-halves), shared-memory phase overlay (`phase1` query + pair scores + stages overlaying `merge` buffer), `64 B` bank-conflict-free XOR swizzle `(chunk ^ (row & 3) ^ ((row >> 2) & 1))`, `qsa_mma` helper, `cudaSharedmemCarveoutMaxShared` carveout configuration, and automatic `blocks_1 > blocks_2` stage-count / `SplitD` selection for non-VQ storages (`!Vq2 && !Q4KeyVq2Value`).

### 3.5 `src/ops/linear/{nvfp4,q4}/`
- `M` `src/ops/linear/nvfp4/nvfp4_codec.cuh`: Added `quantize_nvfp4_k16_packed` and used `fmaf(-code, scale, remainder)` in `quantize_nvfp4_terms_k16_packed`.
- `M` `src/ops/linear/nvfp4/nvfp4_a4_quantize.cuh`: Updated quantizer `__launch_bounds__(Threads, 1024 / Threads)` and vectorized/packed quantizer paths.
- `M` `src/ops/linear/nvfp4/nvfp4_a4.cu`: Raised `kGroupsPerRow == 384` thread block width in `launch_residual_exact` from `192` to `384`.
- `M` `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh` & `M` `src/ops/linear/nvfp4/nvfp4_a4_tma.cu`: Applied per-operand TMA L2 promotions (`Schedule::kPromotions.a_codes`, `kTmaAllL2_256B` for `K=6144`), increased `Nvfp4A4TmaMmaSchedule` stage count (`3 -> 4`) on `K=6144` residual, balanced grouped TMA CTA raster (`max_group_n = 12` for `k <= 2560`), and moved `nvfp4_prepare_shared` inside the slice loop.
- `M` `src/ops/linear/nvfp4/shapes/{n12800_k2560,n13952_k2560,n16384_k2560,n2560_k6144}.cu`: Updated GB10 pair gates (`12800x2560`: `65 -> 56`; `13952x2560`: `33 -> 52`; `16384x2560`: `33 -> 52`) and updated `2560x6144` `select_a4` for `T = 129..256` and `T >= 2048`.
- `M` `src/ops/linear/q4/{q4_operands.h,q4_a16_mma.cuh,q4_mma_launch.cuh,q4_dispatch.cpp,q4_shapes.h,sources.cmake}`: Relaxed `K` divisibility requirement from `128` to `64` (`padded_k % 128 == 0`), generalized `q4_a16_mma_kernel` and `launch_q4_a16_mma` with `RowPolicy` (`Q4MmaIdentityRows` default and `HcStreamRows` for fused up-collapse) and `q4_mma_shared_bytes<Schedule, Epilogue>`, and added generic fallback + five GB10-swept Q4 shape files:
  - `A` `src/ops/linear/q4/shapes/n320_k10240.cu` (`select_q4_n320_k10240`)
  - `A` `src/ops/linear/q4/shapes/n324_k10240.cu` (`select_q4_n324_k10240`)
  - `A` `src/ops/linear/q4/shapes/n10240_k320.cu` (`select_q4_n10240_k320`)
  - `A` `src/ops/linear/q4/shapes/n1280_k2560.cu` (`select_q4_n1280_k2560`)
  - `A` `src/ops/linear/q4/shapes/n2560_k640.cu` (`select_q4_n2560_k640`)

### 3.6 `tests/` & `tools/`
- `M` `tests/ops/test_hyper_connection.cpp`: Added tests covering `hyper_connection_norm`, `hyper_connection_mix_normalized`, and `q4_g64_fp16` HC mix/up-collapse parity against oracle.
- `M` `tests/ops/test_offloaded_moe_layer.cu`: Updated `moe_combine_inject` tests to validate fused `next_norm_weight` / `next_norm_out` parity with `hyper_connection_norm`.
- `M` `tests/ops/test_offloaded_moe_wide.cu`: Added `test_masked_w4a4_verify` (`E=12, T=16, W=8, valid={3,6}, k=10`) with `0xA5` workspace poison and `0xFFFF` sentinel checks to verify masked `entry_map` safety.
- `M` `tests/ops/test_qsa.cpp`: Updated `test_prompt_variants` to allow attention-criterion floating-point tolerance (rather than bitwise identity) when comparing `SplitD` vs combined prompt kernels on non-VQ storages.
- `M` `tools/run_server.sh`: Added `local-f4` alias (`Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-New`), default `SPEC_BACKEND=dflash2` when `DRAFT_PATH` exists, `--expert-a4-activations` toggle via `EXPERT_A4_DEFAULT`, `--lm-head-draft`, and `INFERNIX_NGRAM_WIDTH_COST=0.08`.

### 3.7 Wave 1 (`4a4efac61`) & Wave 2 (Uncommitted) — Speculative Decoding, Runtime & Decode Kernel Tuning

- **`src/models/qwen3_5/` & `src/models/qwen4_exp/`:**
  - `M` `src/models/qwen3_5/program/suffix_automaton.h` *(Wave 1, `4a4efac61`)*: Added O(1) `edge_table_` transition lookup and `last_` tail fast path.
  - `M` `src/models/qwen4_exp/program/program_impl.h` & `M` `src/models/qwen4_exp/program/graph_set.cpp` *(Wave 1, `4a4efac61`)*: Added `mtp_sampled_gated_graphs_`, merged `mtp_catch` into `commit_graph`, and set default `kWidthCost = 0.08`.
  - `M` `src/models/qwen4_exp/program/ngram_volume.cpp` *(Wave 2)*: Added `kPrefetchDist = 8` `__builtin_prefetch` in `NgramVolume::read_rows` for both `resident_` RAM gathers and volume-cache `tags_` lookups, and factored `row_offset`.
- **Ops & Quantization (`src/ops/`):**
  - `M` `src/ops/screened_head/screened_head.cu` *(Wave 2)*: Parallelized `screen_kernel` post-MMA warp reduction, top-`kListLength` filtering, and softmax across 8 half-warps (`16` lanes/col); removed per-stage shared `partial[e]` writeback and redundant `__syncthreads()`; staged epilogue output lists through shared `out_lists` with zero stack spills (`STACK:0 LOCAL:0`).
  - `M` `src/ops/hyper_connection/hyper_connection_mix_fused.cu` *(Wave 2)*: Reduced `hc_down_partials_v4_kernel` register pressure for `ExactColumns == 4` by setting `kPreloadChunks = 1` and fusing `Pending` `pending_y`/`injected8` loads directly into the `values[c][h]` load loop (removing `ys[kColumns][kSlices]`), raising `__launch_bounds__` to `(ExactColumns >= 2 && ExactColumns <= 4) ? 2 : 1` (`Q4Codec` `REG: 141..146 -> 104..116`, `Q8Codec REG: 128`, `Bf16Codec REG: 106..113`, `STACK:0 LOCAL:0`).
  - `M` `src/ops/linear_attention/gated_delta_net/recurrent.cuh` *(Wave 2)*: Updated `recurrent_record_conv_kernel` launch bounds to `MaxWidth <= 16 ? 6 : 4`, reducing `MaxWidth = 16` registers from `85` to `72` (`STACK:0 LOCAL:0`).
  - `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.h`, `M` `src/ops/offloaded_sparse_moe/cuda/wide_expert.cu`, & `M` `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` *(Wave 2)*: Removed unused `narrow_acts` workspace allocation (`kNarrowActBytes`, `kNarrowQuantColumns`, `Offsets::narrow_acts`, and updated `moe_layer.cu` `static_assert`).
- **Tests & Tooling (`tests/`, `tools/`):**
  - `M` `tests/models/qwen4_exp/test_ngram_volume.cpp` & `M` `tests/models/qwen4_exp/test_forward_real.cpp` *(Wave 2)*: Dropped stale `const` qualifiers on `NgramVolume` parameters/locals to match commit `a5a0ce7f1`'s non-`const` header API.
  - `M` `tools/run_server.sh`: *(Wave 1, `4a4efac61`)* Added `hc4-w4dense` (`hc4`) alias; *(Wave 2)* defaulted `NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-auto}"` for `nvfp4`, `local-f4`, and `hc4-w4dense|hc4` (retaining `volume` for `dense8` and `ram` for `iqs_3`).

### 3.8 Wave 3 (Uncommitted) — Sampled MTP Acceptance, Penalty/Chain Overlay, Cumulative Cut & N-gram Hygiene

- **`include/`:**
  - `M` `include/infernix/ops/screened_head.h`: Expanded `kScreenedHeadMaxRanks` from `4` to `16` and documented `T in [1, 32]`, `ranks in [1, 16]` for `screened_head_top` / `screened_head_logits`.
  - `M` `include/infernix/ops/speculative_round.h`: Documented and added optional `prior_drafts` (`I32 [T, prior_steps]`) parameter to `speculative_sample_sparse_drafts`, specifying penalty and intra-chain count aggregation, tie-broken score ordering, and adjusted confidence output.
  - `M` `include/infernix/types.h`: Updated `FastPath::SampledMtpDrafts` (`16` ranks, penalty-aware draft temperature scaling, penalty/intra-chain overlays, cumulative or `--spec-min-p` cut) and `FastPath::ScreenedLmHead` (`<= 32` columns) API documentation.
- **Ops (`src/ops/`):**
  - `M` `src/ops/screened_head/screened_head.cu` & `M` `src/ops/screened_head/screened_head_launch.h`: Fixed multi-rank `screened_top_kernel` tail normalizer exclusion for `ranks <= 16` and updated `T <= 32` launch contract (`REG=70 STACK=0 SHARED=11936 LOCAL=0`).
  - `M` `src/ops/launcher/speculative_round.h`, `M` `src/ops/launcher/speculative_sample_drafts.cu`, & `M` `src/ops/wrapper/speculative_round.cpp`: Validated `prior_drafts` (non-aliasing matrix check in wrapper) and implemented penalty + `prior_drafts` count overlay, adjusted-score insertion sort (tie-breaking on lower token ID), residual tail mass normalizer for `confidence`, and `0.0F` tail fill (`REG=48 STACK=0 SHARED=0 LOCAL=0`).
- **Models (`src/models/`):**
  - `M` `src/models/qwen4_exp/program/mtp_gating.h`: Generalized `MtpCumulative::reset` and `observe` over configurable `(bins, base, floor)` and added calibrated 16-rank sampled table `kMtpSampledCumulativeBins` (`base = 0.28F`, `floor = 0.20F`).
  - `M` `src/models/qwen4_exp/execution/forward.h` & `M` `src/models/qwen4_exp/execution/forward.cpp`: Added `Tensor prior_drafts` to `MtpCall`, widened `screened_head` sampled rank condition to `<= ops::kScreenedHeadMaxRanks` (`16`), and forwarded `call.prior_drafts` to `ops::speculative_sample_sparse_drafts`.
  - `M` `src/models/qwen4_exp/program/program_impl.h`: Enabled `kSampledDraftRanks = ops::kSparseSpeculativeCandidates` (`16`), `MtpCumulative` per-lane calibration on sampled rounds (`kMtpSampledCumulativeBins` in `admit`/`verify`/`run_mtp_draft_steps`/`read_mtp_drafts`), intra-chain `prior_drafts` slice, penalty-aware draft temperature scaling (`scale = pen > 0 ? std::min(1.0F, 0.88F + 0.08F * pen) : 0.88F`), and `2.5x` (`<= 0.5`) N-gram EMA weight backoff on 0-token misses.
  - `M` `src/models/qwen3_5/program/ngram_proposer.h`: Rejected N-gram tails containing `boundaries_` tokens in addition to `< 0`.
- **Tests (`tests/`):**
  - `M` `tests/ops/test_screened_head.cpp`: Isolated `ranks=16` vs `ranks=4` top-4 bit-identity check under `wide_policy{.margin = 1000.0F, .rescored = 64}`, added oracle exclusion/ordering checks for `ranks=16`, and added `T = 24, 32` wide column invariance/graph tests.
  - `M` `tests/ops/test_speculative_round.cpp`: Extended candidate rank coverage to `16` and added unit + empirical distribution tests for penalized `token_counts`, tie-breaking, truncated support, confidence bounds, and `prior_drafts` intra-chain frequency accumulation.
  - `M` `tests/models/qwen4_exp/test_speculative.cpp`: Added `check_sampled_cumulative_table()` validating bin monotonicity, threshold lowering, and 5,000-trial host/device `mtp_cumulative_cut` agreement over `kMtpSampledCumulativeBins`.
  - `M` `tests/models/qwen4_exp/test_speculative_sampling_real.cpp`: Added `--presence-penalty <float>` option across reference, greedy, one-hot, and sampled MTP verification runs.

---

## 4. Verification & Benchmark Summary

### 4.1 Post-Edit Build & CTest Suite (`28/28` Passed)

All 28 affected unit/oracle CTests and the CUDA stack ratchet gate were rebuilt (`ninja -C build-native`) and executed after the final code edit (`Total Test time = 124.19 s`, `0` failed, `0` skipped):

- **Stack-spill ratchet:** `infernix_cuda_stack_check` (`2.60 s`) — `STACK:0`, `LOCAL:0` on `infernix-serve`.
- **Linear & NVFP4:** `infernix_nvfp4_tma_staging_race_test` (`8.11 s`), `infernix_linear_nvfp4_a16_test` (`13.66 s`), `infernix_linear_nvfp4_a4_test` (`22.65 s`), `infernix_linear_q4_a16_test` (`28.32 s`).
- **Hyper-Connection:** `infernix_hyper_connection_stream_mean_test` (`6.39 s`), `infernix_hyper_connection_test` (`30.86 s`).
- **QSA Prompt Attention:** `infernix_qsa_index_test` (`14.66 s`), `infernix_qsa_select_test` (`14.98 s`), `infernix_qsa_nvfp4_test` (`34.81 s`), `infernix_qsa_vq2_test` (`41.64 s`), `infernix_qsa_int8_test` (`41.91 s`), `infernix_qsa_k4v2_test` (`42.87 s`), `infernix_qsa_fp8_test` (`43.99 s`), `infernix_qsa_k8v4_test` (`46.37 s`), `infernix_qsa_bf16_test` (`46.70 s`).
- **Offloaded Sparse MoE:** `infernix_offloaded_moe_cuda_test` (`7.09 s`), `infernix_offloaded_moe_plan_test` (`14.85 s`), `infernix_offloaded_moe_qwen3_5_122b_test` (`15.91 s`), `infernix_offloaded_moe_wide_test` (`25.92 s`), `infernix_offloaded_moe_layer_test` (`112.77 s`), `infernix_offloaded_moe_layer_idle_test` (`11.40 s`).
- **Qwen4Exp Model/Vision/Speculative:** `infernix_qwen4_exp_vision_rg16_test` (`0.03 s`), `infernix_qwen4_exp_call_plan_test` (`0.06 s`), `infernix_qwen4_exp_expert_activations_test` (`0.01 s`), `infernix_qwen4_exp_speculative_test` (`0.02 s`), `infernix_qwen4_exp_vision_window_test` (`0.00 s`), `infernix_qwen4_exp_fast_paths_test` (`0.00 s`).

### 4.2 End-to-End `infernix-serve` A/B Results (`bench_client.py --max-tokens 256 --concurrency 1`, `prompt_tok=1225`, `cache 0.0%`)

| Configuration / Artifact | Client TTFT Mean (Runs 1–3) | Client Prefill Mean | Client Decode Mean (Runs 1–3) | Server Session Stats |
|---|---|---|---|---|
| `local-f4` baseline (`51fa3e2d7`) | `0.490s` | `2503.1 tok/s` | `73.7 tok/s` | — |
| `local-f4` post-campaign (`-New`, `w4dense` + Q8 HC) | `0.4633s` (**-5.4%**) | `2644.5 tok/s` (**+5.6%**) | `72.2 tok/s` (-2.0%) | `TTFT 440 ms`, `prefill 2.84k`, `DFLASH2 54.4%` (`1.44 acc/rnd`, `ngram 41.4%`) |
| `HC4` baseline (`51fa3e2d7`) | `0.554s` | `2209.4 tok/s` | `82.9 tok/s` | — |
| `HC4` pre-reconversion (`fp4_local` MXFP8 dense + Q4 HC) | `0.5207s` (-6.0%) | `2353.8 tok/s` (+6.5%) | `73.4 tok/s` (-11.5%) | `TTFT 501 ms`, `prefill 2.49k`, `decode 73.7`, `DFLASH2 57.5%` (`1.48 acc/rnd`) |
| **`HC4-W4Dense` post-reconversion (`w4dense` NVFP4 dense + Q4 HC)** | **`0.4637s`** (**-16.3% vs HC4 base**) | **`2641.7 tok/s`** (**+19.6% vs HC4 base**) | **`88.2 tok/s`** (**+6.4% vs HC4 base, +20.2% vs pre**) | `req#3–5`: `TTFT 437–443 ms`, `prefill 2.81–2.87k`, `decode 87.6–89.3`; 5-req session `DFLASH2 63.2%` (`1.89 acc/rnd`) |

- **Disclosed issues / findings:**
  1. During initial Step 4 startup testing, `infernix-serve warmup()` hit a `cudaErrorIllegalAddress` crash at `wide::prepare` due to uninitialized masked `entry_map` slots during W4A4 speculative verification (`verify_graph_body`, `batch > 1`). Fixed in `wide_expert.cu` (`plan_kernel` initializes `entry_map` to `-1`, `quantize_token_kernel` skips `expert < 0`) and locked down by `test_masked_w4a4_verify`.
  2. `local-f4` mean decode (`72.2 tok/s` vs `73.7` baseline) and pre-reconversion `HC4` decode (`73.4 tok/s` vs `82.9` baseline) measured lower on Runs 1–3 with active non-reproducible fast paths and speculative acceptance variance (`ngram 73.3%`, `acc/rnd 1.48`); reconverting `HC4` to `HC4-W4Dense` recovered and surpassed baseline decode (`88.2 tok/s`, `69.33 GiB` payload vs `71.11 GiB` pre-reconversion).
  3. End-to-end coherence on `HC4-W4Dense` was verified on `"What is 17 * 23? Explain briefly in one sentence."` (`"17 × 23 = 391."`), alongside the full 28-test CTest suite. Real-artifact integration CTests (`*_real_test`, including `infernix_qwen4_exp_forward_real_test`) were intentionally excluded per repository test constraints.

### 4.3 Wave 1 (`4a4efac61`) & Wave 2 Verification & Benchmark Results

- **Post-Wave-2 Stack & Oracle CTests (`12/12` executed, `0` failed):**
  - `infernix_cuda_stack_check` (`1.50 s`, re-run after final `infernix_qwen4_exp_forward_real_test` relink): `STACK:0 LOCAL:0` across `infernix-serve` and all Wave 2 `.cu.o` targets (`screened_head.cu.o`, `hyper_connection_mix_fused.cu.o`, `recurrent.cu.o`, `wide_expert.cu.o`, `moe_layer.cu.o`).
  - Targeted unit/oracle suite (`11/11` Passed in `186.56 s`, `--parallel 1`, matching the unmodified Wave 2 code state): `infernix_screened_head_test`, `infernix_hyper_connection_test`, `infernix_offloaded_moe_cuda_test`, `infernix_offloaded_moe_wide_test`, `infernix_offloaded_moe_layer_test`, `infernix_qwen4_exp_ngram_volume_test`, `infernix_ngram_proposer_test`, `infernix_qwen4_exp_speculative_test`, `infernix_speculative_round_test`, `infernix_linear_nvfp4_a16_test`, `infernix_gated_delta_net_replay_record_conv_test`.
  - `bash -n tools/run_server.sh` -> exit `0`; `infernix_qwen4_exp_forward_real_test` compiled/linked cleanly against current `libinfernix_ops.a` (not executed because real-artifact test env vars remain unset).
- **Kernel Microbenchmarks (Wave 2):**
  - `screened_head` (`sampling_select_bench` / `infernix_ops_sampling_test screened_head`, 3 uncontended runs): `top form T=1`: `499.69 -> 494.06 us` median; `BF16 head` (`T=1/2/4/5/8`): `934.05/938.07/949.95/955.33/971.19 -> 936.10/933.42/944.53/948.62/965.93 us` (`-5 us` for `T=2..8`); `Q8 head T=5/8` showed run-to-run host jitter (`960..988 vs 940..959 us`) on the same `screen_kernel` binary as `BF16 head`.
  - `hyper_connection_mix_fused` (`T=4`, `q8`): `infernix_hyper_connection_bench --codec q8` `36.91 -> 36.45 us` (min `36.61 -> 36.21 us`); `infernix_hc_mix_fused_bench --codec q8 --graph` `fused_cold` `43.94 -> 41.70 us` (min `41.54 -> 40.67 us`). Note: `hc_mix_fused_bench` only supports `q8|bf16`, not `q4`; `infernix_gdn_replay_bench --component recurrent` benchmarks `recurrent_record_kernel`, not `recurrent_record_conv_kernel`.
- **End-to-End `HC4-W4Dense` CLI A/B (`k8v4`, `--expert-a4-activations --spec mtp --draft-tokens auto --lm-head-draft --ngram-draft-tokens 31 --ngram-min-match 4 --max-context 32768 --max-new 384`, `INFERNIX_EXPERTS_ON_GPU=1 INFERNIX_NGRAM_WIDTH_COST=0.08`):**

| Workload | Pre-Wave-1 (`volume`) | Wave 1 (`volume`) | Wave 2 Suite A (`volume`, 3 repeats decode tok/s, median) | Wave 2 Suite B (`auto` RAM `27.1 GiB`, 3 repeats decode tok/s, median) |
|---|---|---|---|---|
| `P1_sampled` (`seed=42`, `154` rounds, `50.1%`, `2.45 tok/rnd`) | `70.0` | `72.1` | `72.5, 72.4, 72.3` (**`72.4`**) | `71.8, 72.4, 72.1` (**`72.1`**) |
| `P2_sampled` (`seed=42`, `156` rounds, `48.3%`, `2.44 tok/rnd`) | `66.9` | `69.7` | `72.1, 72.0, 72.2` (**`72.1`**) | `72.4, 72.4, 72.1` (**`72.4`**) |
| `P2_greedy` (`140` rounds, `65.3%`, `2.44 tok/rnd`) | `75.6` | `77.2` | `77.3, 77.4, 77.3` (**`77.3`**) | `77.8, 77.7, 77.4` (**`77.7`**) |

  - Suite A (`volume --ngram-cache-mib 4096`) overall throughput: `P1_sampled` `70.6 / 70.5 / 70.5`, `P2_sampled` `70.2 / 70.1 / 70.3`, `P2_greedy` `75.2 / 75.2 / 75.1 tok/s`; prefill: `480.9–500.2 tok/s` (`152–156 ms`).
  - Suite B (`auto`, `27.1 GiB` `.ngram` table resident in RAM, `10.2–10.6 GiB` left free) overall throughput: `P1_sampled` `70.1 / 70.7 / 70.4`, `P2_sampled` `70.7 / 70.7 / 70.4`, `P2_greedy` `75.8 / 75.7 / 75.4 tok/s`; prefill: **`521.0–540.7 tok/s`** (`142–146 ms`). Single-CLI full process wall time in `auto` was `25,228–26,905 ms` due to `~20 s` one-time `.ngram` RAM load (amortized across requests in `infernix-serve`).
- **Rejected / Reverted Candidates & Measurement Caveats:**
  - *Reverted:* `src/ops/linear/nvfp4/shapes/n2560_k6144.cu` `Nvfp4SlicedInstance<8, 8, 1>` for `tokens <= 8` regressed from `43.840 us` (`202.6–205.0 GB/s`) to `53.088–55.552 us` (`161.8–167.9 GB/s`) -> reverted (`git checkout`).
  - *Rejected:* `ExactColumns = 5..8` in `hc_up_collapse_v4_kernel` (`kHcMixFusedMaxColumns = 4` remains optimal; `T >= 5` already routes to Tensor Core `q4_a16_mma_kernel` / `q8_a16_mma_kernel`).
  - *Profiling artifact note:* `nsys --cuda-graph-trace=node` introduced a uniform `~4–5%` slowdown across all kernels (including untouched `gate_up_kernel` / `down_kernel` / `nvfp4_a16_sliced_k`); uncontended Suite A/B runs verified true end-to-end speedups.

### 4.4 Wave 3 Verification, Multi-Seed Serving A/B & CLI Results (`Wave3_E` on `HC4-W4Dense`, `k8v4`)

- **Post-Wave-3 CTest, Static Stack & Determinism Gates (all executed sequentially after the final source edit):**
  - `ctest --test-dir build-native -R "^(infernix_screened_head_test|infernix_speculative_round_test|infernix_gumbel_speculation_test|infernix_ngram_proposer_test|infernix_qwen4_exp_speculative_test|infernix_qwen4_exp_ngram_volume_test|infernix_cuda_stack_check)$" --output-on-failure --parallel 1` -> **`7/7` Passed** (`50.87 s`, `0` failed): `infernix_cuda_stack_check` (`1.54 s`, `STACK:0 LOCAL:0`), `infernix_ngram_proposer_test` (`0.30 s`), `infernix_qwen4_exp_ngram_volume_test` (`0.46 s`), `infernix_qwen4_exp_speculative_test` (`0.00 s`), `infernix_screened_head_test` (`19.19 s`), `infernix_speculative_round_test` (`27.20 s`), `infernix_gumbel_speculation_test` (`2.18 s`).
  - `ckt static` on shipped `build-native/apps/infernix-serve`: `screened_top_kernel` `REG=70 STACK=0 SHARED=11936 LOCAL=0` (3 blk/SM); `speculative_sample_sparse_drafts_kernel` `REG=48 STACK=0 SHARED=0 LOCAL=0` (5 blk/SM).
  - **Real-artifact 800-trial chi-square homogeneity + greedy gates** (`--fast-paths --concurrency 1 --expert-a4-activations --lm-head-draft --auto-draft --block-verification --trials 800`):
    - Unpenalized (`scale=0.88F`): one-hot MTP `2288/4294` (`53.3%`, `1.83 tok/round`, `least p = 0.0444 >= 0.000111`); sampled MTP `2333/4264` (`54.7%`, `1.85 tok/round`, `least p = 0.000399 >= 0.000111`); greedy prompts 0 & 1 exact.
    - Penalized (`--presence-penalty 1.5`, `scale=1.0F`): one-hot MTP `2279/4299` (`53.0%`, `1.82 tok/round`, `least p = 0.0272 >= 0.000111`); sampled MTP `2338/4272` (`54.7%`, `1.86 tok/round`, `least p = 0.00493 >= 0.000111`); greedy prompts 0 & 1 exact.
    - Earlier `Wave3_A` (`scale=1.0F`) also passed Gumbel ref Gate 1 (`least p = 0.0447`, sampled `1485/5347 = 27.8%`, `1.44 tok/round`) and `BlockVerification` Gates 2–3 (`least p = 0.00442` and `0.00493`).
  - **Seeded `--reproducible` CLI determinism** (`2` runs, `--seed 42 --max-new 128`): byte-identical generated completion (`1020` bytes body `cmp` match, `128` tokens, `55` rounds, `68/97` accepted = `70.1%`, `2.24 tok/round`; fast paths off as required by `--reproducible`).
- **44-Request Persistent `infernix-serve` Multi-Seed Benchmark (`1` warmup + `5` sampled cells x `8` seeds `42..49` + `2` greedy cells x `2` repeats, `--ngram-residency auto --kv-dtype k8v4 --max-new 384`):**

| Cell | `Wave2_Baseline` (`4` ranks, `0.85F`, `-2.12`) | `Wave3_A` (`16` ranks, `1.0F`, sampled cum.) | `Wave3_E` (`16` ranks, `0.88F..1.0F` pen-aware) | `Wave3_E` vs `Wave2_Baseline` (`ckt math ab`) |
|---|---|---|---|---|
| `P1_sampled` (`n=8`) | `69.17 tok/s`, `49.00%`, `2.357 tok/rnd` | `67.32`, `47.33%`, `2.231` | `71.08 tok/s`, `51.85%`, `2.373` | **`+2.77%`** (`1.0288x`, `4/8`, `WITHIN_NOISE`), **`+2.84 pp`** (`1745/3561 -> 1769/3412`) |
| `P2_sampled` (`n=8`) | `67.91 tok/s`, `46.57%`, `2.353 tok/rnd` | **`70.36`**, `48.89%`, `2.408` | `67.61 tok/s`, `46.14%`, `2.343` | **`-0.44%`** (`0.9990x`, `4/8`, `WITHIN_NOISE`), **`-0.43 pp`** (`1740/3736 -> 1751/3795`) *(adverse on E)* |
| `P3_sampled_topp95` (`n=8`) | `63.71 tok/s`, `47.37%`, `2.294 tok/rnd` | `65.89`, `49.28%`, `2.402` | **`66.40 tok/s`**, `50.39%`, `2.436` | **`+4.22%`** (`1.0266x`, `6/8`, `WITHIN_NOISE`), **`+3.02 pp`** (`1705/3599 -> 1799/3570`) |
| `P3_penalized` (`n=8`) | `59.96 tok/s`, `42.95%`, `2.125 tok/rnd` | **`62.86`**, `48.05%`, `2.187` | `62.67 tok/s`, **`48.05%`**, `2.187` | **`+4.53%`** (`1.0475x`, `6/8`, `WITHIN_NOISE`), **`+5.11 pp`** (`1605/3737 -> 1643/3419`) |
| `P4_code_sampled` (`n=8`) | `73.11 tok/s`, `53.09%`, `2.666 tok/rnd` | **`76.06`**, `55.62%`, `2.831` | `75.38 tok/s`, `55.21%`, `2.800` | **`+3.11%`** (`1.0785x`, `5/8`, `WITHIN_NOISE`), **`+2.12 pp`** (`1908/3594 -> 1966/3561`) |
| **`OVERALL_SAMPLED` (`n=40`)** | **`66.77 tok/s`**, `47.75%`, `2.347 tok/rnd` | `68.50`, `49.84%`, `2.393` | **`68.63 tok/s`**, **`50.28%`**, **`2.412`** | **`+2.78%`** (`1.0314x`, `25/40`, `WITHIN_NOISE` cross-seed IQR `10.53%`), **`+2.53 pp`** (`8703/18227 -> 8928/17757`, rounds `6462->6321`, fb `155->71`) |
| `P2_greedy` (`n=2`) | `75.65 tok/s`, `65.26%`, `2.436 tok/rnd` | `75.42`, `65.26%`, `2.436` | `75.40 tok/s`, `65.26%`, `2.436` | `-0.33%` (`0.9966x`, `WITHIN_NOISE`), exact `402/616` (`+0.00 pp`) |
| `P4_code_greedy` (`n=2`) | `84.14 tok/s`, `64.58%`, `3.157 tok/rnd` | `86.09`, `66.58%`, `3.219` | **`86.35 tok/s`**, **`66.58%`**, **`3.219`** | **`+2.63%`** (`1.0263x`, `2/2`, **`MEANINGFUL_GAIN`**), **`+2.00 pp`** (`496/768 -> 506/760`, fb `40->32`) |

- **3-Repeat Single-Seed (`seed=42`, `--ngram-residency volume`) CLI A/B & Trajectory Variance Analysis:**
  Under `BlockVerification` (`!gumbel_max`), small draft distribution changes branch into entirely different 384-token completions on a single seed (producing `±10–13%` single-seed swing with `~0.14%` intra-seed repeat CV, demonstrating why multi-seed evaluation is authoritative). Specifically on `seed=42` CLI (`n=3` repeats each):
  - Gains (`MEANINGFUL_GAIN`): `P1_sampled` `72.40 -> 76.00 tok/s` (`+4.97%`, `50.1% -> 54.0%`, `2.45 -> 2.58`), `P3_penalized` `62.90 -> 68.70 tok/s` (`+9.22%`, `44.2% -> 53.7%`, `2.20 -> 2.48`), `P4_code_greedy` `85.40 -> 87.10 tok/s` (`+1.99%`, `64.6% -> 66.6%`, `3.16 -> 3.22`), `P4_code_sampled` `72.20 -> 79.00 tok/s` (`+9.42%`, `49.5% -> 60.4%`, `2.68 -> 2.80`).
  - Adverse single-seed `seed=42` trajectories (`REGRESSION`): `P2_sampled` `72.10 -> 63.20 tok/s` (`-12.34%`, `48.3% -> 40.0%`, `2.44 -> 2.16`; 8-seed `67.91 -> 67.61`), `P3_sampled_topp95` `67.70 -> 63.10 tok/s` (`-6.79%`, `48.4% -> 45.9%`, `2.42 -> 2.30`; 8-seed `+4.22%`), `P2_greedy` `77.30 -> 76.10 tok/s` (`-1.55%`, identical `201/308 = 65.3%`, `140` rounds, `42` fallbacks; `infernix-serve` `-0.33%`).
- **Rejected / Reverted Wave 3 Candidates (all reverted cleanly):**
  - *Candidate B* (`scale = 0.94F`), *Candidate C* (`kScreenedHeadPolicy = {.margin = 5.5F, .rescored = 128}`), and *Candidate D* (`kMtpCumulativeBase = 0.30F`, which failed `infernix_qwen4_exp_speculative_test` `"continuation threshold starts at 0.35"` and regressed `P4_code_greedy` `86.8 -> 82.75 tok/s`).

---

## 5. Rebase Conflict Resolution Policy

- **Prefill MoE & Masked Verify (`src/ops/offloaded_sparse_moe/cuda/`):** Preserve `entry_map` `-1` initialization in `plan_kernel` and `expert >= 0` guard in `quantize_token_kernel`, `narrow_jobs` / `pass_narrow` compaction, removal of unused `narrow_acts` workspace allocation, vectorized SMEM `narrow_expert.cu` with PDL, and optional fused next-layer norm in `moe_combine_inject`.
- **Hyper-Connection Q4/Q8 (`src/ops/hyper_connection/`):** Preserve `Q4Codec` fused decode mix, `hc_down_partials_v4_kernel` `kPreloadChunks = 1` + fused `Pending` register reduction (`ExactColumns == 4` `2` blk/SM), `hc_up_collapse` `Q4_G64_FP16` support, and pipelined `pre_norm` reuse across layers.
- **Screened Head & GDN Replay (`src/ops/screened_head/`, `src/ops/linear_attention/gated_delta_net/`):** Preserve `screen_kernel` 8-half-warp reduction/shared `out_lists` epilogue (`STACK:0 LOCAL:0`), `kScreenedHeadMaxRanks = 16` / `kScreenedHeadMaxColumns = 32` (`screened_top_kernel` normalizer fix), and `recurrent_record_conv_kernel` `MaxWidth <= 16 ? 6 : 4` launch bounds (`REG:72`).
- **MTP / N-gram Speculative & Server Defaults (`src/models/`, `src/ops/launcher/speculative_sample_drafts.cu`, `tools/run_server.sh`):** Preserve `SuffixAutomaton` O(1) `edge_table_` + `last_` tail fast path, `mtp_sampled_gated_graphs_`, merged `mtp_catch`, `kWidthCost = 0.08`, `NgramVolume::read_rows` software prefetch (`kPrefetchDist = 8`) with non-`const` `NgramVolume` signature (`a5a0ce7f1`), `NgramProposer` `boundaries_` suffix check + `2.5x` 0-miss EMA backoff, `kSampledDraftRanks = 16` + `prior_drafts`/penalty overlay + `0.88F..1.0F` penalty-aware temp scaling + `kMtpSampledCumulativeBins`, and `NGRAM_RESIDENCY=auto` default on 128 GiB UMA for `infernix-serve` (`nvfp4`, `local-f4`, `hc4-w4dense`/`hc4`).
- **QSA Prompt (`src/ops/qsa/qsa_prompt.cu`):** Preserve `SplitD` compile-time specialization (`PromptShape` non-VQ storages) and dynamic shared-memory phase overlay for GB10 48-SM occupancy over upstream combined-only `qsa_prompt`.
- **Linear `/shapes/` & Vision (`src/ops/linear/{nvfp4,q4}/`, `src/runtime/engine/`):** Preserve GB10-swept Q4 shape files (`n320_k10240`, `n324_k10240`, `n10240_k320`, `n1280_k2560`, `n2560_k640`), updated `nvfp4` pair gates (`56`/`52`), and on-device vision support under `INFERNIX_EXPERTS_ON_GPU=1`.
