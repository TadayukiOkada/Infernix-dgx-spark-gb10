# GB10 (`sm_121a`) Concurrent & Speculative Decode Optimization Tasks (2026-11-23)

- **Date:** 2026-11-23
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, CC `12.1`, 48 SMs, 24 MiB L2, 121.7 GiB unified LPDDR5X UMA, `~244 GB/s` read / `199 GB/s` write, `~45 W` decode draw under `90 W` TDP)
- **Baseline:** `8edc2c4a8` + working-tree concurrency & startup-capture fixes (`effective_staged_columns` + upfront `verify_target_width` pre-staging behind `mtp_draft`, `kPairMaxDrafts = 7`, `kMultiRowMaxDrafts = 5`, bucketed `capture_graph_set` bounds)
- **Primary Model Scope:** `Qwen3.8-Flash-Next` (`HC4-W4Dense`) on `infernix-serve` / `.infernix` runtime

---

## 1. Context & Baseline Summary

### 1.1 Why Decode Scaling Stalls on GB10
Across `nsys` traces ([`profiles/nsys/kernel_summary.csv`](../../profiles/nsys/kernel_summary.csv)) and live server runs ([`docs/tasks/20261122-gb10-hardware-utilization-tasks.md`](20261122-gb10-hardware-utilization-tasks.md)), `Qwen3.8-Flash-Next` (`HC4-W4Dense`) decode on GB10 is **LPDDR5X memory-bandwidth-bound** (`~244 GB/s` read ceiling, `<1%` steady-state host gaps between GPU launches, ~45 W power draw).

Because dense trunk and routed MoE weight reads saturate memory bandwidth even at small `T`, **pushing decode throughput further (especially at `batch >= 2`) comes from two levers**:
1. **Increasing accepted tokens per round** without increasing weight reads (raising speculation acceptance and depth).
2. **Keeping `batch >= 2` verification rounds off slow fallback kernels** (screened LM head, fused Tensor-Core HC mix) when total columns `batch * W` cross narrow single-row thresholds.

### 1.2 Baseline Changes Already Landed in Tree
- **Zero-stall concurrent PLE pre-staging ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`src/models/qwen4_exp/program/mtp_gating.h`](../../src/models/qwen4_exp/program/mtp_gating.h)):**
  - `effective_staged_columns()` prevents stale/cross-row PLE n-gram leaks for rows `b > 0` when `W > staged_width`.
  - `verify_target_width()` computes the bucketed verify width upfront before `mtp_draft()` / `dflash2_propose()` so `stage_verify_ngram_batch()` stages at `staged_width == W`, keeping `b > 0` pre-staging hidden behind drafting without synchronous cache re-reads.
- **Uncapped multi-row MTP depth ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h)):**
  - Raised `kPairMaxDrafts` default `3.0 -> 7.0` (`INFERNIX_PAIR_MAX_DRAFTS`) and `kMultiRowMaxDrafts` default `4.0 -> 5.0` (`INFERNIX_MULTIROW_MAX_DRAFTS`), letting `choose_pair_draft_length()` / `choose_multirow_draft_length()` + `mtp_draft_gated()` draft deeper on high-confidence concurrent lanes while staying within `kScreenedHeadMaxColumns = 32` for `batch = 2..5` neural rounds.
- **Bucketed startup CUDA graph capture ([`src/models/qwen4_exp/program/graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp)):**
  - Bounded `capture_graph_set()` by `bucket_verify_width(shape.second, max_width_)` so widened multi-row shapes (`3x24`, `7x10`) are captured at startup instead of lazily at first use.

---

## 2. Impact-Ranked Optimization Backlog (`D1`–`D5`)

| Priority | Task ID | Goal | Primary Files / Symbols | Target Scope | Expected GB10 Impact | Status |
|---|---|---|---|---|---|---|
| **P0** | **D1** | Enable & A/B-verify `--block-verification` (`options_.block_verification`) for sampled decode (`temperature > 0`) | [`program.h`](../../src/models/qwen4_exp/program/program.h), [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`speculative_round.cuh`](../../src/ops/kernel/speculative_round.cuh) | `temperature > 0` sampled MTP/DFlash2 rounds (`batch >= 1`) | Higher expected accepted tokens/round at zero extra DRAM weight bandwidth | Open |
| **P0** | **D2** | Raise `kScreenedHeadMaxColumns` (`32 -> 64`) | [`screened_head.h`](../../include/infernix/ops/screened_head.h), [`screened_head/`](../../src/ops/screened_head/screened_head.cu), [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp) | `batch >= 2` wide n-gram (`W = 16..32`) and `batch = 6..8` neural verify rounds (`32 < batch * W <= 64`) | Eliminates full-Q8 LM head (`projection_tall_exact_kernel`, ~2.68 ms/call) + full-vocab top-k fallback (`7.6%` GPU share in `kernel_summary.csv`) | Open |
| **P1** | **D3** | Extend fused Tensor-Core `Q4` `hyper_connection_mix` (`HMMA.16816` + Cluster-DSMEM) from `T <= 16` to `T <= 32` | [`hyper_connection_mix_fused.h`](../../src/ops/hyper_connection/hyper_connection_mix_fused.h), [`hyper_connection_mix_fused.cu`](../../src/ops/hyper_connection/hyper_connection_mix_fused.cu) | `batch = 2..4` verify rounds (`T = 17..32`) | Eliminates unfused `q4_a16` fallback (`~6.8%` of `B=4` decode) and intermediate DRAM writes (`199.3 GB/s` write ceiling) | Open |
| **P1** | **D4** | Extend `univer_trees` root-fork verification from `batch == 1` to `batch == 2` | [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp) | `batch = 2` sampled rounds (`2 * (K + 2) <= 32`) | Recovers step-0 rejections during high-entropy `<think>` decode at negligible extra bandwidth | Open |
| **P2** | **D5** | Long-context & serving tuning: `KV_DTYPE=k8v4` A/B, restore `PREFILL_CHUNK=12288`, batch-aware `RoundCostModel` | [`run_server.sh`](../../tools/run_server.sh), [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`mtp_gating.h`](../../src/models/qwen4_exp/program/mtp_gating.h) | `60K–100K+` context decode, cold prefill TTFT, and skewed `batch >= 2` MoE depth policy | Halves Value KV bandwidth (`k8v4`), reduces chunk overhead (`12288`), calibrates `batch >= 2` `moe_round_cost` | Open |

---

## 3. Task Details

### D1 (P0): Enable & A/B-Verify `--block-verification` for Sampled Decode
- **Observation:** `ops::speculative_accept_compact_drafts` and `ops::speculative_accept_sparse_drafts` already implement exact block verification ([`src/ops/kernel/speculative_round.cuh`](../../src/ops/kernel/speculative_round.cuh)), gated by `options_.block_verification` ([`src/models/qwen4_exp/program/program.h`](../../src/models/qwen4_exp/program/program.h)), which currently defaults to `false`.
- **Action:**
  1. Benchmark `HC4-W4Dense` at `--temperature 1.0 --top-p 0.9 --min-p 0.1 --top-k 20` with `block_verification = false` vs `true` (`batch = 1, 2, 4`), measuring accepted tokens/round and net tok/s.
  2. Verify distributional exactness against `infernix_speculative_round_test` / `infernix_speculative_wide_distribution_test` and `--reproducible` determinism.
  3. Enable under fast paths / default runtime options if net E2E decode tok/s improves without quality regression.

### D2 (P0): Raise `kScreenedHeadMaxColumns` (`32 -> 64`)
- **Observation:** `screened_round()` ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h)) rejects any round with `batch * W > kScreenedHeadMaxColumns` (`32` in [`include/infernix/ops/screened_head.h`](../../include/infernix/ops/screened_head.h)). Multi-row n-gram rounds (`batch = 2, W = 20..32`, `batch = 3, W = 12..24`) and high-concurrency rounds (`batch = 6..8, K = 5`) therefore fall back to the full Q8 LM head (`projection_tall_exact_kernel`, `~2.68 ms` avg) plus full-vocab `speculative_sampling_partial_topk_kernel` (`7.6%` combined GPU time in [`profiles/nsys/kernel_summary.csv`](../../profiles/nsys/kernel_summary.csv)).
- **Action:**
  1. Check `screen_kernel` / `compact_ids_` / `compact_logits_` workspace sizing (`kCompactCols`) and shared-memory/tile bounds in [`src/ops/screened_head/screened_head.cu`](../../src/ops/screened_head/screened_head.cu) and [`screened_head_bf16.cu`](../../src/ops/screened_head/screened_head_bf16.cu) for `columns = 33..64`.
  2. Raise `kScreenedHeadMaxColumns` to `64` and capture screened verify graphs up to `batch * W <= 64` in [`src/models/qwen4_exp/program/graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp).
  3. Qualify against FP32/FP64 screened-head tests and measure `batch = 2..8` decode tok/s.

### D3 (P1): Extend Fused Tensor-Core `Q4` `hyper_connection_mix` to `T <= 32`
- **Observation:** Fused `HMMA.16816` + Cluster-DSMEM `Q4` HC mix currently caps at `T <= 16` ([`src/ops/hyper_connection/hyper_connection_mix_fused.cu`](../../src/ops/hyper_connection/hyper_connection_mix_fused.cu)), so `batch = 3..4` (`K = 5`, `T = 18..24`) and `batch = 2` wide n-gram (`T = 20..32`) rounds drop back to unfused `q4_a16_mma` + `q4_a16_sliced_k_mma` (`~6.8%` of `B=4` decode time).
- **Action:**
  1. Size `ckt math occupancy` / `smem` for `T = 17..32` (2-pass or `32`-column tile without increasing CTA register pressure beyond `<= 96` regs/thread or spilling `STACK`/`LOCAL`).
  2. Gate via `hc_mix_fused_supported()` up to `T = 32` only for shapes where fused beats unfused `q4_a16` in microbenchmarks and E2E `batch = 2..4` runs.
  3. Verify against FP32/FP64 oracle in `test_hyper_connection` and check `infernix_cuda_stack_check`.

### D4 (P1): Extend `univer_trees` Root-Fork Verification to `batch = 2`
- **Observation:** `univer_trees` root-fork verification is gated to `batch == 1` in `verify()` ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h)) and [`src/models/qwen4_exp/program/graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp). At `batch = 2` with `K <= 7`, adding 1 root-fork side column (`2 * (K + 2) <= 18`) still stays well inside `kScreenedHeadMaxColumns = 32` and adds negligible dense weight bandwidth.
- **Action:**
  1. Remove the `batch == 1` restriction where `batch <= 2 && W <= ops::kSpeculativeTreeMaxNodes` and `row_is_tree_` / `tree_rows_host[b]` / compaction planes support per-row tree flags (already indexed by `b`).
  2. Add startup capture for `batch = 2` tree verify graphs in `graph_set.cpp`.
  3. Verify tree acceptance/distribution tests and run `batch = 2` A/B on high-entropy reasoning prompts.

### D5 (P2): Long-Context & Serving Configuration Tuning
- **`KV_DTYPE=k8v4` A/B ([`tools/run_server.sh`](../../tools/run_server.sh)):** Production agentic requests hit `60K–97K` context; evaluate `k8v4` vs `int8` for halved Value KV bandwidth in QSA attention and reduced KV RAM footprint (potentially enabling `NGRAM_RESIDENCY=ram` alongside `HC4-W4Dense` to eliminate NVMe volume misses).
- **Restore `PREFILL_CHUNK=12288` ([`tools/run_server.sh`](../../tools/run_server.sh)):** Working tree currently sets `8192`; restore/benchmark `12288` on `30K–60K` cold prefills to reduce chunk count and prefill TTFT.
- **Batch-aware `RoundCostModel` ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`src/models/qwen4_exp/program/mtp_gating.h`](../../src/models/qwen4_exp/program/mtp_gating.h)):** `observe()` currently only records `batch == 1` round timings and extrapolates `batch >= 2` with a uniform coupon-collector prior; add direct `batch >= 2` cost observations so `choose_pair_draft_length()` / `choose_multirow_draft_length()` adapt to skewed MoE expert routing across concurrent requests.

---

## 4. Non-Goals / Closed Items (Do Not Chase)
- **Host CPU micro-optimizations (`T8.2`):** `nsys` confirms GPU inter-kernel gaps in steady-state decode are `<1%`; scalar host bookkeeping/hashing is not the bottleneck.
- **Zero-copy mapped host-device control buffers (`T2.2`):** Previously tested and reverted after `+12.68 ms` uncached GPU consumer regression exceeded `0.75 ms` transfer savings.

---

## 5. Verification & Acceptance Rules
1. **Analytical Sizing & Stack-Spill Gate:** Run `PYTHONPATH=cuda-kernel-toolkit /opt/etc/Infernix/.venv-convert/bin/python3 -m ckt math ...` before kernel edits and verify `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` (zero new `STACK > 0` or `LOCAL > 0` symbols).
2. **Numerical Oracles & Routes:** Qualify all Op changes against independent FP32/FP64 oracles per [`docs/maintainer/op-development.md`](../maintainer/op-development.md). Keep `--reproducible` deterministic/request-invariant and `--lossless` bit-exact; qualify any fast-path change under [`docs/maintainer/model-quality.md`](../maintainer/model-quality.md).
3. **E2E A/B Measurements:** Require paired A/B benchmarks on `HC4-W4Dense` across `batch = 1, 2, 3, 4` (`tok/s` total & per-lane, MTP/n-gram acceptance rate, and TTFT) rather than Op microbenchmarks alone.
