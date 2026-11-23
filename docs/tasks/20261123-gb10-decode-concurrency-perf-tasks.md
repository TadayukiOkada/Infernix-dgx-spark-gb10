# GB10 (`sm_121a`) Concurrent & Speculative Decode Optimization Tasks (2026-11-23)

- **Date:** 2026-11-23
- **Target Architecture:** NVIDIA DGX Spark GB10 (`sm_121a`, CC `12.1`, 48 SMs, 24 MiB L2, 121.7 GiB unified LPDDR5X UMA, `~244 GB/s` read / `199 GB/s` write, `~45 W` decode draw under `90 W` TDP)
- **Baseline:** `8edc2c4a8` + `1d8cce2cf` concurrency & startup-capture fixes (`effective_staged_columns` + upfront `verify_target_width` pre-staging behind `mtp_draft`, `kPairMaxDrafts = 7`, `kMultiRowMaxDrafts = 5`, bucketed `capture_graph_set` bounds)
- **Primary Model Scope:** `Qwen3.8-Flash-Next` (`HC4-W4Dense`) on `infernix-serve` / `.infernix` runtime

---

## 1. Context & Baseline Summary

### 1.1 Why Decode Scaling Stalls on GB10
Across `nsys` traces ([`profiles/nsys/kernel_summary.csv`](../../profiles/nsys/kernel_summary.csv)) and live server runs ([`docs/tasks/20261122-gb10-hardware-utilization-tasks.md`](20261122-gb10-hardware-utilization-tasks.md)), `Qwen3.8-Flash-Next` (`HC4-W4Dense`) decode on GB10 is **LPDDR5X memory-bandwidth-bound** (`~244 GB/s` read ceiling, `<1%` steady-state host gaps between GPU launches, ~42–46 W decode power draw).

Because dense trunk and routed MoE weight reads saturate memory bandwidth even at small `T`, **pushing decode throughput further (especially at `batch >= 2`) comes from two levers**:
1. **Increasing accepted tokens per round** without increasing weight reads (raising speculation acceptance and depth).
2. **Keeping `batch >= 2` verification rounds off slow fallback kernels** (screened LM head, fused Tensor-Core HC mix) when total columns `batch * W` cross narrow single-row thresholds.

### 1.2 Baseline Changes Already Landed in Tree (`1d8cce2cf`)
- **Zero-stall concurrent PLE pre-staging ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`src/models/qwen4_exp/program/mtp_gating.h`](../../src/models/qwen4_exp/program/mtp_gating.h)):**
  - `effective_staged_columns()` prevents stale/cross-row PLE n-gram leaks for rows `b > 0` when `W > staged_width`.
  - `verify_target_width()` computes the bucketed verify width upfront before `mtp_draft()` / `dflash2_propose()` so `stage_verify_ngram_batch()` stages at `staged_width == W`, keeping `b > 0` pre-staging hidden behind drafting without synchronous cache re-reads.
- **Uncapped multi-row MTP depth ([`src/models/qwen4_exp/program/program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h)):**
  - Raised `kPairMaxDrafts` default `3.0 -> 7.0` (`INFERNIX_PAIR_MAX_DRAFTS`) and `kMultiRowMaxDrafts` default `4.0 -> 5.0` (`INFERNIX_MULTIROW_MAX_DRAFTS`), letting `choose_pair_draft_length()` / `choose_multirow_draft_length()` + `mtp_draft_gated()` draft deeper on high-confidence concurrent lanes while keeping neural rounds compact.
- **Bucketed startup CUDA graph capture ([`src/models/qwen4_exp/program/graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp)):**
  - Bounded `capture_graph_set()` by `bucket_verify_width(shape.second, max_width_)` so widened multi-row shapes (`3x24`, `7x10`) are captured at startup instead of lazily at first use.

---

## 2. Impact-Ranked Optimization Backlog (`D1`–`D5`)

| Priority | Task ID | Goal | Primary Files / Symbols | Target Scope | Expected GB10 Impact | Status |
|---|---|---|---|---|---|---|
| **P0** | **D1** | Enable & A/B-verify `--block-verification` (`options_.block_verification`) for sampled decode (`temperature > 0`) | [`fast_paths.cpp`](../../src/models/qwen4_exp/program/fast_paths.cpp), [`qwen4_exp_instance.cpp`](../../src/runtime/engine/qwen4_exp_instance.cpp), [`speculative_round.cuh`](../../src/ops/kernel/speculative_round.cuh) | `temperature > 0` sampled MTP/DFlash2 rounds (`batch >= 1`) | Higher expected accepted tokens/round at zero extra DRAM weight bandwidth | **Already Enabled** (`FastPath::BlockVerification` qualified & default-on for `HC4-W4Dense`) |
| **P0** | **D2** | Raise `kScreenedHeadMaxColumns` (`32 -> 64`) | [`screened_head.h`](../../include/infernix/ops/screened_head.h), [`screened_head.cu`](../../src/ops/screened_head/screened_head.cu), [`projection_fp32_mma.h`](../../src/ops/projection_fp32/projection_fp32_mma.h), [`test_screened_head.cpp`](../../tests/ops/test_screened_head.cpp) | `batch >= 2` wide n-gram (`W = 16..32`) and `batch = 6..8` neural verify rounds (`32 < batch * W <= 64`) | Eliminates full-Q8 LM head (`projection_tall_exact_kernel`) + full-vocab top-k fallback; BF16 `kProjectionMmaMaxColumns` also raised `32 -> 64` for bit-exact rescore consistency | **Done** (`infernix_screened_head_test`, `infernix_projection_fp32_test`, `infernix_cuda_stack_check` pass) |
| **P1** | **D3** | Extend fused Tensor-Core `Q4` `hyper_connection_mix` (`HMMA.16816` + Cluster-DSMEM) from `T <= 16` to `T <= 32` | [`hyper_connection_mix_fused.h`](../../src/ops/hyper_connection/hyper_connection_mix_fused.h), [`hyper_connection_mix_fused.cu`](../../src/ops/hyper_connection/hyper_connection_mix_fused.cu) | `batch = 2..4` verify rounds (`T = 17..32`) | Eliminates unfused `q4_a16` fallback (`~41.4–41.9 us` at `T=20..32`) | **Deferred / Reverted** (`TCap=32` SMEM `178,840 B > 101,376 B`; 2-slice `16+16` regressed `+23.9%..+60.7%`) |
| **P1** | **D4** | Extend `univer_trees` root-fork verification from `batch == 1` to `batch == 2` | [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`graph_set.cpp`](../../src/models/qwen4_exp/program/graph_set.cpp) | `batch = 2` sampled rounds (`2 * (K + 2) <= 32`) | Recovers step-0 rejections during high-entropy `<think>` decode | **Deferred / Reverted** (`is_tree` disables `screened_head_compact` at `program_impl.h:5486`, spilling full `[248320, T]` logits; `C=2` regressed `-4.1%` to `-5.3%`) |
| **P2** | **D5** | Long-context & serving tuning: `KV_DTYPE=k8v4` A/B, restore `PREFILL_CHUNK=12288`, batch-aware `RoundCostModel` | [`run_server.sh`](../../tools/run_server.sh), [`program_impl.h`](../../src/models/qwen4_exp/program/program_impl.h), [`mtp_gating.h`](../../src/models/qwen4_exp/program/mtp_gating.h) | `60K–100K+` context decode, cold prefill TTFT, and skewed `batch >= 2` MoE depth policy | Halves Value KV bandwidth (`k8v4`), reduces chunk overhead (`12288`), calibrates `batch >= 2` `moe_round_cost` | **Open** (`PREFILL_CHUNK=12288` active in `run_server.sh`) |

---

## 3. Task Execution Notes & Findings

### D1 (P0): `--block-verification` Already Active (`fast_paths.cpp`)
- `FastPath::BlockVerification` is already qualified and enabled by default for `HC4-W4Dense` (`DenseKernels::Nvfp4, true, KvCacheStorage::*`) in [`src/models/qwen4_exp/program/fast_paths.cpp`](../../src/models/qwen4_exp/program/fast_paths.cpp#L533-L539) and wired into `options_.block_verification` in [`src/runtime/engine/qwen4_exp_instance.cpp`](../../src/runtime/engine/qwen4_exp_instance.cpp#L608). No code change needed.

### D2 (P0): `kScreenedHeadMaxColumns` (`32 -> 64`) & `kProjectionMmaMaxColumns` (`32 -> 64`)
- Raised `kScreenedHeadMaxColumns` from `32` to `64` in [`include/infernix/ops/screened_head.h`](../../include/infernix/ops/screened_head.h) and generalized `run_screen()` in [`src/ops/screened_head/screened_head.cu`](../../src/ops/screened_head/screened_head.cu) to iterate 16-column tiles (`for (int col = 0; col < p.columns; col += 16)`).
- Added `T = 48` logits and `T = 40` compact checks in [`tests/ops/test_screened_head.cpp`](../../tests/ops/test_screened_head.cpp).
- Testing at `T = 40, 48` uncovered that `projection_fp32` capped its BF16 tensor-core MMA route at `kProjectionMmaMaxColumns = 32` ([`src/ops/projection_fp32/projection_fp32_mma.h`](../../src/ops/projection_fp32/projection_fp32_mma.h)), forcing BF16 `exact_logits` at `T = 33..63` onto SIMT `projection_kernel` (different K reduction order from `bf16_rescore`). Raising `kProjectionMmaMaxColumns` to `64` lets `HeadSchedule32` serve `T = 33..64` cleanly with bit-exact agreement.

### D3 (P1): `hyper_connection_mix_fused` Q4 at `T = 17..32` — Deferred / Reverted
- **SMEM analysis (`ckt math smem`):** `DownMmaShared<TCap>` stages `bf16 rn[40][TCap][64]` (`92,920 B` total / `1 block/SM` at `TCap = 16`); `TCap = 32` requires **`178,840 B` (`0 blocks/SM`)**, exceeding `sm_121a`'s `101,376 B` dynamic SMEM cap.
- **2-slice microbenchmark (`infernix_hyper_connection_bench --codec q4 --layers 48`, `S=4, H=2560, rank=320`):**
  - Baseline unfused composed: `T=16` fused `37.62 us`, `T=20` **`41.44 us`**, `T=24` **`41.92 us`**, `T=32` **`41.84 us`** (`92–93 GB/s`).
  - 2-slice `launch_q4_mma` (`1..16` + `17..32`): `T=20` **`51.36 us`** (`+23.9%` slower), `T=24` **`53.26 us`** (`+27.1%` slower), `T=32` **`67.23 us`** (`+60.7%` slower).
- Reverted D3 changes; keeping unfused composed fallback for `T >= 17`. Any future D3 revisit requires a tiled-K streaming SMEM redesign so `T = 32` completes in a single cluster/collapse pass.

### D4 (P1): `univer_trees` Root-Fork at `batch = 2` — Deferred / Reverted
- Extending `univer_trees` to `batch <= 2` (`graph_set.cpp` + `program_impl.h`) fired tree verification at `C = 2` (`req#3`: 33 tree rounds / 6 side-branch accepts; `req#4`: 70 rounds / 13 side-branch accepts, ~18.5% hit rate).
- However, [`program_impl.h:5486`](../../src/models/qwen4_exp/program/program_impl.h#L5486) currently sets `compact = !is_tree && use_screened_compact(screened_head) && logit_columns > 0`, disabling `screened_head_compact` on every tree round and forcing full `[248320, T]` FP32 logits writes/reads. Combined with stochastic sampling path divergence (`temperature = 1.0`), `C = 2` decode measured `117.76 tok/s` (`116.28` 3-seed mean) vs `122.76 tok/s` baseline and `123.40 tok/s` greedy D2-only (`127.73` earlier single greedy run was higher-acceptance seed noise). Reverted D4 until sparse/compact tree logits verification is supported.

---

## 4. Measured GB10 E2E Concurrent Decode Results (`HC4-W4Dense`)

Configuration: `./tools/run_server.sh hc4` (`--kv-dtype int8`, `--max-concurrency 8`, `--spec mtp --draft-tokens auto --ngram-draft-tokens 31 --ngram-min-match 8`, `--ngram-cache-mib 4096`), `max_tokens = 512`, distinct prompts per lane, zero lazy CUDA graph capture warnings.

| Workload & Build | `C = 1` sum / lane (`tok/s`) | `C = 2` sum / lane (`tok/s`) | `C = 3` sum / lane (`tok/s`) | `C = 4` sum / lane (`tok/s`) | Notes |
|---|---|---|---|---|---|
| **Baseline `1d8cce2cf` (`PID 3017392`, `temp=1.0`, 1 seed)** | `77.73` / `77.73` (`e2e 76.14`) | `122.76` / `61.38` (`e2e 113.10`) | `138.91` / `46.30` (`e2e 128.58`) | `152.88` / `38.22` (`e2e 140.27`) | Steady 5s `batch 4.00`: `146.4 tok/s` |
| **D2 + D4 (`1d8cce2c-dirty`, `temp=1.0`, same seed)** | `76.98` / `76.98` (`e2e 75.43`) | `117.76` / `58.88` (`e2e 110.86`) | `134.29` / `44.76` (`e2e 126.70`) | `148.49` / `37.12` (`e2e 139.11`) | D4 reverted after root-causing `compact = !is_tree` |
| **D2 + D4 (`temp=1.0`, 3 seeds mean)** | `76.98` / `76.98` | `116.28` / `58.14` | `132.12` / `44.04` | `148.39` / `37.10` | Per-run std variance on sampled `<think>` tokens |
| **Final D2-only (`1d8cce2c-dirty`, `temp=1.0`, same seed)** | `77.71` / `77.71` (`e2e 76.14`) | `118.47` / `59.24` (`e2e 110.82`) | `133.00` / `44.33` (`e2e 123.41`) | `150.98` / `37.75` (`e2e 137.55`) | `5s` `batch 4.00`: `143.2 tok/s` (`0.30 J/tok`) |
| **Final D2-only (`GREEDY_STD`, `temp=0.0`)** | `85.88` / `85.88` (`e2e 83.95`) | `123.40` / `61.70` (`e2e 113.30`) | `142.39` / `47.46` (`e2e 128.40`) | `162.13` / `40.53` (`e2e 144.85`) | `1.89x` C=1 scaling at `C=4`; `5s` `batch 4.00`: `152.2 tok/s` |
| **Final D2-only (`GREEDY_NGRAM`, `temp=0.0`, wide n-gram)** | `94.21` / `94.21` (`e2e 88.83`) | **`156.26`** / `78.13` (`e2e 126.64`) | **`182.32`** / `60.77` (`e2e 133.66`) | **`208.65`** / `52.16` (`e2e 157.23`) | **`2.21x` C=1 scaling at `C=4`**; wide `batch * W <= 64` stays screened (`ngram` up to `193/216`, `5s` window `181.0 tok/s`) |

## 5. Verification Gates Passed (Post-Final Edit)
- `infernix_cuda_stack_check` (zero new `STACK > 0` / `LOCAL > 0` symbols)
- `infernix_qwen4_exp_speculative_test`
- `infernix_qwen4_exp_fast_paths_test`
- `infernix_screened_head_test` (including `BF16 head` & `Q8 head` `wide T=48` and `compact T=40`)
- `infernix_projection_fp32_test`
- `infernix_hyper_connection_test` (verified with D3 reverted prior to D4 revert)
