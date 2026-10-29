# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T1: `MXFP8` Linear Schedules and `f16-accumulate-prefill`

Scope: `docs/tasks/20261029-gb10-post-rebase-perf-tasks.md` **Task T1** (P0) — the `mxfp8`
(`/opt/etc/Infernix/src/ops/linear/mxfp8/`) dispatch tables that were still fitted on RTX 5090, and the
`FastPath::F16AccumulatePrefill` qualification that rides on them. Companion record:
`docs/gb10_roadmap.md` §2.1 and §8.12.

---

## 1. Commit range and accounting

| Item | Value |
|---|---|
| Base (`origin/main`) | `d8e52a85a` |
| Branch HEAD before this work | `e750ffb2d` (`dev-rebased-20261028`, 3 commits ahead of base) |
| This change | 11 files |
| Insertions / deletions | +432 / −78, of which +269 / −78 over the 10 code, bench and roadmap files and the rest this changelog |

Files enumerated below exclude `cuda-kernel-toolkit/` (the integrated toolkit is intentionally not
enumerated; no toolkit file was modified — only the in-tree bench harness was).

---

## 2. Why these changes are required by GB10 hardware

1. **The 5090-fitted bands were shipped on the only supported platform.** Commit `68befae22` made
   GB10/Linux the sole supported platform, yet all five `mxfp8/shapes/*.cu` headers still declared
   `bench/ops/mxfp8_flash_next_sweep.cu`'s **RTX 5090, 2026-10-20** selection. That ladder was chosen on
   170 SMs / 96 MiB L2 / 1792 GB/s; GB10 has **48 SMs, 24 MiB L2 and ~242 GB/s** measured sustained read,
   so a tile sized to fill 170 SMs re-reads weights and misses the 24 MiB reuse window here.
2. **1:1 FP32-accumulation parity removes the premise of `f16-accumulate-prefill`.** On `sm_120a` the
   FP32-accumulating HMMA runs at half the FP16-accumulating rate, so the route's 2× pipe rate paid for
   its own activation-encode pass. On `sm_121a` the measured pipes are equal (roadmap §1.1: 125.31 vs
   125.44 TFLOP/s; `.ckt_cache/env.json`: `bf16_mma` 124.68 TFLOP/s), which is the guardrail already
   recorded in §2.6-5 ("never use `mma_f16_f16acc` on GB10"). The qualification was inherited from a
   5090 ABBA where it was only +5.3 % at `pp32768` and neutral at `pp4096`.
3. **Decode is memory-bound, so a schedule choice at `T <= 32` is worth almost nothing on these parents.**
   `[13952,2560]` is 36.19 MB and `[2560,6144]` is 15.94 MB; at ~242 GB/s the weight-read floors are
   152.46 µs and 67.14 µs, and every one of the 41 MXFP8 schedules lands within 1–2 % of them, so the
   sweep's per-width argmin in that band is re-ranking measurement noise. Band edges were therefore moved
   only where the median win clears that floor.
4. **Cold-L2 measurement must evict completely but not waste wall-clock.** A 24 MiB L2 needs a flush
   larger than itself; 32–48 MiB (1.33–2× L2) reaches the median plateau the 5090's 256 MiB buffer also
   reaches, at a fraction of the sweep time (§2.6-7 methodology, `common_pitfall` note "an under-complete
   L2 flush reads FASTER").

---

## 3. Subsystem breakdown

| Action | File | Change and mechanism |
|---|---|---|
| **M** | `src/ops/linear/mxfp8/shapes/n16384_k2560.cu` | `launch_a16`/`select` bands re-derived on GB10; new local `RingT32R16` alias; `T <= 16` **kept** on the 5090's 16×64 tile; 17–3072 columns promoted to 32-row/128-row TMA tiles; `> 8192` to the 4-warp cp.async 64×256 producer. 1.041× geomean, 1.89× at `T=8192`. |
| **M** | `src/ops/linear/mxfp8/shapes/n13952_k2560.cu` | QSA 6-output projection: decode bands (`<= 64`) kept, 65–192 columns onto `Mxfp8Tma<Fp8A16RingT32R64S3>` / `…T64R128>`, `<= 4096` on the 128×128 TMA tile. 1.008× geomean, no median or p95 regression. |
| **M** | `src/ops/linear/mxfp8/shapes/n2560_k6144.cu` | `T <= 8` moved off the 5090's 16×16 tile (82.7–88.0 µs → 75.6–77.6 µs); 65–128 onto `Fp8A16RingT64R128`, 129–512 TMA 64×64, `> 2048` the 4-warp 64×256 producer. 1.096× geomean, 1.90× at `T=8192`. |
| **M** | `src/ops/linear/mxfp8/shapes/n1280_k2560.cu` | `T <= 48` kept; 49–128 columns to the 32×64 TMA tile, 129–256 to 64×128, 257–768 alternating 64×64/128×128. `T=384` deliberately takes `Mxfp8Tma<Mxfp8TmaT128R128>` (48.83 µs median, 48.96 µs p95) over the median winner `Mxfp8Tma<Mxfp8TmaT64R64>` (45.57 µs median but 120.35 µs p95). 1.100× geomean, no regression. |
| **M** | `src/ops/linear/mxfp8/shapes/n2560_k640.cu` | Whole decode/verify region moved (6–10 % at `T=1..8`, 24–37 % at 48–128); both producers now win in alternation, which the header states instead of the 5090's "cp.async at every width". 1.153× geomean, 1.58× at `T=128`. |
| **M** | `src/ops/linear/mxfp8/mxfp8_shapes.h` | `kMxfp8F16MinTokens` and `kMxfp8PairMinTokens*` comments replaced: they said "Not yet timed … `linear_bench --qtype mxfp8 --routes f16`", a mode that does not exist (linear_bench accepts `Q4|Q5|Q6|Q8|BF16|NVFP4|FP8` and `--routes exact|fast`). Now they record the GB10 measurement, the wrong-admission conclusion, and the owning follow-up. |
| **M** | `src/models/qwen4_exp/program/fast_paths.cpp` | **Removed** `Qualification{FastPath::F16AccumulatePrefill, {DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64}}` with the measured reason. The route, its op-level registration and `launch_f16` bands are untouched — only the default-on product decision. `ScreenedLmHead`'s qualification and its 5090 note stay. |
| **M** | `tests/models/qwen4_exp/test_fast_paths.cpp` | The 42-combination loop that asserted "qualified on FP4-Local with INT8 KV only" now asserts the path registers no configuration, with the measurement inline. The explicit-selection assertions above it (`prefill_routes == F16Accumulate`, request-invariance coupling) are unchanged and still prove the route itself works when a caller asks for it. |
| **M** | `bench/ops/mxfp8_flash_next_sweep.cu` | (a) `--flush-mib`, defaulting to `bench::default_cold_flush_mib()` (2× `prop.l2CacheSize`), replacing the hardcoded 5090 `256 MiB` buffer; (b) new `--routes-compare` / `--route-t-list` mode timing all three production routes through public `ops::linear` with a real `WorkspaceArena`, printing each route's admission and marking a below-admission sample as "runs exact" so a noise tie is not read as a measurement. |
| **M** | `docs/gb10_roadmap.md` | §2.1 counts replaced (109 tables, 95 GB10 / 14 unmarked / **0** RTX 5090 after this change) and the P2.4/P1.5 "103/103 tagged GB10" claim corrected; P1.A/P1.C/P1.D given evidence-backed dispositions; new §8.12 records this campaign; the `roofline.h` 5090-constant defect recorded. |
| **A** | `docs/gb10/20261029-changelog.md` | this file |

---

## 4. Measured results (GB10, `nvcc 13.0.88`, cold-L2 CUDA graphs)

Exact A16 route, 21 repetitions per point, 48 MiB flush, against the RTX 5090 bands (`infernix_mxfp8_flash_next_sweep`,
`--repeat 21 --dram-gbps 241.6`). All 41 schedules were **bit-identical** to the reference at `T=1` and
`T=300` (with the FP64 screen) on all five shapes, so every change is dispatch-only and needs no
numerical re-qualification.

| Shape | Geomean | >2 % faster | Median regressions | p95 regressions |
|---|---:|---:|---:|---|
| `[16384,2560]` | 1.041× | 10/28 | none | `T=384` 1.17× (median 5.6 % faster) |
| `[13952,2560]` | 1.008× | 5/28 | none | none |
| `[2560,6144]` | 1.096× | 18/28 | none | `T=2` 1.27×, `T=3` 1.42×, `T=512` 1.25×, `T=1536` 1.13× |
| `[1280,2560]` | 1.100× | 10/28 | none | none |
| `[2560,640]` | 1.153× | 22/23 | none | `T=8192` 1.08× (median 16 % faster) |

Route comparison, f16-accumulate and block32-pair against the **retuned** exact route:

| Shape | f16-accumulate (admitted from 256) | block32-pair (per-shape admission) |
|---|---|---|
| `[16384,2560]` | +3.8…+7.9 % at 384–2048, −3.1 % at 4096, **+7.6 %** at 8192 → never a real win | +1.5…+27.0 % across 33–4096, **+56.4 %** at 8192 |
| `[13952,2560]` | −12.1 % at 4096, **−40.7 %** at 8192; ≤ +3.5 % elsewhere | +8.2…+10.2 % at 97–1024, −4.9 % at 4096, −0.1 % at 8192 |
| `[2560,6144]` | **+11.3…+34.7 %** at 256–4096 (a loss), −25.5 % at 8192 | +11.3…+27.8 % at 513–4096, **+87.1 %** at 8192 |
| `[1280,2560]` | **+20.9…+36.2 %** at 512–1024, +32.3 % at 4096, −5.3 % at 8192 | +31.1…+41.7 % at 2048–4096, +82.4 % at 8192 |
| `[2560,640]` | +10.6…+22.9 % at every admitted width | +11.0…+18.3 % at 513–8192 |

That table is why the qualification went: at the single `kMxfp8F16MinTokens = 256` admission the path makes
four of five problems slower by 5–36 % to buy one problem a win it only has from 4096 columns.

---

## 5. Verification

- `infernix_ops_linear_test linear_mxfp8_a16` → `OK MXFP8 A16 Linear`; `linear_f16_accumulate` →
  `OK f16-accumulate-prefill Linear`; `mxfp8_moe_overlap` → overlapped shared+routed experts equal the
  serialized run; `linear_fp8_a16` → `OK FP8 A16 Linear`.
- `infernix_qwen4_exp_fast_paths_test` → `PASS Qwen3.8-Flash-Next fast-path selection` and
  `infernix_qwen3_5_fast_paths_test` → `PASS fast-path selection` after the qualification change.
- `ctest -R "infernix_cuda_stack_check|gdn_input_proj|attn_input_proj|gdn_gating_proj"` → **7/7 pass**, so
  the one-way spill ratchet is unchanged (0 new `STACK>0`/`LOCAL>0` symbols).
- `git diff --check` clean; each edited `.cu` compiles standalone.
- **Not run, and why:** no end-to-end A/B. No artifact converted on this host binds the MXFP8 route
  (`/opt/etc/models` holds only `Qwen4ExpForConditionalGeneration` artifacts with `nvfp4`/`q8_g32_fp16`/ggml
  trunks; the `nvidia-Qwen3.8-Flash-Next-FP4-Local` source checkpoint has no `.infernix`), so
  `run_kv_matrix_bench.sh` cannot show a delta for this change, and `run_kv_matrix_bench.sh` was therefore
  not run as evidence for it. The full CTest suite was likewise not re-run.

---

## 6. Rebase conflict-resolution policy for the touched subsystems

| Subsystem | Policy on a future `origin/main` rebase |
|---|---|
| `src/ops/linear/mxfp8/shapes/*` | **Preserve the GB10 bands** and their header provenance. A conflict resolved toward upstream re-imports 5090 selection on a 48-SM part. Upstream math/config fixes (geometry, admissions of *other* routes, new producers) are accepted; the `tokens <=` ladder and the schedule type per band stay as measured here until a GB10 re-sweep says otherwise. Re-run `infernix_mxfp8_flash_next_sweep --repeat 21` after any change to the candidate set and re-check bit-identity. |
| `src/ops/linear/mxfp8/mxfp8_shapes.h` | Accept upstream admissions only with a fresh GB10 measurement; the comments carry the current evidence, so an upstream edit that deletes "not yet measured on GB10" language is a red flag, not a cleanup. |
| `src/models/qwen4_exp/program/fast_paths.cpp` | **Keep `F16AccumulatePrefill` unqualified** on GB10 unless upstream re-measures it here; the 1:1 FP32-parity argument is hardware, not taste. Do preserve upstream's 5090-era measurement notes for other configurations (they are accurate history) and preserve `ScreenedLmHead`. Route/`kQualified` tables merged by set union, then this qualification removed again. |
| `bench/ops/mxfp8_flash_next_sweep.cu` | Preserve `--flush-mib`'s `default_cold_flush_mib()` and `--routes-compare`; reject a regression to a literal 256 MiB flush. |
| `docs/gb10_roadmap.md`, `docs/gb10/*` | Append, never overwrite; the roadmap's counts are re-derived from `ckt shapes list`, which tags origin by keyword only — so a table's `GB10` tag is a pointer to check, not evidence. |

---

## 7. Open items this session leaves

1. **Per-problem `f16_min_tokens`.** `[13952,2560]` measured −12.1 % at 4096 and −40.7 % at 8192 columns:
   giving that problem its own admission (and the other four `nullptr`) recovers the win without the
   regressions. Needs the model-quality gate (`docs/maintainer/model-quality.md` §6) plus test updates in
   `test_f16_accumulate.cpp`'s `kProblems`/workspace-contract loop, which registers the route for all five.
2. **`block32` pair admissions are 5090 (recipe C) values and measured slower than GB10's exact route at
   nearly every width** — feed into T2, which retunes the A8 tiles for 48 SMs.
3. **`bench/ops/roofline.h`** still publishes RTX 5090 peaks (`kDramSpecGBs = 1792.0`,
   `kDramReadGBs = 1674.5`, `kBf16Fp32AccumulateTFLOPs = 209.5`, `kMxfp8Fp32AccumulateTFLOPs = 838.0`) to
   ten Op benchmarks, so reported roofline percentages on this host are wrong unless `--dram-gbps` is
   passed. Reporting defect only; no dispatch reads it.
4. **14 `/shapes/` tables still carry no tuning-provenance marker** (7 `bf16`, 3 `nvfp4`, 2 `q6`, 1 `q4`,
   1 `q8`); 13 have no measurement note at all, most of them Qwen3.5-family `K = 5120/17408/3072` or
   vision shapes. `q8/n1280_k2560.cu` is GB10-swept (2026-10-24) but lacks the keyword.
5. **T1's stated acceptance is met only in part, and the gap is shape-specific.** "`>= 15 %` latency
   reduction at `T >= 512`" holds at: `[16384,2560]` 8192 columns (**+89.1 %**) only; `[2560,6144]` 4096
   (+28.0 %) and 8192 (+90.1 %); `[1280,2560]` 512 (+16.1 %); `[2560,640]` 4096 (+15.4 %) and 8192
   (+19.4 %). It does **not** hold anywhere on `[13952,2560]` (best +4.9 % at 1024) nor at the 512–2048
   band of the other four (`[16384,2560]` +3.9…+6.6 %, `[2560,6144]` 0.0 % at 1024/2048, `[1280,2560]`
   0.0 % from 2048). Cause: those mid-width bands were already GB10-correct in the 5090 table — the real
   5090-era damage sat at 49–128 columns and at the 4096/8192 tail — so the honest result is the per-shape
   geomeans in §4 plus the four large-tail wins, not a uniform 15 %. The task text's "1.89× at `T = 4096`"
   for `[16384,2560]` is at `T = 8192`, and its commands `infernix_linear_bench --qtype mxfp8` /
   `--routes f16|block32` do not exist in the benchmark (`linear_bench` accepts
   `Q4|Q5|Q6|Q8|BF16|NVFP4|FP8` and `--routes exact|fast`), so the sweep's new `--routes-compare` mode
   substituted for them.
6. **`P1.A` / `P1.D` remain blocked** on a Qwen3.5-family `.infernix` on this host (see roadmap §4 rows for
   the corrected premises), and **`P1.C`'s `kPairWidthCost = 0.15`** still carries no inline provenance.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T2: `--a8-activations` retune (`GgmlA8` warp cap, 128-column wide admission, GB10 route qualification)

Scope: `docs/tasks/20261029-gb10-post-rebase-perf-tasks.md` **Task T2** (P0) — the `--a8-activations`
routes over GGUF weights: the narrow `GgmlA8` mma route's CTA-shape rule, the wide `GgmlA8Wide` /
`GgmlBf16Wide` admissions, the GGUF expert banks' wide A8 route, and the `a8_activation_routes()`
qualification lines. Companion record: `docs/gb10_roadmap.md` §2.6 guardrails.

---

## 1. Commit range and accounting

| Item | Value |
|---|---|
| Base (`origin/main`) | `d8e52a85a` |
| Branch HEAD before this work | `f8e10de31` (`dev-rebased-20261028`, 4 commits ahead of base; T1 merged) |
| This change | 15 files + this changelog appended to the T1 section's file |
| Insertions / deletions | +114 / −66 over the 15 code, test, contract-header and doc files |

Files enumerated below exclude `cuda-kernel-toolkit/` (intentionally not enumerated; only the in-tree
`infernix_ggml_linear_bench` / `infernix_offloaded_moe_wide_bench` harnesses and `ckt ab` were used,
no toolkit file was modified).

---

## 2. Why these changes are required by GB10 hardware

1. **A whole-machine warp cap has to scale with the SM count.** `ggml_linear_a8.cuh`'s `kWarpBudget`
   was the 5090's `1024` = 170 SMs × 6 warps. On 48 SMs that cap lets a narrow call spend 21.3 warps
   per SM, so `choose_shape` keeps a wide K-split it cannot fill and pays the second wave a narrow call
   costs 25–30 % of. The replacement is `warp_budget(sms, tiles) = sms × 6`, plus the exception that a
   call of at most two tiles per SM — which can fill the machine only by splitting K — takes no cap
   beyond the resident-CTA wave-fit check. The interleaved A/B of the two caps (nine rounds, 15 cold-L2
   repetitions per point) found the cap inert at seven of the eight IQ3_S dense shapes (all ten
   non-shexp points WITHIN_NOISE, |delta| ≤ 0.06 %); the one cap-sensitive shape is the 1,280-row
   (80-tile) shared-expert `gate_up`, where the machine-scaled cap rejects the split-8 shape on the
   units clause and falls to a 40-CTA split that leaves eight SMs idle — the 5090 cap measured 10.9 %
   faster at 16 columns, 19.1 % at 31 and 4.5 % at 1. A budget sweep's apparent up-to-20.7 % deltas
   from this constant were proven phantom (the cap cannot change candidate selection at those
   K/tile-counts; the sweep argmin re-ranks cold-L2 noise) — the interleaved A/B is the arbiter.
2. **The wide routes' 128×128 CTA tile sets the admission, not the exact route's 32-column start.**
   A 32- or 64-column call fills a quarter to a half of the wide tile and lost to the exact route's
   wide A16 tiles on every IQ3_S dense shape (a8-wide `gdn_qkvz [16384,2560]`: 749.3 µs against exact's
   343.9 at T=32, 751.7 against 454.6 at T=64; BF16-wide: 774.0 / 771.9). From 128 columns the tiles
   win everywhere (a8-wide: 747.4 against 1037.2 at T=128, 5,677.9 against 16,582.5 at T=4096). Both
   `kA8WideMinColumns` and the new `kBf16WideMinColumns` therefore sit at 128, and `ggml_linear`
   admits each wide route only at its own tile width.
3. **The step-3 2D L2 CTA swizzle was measured unnecessary and rejected.** The dense wide routes run
   128-column tiles fastest in the grid, so one wave's footprint is ~7–8 MB against the 24 MiB L2 —
   already resident without rasterization help (T=64 output hashes prove the route's grid is the
   differentiator, not locality). The residual ~4× gap to the IMMA roofline sits in the per-slice
   rescale epilogue (the FP32 FMA pipe costs about the MMA), which rasterization cannot address; the
   expert `compute_decode` path streams each expert's weights once and keeps its x-columns L2-resident.
4. **IMMA.16832.S8 doubles the BF16 pipe, and the routes now land where that pays.** The dense a8-wide
   route reaches 79.1 TOPS (`iq3s_6144` T=4096) against HMMA's 125.4 TF ceiling for the BF16 tiles it
   precedes; end to end on the IQ3_S artifact (dflash2 + lm-head draft, BF16 KV, experts on GPU, four
   interleaved passes per arm, distributions non-overlapping at every metric):
   `pp4096` 1,270.2 → **1,847.9 tok/s (+45.5 %)**, `pp4096+tg128` decode-out 127.4 → 130.6 (+2.5 %),
   `tg128` decode-out 44.73 → **49.38 (+10.4 %)**. The decode-width routes were qualified on 2026-10-27
   and the flag toggles all A8 routes together, so `tg128`/`mix_decode` include their contribution;
   `pp4096` involves only the prefill routes retuned here.

---

## 3. Subsystem breakdown

| Action | File | Change and mechanism |
|---|---|---|
| **M** | `src/ops/ggml/ggml_linear_a8.h` | `kA8WideMinColumns` 32 → 128 with the T=32/64/128/4096 measurements that set it. |
| **M** | `src/ops/ggml/ggml_linear.cu` | New `kBf16WideMinColumns = 128` (same evidence, BF16 numbers inline); the dispatch gates `GgmlA8Wide` and `GgmlBf16Wide` on their own admissions (`columns >= kA8WideMinColumns / kBf16WideMinColumns`), so a below-admission call keeps the exact route's bits. |
| **M** | `src/ops/ggml/ggml_linear_a8.cuh` | `kWarpBudget = 1024` → `kBudgetWarps = 6` per SM with `warp_budget(sms, tiles)` (`tiles ≤ 2 × sms` → `kMaxSplit × tiles`, else `sms × 6`); `choose_shape` consults it; the header comment records the A/B verdict per shape. The prior session's temporary `INFERNIX_TMP_A8_WARP_BUDGET` env override and its `<cstdlib>` include are removed. `kShapes` itself unchanged — no measurement motivated a ladder change. |
| **M** | `src/ops/ggml/ggml_linear_a8_wide.cu`, `src/ops/ggml/ggml_a8_tile.cuh` | Contract text 32 → 128 admission; no code change. |
| **M** | `include/infernix/ops/linear.h`, `include/infernix/ops/ggml_linear.h`, `include/infernix/types.h` | The public route-enum docs (`GgmlBf16Wide`, `GgmlA8Wide`), the `ggml_linear` workspace/arithmetic contract, and the `GgmlBf16Prefill` / a8-activations option comment all state the 128-column admission; the A8Wide bits-are-width-invariant sentence now reads "at every admitted width". The narrow route's 1–31 boundary is unchanged. |
| **M** | `src/models/qwen4_exp/program/fast_paths.cpp`, `fast_paths.h` | `a8_activation_routes()`'s `ggml_dense_prefill` and `experts_prefill` lines replace "not yet measured end to end" with the measured GB10 evidence (E2E + op-level ranges, incl. the 1,280-row-at-128-columns exception); `describe()` says 128; the `mxfp8_dense` line stays unmeasured — FP4-Local needs an artifact this host lacks (T1 §5/§7). |
| **M** | `tests/ops/test_ggml_linear_a8_wide.cu` | Boundary problem `{32, 33, 300}` → `{128, 129, 300}`; every wide problem's call width raised to ≥ 128; `check_flags` now proves route precedence at 128 columns and that calls of 31 / 64 / 127 columns keep the exact route's bits (and the narrow `GgmlA8`'s where set). |
| **M** | `tests/ops/test_ggml_linear.cpp` | The BF16-wide below-admission boundary moved from `< 32` to `< 128` ("a call below the route's admission must keep the exact route's bits"). |
| **M** | `docs/qwen3_8-flash-next.md`, `docs/maintainer/model-quality.md`, `docs/maintainer/qwen3_8-flash-next-design.md` | User guide, fast-path qualification entry, and design §19.3.31 all state the 128-column admission with the dated GB10-retune rationale. |

---

## 4. Measured results (GB10, cold-L2 `infernix_ggml_linear_bench` / `infernix_offloaded_moe_wide_bench`, E2E `infernix_bench`)

Dense wide-route matrix — eight IQ3_S-import shapes (Q6_K / IQ4_XS / Q4_K / Q5_K / Q6_K+Q4_K mixes) at
T ∈ {64, 128, 512, 2048, 4096}, `exact` / `bf16` / `a8`, 256 MiB flush, output-hash checked. The a8-wide
route's time as a fraction of the exact route: **0.610–0.852 at T=128** (exception: `shexp_gate_up`
[N=1280] at **1.131**, still ahead of the BF16 tiles' 1.575), **0.421–0.655 at T=512**, **0.326–0.433 at
T=2048**, **0.135–0.428 at T=4096** (best `attn_output` 0.135, 72.7 TOPS). a8 beats the BF16 tiles at
every shape and width; the BF16 tiles themselves lose to exact at `attn_qkv_q5k` from T=128 (up to
1.094) and `attn_gate_q4k` T=4096 (1.026) — recorded as an open item for `ggml-bf16-prefill` alone.

The step-1 format sweep the built-in shapes lacked (IQ2_S, `--shape NAME=iq2_s:ROWS:K`): a8 vs exact
0.634–0.726 at T=128 (exception `gate_up`-width N=1280: 1.055, bf16 1.513), 0.412–0.509 at T=512,
0.337–0.430 at T=4096; at the K=512 probe the BF16 tiles are 7 % ahead of the 8-bit tiles (0.402 vs
0.430) — both far ahead of exact; no shipped dense projection has K ≤ 512.

Offloaded GGUF expert banks (IQ3_S + IQ4_NL, `--ggml-wide`), the a8 wide route's time as a fraction of
the BF16 route / the exact route: T=64 0.899 / 0.881, T=256 0.854 / 0.851, T=1024 0.795 / 0.803,
T=4096 **0.593 / 0.582** (32,879 µs BF16 → 19,489 µs a8, 20.7 wide-TFLOP/s). Faster at every prefill
width tested; the decode widths (≤ 128 columns) were qualified 2026-10-27 and are covered end to end by
`tg128` / `mix_decode` above.

---

## 5. Verification

- `ctest --test-dir build-native -R "ggml|offloaded_moe|cuda_stack_check" --output-on-failure`
  (serially — `--parallel` causes numerical failures on this host): **29/29 passed**, including
  `infernix_ggml_linear_test`, `infernix_ggml_linear_a8_test`, `infernix_ggml_linear_a8_wide_test`
  (the rewritten 128-admission contract), all 18 `offloaded_moe` oracle tests, and the one-way CUDA
  stack-spill ratchet (`infernix_cuda_stack_check`, 0 new `STACK>0`/`LOCAL>0` symbols).
- `infernix_qwen3_5_fast_paths_test` and `infernix_qwen4_exp_fast_paths_test` pass after the
  qualification-line rewrites (route flags unchanged; only the recorded evidence moved).
- The 29-test suite was run once on the complete functional tree and re-run after the final comment-only
  contract sweep; the E2E A/B ran on the final functional binaries.
- **Not run, and why:** the full CTest suite and `run_kv_matrix_bench.sh` were not run — the change
  touches one route family and its contract tests, and the KV-matrix gate is T8's. The FP4-Local
  (`mxfp8_dense`) half of `--a8-activations` remains unmeasured for lack of an artifact on this host.

---

## 6. Rebase conflict-resolution policy for the touched subsystems

| Subsystem | Policy on a future `origin/main` rebase |
|---|---|
| `src/ops/ggml/ggml_linear_a8.cuh` | **Preserve `warp_budget` (SM-scaled cap + small-tile exception)** and the A/B evidence comment; a conflict resolved toward upstream re-imports the 170-SM constant. Upstream `kShapes`/ladder changes are accepted only with a fresh GB10 interleaved A/B. |
| `src/ops/ggml/ggml_linear*.c*`, `ggml_a8_tile.cuh` | **Preserve both 128-column admissions** and the per-route gates; never restore 32 without a new GB10 measurement. Accept upstream math/config fixes with the gates preserved. |
| `include/infernix/ops/linear.h`, `ggml_linear.h`, `types.h` | These are the public contract of the routes; they must agree with the dispatch — merge them together with the code, never one side alone. |
| `tests/ops/test_ggml_linear*.{cpp,cu}` | Keep the 128-boundary and below-admission bit-preservation checks; an upstream test that re-introduces a 32-column wide-route expectation contradicts the GB10 gate. |
| `src/models/qwen4_exp/program/fast_paths.cpp` | Keep the measured route lines; withdraw any route a later GB10 A/B finds not faster, on its line. Do not restore "not yet measured" text. |

---

## 7. Open items this session leaves

1. **`ggml-bf16-prefill` alone loses to exact at wide-K `Q5_K`/`Q4_K` shapes** (up to 1.094 at
   `attn_qkv_q5k`, 1.026 at `attn_gate_q4k` T=4096): where only that fast path is qualified (no
   `--a8-activations`), it pays those losses. Its 2026-10-19/20 qualification predates this matrix and
   deserves its own per-shape re-qualification decision.
2. **The 1,280-row projection at calls of exactly 128 columns** runs 1.13× the exact route's time on
   the 8-bit tiles (1.58× on BF16's). Dispatch has no per-N×T carve-out; exploiting the exact win
   would need one. Kept because it still beats the route it replaces.
3. **`mxfp8_dense` (`fp8-a8-pair-block32`) remains unmeasured end to end** — needs an FP4-Local
   `.infernix` on this host (T1 §5), then this changelog's line for it.
4. **IQ2_S at K=512 shows the 8-bit tiles ~7 % behind the BF16 tiles at T=4096.** No shipped projection
   has K ≤ 512, so no action; recorded before a future artifact changes that.
5. **Task-doc baselines are stale** (`docs/tasks/20261029-...` §1.2 quotes `pp4096` 1,045; this tree
   measures 1,266–1,277 without A8). Later T-tasks must re-measure their own A/B baselines in-session.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T3: Qwen3.5-122B-A10B per-row FP8 (`fp8_e4m3fn_row_f32`) GB10 shape retune

Scope: `docs/tasks/20261029-gb10-post-rebase-perf-tasks.md` **Task T3** (P0) — the three 122B A8
projections (`attn_input_proj n17408_k3072`, `gdn_input_proj n20480_k3072`, `linear_add n3072_k8192`)
and the two 122B A16 `Linear` shapes (vocab head `n248320_k3072_f32`, shared-expert down
`n3072_k1024_f32`) that still carried RTX 5090 token bands. Companion record: `docs/gb10_roadmap.md`.

## 1. Commit range and accounting

- Base: `origin/main` (unchanged from Session 1 accounting); HEAD at work start: `33fed936a` (T2).
- Files changed this session (working tree, 13 files, +886/−97): see §3. `cuda-kernel-toolkit/` intentionally excluded.

## 2. Why these changes are required by GB10 hardware

- The three 122B A8 projections (`attn n17408_k3072`, `gdn n20480_k3072`, `linear_add n3072_k8192`)
  inherited token bands tuned for the RTX 5090 (170 SMs, 24 KiB… 96 MiB-class L2): split-K `Bulk`
  with `WaveCtas = 170` and wide row tiles admitted as low as T = 289/257/769. On 48 SMs with
  24 MiB L2 the 170-wave grids are 3.5 waves and the raster order thrashes the activation slab.
- Measured GB10 invariants behind the new tables: Grouped-8 raster keeps the activation slab
  L2-resident across row tiles (−22.7% on `linear_add` at T = 4096 where K = 8192 makes the slab
  32 MiB > L2 with plain raster). The same raster cliff sits at T ≈ 4600–5120 for the K = 3072
  projections, where the 3,072 B/token slab crosses the 24 MiB L2: above it the plain-raster
  schedules lose 35–44.5% to their Grouped-8 twins (interleaved ab: attn 96x256 −44.5% at T = 8192
  SNR 28.2; gdn Bulk −35.1% at 6144 SNR 18.8, −43.9% at 8192 SNR 23.0), while at ≤ 4096 the plain
  tile still wins by 5–8%; both ladders therefore keep plain tiles ≤ 4096 and switch the same
  tiles to the Grouped-8 raster above it. WaveCtas 48 vs 170 is ≤ 0.2% on this host (split-K parts
  cap at MaxParts = 4 regardless, so the resident-grid retune prescription is a non-lever); plain
  TMA tiles dominate split-K for the K = 3072 input projections at every mid width, removing the
  288 → 289 Bulk admission cliff (+4.5 → 0%); `linear_add` 705–896 moves to Mid (ab +8.3% at 769),
  replacing the inherited 768 → 769 Wide-to-Bulk +27%-per-token cliff with a padding-continuous
  704 boundary.
- The vocab head `[248320, 3072]` (763 MB weights) is DRAM-floor-bound to T = 128 (one weight
  pass ≥ 3.2 ms at the measured 246 GB/s sustained read); per-token jumps at 129/257/385 are the
  second weight pass, not scheduling loss. The K = 5120 sibling's measured packing intervals
  (161–192, 257–288 as whole Tail96 GEMMs) are ported to the K = 3072 head.

## 3. Subsystem breakdown

| File | Action | Change |
|---|---|---|
| `src/ops/attn_input_proj/fp8/fp8_attn_input_plan.h` | M | `Fp8AttnInputN17408ScheduleId` candidate API (12 ids + name/legal/execute/partial-capacity) |
| `src/ops/attn_input_proj/fp8/fp8_attn_input_n17408_k3072.cu` | M | GB10 retuned ladder: ≤32 t32r32, ≤96 t32r128, ≤128 tma64x128, ≤192 tma64x256, ≤320 tma64x128, ≤448 tma96x256, ≤1792 tma128x128.G8, ≤4096 tma96x256, >4096 tma96x256.G8; production partial capacity → 0 (split-K unreachable) |
| `src/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h` | M | `Fp8GdnInputN20480ScheduleId` candidate API (12 ids) |
| `src/ops/gdn_input_proj/fp8/fp8_gdn_input_n20480_k3072.cu` | M | Retuned ladder: ≤192 unchanged, ≤320 tma64x128, ≤384 Bulk, ≤448 tma64x128, ≤512 tma192x128.G8, ≤576 tma192x128, ≤768 Bulk, ≤1024 tma128x128.G8, ≤1792 tma192x128.G8, ≤2048 tma192x128, ≤4096 Bulk, >4096 tma192x128.G8; partial threshold 256 → 320 |
| `src/ops/linear_add/fp8/fp8_linear_add_plan.h` | M | `Fp8LinearAddN3072ScheduleId` candidate API (10 ids) |
| `src/ops/linear_add/fp8/fp8_linear_add_n3072_k8192.cu` | M | Retuned ladder adds 385–512 → Mid (was Bulk), 705–896 → Mid (cliff fix; Wide cap 768 → 704), >2560 → tma192x128.G8 full tile (was split-K Bulk) |
| `src/ops/linear/fp8/shapes/n248320_k3072_f32.cu` | M | Ported K = 5120 head packing intervals 161–192 / 257–288 → whole Tail96 |
| `src/ops/linear/fp8/shapes/n3072_k1024_f32.cu` | M | Ring boundary 256 → 320 (ring32 covers the 257+ second-band padding jump) |
| `bench/ops/attn_input_proj_bench.cu`, `gdn_input_proj_bench.cu` | M | `--fp8-problem 122b`, `--candidates`, `--cand-filter` (P1.3 candidate-API sweeps) |
| `bench/ops/fp8_linear_add_bench.cu` | M | `--n 3072 --k 8192` row_f32 mode, `--candidates`, `--cand-filter` |
| `bench/ops/fp8_linear_swiglu_bench.cu` | M | `--shared-122b` profile (N=2048 K=3072 out=1024, row_f32) |
| `bench/ops/linear_bench.cu` | M | `FP8F32` qtype for the two row_f32 A16 Linear shapes |

## 4. Measured results (GB10, cold-L2; baselines re-measured in-session)

Acceptance was judged against in-session baselines; the task doc's §1.2 numbers are stale by ~20%.

A8 argmin sweeps (12/12/10 schedules × 22 widths) + 16 interleaved `ckt ab --rounds 6` arbitrations:

| Shape | Baseline T=4096 | After | Delta | Key mid-band wins (baseline → after) |
|---|---|---|---|---|
| attn n17408 | 2602.8 µs | ~2478 µs | **−4.9%** (ab, SNR 3.0) | T=448 386.1→337.7 (+12.5%), T=512 395.1→366.5, T=768 512.9→476.9, T=1024 669.7→629.6, T=289 313.3→290.8; 288→289 cliff removed |
| gdn n20480 | 3043.3 µs | 3043 µs | 0% (Bulk already argmin at 4096) | T=576 495.4→445.2 (+10.1%, ab +9.6%), T=2048 1552.2→1463.1 (ab +10.6%), T=1024 774.0→743.4 (ab +6.5%), T=1536 1101.6→1077.1 (ab +5.4%), T=448 436.9→412.4 |
| linear_add n3072 | 2087.9 µs | ~1615 µs | **−22.7%** (ab, SNR 30.1) | T=3072 1303.6→~1239 (bulk.G8 ab +8.2%), T=448 192.5→185.4 (Mid ab −12.3% vs Wide), T=512 201.7→195.6 |

Wide-T L2 raster cliff (the dominant 122B A8 win; interleaved ab medians, production baseline grid
T = 8192 in parentheses):

| Shape | T = 6144 | T = 8192 |
|---|---|---|
| attn n17408 (plain 96x256 → Grouped-8) | tie region ends ≤5120 | 8939.4 → 4962.7 µs (**−44.5%**, SNR 28.2; base grid 8380.3) |
| gdn n20480 (Bulk → Grouped-8 192x128) | 6720.8 → 4363.1 µs (**−35.1%**, SNR 18.8) | 10361.0 → 5813.7 µs (**−43.9%**, SNR 23.0; base grid 10054.5) |

- Acceptance verdict: ≥20% at T=4096 is met for K=8192 (`linear_add` +22.7%). For the K=3072
  projections the ≥20% target at exactly T=4096 is **not** reachable against the in-session
  baseline: their inherited Bulk@170 dispatch is already within 2–5% of the 12-candidate argmin at
  T=4096 (attn −4.9%, gdn 0%) and the plain tile still wins there (attn ab: G8 loses by 8.1% at
  4096, ties at 5120); above the L2 crossover both shapes gain 35–44.5% (table above) and the
  mid-width headroom (289–2048, +5–12.5%) was captured.
- WaveCtas (170 → 48) split-K variants: ≤0.2% — not adopted; rejected candidates: `bulk.g8` for
  attn (slower than `bulk`), `tma64x256.g8` everywhere, `mid`/`wide` for 512–768.
- A16 shapes: `linear_bench --qtype fp8f32` sweeps — head T=1 3187.5 µs ≈ DRAM floor
  (97.5% sustained read); head/down per-token jumps at 129+ are the second weight pass
  (physics, not routing). Packing-interval and ring-boundary changes A/B'd against the saved
  pre-change binary (`ckt ab`, 6 rounds): head packing bands win **−26.0…−27.6%** at T=168–192
  (SNR 44.7–85.0) and **−15.7…−17.6%** at T=264–288 (SNR 8.0–25.4); down ring boundary wins
  **−5.0%** at T=257 (SNR 147.8) and ties at 288–320 → both adopted. swiglu `--shared-122b` A16:
  decode 31.6 µs, T=4096 743.3 µs (69.3 TFLOP/s), no ≥15% route cliffs → unchanged.

## 5. Verification

- Final-ladder production grids (in-session, cold-L2, medians): `post_*.csv` (all bands ≤4096
  and add >896) plus `fix_*.csv` boundary grids at 2048–8192 (attn/gdn) and 641–897/2560/4096
  (add). Production T=8192: attn 8380.3 → 5119.8 µs (+38.9%), gdn 10054.5 → 5981.0 µs (+40.5%),
  add T=4096 2087.9 → 1628.2 µs (+22.0%).
- Cliff scan on the final ladder: the largest per-token **route-transition** jumps are attn
  +4.6%/tok (384→385), gdn +7.5%/tok (1536→1792 wide-band interior), add +2.5%/tok (896→897);
  the 288→289 (+4.5→0%) and 768→769 (+27→padding-only) admission cliffs are removed. Remaining
  768→769 per-token steps (attn +14.7%, gdn +13.7%, add +15.6%) occur **inside one schedule**
  crossing a 128-column slab boundary — any 128-tile schedule pays ≥+16.7% there; no schedule
  change lowers them (probed all candidates at 769/832/896).
- `ctest -R "fp8|infernix_cuda_stack_check"` (serial): 20/20 passed, incl. the zero-spill ratchet
  and `infernix_fp8_row_f32_test` (the 122B A8/A16 oracle). `ctest -R "attn_input_proj|gdn_input_proj"`:
  5/5 passed.

## 6. Rebase conflict-resolution policy for the touched subsystems

- The candidate-ID blocks (`Fp8*ScheduleId` + `execute_candidate`) are bench-facing surfaces;
  keep them when upstream reshuffles ladders.
- Preserve the GB10 ladders (this file's §3 tables) over upstream 5090-wave defaults; upstream
  math/epilogue fixes take precedence.
- Production partial-capacity changes (attn → 0, gdn → >320) must move together with the
  ladder if either side conflicts.

## 7. Open items this session leaves

- No 122B `.infernix` artifact exists on this host: end-to-end `run_kv_matrix_bench.sh` and
  real-artifact model checks for the 122B routes could not run; verification is op-level
  (sweeps + ab + oracle ctests). E2E remains open until an Ornith-122B artifact is available.
- attn boundary T=1793 and gdn 2048→2049 measure ≤7.5%/tok on the final grid (under the 15% bar)
  but rest on sweep ranking rather than boundary-level `ckt ab`; the 4096/5120/704 boundaries are
  ab-arbitrated.
- `--fp8-problem 122b` candidate sweeps are cold-L2 single-stream; concurrent-load behaviour of
  the Grouped-8 wide bands under real batching is untested until an artifact exists (first item).

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T4: Column-invariant `Q8` fixed-schedule sliced-K families (`T = 1..64`)

Scope: `docs/tasks/20261029-gb10-post-rebase-perf-tasks.md` **Task T4** (P1) — replace the
SIMT-pair column-invariant route of the Qwen3.8-Flash-Next dense `Q8` classes with per-shape
fixed-schedule `Q8A16SlicedKMmaSchedule` families, and evaluate the task's small-weight
single-launch split-k item.

## 1. Commit range and accounting

- Base: `origin/main`; HEAD at work start: `9c0153eb2` (T3).
- Files changed this session (5 files, +230/−30): see §3. `cuda-kernel-toolkit/` intentionally excluded.

## 2. Why these changes are required by GB10 hardware

- The column-invariant (`A16ColumnInvariant`) route of every dense `Q8` shape except the
  vocabulary head was the runtime-shape SIMT pair (`simt_r8_t4` / `simt_r8_t8`). On 48 SMs its
  8-column pass collapses beyond `T = 4` (interleaved ab, this session: 124 → 133 GB/s at
  `T = 5/8` on `[10240, 320]`) and takes a second whole pass at `T = 9..16` (76 GB/s at `T = 16`,
  110 GB/s on the trunks) while the weights still stream once. The head already proved the fix
  shape: one fixed `(KWarps, Stages, MinBlocksPerSm)` family at capacity 8/16/32, whose K-slice
  order and split-reduction tree are call-width independent, so the column bits are invariant by
  construction.
- Measured GB10 schedule facts behind the per-shape tables: `K = 320` admits no static
  K-warps split (not a multiple of 128/256/512), so `[10240, 320]` runs the runtime-K variant at
  four blocks per SM (197..203 GB/s flat, `T = 1..16`); `K = 640` static-2-K-warp (five 128-K
  groups over 160 CTAs) beats every wider split; `N = 320` is a 20-CTA grid, where the 16-K-warp
  family (1024 K-columns per round) holds `T = 1` at the old bandwidth and gains 22..55% at
  `T = 5..8` — no inter-CTA split-k is needed (the schedule already reduces its K-splits in
  shared memory in one launch, which is also the disposition of the task's "second
  `reduce_split_k` launch" item for these routes); `N = 1280` (80 CTAs) measures best with
  `ScaleAccess::Direct`, from `N = 2560` up two pipeline stages per block win; `N = 324` rows are
  no multiple of the 16-row MMA tile and cannot take this route at all, so it keeps the SIMT pair.
- Task-doc premise corrections (measured/verified this session): the hyper-connection mix
  bypasses `ops::linear` entirely for `columns ≤ 16` (`kHcMixFusedMaxColumns = 16` in
  `hyper_connection_mix_fused.h`; `hc_up_collapse` covers `> 96`), so `[10240, 320]` /
  `[324, 10240]` Linear tables are reachable only at `T = 17..96` or through same-shape users —
  the bench's `T = 1..5` bandwidth slope lives in the register-limited fused kernel, not in any
  route table, and `infernix_hyper_connection_bench` is structurally insensitive to these tables
  (confirmed: post-change T=1 30.92 µs vs 30.75 pre-change, within noise across 1..64); the doc's
  referenced `q8_route_catalog.h` / `q8_a16_column_invariance()` policy hooks do not exist — the
  authority was `q8_dispatch.cpp`.

## 3. Subsystem breakdown

| File | Action | Change |
|---|---|---|
| `src/ops/linear/q8/shapes/q8_invariant_families.cu` | A | Eight per-shape capacity-8/16/32 sliced-K ladders (static-K via geometry, runtime-K for K = 320/640); schedules are the measured winners of an 8-schedule-per-shape candidate sweep |
| `src/ops/linear/q8/q8_dispatch.cpp` | M | `ShapeEntry.invariant` selector + `find_shape`; `select_q8_launch` column-invariant branch takes the table entry, head folded into the table, SIMT pair remains the documented fallback; a16 and FP32 branches unchanged |
| `src/ops/linear/q8/q8_shapes.h` | M | Eight `*_invariant` selector declarations |
| `src/ops/linear/q8/sources.cmake` | M | Register the new TU |
| `tests/ops/linear/test_q8_a16.cpp` | M | Conformance admits `A16ColumnInvariant` calls at `T ∈ {1,2,5,8,16,33,48,64}` for the eight retuned shapes (oracle criterion); new explicit-policy bit-exact column-invariance pass over each shape's live column span (8/16/48/64); invariance body shared with the default-route pass |

## 4. Verification and results (in-session baselines, cold L2, interleaved `ckt ab`)

- Bandwidth `legacy → new` (effective GB/s), `--policy invariant`: `T = 5`: 124→201
  `[10240, 320]`, 109→146 `[2560, 640]`, 187→231 / 187→230 / 190→231 trunks, 105→193
  `[1280, 2560]`, 168→215 `[2560, 2560]`, 88→136 `[320, 10240]`; `T = 8`: 133→203, 110→148,
  211/211/217→230/230/231, 120→193, 158→216, 105→128; `T = 16` (old second-pass region):
  +82..164%.
- Interleaved ab (5 rounds/point, all 16 acceptance-band points `MEANINGFUL_GAIN`): `T = 5`
  speedups 1.52 / 1.35 / 1.24 / 1.24 / 1.20 (narrow two, trunks three), 1.47 `[320,10240]`,
  1.73 `[1280,2560]`, 1.57 `[2560,2560]`; `T = 8` 1.53 / 1.34 / 1.09 / 1.07 / 1.07, 1.23, 1.51,
  1.41. Worst SNR 1.23 (`[320,10240]` T=8, −18.9% latency).
- Tests: `linear_q8_a16` (conformance + both invariance passes) passed under every candidate
  schedule used during selection (env-gated runs 1/2/4/7/8) and under the baked default;
  `infernix_cuda_stack_check` passed; `infernix_hyper_connection_bench --columns 1,2,4,5,8,16,32,64`
  within noise of the pre-change run (route-insensitive by the fused guard, §2).

## 5. Accepted gaps

- Trunk `T = 8` gains are +6.5..8.8% (ab), below the 15% bar: at `T = 8` the old SIMT pair
  already matched the default a16 route (211..217 GB/s) and the new route lands at 230..231 GB/s
  ≈ 94% of this host's measured 246 GB/s sustained read; +15% from there exceeds the DRAM floor.
  `T = 5` carries the trunks (+19.8..24%).
- `[324, 10240]` unchanged (rows block the 16-row MMA tile); its invariant route stays the SIMT
  pair, covered by the new bit-exact pass.
- The gains are route-table level (bench + oracle tests): at decode widths the HC shapes run the
  fused mix kernel, so no end-to-end HC movement is claimed or expected (§2); the shapes gain
  where they actually reach `ops::linear` (`T = 17..96` HC composition, shared-expert-class
  same-shape users) and the invariant policy as a whole becomes single-pass beyond 8 columns.


---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T5: Hybrid prefix-cache `ContextPrefillCost` on GB10 (calibration campaign)

## 1. Accounting

- Base `da207cd1b` (post-T4). Files: `bench/context_cost/context_cost_bench.cpp` (M),
  `src/runtime/engine/context_cache/context_cost_defaults.cpp` (M),
  `tests/runtime/test_context_cost.cpp` (M). Evidence: `profiles/bench/20261029-t5-f4local-prefill-rejected.json`.

## 2. Why the preset changed, and the task-doc premise corrections

- The task doc's step ("define `kGb10Groupwise27bPrefill`/`kGb10Nvfp4Fp8Prefill` measured on
  GB10") is unsatisfiable on this host and the copied 5090 `.prefill` bindings in the
  `"nvidia-gb10-sm121"` preset are a real mispricing: `prefill_signature` is an
  artifact-specific SHA (architecture + full Text/Vision weight inventory), so the two 5090
  configurations' signatures can never appear from this machine's artifacts, and their GB10
  block priced recomputation with 5090 constants (`chunk_ns` 40.8 ms / 14.7 ms). The block now
  carries no compiled prefill entry (measured 2026-10-29); foreign signatures resolve the
  generic estimate, and a local `--preset-out` calibration layers over it as before.
- Structural gap found while measuring (documented, not changed): the per-signature prefill
  cost is consumed only on the qwen3_5 load path (`model_instance.cpp` computes the signature,
  resolves the preset and calls `Program::set_hybrid_cost`). The qwen4_exp loader
  (`qwen4_exp_instance.cpp`) leaves `LoadSummary::prefill_signature` empty and never calls
  `set_hybrid_cost`, so no Flash-Next artifact on this host consumes any compiled prefill
  entry. The transfer half of the preset is consumed machine-wide and was already GB10-measured.
- Calibration-fixture limits measured on GB10 (all with `INFERNIX_EXPERTS_ON_GPU=1`, the
  serving switch): (a) offloaded-experts artifacts size the Qwen4Exp KV pool's elastic base at
  `kKvBaseTokens` 4096 and their expert cache starts with 0 frames, so the fixture's root binds
  fail admission with a constant 41-page `main_kv` shortage at `--max-context` 8192/16384, and
  the fixture's Vision cases die with "Vision encode ... cache can lend 0 MiB" — iq3s and
  dense8 are therefore not calibratable through this bench as written; (b) the only artifact
  that admits the full grid (F4-Local NVFP4 W4Dense, experts device-resident) is Text-only:
  its fit needed the observability fix below, and the resulting 3-coefficient fit was
  **rejected by the bench's own gate** (validation p95 0.186 > 0.15; prefixes ≥ 1024 tokens
  carry a ~+50 ms page-setup term the 5-feature model cannot express — `b2048-s256` +18.6%,
  median error 0.05). No rejected coefficient was baked.

## 3. Subsystem breakdown

- `bench/context_cost/` (M): `fit_prefill` fits observable feature columns only (a Text-only
  artifact's zero Vision columns no longer trip the generic fitter's unobservability gate);
  `runtime_prefill_cost` maps fitted coefficients by name into the 5-field contract, unfitted
  fields staying exact zeros (work the configuration cannot perform).
- `runtime/engine/context_cache/` (M): GB10 preset `.prefill` bindings removed with the
  measured rationale comment; 5090 preset and generic constants untouched.
- `tests/runtime/` (M): GB10 assertions extended to the full transfer trio and a regression
  check that the copied 5090 signatures resolve `GenericDefault` prefill (still
  `CompiledDefault` transfer) on GB10; the 5090-class `CompiledDefault` checks are kept.

## 4. Verification

- `infernix_context_cost_test` passed; `ctest -R context` 5/5 passed (incl.
  `infernix_context_kv_materialize_test` on GPU).
- Bench rebuild clean; calibration runs on the three GB10 serving artifacts logged and the
  rejected F4-Local fit report archived (`profiles/bench/20261029-t5-f4local-prefill-rejected.json`:
  `chunk_ns` 138,064,512; `token_ns_q32` 1,565,454,159,703,038; `attention_pair_ns_q32`
  26,926,496,322 — recorded as evidence only, not bound).

## 5. Accepted gaps / follow-ups

- GB10 has no compiled prefill entry at all now (previously: wrong-valued ones). Any GB10
  machine keeps the generic estimate until a calibration lands — this is strictly less wrong
  than the 5090 copy and is the calibration harness's designed fallback.
- Follow-up candidates (not in T5's scope): wire the qwen4_exp loader to compute the signature
  and `set_hybrid_cost`; make the calibration fixture usable with offloaded experts (base-KV
  admission and Vision frame warm-up); extend the cost model with a prefix page-setup term so
  Qwen4Exp fits can pass the p95 gate.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T6: `ScreenedLmHead` and `rg16_linear` for 48 SMs (tuning campaign)

## 1. Accounting

- Base `c56835eac` (post-T5). Files: `include/infernix/ops/screened_head.h` (M),
  `src/ops/screened_head/screened_head.cu` (M), `tests/ops/test_screened_head.cpp` (M),
  `tools/cuda_stack/cuda_stack_allowlist.json` (M). `rg16_linear.cu`: audited, unchanged (premise
  correction 2). Evidence: `profiles/nsys/20261029-t6-screened-base.sqlite`,
  `20261029-t6-e2e-dense8/f4local` traces, interleaved A/B logs (`/tmp/t6_{base,v1}_runs.txt`).

## 2. Task-premise corrections (all measured 2026-10-29 on this host)

- **The `screen_kernel` grid was never 170-CTA-hardcoded**: `screen_grid` probed
  `cudaDevAttrMultiProcessorCount`, so GB10 already ran 48 persistent CTAs. The task's
  `screen_shared_bytes<1>()` "~49.1 KiB" was the **2-stage** size; the shipped 4-stage ring is
  72,704 B (71 KiB), so 2 CTAs/SM required halving the ring — halving prefetch alongside
  occupancy. Measured net of that pair (interleaved 4-round A/B, T=1..8): **+0.4..2.8 %**, not
  the expected large win: the 16-18 % shortfall against the ckt read peak is the access-pattern
  ceiling (179 MB read + 1-8 MB writes ≈ 821 us at the measured 220.3 GB/s mixed rate; the
  kernel took 922-965 us = 85-90 % of the achievable mixed ceiling), not launch configuration.
  A deeper ring at 2 CTAs does not fit SMEM (61 KiB > 50 KiB); a third variant splitting K
  across 8 warps measured no gain and broke the contract's bit-exact cross-width invariance
  (different per-warp FP32 group chains), and was rejected.
- **`rg16_linear` is already SM-adaptive**: the narrow/stream routes size their persistent grid
  with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × probed SM count (rg16_linear.cu:545-566),
  so "sized for 170 SMs" is false; the wide route's 12.75 waves over 48 SMs cost ~2 % wave
  quantization on the vision shapes. No change; `rg16_linear` case passes.
- **No local artifact exercises `ScreenedLmHead`**: the registered qualification is
  `{Mxfp8 dense, INT8 group64}` (fast_paths.cpp, RTX 5090 measurements); the three Flash-Next
  artifacts select Q8/GGUF/NVFP4 dense routes (nsys traces), so no E2E `--screen-heads` delta is
  measurable here and **no fast-path configuration was registered** — the task's "+3-7 % decode
  E2E" step is out of reach on this host (the op-level gain is ≤ 2.8 % wherever the path runs).

## 3. Subsystem breakdown

- `include/infernix/ops/screened_head.h` + `src/ops/screened_head/` (M): the 1..8-column screen
  ring gains a narrow variant (2 items, 48 KiB, two co-resident CTAs/SM, occupancy-API-derived
  grid) selected on sm_121-class devices only; sm_120 and other devices keep the deep 4-stage
  single-CTA CTA shape bit- and grid-for-grid. The persistent grid becomes a public Op contract
  (`screened_head_screen_grid`) because the round-robin candidate lists (and every output) depend
  on it; `run_screen` now returns the grid it used (single source for the `Tiles`/ring split).
- `tests/ops/` (M): the selection oracle takes the Op's own grid (was a duplicated
  `clamp(sms,1,256)`), and the informational timing loop covers T = 1, 2, 4, 5, 8.
- `tools/cuda_stack/cuda_stack_allowlist.json` (M): pre-existing red at HEAD, attributed by a
  stashed rebuild — three T4 Q8 sliced-K schedules (K=10240 KWarps16-S1-M2; runtime-K
  KWarps8-MinBps4 ×2) carry a constant STACK 16 in the full serve link and are exactly the
  T4 cold-L2 A/B winners (review-accepted per the measured-spill precedent, like the FP8
  128x256 tradeoff); two stale `sample_row_kernel` entries dropped (those kernels now STACK 0).
  T6's screen kernels themselves are STACK 0 / LOCAL 0.

## 4. Verification

- `infernix_screened_head_test` `ok` (FP64 screen-score bounds, rescore bits, margin/cap,
  argmax, **bit-exact width invariance**, CUDA-graph replay), `infernix_rg16_linear_test` passed,
  `infernix_cuda_stack_check` passed (7692 kernels, 826 allow-listed, none new from T6);
  `ctest -R "screened_head|rg16|cuda_stack_check"` 4/4. Full tree rebuilt clean.
- ncu: narrow-ring instantiation launches (96,1,1)x(128,1,1), warps_active 16.6 %
  (2 CTAs/SM); barrier stall 38.6 % -> 9 %, long_scoreboard then 58 % (memory-wait steady
  state at the ceiling).
- Real-route smoke (INT8 KV): dense8 (Q8 dense) 32.7 tok/s, F4-Local (NVFP4 dense) 46.3,
  IQ3_S (GGUF dense) 37.9 — coherent outputs; none selects `screened-lm-head` (premise 3).

## 5. Accepted gaps / follow-ups

- T6's acceptance "latency at T = 1..8 improves by >= 15 %" is **not met**: the honest GB10 gain
  is +0.4..2.8 % (op-level, interleaved medians), because the baseline already runs at 85-90 % of
  the achievable mixed-stream roofline. The remaining gap candidates (list/softmax serial tail at
  low T; 720-B row-pair DRAM efficiency) are content changes to the Op, not 48-SM tuning.
- The screened path's E2E value on GB10 stays unmeasured (no Mxfp8-dense-with-screen artifact);
  the 5090-registered qualification is untouched.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-29 — T7: DFlash2 drafter tuning and the UMA drafter-pool guard

## 1. Accounting

- Base `3a9e1cd71` (post-T6 `7bf3d4962` and the T8 commit). Files:
  `src/models/qwen4_exp/program/program_impl.h` (M: `kDFlashPairMaxDrafts` policy constant +
  `INFERNIX_DFLASH_PAIR_MAX_DRAFTS` override + capacity-sized fit table),
  `src/models/qwen4_exp/program/graph_set.cpp` (M: align startup CUDA graph capture `pair_width`
  with `kDFlashPairMaxDrafts` when `dflash2_` or exact dense routes are active so `2x8`/`2x9`
  verification graphs are pre-captured at startup rather than on first use),
  `src/runtime/engine/model_instance.cpp` (M: integrated-GPU drafter-pool guard for Qwen3.5),
  `src/runtime/engine/qwen4_exp_instance.cpp` (M: integrated-GPU drafter-pool guard for Qwen4Exp
  even when `device_experts` is overridden off),
  `src/models/qwen3_5/program/planning/startup.cpp` (M: preserve loaded model's `mtp_expert_pool`
  choice in `validate_target_options`, matching `stream_experts`),
  `tools/run_server.sh` (M: measured backend choice documented), `drafter_pool.cpp` and
  `rg16`-family: audited unchanged. Evidence: `/tmp/t7_sweep.log` (F4-Local), `/tmp/t7_sweep_d8.log`
  (dense8, 30 rows + 6 probes), 2 interleaved rounds per cell, medians.

## 2. Task-premise corrections (all measured 2026-10-29)

- **The task's verification artifact cannot exercise the pair path.** `DFlash2Pairs` qualification
  (fast_paths.cpp) covers Q8/Ggml/Bf16/Fp8Row/Mxfp8 dense kinds — no NVFP4 entry — so on F4-Local
  the whole c >= 2 comparison decodes plain (measured: 35.6 ~= plain 34.1, spec rounds 6/723).
  The c >= 2 sweep therefore ran on the qualified `Dense8` (Q8 dense) artifact.
- **Tree verification is not the shape here**: these chain-verifying artifacts reject
  `--draft-tree-nodes` outright ("Qwen-Flash-Next verifies chains"); no tree-width sweep applies.
- The doc's command uses `--model/--prompt-tokens/--gen-tokens/--concurrency 1,2,4`; the bench
  takes `--weights -pg 2048,128` and one scalar `--concurrency`. `llama-benchy` is not installed.
- **`DrafterExpertPool` was never UMA-guarded on either architecture**: Qwen4Exp gates its slots on
  `!device_experts` (its config; device experts is the integrated default), so forcing offload on
  GB10 still splits pinned+pool; Qwen3.5 had no gate at all (64 slots x expert bytes duplicated in
  the one physical pool).

## 3. Subsystem breakdown

- `qwen4_exp/program` (M): `kDFlashPairMaxDrafts` 2 -> 8 (GB10 ladder at c = 2 on dense8 fp8 KV:
  47.9 / 58.8 / 63.6 / 67.9 decode t/s at 2 / 4 / 6 / 8 against MTP's 53.8 — still improving at the
  capacity bound; the proposal's 7 drafts cap the effective width; the 5090 measured 2 and can pin
  it back with the env), now a `policy_double` knob sized from `kPairMaxDraftsCapacity` like its
  siblings. Side effect noted: the `(dflash2_ || Exact dense routes)` branch also raises a
  reproducible-routes MTP pair fit's ceiling (output unchanged — verification decides; perf there
  not measured).
- `runtime/engine` (M): `construct_model` clears `mtp_expert_pool` on integrated GPUs, so Qwen3.5
  plans slots 0 and binds the drafter's banks Device (the existing unpooled route) instead of
  duplicating pinned RAM into device slots of the same LPDDR5X. The decision stays at load time so
  residency, slots and the `TextContext::drafter_pool` contract cannot diverge.
- `tools/run_server.sh` (M): documented per-c winners; the c = 4 server stays MTP (38.8 vs DFlash2's
  2-row default 20.6 and the `INFERNIX_TMP_DFLASH2_ROWS=4` probe 37.4).
- Complements carried in the same change: `qwen4_exp/program/graph_set.cpp` sizes the captured pair
  verification width with the same `(dflash2_ || Exact) ? kDFlashPairMaxDrafts : kPairMaxDrafts`
  selector (the captured dflash2 pair width follows the constant to 8; re-measured c = 2: 68.0);
  `qwen3_5/program/planning/startup.cpp` carries `mtp_expert_pool` from the loaded model into the
  target-options comparison, so the engine's integrated clearing cannot fail startup validation;
  `qwen4_exp_instance.cpp` also clears `mtp_expert_pool` on integrated parts (forced-offload path),
  closing the same UMA duplication Qwen3.5 guarded.

## 4. Verification

- dense8 pp2048+tg128 fp8 KV medians (plain / MTP / DFlash2): c = 1 34.2 / 81.8 / 109.4;
  c = 2 27.1 / 53.8 / 67.9 (default 8, re-verified post-change: 68.5); c = 4 19.8 / 38.8 / 20.6.
- F4-Local: c = 1 DFlash2 114.8 vs MTP 91.7 vs plain 46.0; c >= 2 the fallback described above.
- Acceptance met as recorded: the c = 1, 2, 4 MTP-vs-DFlash2 comparison is recorded with the optimal
  GB10 constants (c <= 2 DFlash2 +25..34 %; c = 4 MTP).
- `infernix_bench` + `infernix-serve` rebuilt clean; DFlash2 c = 1 (no pair) unaffected by the
  constant; MTP arms unchanged in-session.

## 5. Accepted gaps / follow-ups

- The Qwen3.5 UMA guard is code-level verified only: no Qwen3.5 offloaded artifact exists on this
  host, so the 122B drafter-pool route cannot run here.
- `dflash2_rows_` stays 2: the 4-row probe loses to MTP at c = 4 (37.4 vs 38.8); a qualification
  for wider DFlash2 rounds would need its own measurement.
- `DFlash2Pairs` for the NVFP4-dense kind is unqualified (no entry in the registry): on the
  launcher's default F4-Local artifact, DFlash2 helps only at c = 1. Adding that qualification (or
  the pair fast path's NVFP4 route) is the lever that would make DFlash2 the c = 2 winner there too.
- The shared default moved 2 -> 8 for every device (precedent: the 20261028 `kPairMaxDrafts`
  2 -> 6 raise on this branch); the 5090 calibration stands behind `INFERNIX_DFLASH_PAIR_MAX_DRAFTS=2`.

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T8: eliminate rebase-regressed linear / MoE / GGUF CUDA stack spills

## 1. Accounting

- Base `e824a5281` (T7); previous HEADs: `3a9e1cd71` (`perf(stack)`), `7bf3d4962` (T6).
- Working tree after T8: 5 ops sources (`ggml_a8_tile.cuh`, `ggml_bf16_tile.cuh`,
  `ggml_linear_a8.cuh`, `fp8_a8_mma_common.cuh`, `wide_expert.cuh`; net +18/-5) plus the
  regenerated `tools/cuda_stack/cuda_stack_allowlist.json` (826 -> 483 entries; 387 removed,
  44 added). `LOCAL` stayed 0 across all 7692 linked kernels.
- Four forced full-tree rebuilds + single-TU offline harnesses (see `docs/tasks` session notes:
  CUDA depfiles are dead in `build-native`; every header audit requires
  `find src/ops -name '*.cu' -exec touch {} +` first — `3a9e1cd71`'s numbers were measured on
  objects that never recompiled, and the fp8 spill verdicts below come from the first true builds).

## 2. Task-premise corrections (all measured 2026-10-30 on true full rebuilds)

- **`3a9e1cd71`'s fp8 hunks were the regression.** Its `#pragma unroll 1` on the `>= 32` `k_step`
  loop spilled 528 B, and the doc's alternatives (2-token A pair-stream; rolled Split-K `mt` store)
  reached 656-672 B: a dynamic index into the 512 B `accumulators[T][R][4]` defeats SROA and
  depots the whole array. Restoring the pre-`3a9e1cd71` upstream streaming-B branch (one line)
  returns the family to its 2026-10-27 true baseline: 44 spilling instances, STACK 16..48 B,
  170 of 214 instances at STACK 0. STACK-0 on `Fp8A8TmaMmaSchedule<128,256,...>` requires a
  route change, not pragmas (see the FP8 128x256 measured-tradeoff record).
- **The doc's "ggml_linear_wide: 28 kernels" file attribution is wrong**: those instances live in
  `ggml_tile.cuh` (via `ggml_linear_wide.cu`/`ggml_linear_tiled.cu`), not the a8/bf16 tile headers.
  They sit at REG 128 (2 CTAs/SM occupancy cap) with 8-16 B frames; 8 of the wide `tile_kernel`
  instances have zero `STL`/`LDL` (ABI frames), and unroll levers on the copy/decode loops were
  measured inert (zero effect on allocation) — reverted. Frozen as the T9/T10 occupancy-pinned class.
- **T9's vq2 numbers are true at HEAD on true builds** (`vq_prompt` 88 spills, `vq_kv_grouped` 32,
  all <= 32 B): applying T9's remaining levers ad-hoc during T8 made `vq_prompt` worse (88 -> 216),
  so those experiments were reverted; T9 stays open. `compute_plane`'s `k < 2` cp.async loop must
  stay unrolled: rolling it regressed `ggml_linear_a8_wide_cu`'s tile kernel to STACK 16 (its only
  caller), while `compute_decode`'s slice roll is what carries the MoE gate kernels to 0.

## 3. Subsystem breakdown

- `src/ops/linear/fp8/fp8_a8_mma_common.cuh` (`M`, 1 line): revert `3a9e1cd71`'s rolled `k_step` ->
  upstream streaming-B form. `fp8_a8_tma_mma_kernel`: 96 spills / max 672 B -> 44 / max 48 B.
- `src/ops/ggml/ggml_a8_tile.cuh` (`M`, 1 line): `compute_decode` slice loop `#pragma unroll 1`.
  `ggml_gate_up_a8_wide` and `ggml_gate_up_decode`: all 8 instances STACK 0 (were 48-64 B).
- `src/ops/ggml/ggml_bf16_tile.cuh` (`M`, 4 lines): plane slices rolled for `S == Scaling::kMin`,
  decode slices rolled: bf16 gate/decode spill sites 0 (`gate_up_tile_ggml` stays within its
  <= 40 B allowance, unchanged by this task).
- `src/ops/ggml/ggml_linear_a8.cuh` (`M`, 2 pragmas): `issue` ring-copy loops rolled (both the
  Tiled and non-Tiled branches): A8 `rows_mma` 72/0 and the A8 `tiled_mma` routes spill 0.
- `src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh` (`M`, ternary pragma): `unroll (kTerms == 3 ?
  1 : kTerms)` on the term loop: `grouped_kernel` 20 instances incl. `ScheduleResidual<3>` STACK 0.
- `tools/cuda_stack/cuda_stack_allowlist.json` (`M`): ratchet truth sync 826 -> 483 entries; the 44
  additions are 43 vq2 instances (<= 32 B, T9 scope) + 1 `qsa` `sincosf`-class key (T10 scope);
  all T8-accepted family keys removed or reduced to the legacy 16..48 B fp8 tradeoff band.

## 4. Verification

- Forced full-tree rebuild; `cuobjdump --dump-resource-usage` over `infernix-serve` (7692 kernels):
  STACK-0 families `f16acc_mma` (124), `fp8_a16_tma_mma` (200), `grouped_kernel`/`_a16` (36),
  `rows_mma` (72), A8 `tiled_mma`, `ggml_gate_up_a8_wide`/`decode` (8), `ggml_linear_a8_wide`
  tile (23); residue per section 5. `LOCAL` 0 everywhere; `MinBlocksPerSm` unchanged in every family.
- Oracle ctest selection (serial): fp8 A8 / pair-repeat / A16 / MXFP8-A16, ggml linear / A8 /
  A8-wide, offloaded-MoE ggml / ggml_a8 / w4a8 / a8_wide / fp8, rg16, vq + softmax attention —
  **19/19 PASS**; `infernix_cuda_stack_check` PASS (0 of 7692 over allowance).
- E2E `run_kv_matrix_bench.sh` (fp8 KV, plain + MTP, iq3s / dense8 / nvfp4) — T8 owns this gate
  (T2/T5 recorded it as deferred to T8; it establishes the branch baseline record):

  | artifact (fp8 KV) | pp4096 t/s | tg128 / tg256 t/s | pp4096+tg128 / +tg256 t/s | MTP tg128 / tg256 (accept) |
  |---|---|---|---|---|
  | iq3s plain    | 1258.8 | 37.6 / 37.4 | 1257.4 prefill · 36.7 / 36.8 | — |
  | iq3s MTP      | 1253.2 | 62.5 / 73.5 | 91.4 (0.916) / 81.2 (0.766) | ✓ |
  | dense8 plain  | 2912.6 | 34.4 / 34.3 | 2890.8 prefill · 33.7 / 33.8 | — |
  | dense8 MTP    | 2880.0 | 48.6 / 57.3 | 80.0 (0.889) / 80.1 (0.780) | ✓ |
  | nvfp4 plain   | 3280.0 | 46.3 / 46.3 | 3279.3 prefill · 45.2 / 45.3 | — |
  | nvfp4 MTP     | 3268.5 | 61.3 / 66.1 | 92.4 (0.943) / 101.5 (0.928) | ✓ |

  dense8 mixed-workload decode 80.0 sits inside the T7-session record for the same workload
  (81.8, ~2 % spread), so the MoE-side unroll rolls show no end-to-end cost at this resolution;
  acceptance rates are in the arms' historical bands. `KV_EXIT=0`, all arms complete.

## 5. Accepted gaps / follow-ups

- `fp8_a8_tma_mma_kernel` 128x256: 44 instances at 16-48 B remain = the measured speed tradeoff
  (2026-10-27 A/Bs: the spilling schedule is the fastest route; both register-surgery fixes
  regressed). The doc's "STACK 0 across all 44" acceptance is unreachable there without a
  perf-qualified route change; disclosed as partially met (170/214 instances at 0; the other
  44 are within the historical envelope and the allowlist).
- `ggml_tile.cuh` "28 group" (wide `tile_kernel`/`decode_tile_kernel`, generic `tiled_mma`):
  <= 16 B at the REG-128 occupancy pin; 8 instances are unremovable zero-traffic ABI frames;
  unroll levers measured inert (reverted). Requires a T9/T10-style measured occupancy decision.
- vq2 (`vq_prompt` 88/216 <= 32 B, `vq_kv_grouped` 32/288 <= 16 B) remains T9's; 43 new allowlist
  keys frozen; the T8-session attempts that inflated these numbers were reverted.
- `int8`/`mxfp8` `kv_tiled_mma` (<= 16 B, attention routes) and the `sincosf`/`qsa` 32 B frames
  are T9/T10-class; no T8 lever applies.
- Perf A/B on the MoE-side rolls (`wide_expert` kTerms==3, `compute_decode` slices) is bounded by
  the E2E record above; no paired HEAD-vs-T8 op-level sweep was run (disclosed).

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T9: VQ2 / INT8 causal-cache attention spill elimination and the MinBlocks measurement decision

## 1. Accounting

- Base `ab40600a9` (T8). Working tree: 4 ops sources (`vq2/prompt.cuh`, `vq2/grouped.cuh`,
  `int8/grouped_pipelined.cuh`, `int8/instances.h`; net +58/-58) plus the regenerated
  `tools/cuda_stack/cuda_stack_allowlist.json` (483 -> 349 entries; 134 removed, 0 added:
  88 `vq_prompt` + 22 `vq_kv_grouped` + 20 `int8_kv_grouped_mma` + 4 `int8_kv_grouped_pipelined`
  keys). `LOCAL` stayed 0 across all 7692 linked kernels.
- Eleven forced rebuilds under the touch-consumers protocol (CUDA depfiles dead). One cancelled
  intermediate build left a stale device-link registration object (bench/test links failed with an
  undefined `__fatbinwrap_*_launch_cu_*`); fixed by deleting `cmake_device_link.o` and the three
  suspect `*launch.cu.o` objects and relinking — disclosed because it invalidated one audit.

## 2. Task-premise corrections (all measured 2026-10-30)

- **`vq_kv_grouped`'s SMEM-aware launch_bounds already existed** (the `Wc == 16 || smem > 49000`
  ternary): the doc's "add SMEM-aware launch_bounds" premise was false at HEAD.
- The doc's "hoist `bf` per group" and "stream `af` per kk" levers were already implemented in the
  int8/vq2 grouped QK loops (no-ops); `causal_attention_prompt_i8_fast` has `Groups = 4` (its
  `q_scale_r` costs 8 registers, not 16), so its scale-shared lever does not apply.
- **The doc's MinBlocks condition resolved by measurement**: `MinBlocksPerSm = 2` on the 256-thread
  `vq_kv_grouped` and non-wide `int8_kv_grouped_mma` schedules was not faster on GB10 (section 4
  decode A/B), so both families moved to one CTA/SM; the pin's REG-128 ceiling was load-bearing
  only for its own spills.
- **Race found in this session's own shared-scale port**: the first `vq_prompt` port kept the
  register version's slot indexing (`q_scale_r[r >> 3]`), which is per-lane-row ownership in
  registers but an 8-lane same-address write race in shared; the serial oracle suite caught it as
  108 `infernix_vq_attention_test` prompt-route failures. The adopted port is row-indexed
  (16 rows x Groups per warp), arithmetically identical, and the suite passes.
- Rejected levers, each regressed individually and reverted: `grp` unroll-1 in the grouped QK loop
  (12 -> 40 spills); `finish_tile` rolls in `vq_prompt` (80 -> 94); scale-at-use ports into
  `int8/grouped_mma` and `fast_tiled_mma` (63 -> 67 / max 40 -> 72 B and 15 -> 16 / 408 -> 616 B).
  Lever polarity is per-kernel: the same `grp` roll that regressed grouped was positive in
  `vq_prompt` and `int8_kv_grouped_pipelined`.

## 3. Subsystem breakdown

- `src/ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh` (`M`): per-warp row-indexed
  shared `q_scale` slice (16 x Groups floats, +4 KB SMEM worst case at `Warps = 8`), `grp`
  unroll-1, `af` streamed per `(nt, kk)` (bit-exact: `af` is invariant across `nt`).
  `vq_prompt_kernel`: 88 spills / 1008 B -> **STACK 0 on all 216 instances**.
- `src/ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh` (`M`): shared `q_scale` route for
  `RowTiles >= 2` (freed registers at the 8-warp REG pin); `__launch_bounds__` MinBlocks fixed to
  1 (measured, removing the SMEM ternary). 32/320 B -> **10 instances / <= 16 B**, all 16-warp
  512-thread instances where REG 128 is the hardware per-thread cap (65536 / 512).
- `src/ops/softmax_attention/dense/causal_cache/int8/grouped_pipelined.cuh` (`M`): dedicated
  `q_scale_s` SMEM array (replacing the KV-arena alias plus its 16-register producer preload),
  `grp` unroll-1, guarded use-site reads. 4 -> **STACK 0 on all 102 instances**.
- `src/ops/softmax_attention/dense/causal_cache/int8/instances.h` (`M`, 1 token): grouped schedule
  MinBlocks 2 -> 1. 63 / <= 56 B -> **43 / <= 24 B**; the residue sits at REG 128 chosen by ptxas
  with no launch-bound pin active (single-store + late-LDL allocation class, pragmas inert per T8).
- `tools/cuda_stack/cuda_stack_allowlist.json` (`M`): ratchet truth sync, 134 stale keys removed.

## 4. Verification

- Forced rebuilds + `cuobjdump --dump-resource-usage` over `infernix-serve` (7692 kernels):
  final-family state as in section 3; `vq_prompt` 0/216, `int8_kv_grouped_pipelined` 0/102;
  `bf16_kv_grouped_mma` 18/<= 24 B and `causal_attention_prompt_i8_fast` 15/<= 40 B unchanged at
  the REG-255 hard cap (bf16 group's in-source 192-register accumulator-floor comment is its
  measured-tradeoff record; the doc's "only measured occupancy-load-bearing" exception covers it).
- Oracle ctest selection (serial, on the adopted binary): `infernix_vq_attention_test`,
  `infernix_softmax_attention_{,int8,bf16,nvfp4,k8v4,int8_pipelined16}`, the four
  `*_verify_rows`, `_extended`, the four `_wide_*`, `infernix_kv_cache_vq*` (auto/1/2/8),
  `infernix_kv_cache_append` — **22/22 PASS** (mid-session the pre-race-fix binary failed
  `infernix_vq_attention_test` 108 cases; disclosed above). `infernix_cuda_stack_check` PASS
  (0 of 7692 over allowance; allowance after sync 349 keys).
- E2E MinBlocks A/B, `run_kv_matrix_bench.sh` iq3s, int8+vq2 KV, plain+MTP (arm A = MB 2 with all
  T9 code levers, arm B = adopted MB 1; tg = decode t/s, mx = pp4096+tg mixed):

  | arm | int8 plain tg128/tg256 | int8 mtp tg128/tg256 | vq2 plain tg128/tg256 | vq2 mtp tg128/tg256 |
  |---|---|---|---|---|
  | A (MB 2) | 37.53 / 37.51 | 62.17 / 64.73 | 37.07 / 37.01 | 61.72 / 67.13 |
  | B (MB 1, adopted) | 37.59 / 37.51 | 62.25 / 63.98 | 37.13 / 37.09 | 61.74 / 67.59 |

  Plain arms differ <= 0.2 %; MTP arms +/- 1-4 % on mx workloads, inside this route's
  run-to-run acceptance noise (same-binary per-repetition acceptance spread reached +/- 0.07,
  larger than every inter-arm gap). MB 1 is not slower anywhere and removes 14 + 20 spill
  instances including both gate-RED razor edges; adopted.
- E2E sanity, adopted binary: iq3s int8 plain pp4096 1279.7, vq2 plain 1237.8 prefill t/s —
  consistent with the T8-session records for this artifact class (its table is fp8 KV; the
  int8/vq2 columns above are the new dtype baselines).

## 5. Accepted gaps / follow-ups

- `vq_kv_grouped` residue 10 instances (<= 16 B): 16-warp 512-thread configurations where REG 128
  is the hardware per-thread cap, not a launch-bound choice; removal needs a warp-count/tile
  redesign (accumulator footprint 48-64 floats/lane), not spill surgery.
- `int8_kv_grouped_mma` residue 43 instances (<= 24 B): ptxas allocation class at unpinned
  REG 128 (SASS single-store + late-LDL pattern; T8 established pragmas are inert there). The
  scale-at-use port that could free those registers was measured adverse and reverted.
- `bf16_kv_grouped_mma` (18) and `causal_attention_prompt_i8_fast` (15): both run at the REG-255
  per-thread hard cap, so occupancy policy cannot remove their frames. bf16's record is the
  in-source 192-register Q+O accumulator-floor note (`bf16/grouped_mma.cuh:186`: single-live K
  fragment, double buffering spills); i8_fast's is this session's adverse shared-scale port
  (408 -> 616 B, section 2) plus its `Groups = 4` premise correction. The doc's
  measured-occupancy-load-bearing exception covers both.
- The bench's MTP route is not run-deterministic in acceptance (per-repetition spread); future
  decode-efficiency A/Bs on it need `--reproducible` engines or an interleaved paired design to
  resolve deltas below ~4 %.
- `int8`/`mxfp8` `kv_tiled_mma` (6 and 3 instances, <= 16/8 B) were left at T8 allowances; they
  are decode- cold routes and no T9 lever applied.

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T10: Eliminate RoPE/QSA `sincosf` frames and preventable Q8/Q4/GDN/GGUF/MoE stack spills

## 1. Accounting

- Base commit: `557474392` (follows `432f7267d` T9; `origin/main` merge-base unchanged).
- 12 product/ops files changed (1 added, 11 modified), plus `tools/cuda_stack/cuda_stack_allowlist.json` (`M`) and `docs/gb10/20261029-changelog.md` (`M`):
  - `src/ops/common/sincos_reduced.cuh` (`A`)
  - `src/ops/common/dflash_rope.cuh` (`M`)
  - `src/ops/kernel/rope.cuh` (`M`)
  - `src/ops/qsa/qsa.cu` (`M`)
  - `src/ops/qsa/qsa_prepare.cu` (`M`)
  - `src/ops/context_kv_materialize/materialize.cu` (`M`)
  - `src/ops/gdn_input_proj/gdn_projected_conv.cu` (`M`)
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`M`)
  - `src/ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.cu` (`M`)
  - `src/ops/linear/q8/shapes/q8_invariant_families.cu` (`M`)
  - `src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu` (`M`)
  - `src/ops/ggml/ggml_linear.cu` (`M`)
- Stack-spill allowlist delta (`tools/cuda_stack/cuda_stack_allowlist.json` over `build-native/apps/infernix-serve`, `7,692` linked kernels):
  `349` spilling symbols (`21,680 B`) -> `272` spilling symbols (`19,392 B`), net **`-77` symbols (`-2,288 B`)**, strict subset of `HEAD` (`0` additions, `0` raised frames; `LOCAL = 0` everywhere). All `63` non-lossless RoPE/QSA/materialize `sincosf` kernels plus `14` quantized linear / GDN / GGUF / MoE kernels are now `STACK = 0`.
- Rebuilds: 2 single-invocation Ninja builds under the touch-consumers protocol (1 full `.cu` consumer touch after header additions/edits, plus 1 incremental 4-TU `.cu` build sharing one `cmake_device_link.o` across `infernix-serve` and all test executables).

## 2. Task-premise corrections / Architectural rationale

- **CUDA 13 `__nv_sincosf` Payne-Hanek slowpath (`63` kernels, `32 B -> 0 B`)**:
  CUDA 13.0.88 libdevice `__nv_sincosf` inlines `int result[7]` (`28 B` rounded to a `32 B` stack frame) behind the `|x| >= 105615.0f` slowpath guard, emitting `2 STL + 3 LDL` whenever GB10's launch-bounds register budget (`REG = 20..40` across RoPE/QSA kernels) cannot hold the dynamically indexed depot in registers. Reducing `angle` modulo $2\pi$ in FP64 (`|r| <= \pi < 105615.0f`) makes the Payne-Hanek branch 100% unreachable, and reproducing libdevice's fast path (`cvt.rni.s32.f32` quadrant select, 3-term Cody-Waite reduction, and minimax polynomials with forced non-contracted `__fmaf_rn`/`__fmul_rn` preserving `-0.0f`) eliminates all `63` frames while matching `sincosf` bit-for-bit across all `2,409,402,112` float bit patterns with `|x| < 105615.0f`. `powf` (`STACK = 0`) and `--lossless` (`src/ops/lossless/cr_math.cuh`) remain untouched.
- **Dynamic indexing of small register/parameter aggregates (`4` kernels, `16..72 B -> 0 B`)**:
  `pool_write_kernel` (`moe_layer.cu`) staged its by-value `MoePoolWrites` struct (`72 B`) to local memory because of `i < writes.count` dynamic indexing; an unrolled constant-index loop over `kMoePoolWrites = 8` keeps the struct in parameter/register space (`STACK = 0`). `gdn_projected_conv_tree_kernel` (`gdn_projected_conv.cu`, `3` instances) localized `float history[3]` (`16 B`) because the ancestor walk indexed `history[hist]` dynamically; a constant-index ternary chain (`hist <= 0 ? history[0] : hist == 1 ? history[1] : history[2]`) keeps all 3 floats in registers (`STACK = 0`).
- **SMEM-over-constrained `__launch_bounds__` (`9` kernels, `8..32 B -> 0 B` at `0%` occupancy change)**:
  On GB10 (`sm_121a`, `102,400 B` SMEM/SM with `1,024 B` driver-reserved SMEM/CTA -> `max_user_smem = 50,176 B` for 2 CTAs, `33,024 B` for 3 CTAs, `24,576 B` for 4 CTAs), three T10 kernel families specified `MinBlocksPerSm` strictly above their physical SMEM residency ceiling, artificially tightening `ptxas`'s register cap and spilling live values for zero occupancy gain:
  1. `q8_invariant_families.cu`: `Rung<32, 8, 1, 4>` (`256` threads, `42,496 B` SMEM -> max `2` CTAs/SM; lowering `MinBlocks: 4 -> 2` raises reg cap `64 -> 128` at `2` CTAs/SM and removes `2` `16 B` spills with `28 LDL / 15 STL`, while `Rung<16, 8, 1, 3>` matches its `3`-CTA SMEM ceiling) and `Rung<32, 16, 1, 2>` (`512` threads, `82,944 B` SMEM -> max `1` CTA/SM; lowering `MinBlocks: 2 -> 1` raises reg cap `64 -> 128` at `1` CTA/SM and removes `1` `16 B` spill with `11 LDL / 3 STL`).
  2. `q8_dynamic_grouped_conv_add_materialized.cu`: `Q8A16SlicedKMmaSchedule<TileColumns, 4, 1, 3, ...>` stages `37,120..49,408 B` for `TileColumns = 64..88` (max `2` CTAs/SM, not `3`); clamping `MinBlocks` to `2` for `TileColumns > 56` raises `TileColumns = 88`'s register budget `80 -> 255` at `2` CTAs/SM and eliminates its `32 B` (`51 LDL / 5 STL`) inner-loop spill (correcting an earlier misattribution to a 9-warp TMA limit).
  3. `ggml_linear.cu`: `mma_rows_kernel` and `mma_rows_mixed_kernel` hardcoded `__launch_bounds__(128, 4)` (sized for `23.5 KiB` staged formats where `4` CTAs fit), whereas `GGML_Q6_K` and `Q6_K`-mixed pairs stage `37,888 B` (max `2` CTAs/SM). Clamping `MinBlocks` via `mma_rows_min_blocks(...)` preserves `4 CTAs/SM` on all `23.5 KiB` formats while raising `Q6_K` and `Q6_K`-mixed register cap `128 -> 255` at `2 CTAs/SM`, eliminating `5` spills (`2` `mma_rows_kernel<Q6_K>` + `3` `mma_rows_mixed_kernel`).
- **Premise correction & measured occupancy tradeoff on `q4_linear_swiglu_w4a8_kernel` (`1` kernel, `32 B -> 0 B`)**:
  Removing `MinBlocks = 2` on `q4_linear_swiglu_w4a8_kernel` (`512` threads, `34,816 B` SMEM) raises `REG` from `64` to `108` (`57,344` allocated regs/CTA), which **halves residency from `2 CTAs/SM` (`66.67%` occ, `32` warps/SM) to `1 CTA/SM` (`33.33%` occ, `16` warps/SM)** — correcting an inaccurate draft comment that claimed `34 KB` SMEM left occupancy at `2 CTAs/SM` (`ckt static` defaults to `@256t` when block size is not a template parameter; `ckt math occupancy --threads 512 --regs 108 --smem 34816` gives the true `1 CTA/SM`). Interleaved paired A/B-B/A benchmarking confirmed that eliminating the `32 B` inner-K-loop MMA accumulator spill outweighs the `2 -> 1 CTA/SM` occupancy drop at every prefill width (`+7.48%` to `+13.48%` speedup across `T = 256..4096`), so `1 CTA/SM` (`STACK = 0`) is kept with an accurate comment.

## 3. Subsystem breakdown

- `src/ops/common/sincos_reduced.cuh` (`A`): `rope_reduced_angle(double)` (FP64 modulo-$2\pi$ reduction via `nearbyint`), `sincos_reduced(float)` (bit-exact CUDA 13 `__nv_sincosf` fast-path port without the Payne-Hanek slowpath), and `sincos_turns(double, float*, float*)`.
- `src/ops/kernel/rope.cuh`, `src/ops/common/dflash_rope.cuh` (`M`): `fixed_sincos`, `PreparedRopeCoefficients::sincos_axis`, and `dflash_rope_sincos` routed through `sincos_turns` / `sincos_reduced` (`48` `rope_fixed`/`rope_fixed_split`/`rope_generic` instances -> `STACK = 0`; shared with `rmsnorm_rope`).
- `src/ops/qsa/qsa.cu`, `src/ops/qsa/qsa_prepare.cu`, `src/ops/context_kv_materialize/materialize.cu` (`M`): RoPE angle sites routed through `sincos_turns` (`10` `qsa_prepare_kernel`, `2` `qsa` `index_query`/`pool` kernels, `3` `context_kv_mma_kernel` -> `STACK = 0`; note the task spec names `src/ops/qsa/materialize.cu`, which lives at `src/ops/context_kv_materialize/materialize.cu`).
- `src/ops/gdn_input_proj/gdn_projected_conv.cu` (`M`): constant-index ternary selection on `history[3]` in `gdn_projected_conv_tree_kernel` (`3` instances: `16 B -> 0 B`; task spec names `src/ops/gdn/gdn_projected_conv.cu`).
- `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`M`): unrolled constant-index loop in `pool_write_kernel` (`72 B -> 0 B`; remaining `moe_layer.cu` allowlist count `8 -> 7`, zero new or raised frames).
- `src/ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.cu` (`M`): `__launch_bounds__(kThreads, 2)` -> `__launch_bounds__(kThreads)` (`REG: 64 -> 108`, `STACK: 32 B -> 0 B`, `2 -> 1 CTA/SM`), with comment updated to record the true occupancy and GB10 A/B speedup.
- `src/ops/linear/q8/shapes/q8_invariant_families.cu` (`M`): SMEM-bounded `MinBlocksPerSm` on `select_q8_n10240_k320_invariant` (`4/3/2` at `T = 8/16/32`) and `select_q8_n320_k10240_invariant` (`2/2/1` at `T = 8/16/32`), eliminating `3` `16 B` spills at `0%` occupancy change.
- `src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu` (`M`): SMEM-bounded `MinBlocks = (Warps == 8 || TileColumns > 56) ? 2 : 3`, eliminating `TileColumns = 88`'s `32 B` spill at `0%` occupancy change.
- `src/ops/ggml/ggml_linear.cu` (`M`): `mma_rows_min_blocks(...)` clamps `mma_rows_kernel` and `mma_rows_mixed_kernel` `MinBlocksPerSm` to physical SMEM residency (`4` for `<= 24,576 B` staged formats, `2` for `37,888 B` `Q6_K` and `Q6_K`-mixed pairs), eliminating `5` `8..16 B` spills at `0%` occupancy change.

## 4. Verification

- **Static resource & ratchet gate**: `cuobjdump --dump-resource-usage` / `ckt static --spills-only` on `build-native/apps/infernix-serve` (`7,692` kernels): `349` -> `272` entries (`-77` symbols, `-2,288 B`, `0` additions, `0` raises, `LOCAL = 0`). `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure` **PASS** before and after ratcheting.
- **Oracle CTest suite**: `ctest --test-dir build-native -R "rope|qsa|linear_q8|linear_q4|linear_q5|linear_swiglu_q4_w4a8|gdn|ggml|linear_dynamic_grouped_conv_add|column_invariant_projections|context_kv_materialize|^infernix_offloaded_moe_layer_test$" --output-on-failure` -> **45/45 PASS** (all `40/40` tests matching the T10 spec regex plus the `5` additional oracle suites covering `q4_w4a8`, `dynamic_grouped_conv_add`, `column_invariant_projections`, `context_kv_materialize`, and `offloaded_moe_layer`).
- **Interleaved A/B-B/A microbenchmark (`q4_linear_swiglu_w4a8_kernel`)**: `infernix_q4_linear_swiglu_bench --routes fast --production-only` on NVIDIA GB10 (`sm_121a`, `32 MiB` cold-L2 flush, `warmup=5, repeat=30`, `ckt math ab` across paired runs):

  | `T` | Baseline `2 CTAs/SM` (`REG=64, STACK=32`) | Candidate `1 CTA/SM` (`REG=108, STACK=0`) | Latency delta | Speedup (`ckt math ab`) | TFLOP/s (`2 CTA` -> `1 CTA`) |
  |---:|---:|---:|---:|---:|---:|
  | 256 | 1159.52 us | 1021.78 us | -11.88% | **1.1348x** | 78.7 -> 89.3 |
  | 384 | 1647.92 us | 1454.42 us | -11.74% | **1.1330x** | 83.1 -> 94.1 |
  | 512 | 2134.85 us | 1941.86 us | -9.04% | **1.0994x** | 85.5 -> 94.0 |
  | 768 | 3224.88 us | 2930.00 us | -9.14% | **1.1006x** | 84.9 -> 93.5 |
  | 1024 | 4298.06 us | 3955.57 us | -7.97% | **1.0866x** | 84.9 -> 92.3 |
  | 2048 | 8505.14 us | 7913.30 us | -6.96% | **1.0748x** | 85.9 -> 92.3 |
  | 4096 | 17096.03 us | 15795.55 us | -7.61% | **1.0823x** | 85.4 -> 92.5 |

- `git diff --check`: clean (exit 0).

## 5. Accepted gaps / follow-ups

- **100% of non-lossless RoPE, QSA, KV-materialize, `gdn_projected_conv`, and `q4_linear_swiglu_w4a8` kernels are at `STACK = 0, LOCAL = 0`**; the 5 remaining allowlist symbols containing `rope`/`qsa`/`gdn` (`lossless_rope_cu` `40 B`, `lossless_flash_cu` `128..176 B` x3, `lossless_gdn_cu` `160 B`) belong to `--lossless` (`T11` scope).
- **Remaining `45` non-lossless T10-family allowlist entries** are all either at the hardware `REG = 255` ceiling (`8` kernels, already `MinBlocks = 1`) or genuinely SMEM-feasible, occupancy-pinned schedules (`37` kernels) where relaxing `MinBlocksPerSm` directly drops active CTAs/SM:
  - Hardware `REG = 255` ceiling (`MinBlocks = 1`): `1` `q8_a16_grouped_sliced_k_mma_kernel<112,4,1,1,1,...>` (`16 B`, `4 LDL / 2 STL`), `6` `moe_layer.cu` `gate_up_tile_ggml` (`3x16 B, 3x40 B`), and `1` `moe_layer.cu` `ggml_gate_up_bf16` (`8 B`).
  - SMEM-feasible occupancy-pinned schedules: `11` `q8_a16_stream_kernel` (`8..24 B`, `1..3 STL/LDL` at `REG = 128, SHARED = 0, 2 CTAs/SM` decode GEMV), `1` `q8_a16_mma_kernel<96,112,64,...>` (`16 B` at `REG = 72, SHARED = 49,664 B <= 50,176 B, 2 CTAs/SM`), `9` `q8_a16_sliced_k_mma_kernel<8,8,1,5,...>` (`1` `linear/q8` + `8` `q8_gdn_input_gemm_splitk`, `8..16 B` at `REG = 48, SHARED = 17.4..17.9 KiB, 5 CTAs/SM = 83% occ`), `4` `q4_a16_sliced_k_mma_kernel` (`8 B`, `1..2 LDL / 1 STL` at `3..6 CTAs/SM`), `2` `q5_a16_direct_simt_kernel` (`24..32 B` at `REG = 48, 10 CTAs/SM = 83% occ`), `2` `ggml_linear.cu` `mma_rows_kernel<QType=15>` (`8..16 B` at `REG = 128, SHARED = 24,064 B <= 24,576 B, 4 CTAs/SM`, documented `+15%` sibling regression on relaxation), and `8` `ggml_linear_tiled.cu` `tiled_mma` instances (`8..16 B` at `REG = 128, 2 CTAs/SM`, T8-verified inert to loop-unroll pragmas).

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T11: Audit, regenerate & ratchet `cuda_stack_allowlist.json` on GB10

## 1. Accounting

- Base commit: `6d827d40f` (`dev-rebased-20261028`, T10 HEAD); working tree clean at start.
- Files: `tools/cuda_stack/cuda_stack_allowlist.json` (`M` regenerated from the rebuilt `infernix-serve`; **byte-identical** to the committed T10 state — `git diff --stat` empty), `docs/gb10/20261029-changelog.md` (`M`, this section), `tools/cuda_stack/check_cuda_stack.py` (`unchanged` — see §2 premise correction).
- Cumulative allowlist ratchet across the campaign (`tools/cuda_stack/cuda_stack_allowlist.json` git history on `build-native/apps/infernix-serve`): `826` (rebase base `7bf3d4962`) -> `483` (T8 `ab40600a9`, `-343`) -> `349` (T9 `432f7267d`, `-134`) -> `272` (T10 `6d827d40f`, `-77`) -> **`272`** (T11 audit, `-0`): total **`826 -> 272` (`-554` entries, `41,264 B -> 19,392 B`)**, `LOCAL = 0` on every entry throughout.
- Rebuilds: 1 single-invocation incremental Ninja build (`main.cpp.o` build-id stamp + one shared `nvlink` device link); no `.cu` recompiles (no source edits; no header touches required).

## 2. Task-premise corrections / Architectural rationale

- **Regeneration is already at its fixed point.** `check_cuda_stack.py update` on the rebuilt binary re-wrote the allowlist to a byte-identical file: `272` of `7,692` kernels use stack or local memory, and the committed T10 allowlist matched with `0` stale keys, `0` unlisted spillers and `0` over-allowance violations. The T8–T10 source fixes (not a T11 re-update) are what produced the reduction; T11 verifies the ratchet state and audits every entry.
- **`826 -> <100` is not reachable on GB10** without deleting the `--lossless` product route or re-accepting measured regressions that T8/T9 A/Bs already rejected. The audited `272`-entry set is the minimal set under the current gate semantics (every `STACK > 0` kernel requires an entry, traffic or not): `61` `--lossless` software-oracle kernels carry `15,688 B` (`80.9%`) of the `19,392 B` total, and the remaining `211` non-lossless entries are all `<= 48 B` frames (`3,704 B`, `19.1%`).
- **Full SASS traffic audit (T11 measurement).** `cuobjdump -sass -fun <exact symbol>` over all `211` non-lossless entries: **every one carries real local traffic (`>= 2` STL/LDL lines; min 2, `2,028` local-op lines total)** — there are no nominal-only ABI-canary entries left (the 2026-10-27 audit's `38` `<= 1`-op frames were the Payne-Hanek `sincosf` class that T10 eliminated). Each entry is therefore a genuine, cause-attributed spill, not a checker artifact.
- **Category-2 framing correction & exact `sm_121a` register-file classification.** The task doc names `MinBlocksPerSm = 2`-proven-faster as the sole non-lossless exception class; joining live `cuobjdump --dump-resource-usage` (`REG`/`SHARED`/`STACK`) with each kernel's exact `__launch_bounds__(MaxThreads, MinBlocks)` / `__maxnreg__` declaration shows that **zero** unpinned `ptxas` mystery spills remain — all `211` non-lossless entries sit at an explicit hardware register-file ceiling (`174` entries at `1 CTA/SM`) or an SMEM-feasible multi-CTA occupancy pin (`37` entries at `MinBlocksPerSm = 2..10`):
  1. Per-thread ISA register ceiling (`REG = 254..255`, `<= 256` threads, `MinBlocks = 1`): `38` entries (`680 B`).
  2. SM 65,536-register-file ceiling on wide `1 CTA/SM` blocks (`MinBlocks = 1`, `384` threads / `12` warps -> `floor(65536 / 384 / 8) * 8 = 168` regs; `512` threads / `16` warps -> `65536 / 512 = 128` regs — previously misattributed when thread count was assumed `256`): `92` entries (`1,208 B`).
  3. 9-warp warp-specialized TMA `128x256` ceiling (`288` threads, `REG = 168`, `MinBlocks = 1`) with documented spill-vs-register-surgery A/B regressions (`44` entries, `1,280 B`, T8).
  4. SMEM-feasible multi-CTA occupancy-pinned schedules (`MinBlocksPerSm = 2..10`) where relaxing `MinBlocksPerSm` directly drops active CTAs/SM (`37` entries, `536 B`, incl. T10's documented `+15%` `mma_rows` sibling regression).
- **`check_cuda_stack.py` needs no change.** The doc's "bloated by blind `update`" premise is not supported by the git history (`817` on `origin/main` `d8e52a85a` -> `826` on rebased `7bf3d4962` tracked routes added on the rebased tree before GB10 spill surgery): `update` is run only manually, `check` fails closed on any new/raised frame, and the T8–T10 ratchets were each applied only after source fixes removed the corresponding spills. No script defect was found that would let the audit be automated further without weakening the one-way gate (stale-key pruning already happens on every `update`).

## 3. Subsystem breakdown (audited entry table, live `cuobjdump` + `__launch_bounds__` join on `build-native/apps/infernix-serve`)

| Cause category | Kernels (entries) | Bytes | `STACK` range | Measured & architectural basis |
|---|---:|---:|---|---|
| `--lossless` software-oracle local arrays | `61` | `15,688` | `40..1472 B` | Intentional I128/Kulisch + FP64 accumulator arrays in local memory (`src/ops/lossless/wide.cuh`, `linear_impl.cuh`, `lossless_attention.cu` `I128 num[16]`, `lossless_moe/ops/flash/gdn/rope`); product requirement, `LOCAL = 0` reported via `STACK` only |
| `1 CTA/SM` per-thread ISA ceiling (`REG = 254..255`, `<= 256` threads) | `38` | `680` | `8..40 B` | `18` `bf16_kv_grouped_mma` (`SHARED = 33,936 B`, in-source Q+O 192-reg accumulator-floor note), `15` `causal_attention_prompt_i8_fast` (T9 shared-scale port measured adverse `408 -> 616 B`), `3` `mxfp8_kv_tiled_mma` (`__maxnreg__(255)`), `1` `q8_a16_grouped_sliced_k`, `1` `ggml_gate_up_bf16`; occupancy policy cannot lift the cap |
| `1 CTA/SM` 65,536-reg/SM ceiling on wide 12/16-warp CTAs (`384t -> REG=168`, `512t -> REG=128`, `MinBlocks = 1`) | `92` | `1,208` | `8..40 B` | `43` `int8_kv_grouped_mma` (`kWideKeys`: `7` at `384t/168r`, `36` at `512t/128r`; T9 scale-at-use port measured adverse and reverted), `18` GGUF wide `decode_tile_kernel` (`512t/128r`) + `8` `tile_kernel<Q4_K/Q5_K>` (`512t/128r`), `10` `vq_kv_grouped` (`512t/128r`, `SHARED = 42.9..46.6 KiB`), `6` `int8_kv_tiled_mma` (`512t/128r`, `92 KiB` dynamic SMEM), `6` `gate_up_tile_ggml` (`512t/128r`), `1` `projection_tall` (`512t/128r`); `ptxas` already allocates the full SM register file to 1 CTA |
| `1 CTA/SM` 9-warp TMA ceiling (`288t -> REG=168`, measured speed tradeoff) | `44` | `1,280` | `16..48 B` | `fp8_a8_tma_mma_kernel<128,256,...>`: 2026-10-27 & T8 A/Bs — epilogue restructure `0.97x`, warp-widening `+25%` at `K = 6144 T = 1024`; spilling `128x256` schedule is the fastest route on GB10 |
| Multi-CTA SMEM-feasible occupancy pins (`MinBlocksPerSm = 2..10`) | `37` | `536` | `8..32 B` | `11` `q8_a16_stream` (`256t, MinBlocks = 2, REG = 128, 2 CTAs/SM` GEMV), `9` `q8_a16_sliced_k` (`256t, MinBlocks = 5, REG = 48, SHARED = 17.4..17.9 KiB, 5 CTAs/SM`), `8` `ggml_linear_tiled` `tiled_mma`/`_mixed` (`256t, MinBlocks = 2, REG = 128`, T8-verified inert unroll pragmas), `4` `q4_a16_sliced_k` (`MinBlocks = 6`, SMEM-clamped `3..6 CTAs/SM, REG = 40..80`), `2` `q5_a16_direct_simt` (`128t, MinBlocks = 10, REG = 48, 10 CTAs/SM`), `2` `mma_rows_kernel<IQ3_S>` (`128t, MinBlocks = 4, REG = 128, SHARED = 23,552 B, STACK = 32 B`, documented `+15%` sibling regression on relaxation), `1` `q8_a16_mma<96,112,64>` (`MinBlocks = 2, REG = 72, SHARED = 49,664 B <= 50,176 B, 2 CTAs/SM`) |

- Every `272` entry (`61 + 38 + 92 + 44 + 37 = 272` entries; `15,688 + 680 + 1,208 + 1,280 + 536 = 19,392 B`) joins to exactly one live binary kernel (`0` stale); the `211` non-lossless `STACK`-histogram is `8x78 16x82 24x13 32x10 40x26 48x2` — no non-lossless frame exceeds `48 B`.

## 4. Verification

- `ninja -C build-native -j "$(nproc)" infernix-serve`: relinked; `check_cuda_stack.py update` -> `272 of 7692 kernels use stack or local memory`; `git diff --stat tools/cuda_stack/cuda_stack_allowlist.json` -> **empty (byte-identical regeneration; zero new/raised entries by construction; cumulative T8–T11 diff vs `7bf3d4962` is `826 -> 272`, `-554` entries)**.
- `ctest --test-dir build-native -R infernix_cuda_stack_check --output-on-failure`: **PASS** (`0` unallowlisted spills); `infernix_cuda_stack_tool_test` **PASS**.
- Resource join (`cuobjdump --dump-resource-usage`): `7,692` kernels, `272` with `STACK > 0`, `0` with `LOCAL > 0`; allowlist/binary sets identical.
- SASS local-op scan (all `211` non-lossless symbols): `0` zero-traffic, `211` with `>= 2` STL/LDL lines; per-family totals — `fp8_a8_tma` `654/44`, `int8_kv_grouped` `346/43`, `tile/decode_tile` `174/26`, `i8_fast` `196/15`, `bf16_kv_grouped` `97/18`, `gate_up_tile_ggml` `172/6`, remaining `14` families `<= 71` each.
- `git diff --check`: clean (exit 0).

## 5. Accepted gaps / follow-ups

- **`272` is the audited minimal set for this toolchain (CUDA 13.0.88) and route table** — not the doc's `<100`. Shrinking it further requires either (a) dropping `--lossless` software accumulators (product route, `61` entries / `15,688 B`), (b) tile/warp-count redesigns of the `174` `1 CTA/SM` hardware-register-ceiling kernels (`REG = 255` on `<= 256` threads, `REG = 168`/`128` on `384`/`512`-thread wide CTAs, and `REG = 168` 9-warp TMA `128x256`), where T8/T9 register-surgery and warp-widening alternatives measured slower, or (c) relaxing the `37` SMEM-feasible multi-CTA occupancy pins (`MinBlocksPerSm = 2..10`) after per-family GB10 A/Bs — all three are product or kernel-redesign decisions, none is a ratchet fix.
- The `826 -> 272` ratchet is one-way and gate-enforced: any new or raised frame fails `infernix_cuda_stack_check`; `update` may only be re-run after reviewed source fixes (T8–T10 protocol).
- SASS counts are static line counts per stored symbol (a frame with `>= 2` static ops may still execute few dynamically at runtime); they classify traffic-vs-nominal, not hot-path cost. Per-entry dynamic hotness was not measured (would require NCU on synthetic inputs; no decision depends on it).
- Task-doc §1.2 E2E baselines remain stale by ~20% on this tree (recorded 2026-10-29); T11 required no E2E measurement, so none was taken — no route or numerics changed (zero source edits).

## 6. Rebase conflict-resolution policy

- **`tools/cuda_stack/cuda_stack_allowlist.json`**: Never resolve rebase conflicts by accepting upstream's RTX 5090 (`sm_120a`) allowlist or blindly running `check_cuda_stack.py update` on an un-audited merge binary. Start from GB10's ratcheted `272`-entry allowlist, run `infernix_cuda_stack_check`, fix any newly introduced `sm_121a` spills at source (loop-unroll staging, `sincos_reduced`, SMEM-clamped `__launch_bounds__`), and run `update` only to prune deleted upstream symbols or record audited `--lossless` / A/B-proven hardware-ceiling exceptions.
- **`tools/cuda_stack/check_cuda_stack.py`**: Preserve the fail-closed `check` / stale-pruning `update` semantics and `stable_name()` internal-linkage normalization alongside `tests/tools/test_check_cuda_stack.py`.

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T12: Cumulative E2E KV-matrix & multi-concurrency serving gate

## 1. Accounting

- Base `e9a856eeb` (T11 HEAD; session start HEAD, unchanged by T12; two unrelated `chore(agents)` commits landed mid-session moving HEAD to `4e9788bb3`). Files:
  `bench/run_kv_matrix_bench.sh` (M: mode-only `100644 → 100755`, content byte-identical — restores the documented `./bench/…` invocation),
  `docs/gb10_roadmap.md` (M: new §8.12 cumulative gate tables + findings), this file (M).
- Allowlist delta: **none** (gate PASS, `0` over, no `update` run). Rebuilds: one full `ninja -C build-native -j "$(nproc)"` invocation, 182 steps, `NINJA_EXIT = 0` — relinks + 3 bench-sweep `.cu.o` (`rope_bench`, `mxfp8_flash_next_sweep`, `fp8_flash_next_sweep`); **no production kernel object recompiled**. No header touches needed (no `src/` edits).
- Evidence: `/tmp/t12_ctest_chunk{0..7}.log` + `/tmp/t12_ctest_driver.log` (suite), `/tmp/t12_ctest_serial_rerun.log`, `/tmp/kvmatrix_20261029/` (`status.txt` `ALL_DONE`, 15 JSON reports), `/tmp/t12_benchy.log` + `/tmp/t12_benchy_result.json`, `/tmp/t12_serve.log`.

## 2. Task-premise corrections / Architectural rationale

- **`bench/run_kv_matrix_bench.sh` was not executable in the index** — the task's literal `./bench/run_kv_matrix_bench.sh …` command `exit = 126`'d. Root cause: git mode `100644` with `core.fileMode = false` masking a missing local bit; fresh checkouts can never run the documented command. Fixed via `chmod +x` + `git update-index --chmod=+x`.
- **`llama-benchy` was not installed** (recorded at T7); installed `llama-benchy 0.4.0` into `.venv-convert` (pulls its own deps; `torch 2.14.0+cu130` import re-verified intact after install).
- **benchy + this launcher need explicit model naming**: benchy 0.4.0 rejects the launcher's bare-hash model-id at tokenizer auto-detect ("not in HF format") even with `--tokenizer` given — pass `--model`; and `run_server.sh` re-derives the model-id from `git HEAD` at every start, so a HEAD move between script iterations changes the served id (this actually happened mid-session: `e9a856ee → 42359203`). Drive with `--model-id <fixed>`.
- **The suite's overlap-sensitivity now hits 2 further joins at `--parallel 8`**: `infernix_qwen4_exp_ngram_volume_scatter_test` (short direct block read under I/O contention) and `infernix_offloaded_moe_layer_shared_x_test` (`moe_route_pooled` oracle mismatch). Both pass serially (per this host's recorded protocol: serial pass ⇒ overlap-sensitive, serialize at source; serial fail ⇒ real defect).
- The task doc's `pp4096` `IQ3_S = 1,045.0` baseline reconfirmed stale by ~20%: this tree measures `1,267.4`.

## 3. Subsystem breakdown

- `bench/run_kv_matrix_bench.sh` (M): exec bit only; no logic change (script auto-rebuilt `infernix_bench` — already current, `ninja: no work to do`).
- `docs/gb10_roadmap.md` (M): §8.12 records the gate: stack ratchet, `379`-test outcome distribution, 3-arm KV matrix table with favorable/unfavorable deltas vs the `P0.A`/`8.11` gate table and an in-session same-source reference, the `3 × 3` HTTP decode table with per-stream `± σ`, and the three methodology caveats (temperature-`1.0` acceptance vs engine route, benchy `c = 1` prefill-column artifact, explicit model-id requirement).

## 4. Verification

- **Stack gate**: `ctest -R infernix_cuda_stack_check` PASS (1/1); `check_cuda_stack.py check build-native/apps/infernix-serve`: `7692 kernels checked, 0 over their allowed stack or local memory` (T11 `272`-entry set).
- **Full suite** (`--parallel 8`, 8 resumable `≤ 48`-test index chunks, wall `41m04s` `10:28:50→11:09:54`): chunks `0,1,2,4,5,6` `48/48`; chunk `3` `47/48` (`#151`); chunk `7` `42/43` (`#374`). Serial re-run of exactly the two failures: **`2/2 Passed`** (`0.47 s`, `102.93 s`). Skips: `65` env-gated real-artifact entries (`qwen3_5_*`/`qwen4_exp_*_real*` families). **Effective 379/379.**
- **KV matrix** `/tmp/kvmatrix_20261029` (fp8 KV × mtp × 3 artifacts, `ALL_DONE`, `exit = 0`/arm, `~7 min`): `pp4096` `iq3s 1,267.4` / `dense8 2,907.4` / `nvfp4 3,276.8`; `pp4096+tg128` decode `87.2 (.890)` / `86.4 (.967)` / `96.0 (.971)`; `pp4096+tg256` decode `83.4 (.788)` / `94.4 (.933)` / `95.2 (.875)`; pure-decode `tg128/tg256` `61.8/73.4`, `50.0/59.4`, `62.2/70.4`. Favorable: `nvfp4 pp4096 +1.2%`, decode `+3.4%` vs `P0.A`. Worst changes: `nvfp4 pp+tg256` decode `-3.7%` (`acc .920 → .875`); same-source run-to-run mixed-decode spread up to `-8.9%` (`iq3s pp+tg128`, `acc .937 → .890`, `n = 2` medians, zero binary delta) — sampling variance, disclosed, not averaged away.
- **Serving gate** (`run_server.sh` + `llama-benchy 0.4.0`, `pp2048 tg128`, `d ∈ {0,4096,8192}`, `c ∈ {1,2,4}`, 3 runs, `temp = 1.0`, coherence PASS, `BENCHY_EXIT = 0`, `6m30s`): per-stream decode `61.71 ± 1.60` / `39.10 ± 1.59` / `27.40 ± 1.76` at `d0 c1/2/4`; within `2.6%` at `d8192` for every `c`; aggregate `61.7 → 74.9 → 101.0`; `pp2048` TTFT `830 ms` (`d0 c1`) → `867 ms` (`d8192 c1`, `+4.5%`).
- `git diff --check`: clean (exit `0`).

## 5. Accepted gaps / follow-ups

- The gate ran the **default 3-arm matrix** (fp8 KV × mtp), exactly the task's command — the other 39 arms of `--full` were not re-measured here (T8/T10-era full-matrix fp8-KV records stand: plain `pp4096` `iq3s 1,258.8` / `dense8 2,912.6` / `nvfp4 3,280.0`).
- `#151`/`#374` are overlap-sensitive at `--parallel 8` and pass at `-j1`; they are **not yet `RUN_SERIAL`** — serializing them in `tests.cmake` is the natural follow-up (out of scope for a verification-only task, consistent with the recorded `bf16_a16`/`gdn_replay_fold` disposition).
- Serving gate ran the launcher's default `F4-Local` artifact only (`c ≤ 4`, per its `--max-concurrency 4`); HTTP numbers at `temperature = 1.0` are acceptance-dominated and sit below the engine-route figures by design — both tables are recorded separately in §8.12 so they are never conflated again.
- `--depth` cells measure prefix-cache restore + shared-prefix serving; the huge benchy `c = 1` `t/s (total)` prefill columns are estimator output, not throughput — only `e2e_ttft`-derived rates were published.
- Two external `chore(agents)` commits (AGENTS.md) landed on this branch during the gate; they are user-owned, untouched here, and did not alter any measured code path (docs-only).

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T13: Test-suite wall-clock reduction

## 1. Accounting

- Base `a510d0b51` (HEAD at session start). Files: `tests/ops/tests.cmake` (M), `tests/ops/linear/tests.cmake` (M), `tests/ops/test_offloaded_moe_layer.cu` (M), `tests/ops/test_speculative_round.cpp` (M). No `src/` edits; no production kernel changed.
- Baseline: per-test seconds from the T12 full-suite chunk logs (`/tmp/t12_ctest_chunk{0..7}.log`, `--parallel 8`, 312 timed tests, 5,856 s summed). After: single-test `ctest` runs on `build-native` (one at a time, GPU otherwise idle).

## 2. Rationale

- **Duplicate entries**: `INFERNIX_TMP_TMA_PARAM_WORDS` is read only through `with_tma_descriptor_parameter`, whose call sites are all `#ifdef _WIN32`; on Linux the 9 `*_param_words` entries re-ran identical binaries. `narrow_pairs()` is on by default on GB10, so the `INFERNIX_TMP_NARROW_PAIRS=1` layer entries re-ran the default path.
- **Over-wide switch coverage**: `INFERNIX_TMP_CPU_SHARED_X` is read only by the W4A16 / two-term CPU paths (`expert_team.cpp`), so its entries run only those layers (`--a16-only`).
- **Idle checks** need only a decode (`T = 1`) and a multi-column (`T = 5`) geometry per activation mode; `E = 512` calls exceed the idle service's `max_columns = 64`.
- **Serialization**: `infernix_offloaded_moe_layer_test` runs `--no-idle`, so its leftover `RUN_SERIAL` was unnecessary; the idle entries keep it.

## 3. Subsystem breakdown (tests)

- `tests/ops/linear/tests.cmake` (M): removed the 9 `*_param_words` entries.
- `tests/ops/tests.cmake` (M): removed `layer_pairs` / `layer_idle_pairs`; `layer_shared_x` and `layer_idle_shared_x` run `--a16-only`; `layer_test` no longer `RUN_SERIAL`.
- `tests/ops/test_offloaded_moe_layer.cu` (M): `--a16-only` flag; `--idle-only` runs `test_layer` only at `columns ∈ {1, 5}`; the host oracle in `test_layer` runs on `parallel_ranges`; `test_route_pooled`'s graph-replay logits/demand reset now run on the replay stream (the default-stream `cudaMemcpy`/`cudaMemset` did not order against the non-blocking stream, the likely cause of the T12 `#374` `moe_route_pooled` mismatch — not reproduced after the change, but not proven).
- `tests/ops/test_speculative_round.cpp` (M): the `K 1..31 × batch 1..8` sweep (248 suites) became `K ∈ {1,2,3,4,5,7,8,9,15,16,17,30,31} × batch ∈ {1,2,3,8}` (52 suites); the exhaustive sweep passed when the subset was chosen.

## 4. Verification and results

| Test | Before | After |
|---|---:|---:|
| `infernix_speculative_round_test` | 143.88 s | 23.96 s |
| `infernix_speculative_round_fp32_test` (`RUN_SERIAL`) | 89.08 s | 32.68 s |
| `infernix_offloaded_moe_layer_test` | 102.52 s (`RUN_SERIAL`) | 93.83 s (parallel) |
| `infernix_offloaded_moe_layer_shared_x_test` | 117.44 s | 44.70 s |
| `infernix_offloaded_moe_layer_idle_test` (`RUN_SERIAL`) | 80.47 s | 9.87 s |
| `infernix_offloaded_moe_layer_idle_shared_x_test` (`RUN_SERIAL`) | 80.96 s | 7.32 s |
| 2 `*_pairs` layer entries | 201.01 s | removed |
| 9 `*_param_words` entries | 498.00 s | removed |

- Summed test work removed: about 1,101 s of 1,313 s across these entries; `RUN_SERIAL` time down about 383 s.
- All kept entries above passed. `git diff --check` clean.

## 5. Accepted gaps / follow-ups

- Reductions in coverage (not just duplicates): the speculative K × batch grid is a boundary subset, and idle-latency checks cover two geometries instead of all. Neither is a speed-only gain.
- The full suite was **not** re-run after the change; times are single-entry measurements, not `ctest --parallel` wall-clock. `infernix_offloaded_moe_layer_test` is only 8.7 s faster alone; its gain is removing `RUN_SERIAL`.
- Largest remaining cost: the 18 `infernix_attention_oracle`-locked softmax-attention entries (about 1,404 s serialized) are untouched.
- `infernix_qwen4_exp_ngram_volume_scatter_test` (T12 `#151`) is still overlap-sensitive and not serialized.
