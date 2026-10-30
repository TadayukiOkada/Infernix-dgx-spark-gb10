# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T1: Enable & qualify `DFlash2Pairs` and `Q8SlicedK` for `DenseKernels::Nvfp4`; `GdnConvFused` measured and held as candidate

Scope: `docs/tasks/20261030-performance-tasks.md` **Task T1** (P0) — the three `Qwen4Exp` fast-path
gates that penalized `Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix` (`DenseKernels::Nvfp4`,
4-bit routed experts). Artifact: `/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/
Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix`, drafter `Qwen3.8-Flash-Next-DFlash2-Infernix.
infernix`, `--kv-dtype fp8`, `infernix_bench pp2048+tg128 -r 3` medians (decode out t/s).

---

## 1. Accounting

| Item | Value |
|---|---|
| Base commit | `66466fd26` (`dev-rebased-20261028`, clean tree) |
| Files changed | 5 modified + 1 added (this file) |
| Diff | +105 / −17 across sources, tests, docs, launcher |
| Stack-spill allowlist delta | none (`infernix_cuda_stack_check` pass; no CUDA source touched; `update` not run) |
| Rebuilds (single-invocation ninja) | 3 (pre-change baseline arm, fused-registered build, final build) |

| File | A/M | Change |
|---|---|---|
| `src/models/qwen4_exp/execution/forward.cpp` | M | `Forward::gdn` `fused_conv` admits `QType::NVFP4` beside `Q8_G32_FP16` |
| `src/models/qwen4_exp/program/fast_paths.cpp` | M | `fast_path_applies` (`Q8SlicedK`, `GdnConvFused`): `Q8 || Nvfp4`; `kQualified`: +14 `DFlash2Pairs` + 14 `Q8SlicedK` entries `{Nvfp4, both a4 kinds, all 7 KV}`; `GdnConvFused` Nvfp4 unregistered with its measured verdict |
| `tests/models/qwen4_exp/test_fast_paths.cpp` | M | `DFlash2Pairs` registration loop +`Nvfp4`; new W4Dense block: applies/qualified matrix (both a4 kinds × 7 KV), Forward route flags, reproducibility (request-invariance) behavior |
| `docs/maintainer/model-quality.md` | M | §6.7 rows `q8-sliced-k/gdn-conv-fused/qsa-prepare-fused` and `dflash2-pairs` updated with the T1 registrations, the measured fused-route verdict, and the follow-ups |
| `tools/run_server.sh` | M | T7-era "pair path is NOT qualified on F4-Local" guidance replaced by the measured post-T1 winners |

Evidence files: `/tmp/t1_armA_c*_*.log` (pre-change arm A), `/tmp/t1_armB_c*_*.log` (all-three-on
intermediate), `/tmp/t1_final_*.log` (shipping configuration), `/tmp/t1_iso_B_c1_plain_repro.log`
(all-fast-off reference), NSYS graph-node traces `/tmp/t1_nd_default.*` / `/tmp/t1_nd_repro.*`.

## 2. Task-premise corrections / architectural rationale

- **Premise 2 holds (fixed): `DFlash2Pairs` had zero `{Nvfp4, *, *}` entries** although
  model-quality §6.7 had registered it on "every Flash-Next configuration" (owner's decision
  2026-10-24) — the dense-kind list simply never carried NVFP4. With the 14 entries added,
  `qwen4_exp_instance.cpp:543` (`pair_speculation = AdaptiveSpeculation || DFlash2Pairs`) is the
  entire wiring; no Program-side gate remained (the `(dflash2_ || Exact)` graph-fit selector of T7
  already captures the `2 × kDFlashPairMaxDrafts = 8` pair width). GB10 rationale: decode is
  weight-LPDDR5X-bound (~273 GB/s shared), so verifying 2 lanes × 9 rows per round amortizes each
  weight read across twice the tokens; the single-lane fallback left the second lane plain-decoding.
- **Premise 3 holds with a caveat (fixed): `Q8SlicedK` was gated out for the whole Nvfp4 kind**
  even though the flag serves only the artifact's 303 Q8 weights (shared expert `[1280,2560]` /
  `[2560,640]` ×49, hyper-connections, MTP `[16384,2560]/[2560,6144]/[5120,2560]`, LM head
  `[248320,2560]`) and the linear dispatch ignores it on NVFP4 problems. Measured E2E at this
  artifact's decode widths (all T=1): **neutral** — the T4 winners were T≥5 gains; at T=1 the
  routes tie (`ckt ab` 3 rounds, WITHIN_NOISE on `[1280,2560]`, `[2560,640]`; LM head fast +1.0 %).
  Registered per the task; no regression (final plain decode equals baseline within σ).
- **Premise 1 is FALSE on GB10 (corrected by measurement): the fused NVFP4 GDN decode convolution
  is slower, not faster.** NSYS graph-node kernel sums (`--cuda-graph-trace=node`, plain decode,
  default vs all-fast-off, 9,324 = 259×36 GDN projection calls per trace):
  the fused epilogue kernel (`nvfp4_a16_gemv_kernel`, same `Nvfp4A16GemvSchedule<8,2,16,4,…>`
  instantiation) averages **120.2 µs/call** against the composed route's **96.2 µs** projection
  + **2.8 µs** `causal_conv1d_snapshot_decode` = **99.0 µs**: +21.2 µs per layer call,
  ≈ +0.76 ms/token, ≈ the measured **45.57 → 44.09 decode t/s (−3.3 %) at c = 1**
  (−0.7 % at c = 2; the op microbench's near-tie did not reproduce because its fixture is the
  `[16384,5120]` 27B parent, not Flash-Next's K=2560). The fused kernel is functionally correct
  (FP64 oracle, B=1..8) and removes 36 launches, but on the GEMV decode shape the state-window
  epilogue serializes inside the GEMV's CTA tails where the standalone kernel runs the same
  traffic in 2.8 µs fully parallel. Per the shipping rule (a numerics-changing fast path registers
  only where it is measured faster), **`GdnConvFused` for NVFP4 ships applied-but-unregistered**:
  the `forward.cpp`/wrapper wiring and the oracle test are delivered, and it runs under
  `--fast-path gdn-conv-fused` (re-measured on the final build: 44.16 vs 45.78, −3.5 % — the flag
  works, the kernel does not win). Its Q8 fusion (Dense8) is unaffected and stays qualified.
- **The doc's KV list (`FP16/BF16/FP8_E4M3/INT8/INT4`) does not exist as such**: the registry keys
  on `KvCacheStorage` (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2); all seven were registered, the
  same set every other Flash-Next configuration uses (none of these paths reads the KV format).

## 3. Subsystem breakdown

- `qwen4_exp/execution` (M): `fused_conv` admits NVFP4 (`[16384,2560]`-validated by the wrapper)
  so the implemented route executes when selected; decode-only (`batch.width <= 1`, no verify),
  prefill and verification stay composed.
- `qwen4_exp/program` (M): `fast_path_applies(Q8SlicedK/GdnConvFused)` = `Q8 || Nvfp4` (the NVFP4
  fused route is an implemented candidate, precedent: `expert-a4-residual-decode`); `kQualified`
  gains `DFlash2Pairs` × 14 and `Q8SlicedK` × 14; the block comment carries the measurements.
- `tests` (M): 116 assertions pass, incl. the new W4Dense matrix and the reproducibility pair:
  `{GdnConvFused}` alone stays request-invariant (the reproducible route keeps it as candidate);
  `+Q8SlicedK/DFlash2Pairs` flips `request_invariant()` false (excluded under `--reproducible`).
- `docs/maintainer/model-quality.md` (M), `tools/run_server.sh` (M): authority rows updated from
  measured values; no parallel new docs.

## 4. Verification

- Unit/contract: `infernix_qwen4_exp_fast_paths_test` **PASS**; `infernix_ops_recurrent_test
  gdn_input_proj` **OK** (includes NVFP4 `[16384,2560]` routes-form conv snapshot, B=1..8, in-place
  + distinct slots + graph replay, against the complete FP64 oracle); `gdn_input_proj_conv_snapshot`
  **OK**; `ctest -R infernix_cuda_stack_check` **Passed** (no allowlist change — zero new spills,
  none removed: no CUDA source touched, so no ratchet-down).
- E2E serving bench (medians of `-r 3`, decode out t/s; A = `66466fd26` binaries, final = this
  change; `spec round/fb` from the report):

| cell | plain A → final | mtp A → B | dflash2 A → final |
|---|---|---|---|
| c = 1 | 45.75 → **45.78** (fast_paths header carries `q8-sliced-k`) | 79.65 → 79.88 (noise) | 115.55 → 115.47 (noise) |
| c = 2 | 33.90 → **33.96** | 60.38 → 60.29 (noise) | **35.37 (rounds 6/723) → 72.03 (rounds 136/4), +103 %** |

  Intermediate build (all three registered) showed 44.09/33.65 plain — the fused-conv regression
  that drove the §2 correction; `--reproducible` (all fast paths off) reference: 45.57.
- Acceptance mapping: (1) both tests pass 100 % — **met**. (2) c=2 `--spec dflash2` executes
  2-lane pair speculation (`spec round/fb 136/4` across multi-lane rounds, acceptance 0.918) at
  **72.03 t/s vs the re-measured 35.37 non-paired baseline** (also > MTP 60.3) — **met**.
  (3) plain decode fused-conv on NVFP4 — **delivered as candidate, not shipped default**: the
  premise (fusion faster) was measured false (§2); kernel is oracle-correct within the
  `GdnConvFused` A16 criterion. (4) stack-spill gate green, zero new spills — **met**.

## 5. Accepted gaps / follow-ups

- The model-quality **joint gate (256-segment decode arms against the precise reference) was not
  run** for `dflash2-pairs`/`q8-sliced-k` on this artifact: registered on the task's directive,
  following the owner's 2026-10-24 dflash2-pairs ahead-of-gate precedent (and `q8-sliced-k`'s
  per-kernel op criteria, which it enforces). Recorded in both the registry comment and §6.7.
- NVFP4 fused-conv decode route: −21 µs/layer to recover; the lever is the conv-epilogue CTA-tail
  serialization (the Q8 fused route does not pay it), candidate flag keeps it measurable.
- `c = 4` DFlash2-vs-MTP on W4Dense was not re-measured (T7 dense8 evidence kept; the launcher's
  c = 4 server stays MTP). MTP and c=1-DFlash2 arms were measured on the intermediate build; the
  final build differs only by the fused-conv registration, which never fires in verify rounds
  (`batch.verify != nullptr`), so those cells carry over — not re-run.
- No `run_kv_matrix_bench.sh` campaign was run: the task's verification is the serving matrix
  above, and the change registers no kernel, only routes (prefill t/s unchanged, ~2,650–2,690).

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T2: `[2560,6144]` T=17..24 sliced-K retune (WarpsK 8) and `Nvfp4A4Pair` admission lowered 128 → 65 on the four Flash-Next NVFP4 shapes

Scope: `docs/tasks/20261030-performance-tasks.md` **Task T2** (P0) — the `[2560, 6144]` ×48
verification-width dispatch cliff and the `pair_min_tokens` crossover on the
`Qwen3.8-Flash-Next-Local-NVFP4-W4Dense` shapes `[2560,6144]`×48, `[16384,2560]`×36,
`[13952,2560]`×12, `[12800,2560]`×1.

## 1. Accounting

| Item | Value |
|---|---|
| Base commit | `8c4ebdbf4` (`dev-rebased-20261028`, clean tree) |
| Files changed | 8 modified; +19 / −12 |
| Stack-spill allowlist delta | none — `check_cuda_stack.py update` regeneration **byte-identical** (272/7692 STACK>0, unchanged); gate passes |
| Rebuilds (single-invocation ninja) | 6 (4 candidate bench iterations, unified final build, HEAD-baseline bench rebuilt for the interleaved A/B) |

| File | A/M | Change |
|---|---|---|
| `src/ops/linear/nvfp4/shapes/n2560_k6144.cu` | M | `select_a16`: `T≤20 → Nvfp4SlicedInstance<24,8,1,1>`, `T≤24 → Nvfp4SlicedInstance<24,8,2,2>` (both WarpsK=8, replacing `Nvfp4SlicedInstance<32,4,2,2>`); `pair_min_tokens` 128 → 65; rationale comment |
| `src/ops/linear/nvfp4/shapes/n16384_k2560.cu` | M | `pair_min_tokens` 65; `launch_nvfp4_n16384_k2560_split` pair gate `x.ne[1] >= 65` |
| `src/ops/linear/nvfp4/shapes/n13952_k2560.cu` | M | same two changes for `[13952,2560]` |
| `src/ops/linear/nvfp4/shapes/n12800_k2560.cu` | M | `pair_min_tokens` 65 |
| `src/ops/wrapper/gdn_input_proj.cpp` | M | `[16384,2560]` pair workspace capacity: `max_tokens >= 65` |
| `src/ops/wrapper/attn_input_proj.cpp` | M | `[13952,2560]` pair workspace capacity: `max_tokens >= 65` |
| `tests/ops/linear/test_nvfp4_a4.cpp` | M | `Nvfp4A4Pair` invocation arms +`T=65` (admission floor) and +`T=96` across the 8 pair-registered shapes |
| `tests/ops/test_gdn_input_proj.cpp` | M | pair sweep +`{65, 96}` on `[16384,2560]`; A4-tolerance criterion gate follows the admission: `Nvfp4A4Pair && tokens >= 65` |

Evidence: `/tmp/t2/base_*.log` (HEAD in-session baselines), `/tmp/t2/ab_exact_*.{r1..r5}`
(5 alternating HEAD/final bench-binary rounds, per-point medians), `/tmp/t2/ab_fast_*.{r1..r3}`
(3 alternating rounds per shape), smoke log `/tmp/t2/serve.log`.

## 2. Task-premise corrections / architectural rationale

- **The `[2560,6144]` cliff is real but the doc overstates it ~1.5x.** The doc quotes `T=17` at
  90.3 µs and `T=19..20` at 115 µs; the in-session HEAD baseline measures `T=17..24` at
  **67.4–69.8 µs** (+57.8 % over `T=16` 44.9 µs), not the 2.56x. The doc's command uses
  `--iters`, which this `infernix_linear_bench` does not accept (`--repeat`); its figures come
  from a different build/flush configuration and cannot be reproduced here. The stated mechanism
  held exactly: `Nvfp4SlicedInstance<32,4,2,2>` halves K-reduction warps (4 vs 8) on `K=6144`
  and the 8-warp tiles remove the cliff.
- **`T=25..32` was NOT slow.** The doc claims 63.3–110.5 µs there and asks to replace the
  dispatch; the in-session HEAD baseline measures **59.2–61.4 µs**, at or below every sliced
  candidate measured in that band (`<32,8,2,2>`: 61.3–67.5). The `Nvfp4A16MmaSchedule<32,32,128,16,16,2,2>`
  dispatch **stays unchanged** — replacing it was a measured regression, so the accepted fix
  spans `T=17..24` only.
- **The doc's candidate parameters don't exist as written.** `Nvfp4SlicedInstance<24,8,2,2>` /
  `<32,8,2,2>` resolve to derived `MinBlocksPerSm=1`, not 2 (the partial-reduction SMEM of the
  RowTiles=2 tiles dominates), and `Nvfp4A16MmaSchedule<32,64,128,16,32,2,2>` swaps
  WarpRows/WarpTokens relative to the template order. Measured winners instead:
  **`<24,8,1,1>` for 17..20** (Stages=1, RowTiles=1 — the only variant whose derived
  MinBlocksPerSm=2) and **`<24,8,2,2>` for 21..24**; `<32,8,2,2>` was worse everywhere below its
  capacity, `<24,8,2,1>` lost ~1 tile at 17..20.
- **Pair premise holds in direction, magnitude is 13–30 % not 18–31 %.** Interleaved
  3-round medians (baseline binary rebuilt at HEAD via stash) give `T=72..120` reductions
  `[16384,2560]` −15.1…−28.2 %, `[13952,2560]` −13.9…−29.7 %, `[2560,6144]` −19.8…−23.6 %,
  `[12800,2560]` −13.4…−26.7 %. The doc's single-run figures overstated the low end.
- **Clock-state drift matters at these widths.** Identical dispatch measures `[2560,6144]`
  `T=23..24` at 57.2 µs on a cold GPU (candidate-C link) and 61.3 µs warm (final link; the two
  builds carry the same dispatch code for that band); unchanged bands drift the same way, so all
  headline numbers here are **interleaved A/B medians**, and the doc's absolute `≤58 µs` bar is
  only interpretable mod clock state (§4/§5).

## 3. Subsystem breakdown

- `linear/nvfp4/shapes` (M): one dispatch-table retune (zero new kernel source — all four
  instances were already instantiated families), plus the four `pair_min_tokens` constants.
  The split launchers' `x.ne[1] >= 65` gate is the wrapper-route twin of the same admission;
  the A4 TMA machinery needed no change: `allocate_nvfp4_a4_workspace` already pads the scale
  planes to `Tiled128`, the row-scaled residual quantizer zeroes padding scale rows, the A-code
  TMA box reads zeros past `p.tokens`, and every epilogue store is guarded by
  `token_begin + t < token_count` — verified by reading those three sites, then confirmed
  numerically at `T=65/96/127` (not multiples of 128) against the FP64 oracle.
- `wrapper` (M): the two pre-allocated pair-workspace capacity gates; every other consumer
  (`nvfp4_linear_workspace_capacity_bytes`, engine planning, `mtp.cpp`) reads
  `pair_min_tokens` dynamically, so no further sizing sites exist (grepped).
- `tests` (M): pair arms at the exact new admission floor (`T=65`) and mid-band (`T=96`); the
  gdn wrapper's A4-tolerance selection tracks the admission constant so `T=65..127` pair runs
  are judged under `kGdnInputProjA4Tolerance`, not the stricter A16 criterion they met while
  dispatched to A16.

## 4. Verification

- Oracles: `ctest -R "infernix_linear_nvfp4_a16_test|infernix_linear_nvfp4_a4_test|
  infernix_gdn_input_proj_test|infernix_attn_input_proj_test"` **4/4 Passed** (serial), plus
  `infernix_ops_linear_test linear_nvfp4_a16 --shape 2560 6144` **OK**. `infernix_cuda_stack_check`
  **Passed**; allowlist regeneration byte-identical → zero new spills, none eliminated.
- Serve smoke on the real W4Dense artifact (`tools/run_server.sh`, this build): chat requests
  complete (prompt 378/57, prefill 381.8 tok/s, decode 58.1 tok/s, MTP accepted) — no workspace
  or graph-planning exception from the resized pair capacities.
- `[2560,6144]` exact, **5 alternating HEAD/final rounds, per-point median (µs)** — fixed band:

| T | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|
| HEAD | 67.4 | 67.4 | 67.5 | 67.4 | 69.5 | 69.5 | 69.5 | 69.5 |
| T2 | **55.1** | **55.1** | **55.1** | **55.1** | **53.1** | **53.1** | **61.3** | **61.3** |
| Δ | −18.2 % | −18.2 % | −18.3 % | −18.2 % | −23.6 % | −23.6 % | −11.9 % | −11.8 % |

  Unchanged bands `T=25..40` read −0.2…+3.9 % (±1 tick A/B noise floor; no regression — those
  schedules are byte-identical dispatch). Cold-state run of the identical-dispatch candidate
  build: 17..20 = 49.0–51.0, 21..24 = 55.1–57.2 µs.
- Pair admission, **3 alternating rounds** (`--routes fast`, medians): every `T=72..120` point
  on all four shapes drops −13.4…−29.7 % (tables in §2); `T=64` and `T≥128` are neutral
  (identical dispatch in both arms, |Δ| ≤ 3.2 % = noise). `T=65/96/127` pass the A4-pair oracle
  on all 8 pair shapes and the gdn wrapper bit-matches `linear` there.

## 5. Accepted gaps / follow-ups

- **Acceptance bar 1 is met at `T=17..22` (≤55.2 µs warm, −18…−24 %) but `T=23..24` measures
  61.3 µs warm (−11.9 %), 57.2 µs cold** — the absolute `≤58 µs` sits exactly in the clock-state
  drift band. No candidate schedule beat the shipped one at 23..24 (tile-24 full tile,
  32×32 MMA, `<32,8,2,2>` and the Stages/RowTiles variants all plateau at 57–64 µs across the
  two machine states): padding/reduction quantum, not a routing lever. The cliff itself
  (T=16→17 inversion) is eliminated; the ladder is monotone non-decreasing across 16→32 in the
  warm A/B (44.9 → 55.1 → 61.3 → 61.3).
- Acceptance bar 2's stated −18…−31 % becomes **−13.4…−29.7 %** under interleaved medians; the
  two weakest points (`[12800,2560]` T=72 −13.4 %, `[13952,2560]` T=72 −13.9 %) are where HEAD's
  A16 64-token MMA band was already fastest. Reported as measured, not as met.
- No end-to-end model benchmark: the acceptance criteria are op-level, the kernels and route
  registrations are unchanged (only an admission constant + one dispatch table), and the pair
  route's numerics profile at 65..127 is the already-qualified `Nvfp4A4Pair` two-term row-scaled
  form; the serve smoke covers engine sizing. A pp/tg A/B on W4Dense is cheap follow-up if the
  campaign owner wants E2E attribution of the ≈2–3 ms/pass derived saving.
- `attn_input_proj`'s `[13952,2560]` pair path has no dedicated wrapper-level test (same gap as
  before this change); covered by the linear pair test (same kernel, same schedule) and the
  capacity contract. The four 27B-only NVFP4 pair shapes keep `pair_min_tokens=128` — outside
  the task's Flash-Next scope.

---

# GB10 (sm_121a) Branch Changelog — 2026-10-30 — T3: `hyper_connection_mix` at speculative and medium widths (T=2..16)

Scope: `docs/tasks/20261030-performance-tasks.md` **Task T3** (P0) — `hyper_connection_mix`
(Qwen3.8-Flash-Next, S=4 streams x H=2560, rank 320; down [324,10240] + up [10240,320]
Q8_G32_FP16 RowSplit or BF16 contiguous, 7.70 MB weights read per call, 98 calls per forward
pass) at the execution lanes' column counts T=2..16 on GB10. Evidence host: this GB10
(48 SMs, UMA LPDDR5X), CUDA 13; cold-L2 `infernix_hyper_connection_bench` medians, interleaved
rebuilt-HEAD A/B; `bench/run_kv_matrix_bench.sh` default gate (fp8 KV x mtp x iq3s/dense8/nvfp4).

---

## 1. Accounting

| Item | Value |
|---|---|
| Base commit | `08fe38f35` (`dev-rebased-20261028`, clean tree) |
| Files changed | 7 modified + this changelog section |
| Diff | +123 / -75 |
| Stack-spill allowlist delta | none — `infernix_cuda_stack_check` pass, no new STACK/LOCAL symbols, nothing eliminated so `cuda_stack_allowlist.json` untouched and `update` was not run |
| Rebuilds (single-invocation ninja) | ~12 (4 dispatch-band probe binaries `thr8`/`thr4`/`thr2`/`ext1`, 1 up-collapse-boundary probe, 3 candidate builds, 2 oracle-test iterations, HEAD binary rebuilt after the final tree for the E2E baseline arm, final consistent rebuild) |

| File | A/M | Change |
|---|---|---|
| `src/ops/hyper_connection/hyper_connection_mix_fused.h` | M | single `kHcMixFusedMaxColumns = 16` replaced by measured per-codec admission `kHcMixFusedMaxColumnsQ8 = 3` / `kHcMixFusedMaxColumnsBf16 = 4` (comments cite the crossover measurements); `kHcMixFusedMaxColumns = 4` kept as the shared sizing cap for arrays/workspace/attributes |
| `src/ops/hyper_connection/hyper_connection_mix_fused.cu` | M | `hc_up_collapse_v4_kernel` static shared arrays -> one dynamic `extern __shared__` slab (static SHARED 38,144 B -> 0; per-call dynamic <= 9,280 B, `cudaFuncAttributeMaxDynamicSharedMemorySize` set in the `configured` launcher); `hc_down_partials_v4_kernel` per-column-CTA `ColumnGrid` template arm and its `INFERNIX_TMP_HC_DOWN_COLUMNS` env branch removed (measured slower); `hc_mix_fused_supported` gates per codec |
| `src/ops/hyper_connection/hyper_connection_up_collapse.cu` | M | `columns <= 16` u-fusion upper gate kept after measurement; speculative comment replaced by the measured per-width verdict (separate linear+collapse beats this route at 5 and 9..16; it wins only at 6..8, so fusion starts at 17) |
| `src/core/temporary_switches.cpp` | M | `INFERNIX_TMP_HC_DOWN_COLUMNS` registration removed (superseded path deleted per change-consistency) |
| `tests/ops/test_hyper_connection.cpp` | M | oracle tolerances keyed to the codec's fused band; column-invariance contract rewritten to measured reality (fused bits bit-exact within the band; composed route asserted only within one linear capacity tier — same-tier 9-vs-16 bit-exact — cross-tier assertions removed after they failed); graph-capture checks at T=9 and T=3 |
| `tests/ops/tests.cmake` | M | env-var `infernix_hyper_connection_columns_test` arm removed; header comment states the fused bands |
| `bench/ops/hc_mix_fused_bench.cu` | M | fused arm / check guard / weight-construction assertion use the per-codec `fused_max` |

## 2. Task-premise corrections (all measured on this tree)

1. **The T=4 cliff does not exist.** The task doc claims HEAD T=4 = 61.04 us against a T=3 ~39 us
   cliff and sets the bar <= 48.5 us. Rebuilt HEAD measures T=4 at 44.96-45.22 us cold — the bar was
   already met; the doc's T=17 = 103.23 likewise measured 92.81. (Matches the stored
   "task-doc baselines stale" finding: those figures were hot-state/wrong-build artifacts.)
2. **The claimed ColumnGrid win is inverted.** The doc credits the `INFERNIX_TMP_HC_DOWN_COLUMNS`
   per-column-CTA experiment with -20.9%; interleaved A/B measured that arm *slower* than the
   default column loop (T=2 +3.1%, T=3 +4.6%, T=4 48.46 vs 44.96). The default loop shipped and the
   env hack was deleted.
3. **The SMEM/occupancy premise is inert and its REG target is impossible.** `hc_up_collapse_v4_kernel`
   is register-limited to 1 CTA/SM at its natural REG (168-202 by instantiation), so shrinking its
   38,144 B static reservation cannot add a block; and the doc's "verify REG <= 64" cannot hold —
   forcing `__launch_bounds__(_, 2)` (128-reg ceiling) compiled `STACK:128` stack spills in *every*
   v4 instantiation (cuobjdump), which the ratchet gate forbids. The SMEM conversion to dynamic was
   still done (correct sizing per call, no 38 KB per-SM reservation) but is E2E-neutral; the kernel
   keeps `__launch_bounds__(kUpV4Threads, 1)` with a rationale comment.
4. **The named test target does not exist.** `infernix_ops_state_recurrent_test` is not a target in
   this tree; the `hyper_connection` CTests register in `infernix_ops_projection_test`.
5. **The real lever is the SIMT-vs-MMA crossover, not the kernels.** Interleaved probe builds
   moving `kHcMixFusedMaxColumns` (8/4/2 + up-collapse widening) show the composed MMA route beats
   the fused SIMT mix from T=4 for Q8 (T=4 -2.9% down to T=16 -24.6%) and from T=5 for BF16 (fused
   still wins T=4 by +4.7% because the BF16 linear reads 2x the bytes). The T=16 > T=17 inversion
   was real and is fixed by dispatch: fused band -> composed 4/5..16 -> u-fused up+collapse >= 17.
   Widening `hc_up_collapse_supported` to >1 was probed and rejected (-3..-4% at 5 and 9..16; it
   wins only at 6..8).
6. **The composed route is not column-invariant across linear capacity tiers.** 5-vs-16 columns
   differ bit-wise for both codecs (the [10240,320] linear swaps its 8- vs 16-column SlicedK
   ladder), and 16->17 changes Q8 bits (16->32 tier) while BF16 keeps them. The doc's
   "swap-fused-mix preserves bit-identical outputs across T" oracle contract therefore only
   survives same-tier comparisons; the test now asserts that measured reality.

## 3. Subsystem breakdown (why it diverges / mechanism)

- **`src/ops/hyper_connection` (M).** Dispatch, not arithmetic: per-codec fused admission
  (GB10-measured crossover 3 Q8 / 4 BF16, replacing the inherited 16 that left every speculative
  lane — MTP c=4, DFlash2 pair c=8/9, DFlash2 single c=16 — on the losing side). v4 up kernel:
  dynamic SMEM slab (`m`/`u`/`inv` over one `extern __shared__` buffer, index layout unchanged ->
  bit-identical outputs inside its remaining band); v4 down kernel loses the measured-slower
  per-column grid. No math, rounding order, or oracle semantics changed in any kernel.
- **`src/core` (M).** Temporary-switch registry entry deleted with its path (no backward-compat
  fallback per repo policy).
- **`tests/` (M).** Contract rewritten to the measured invariance classes (fused bit-exact,
  composed same-tier bit-exact); the env-var test arm deleted. 2/2 hyper_connection CTests pass.
- **`bench/` (M).** The fused/composed A/B bench follows the per-codec band so both routes stay
  measurable on GB10.

## 4. Verification

- **Op-level interleaved A/B** (`/tmp/t3/bench_head` vs `bench_final2`, cold L2, 3 rounds, medians;
  unchanged-dispatch control points T=17..64 drift +/-0.4%):

  | T | HEAD us | final us | delta | | T | HEAD us | final us | delta |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 31.12 | 30.98 | -0.4% | |(fused, no regression)| 9 | 76.97 | 64.65 | -16.0% |
  | 2 | 35.47 | 34.77 | -2.0% | | | 12 | 94.26 | 70.32 | -25.4% |
  | 3 | 41.14 | 39.66 | -3.6% | | | 16 | 113.65 | 79.87 | -29.7% |
  | 4 | 45.12 | 43.86 | -2.8% | | | 17 | 92.81 | 92.81 | ~0 |
  | 5 | 53.32 | 44.40 | -16.7% | | | 24/32/48/64 | — | — | +/-0.4% |
  | 6 | 59.84 | 52.82 | -11.7% | | | | | | |
  | 8 | 67.84 | 57.24 | -15.6% | | | | | | |

  Acceptance: T=4 43.86 <= 48.5 us pass; T=8/12/16 monotonic (57.24 < 70.32 < 79.87) and T=16 79.87
  <= 98 us with the T=16 > T=17 inversion gone; T=1 within noise.
  BF16 codec: T=4 -0.1% (band restored), T=5 -3.5%, T=6 -10.7%, T=8 -18.8%, T=12 -44.8%,
  T=16 -53.1%; T=2 +3.0% / T=3 +0.7% inside the arms' own 56-61 us spread (noise; both stay fused).
- **CTests**: `ctest -R hyper_connection` 2/2 pass (oracle incl. new invariance/graph checks);
  `infernix_cuda_stack_check` pass; `infernix_hc_mix_fused_bench --check` both codecs pass
  (max |fused-composed| within the FP32-vs-BF16-intermediate tolerance, 0.0 diffs reported on
  inject).
- **Resources** (`cuobjdump --dump-resource-usage` on `infernix-serve`, per exact symbol): v4
  down/up kernels `REG:168-202 STACK:0 LOCAL:0`; `hc_up_collapse_v4_kernel` static `SHARED:0`
  (was 38,144 B), dynamic <= 9,280 B per launch.
- **E2E KV matrix** (`bench/run_kv_matrix_bench.sh` default gate fp8 x mtp, final binary vs
  rebuilt HEAD, back-to-back arms; decode t/s median per test):

  | model | tg128 | tg256 | pp4096+tg128 | pp4096+tg256 |
  |---|---|---|---|---|
  | iq3s | +1.5% | +1.4% | -2.7% | -10.4% |
  | dense8 | -4.9% | -7.2% | +1.7% | +1.9% |
  | nvfp4 | -2.8% | +11.1% | +3.2% | +7.8% |

  All 12 decode points: geomean **-0.1%** (range -10.4%..+11.1%), no arm failed. Per-point deltas
  are *not* clean performance deltas: the dispatch change alters the mix's output bits at T=4..16
  (Q8) / T=5..16 (BF16), so each arm follows a different accepted-token trajectory (MTP
  acceptance rates diverge both ways, e.g. iq3s tg128 0.802 -> 0.738, nvfp4 tg256 0.739 -> 0.814),
  which moves decode t/s more than the ~2.5% of step time the mix occupies. The Op-level interleaved
  A/B above is the performance evidence; E2E shows the change is gate-clean and trajectory-divergent.

## 5. Accepted gaps

- The T=4 <= 48.5 us bar was already met at HEAD (44.96-45.22); the shipped T=4 margin (-2.8%) comes
  from the route swap, not a kernel win.
- BF16 T=2/T=3 A/B deltas (+3.0%/+0.7%) are inside the arms' own spread and those widths keep the
  fused route unchanged — not claimed as regressions or wins.
- The reproducible-route bits change at Q8 T=4..16 and BF16 T=5..16 (FP32 -> BF16-rounded
  intermediates of the composed route; cross-tier Linear ladders). Output stays request-invariant;
  the composed route is the pre-existing production path (it already served T>=17 and BF16 T>=5 at
  HEAD), so no new kernel ships unqualified. Per Op-qualification policy the fused-route kernels
  inside the shrunken band are unchanged and bit-identical there.
- E2E acceptance-rate divergence above means this matrix cannot certify per-point E2E perf; it
  certifies the gate runs clean on all three artifacts. A model-quality sign-off on the changed
  speculative-path bits is the owner's call, consistent with the fast-path policy.
- The down/up kernels themselves were not optimized (bandwidth floor ~7.7 MB/call at ~280 GB/s
  caps T=1 near 27-31 us; T=1 measured 30.98). Kernel-level headroom (e.g. a T>=4 SIMT collapse
  that beats the MMA linear) remains open work.
- The GB10-calibrated band constants are shared with the 5090 config on this branch, as every other
  band constant in this file already is.

# GB10 (sm_121a) Branch Changelog — 2026-10-30 — T4: Wire & retune Flash-Next shared-expert / DFlash2 Q8 projections (`[1280,2560]`, `[2560,12800]`) and the last three NVFP4 `/shapes/` on GB10

## 1. Accounting

- Base commit: `7b96b64e0` (`origin/main` rebase line, `dev-rebased-20261028`); HEAD before this task's edits: `7b96b64e0`; worktree clean at start.
- Files changed (9 tracked, 1 added; `cuda-kernel-toolkit/` intentionally excluded — untouched):
  - `src/ops/linear/q8/shapes/n1280_k2560.cu` (M) — GB10 exact + fast tables
  - `src/ops/linear/q8/shapes/n2560_k12800.cu` (A) — new DFlash2 `feature_projection` shape
  - `src/ops/linear/q8/shapes/q8_invariant_families.cu` (M) — `[2560,12800]` family rung set
  - `src/ops/linear/q8/q8_shapes.h` (M) — `select_q8_n1280_k2560_fast`, `select_q8_n2560_k12800(_fast|_invariant)` declarations
  - `src/ops/linear/q8/q8_dispatch.cpp` (M) — fast wired for `{1280,2560}`; `{2560,12800}` entry (exact/fast/invariant)
  - `src/ops/linear/q8/sources.cmake` (M) — new shape registered
  - `src/ops/linear/nvfp4/shapes/n5120_k6144.cu` (M), `n5120_k17408.cu` (M), `n34816_k5120.cu` (M) — A16 retune + provenance headers
  - `tests/ops/linear/test_q8_a16.cpp` (M) — `[2560,12800]` geometry, fast seams, both invariance lists, invariant-policy branch
- Allowlist delta: none — `check_cuda_stack.py update` output is byte-identical to the committed allowlist; the ratchet gate passed unchanged (zero new spills, zero removed).
- Rebuild count: 3 unified Ninja invocations (shared device link); header edits followed by `find src/ops/linear/q8 -name '*.cu' -exec touch {} +`.

## 2. Task-premise corrections / architectural rationale

- **The doc's "wire the dead `SlicedK_*` defs into the exact selector" premise fails the shipped column-invariance contract.** The `[1280,2560]` shape carries a 48-column default-route invariance contract (`q8_a16_column_invariance`). The route sweep's per-width gates cannot certify that contract across schedules: `bit_exact` checks a candidate at its own width against the pre-change route, and `t_invariant` checks a candidate against *itself* at width 1 — neither chains candidate-vs-candidate across widths or schedules. The first shipped table (swept sliced tiles T=1..33 + the SIMT pair T=34..48) passed every sweep gate yet failed the contract at T=34..48 (8–9 tail outputs differ): SIMT and sliced-K reduce a column differently. Fix: the exact table delegates its whole contracted band to one fixed sliced-K family — `[1280,2560]` T≤48 and `[2560,12800]` T≤32 run `q8_invariant_families` ladders (one K-warps-8 tree at capacities 8/16/32), the same device every other Flash-Next shape uses. The T-dependent 4-K-warp winners stay on the fast route (bits depend on call width there).
- **`[2560,12800]` had no kShapes entry at all — confirmed.** DFlash2 `feature_projection` fell to the runtime-shape generic route. The SIMT variants `simt_r1_t4_w8`/`r8_t5`/`r8_t6`/`r8_t8` each reproduce the fallback's bits *at their own width* (sweep b1t1) yet differ from *each other* by 1–6 tail outputs across widths — the new 8-column contract test caught exactly that, and the family ladder replaced them.
- **NVFP4 premise held**: the three files carried the 5090 fit; A16 retuned under `infernix_nvfp4_route_sweep --policy a16` (adoption gate = numerical agreement against the oracle; the NVFP4 A16 tables are oracle-bound, so no bit preservation is required — production itself is not t-invariant on this family). A4 admissions and routes unchanged.
- GB10 invariants kept: 1:1 FP32 accumulation routes untouched (all adopted Q8 tiles are the F32-acc sliced-K/MMA families already in production use); decode bands stay DRAM-streaming-optimized (the `[2560,12800]` 33.2 MB weight is one UMA pass: 156–168 us ≈ 198..213 GB/s against the 246 GB/s sustained ceiling at T≤16).

## 3. Subsystem breakdown

- **Q8 shared-expert gate `[1280,2560]` (49x/pass)** — `select_q8_n1280_k2560`: family (cap 8/16/32 Direct rungs) T≤48, then the pre-existing `mma_r64x32_t64_k128_a1`/`r64_t96`/`r64_t128` tail; dead `Stream<2,1,4,…>` and orphan `SlicedK_T8/T16` defs removed. `select_q8_n1280_k2560_fast`: family @T=1, `sliced_r16_t16_w8_s2` ≤16, `sliced_r32_t16_w4_s2` ≤32, `sliced_r16_t16_w4_s2` ≤44, `sliced_r32_t16_w4_s2` ≤64, exact tail beyond (sweep: 19.2–19.4 us over T=17..32, 25–28 us over T=33..64 where the pre-change table paid 56..80).
- **Q8 DFlash2 feature_projection `[2560,12800]`** — new `select_q8_n2560_k12800`: family ≤32, `mma_r64x16_t128_k128_a2` ≤48 (211–214 vs fallback 265–270; fallback T=12 cliff 403 → family band), `mma_r64x16_t48_k128_a1` ≤64 (194–199), then the `[2560,6144]`-pattern tail (unmeasured, disclosed). `select_q8_n2560_k12800_fast`: `simt_r1_t4_w8` @T=1 (155.5 vs family 162–164), `sliced_r16_t16_w8_s2` ≤16 (162–166), `sliced_r16_t32_w4_s2` ≤32 (182–184), exact beyond. Family: `static_ladder<G2560x12800, Rung<8,8,2,1>, Rung<16,8,2,1>, Rung<32,8,2,1>>` (K=12800 = 25×512 static-split legal).
- **NVFP4 A16 retunes** — `n5120_k6144`: RowTiles-2 sliced tiles + K64 64-row MMA at T≤64 + T≤256 `64,64,64,32,16,2,2` band; `n5120_k17408`: T≤16 w8 RowTiles-2 sliced, T≤32 w4 tiles, T≤192 `64,96,128,64,…`, T≤256 `128,64,64,64,…`, T≥257 `64,128,64,64,16,2,2` tail replacing the t256 5090 tile; `n34816_k5120`: 32-row sliced tiles T≤32, `32,64,128,16,16,1,3` at T≤64, `64,64,128,32,16,1,2` at T≤128, `128,64,64,64,16,2,2` prefill band. C2/C4/C5 SIMT schedules and the 5090 t256 tail deleted; A4 route tables and `uses_a4` admissions byte-identical.
- **Test** — geometry `{2560,12800,383U}` + fast-seam list (T=1..4096) + default-route 8-column invariance + A16ColumnInvariant invariance entry + invariant-policy conformance (T=1..64 ladder seams).

## 4. Verification

- Oracles (unified binary, current build): `linear_q8_a16` **OK** (33-geometry conformance incl. fast seams + default invariance to 48 cols at `[1280,2560]` and 8 cols at `[2560,12800]` + policy-invariance suite), `linear_nvfp4_a16` **OK**, `linear_nvfp4_a4` **OK**.
- Ratchet gate: `ctest -R infernix_cuda_stack_check` Passed (1.45 s); allowlist update byte-identical (zero new spills).
- `ckt shapes list`: q8 32/32 and nvfp4 15/15 report `origin=GB10` (criterion 1: 100%).
- Q8 A/B vs the HEAD binary (cold-L2, `--repeat 10`, alternating arms 3 rounds, per-point medians; ckt ab interleaved aggregate agrees):

  | shape × route | mean T=1..32 | range | worst point | ckt ab aggregate |
  |---|---|---|---|---|
  | `[1280,2560]` exact | **−54.0%** | −5..−71% (T=1 +0.2% flat) | T=1 +0.2% | −60.7% (2.54x) MEANINGFUL_GAIN |
  | `[1280,2560]` fast | **−55.8%** | −10..−72% | none >+2% | −58.9% (2.44x) MEANINGFUL_GAIN |
  | `[2560,12800]` exact | **−31.2%** | −12..−41% | none >+2% | −33.6% (1.51x) MEANINGFUL_GAIN |
  | `[2560,12800]` fast | **−29.5%** | −14..−41% | none >+2% | −28.6% (1.40x) MEANINGFUL_GAIN |

  HEAD `fast` equals HEAD `exact` on both shapes (both were `fast=nullptr`), so the fast deltas are the admission effect.
- NVFP4 (single-process interleaved sweep, argmin-vs-pre-change production, µs medians): `[5120,6144]` T=17..32 −13.4..−20.8%, T=192 −26.4%, T=16 −4.5%, rest ±0.4..2.7%; `[5120,17408]` T=2..16 −4.1..−10.6%, T=17..32 −18.9..−36.5%, T=192 −23.7%, T=384 −43.9%, T=512 −21.9%; `[34816,5120]` T=3 −9.8%, T=17..32 −6.1..−8.6%, T=64 −7.0%, T=128 −8.6%, T=192/256/384 −36.9/−16.6/−37.3%, T=512 −13.5%. At every width the tables keep the pre-change tile where no candidate beat it (no adopted point is slower than the pre-change table).
- `git diff --check` clean. No commit made.

## 5. Accepted gaps / follow-ups

- The `[1280,2560]` exact **default-route bits change at T=2..48** (family replaced the streamed/SIMT family; family was already the qualified A16ColumnInvariant route for this shape since 2026-10-29). Reproducible-route bits for any model using the shared-expert gate shift accordingly; request-invariance holds. `[2560,12800]` bits change vs the generic fallback. Model-quality sign-off per fast-path policy is the owner's call.
- `[2560,12800]` T=17..32 runs the family cap-16/32 rungs, adopted from the `[2560,6144]`/`[2560,2560]` pattern rather than a shape-specific rung sweep — A/B confirms −27..−34% vs HEAD at those widths, but a rung-level sweep could still find more. The T>64 tail is a shape-pattern ladder, unmeasured here.
- `[1280,2560]` T=1 exact is flat vs HEAD (family 16.6 vs streamed 17.8 us in the sweep; A/B +0.2% within noise): the sweep's T=1 sliced winner was not adopted on the exact route because a lone fast T=1 tile would split the family's bit contract; the fast route carries the T=1 win (−10%).
- NVFP4 A4 routes, admissions, and the pair/TMA ladders are untouched (T4 scope is the A16 tables); `[5120,17408]` T=96 and `[34816,5120]` T=96 kept the production MMA tile — the sweep's argmin there did not beat it beyond noise.
- No E2E KV-matrix run for T4 (Op-level + oracle contract evidence; no engine route or admission changed beyond the two Q8 shape entries).

---

# GB10 (`sm_121a`) Branch Changelog — 2026-10-30 — T5: Wire the `Qwen4Exp` hybrid prefix-cache cost model (`prefill_signature` + resolved `ContextPrefillCost`); GB10 prefill calibration measured and held

## 1. Accounting

- Base `ae716a49b` (post-T4). `src/models/qwen4_exp/measurement.h` (A),
  `src/models/qwen4_exp/measurement.cpp` (A), `src/models/qwen4_exp/program_sources.cmake` (M),
  `src/runtime/engine/qwen4_exp_instance.cpp` (M),
  `src/runtime/engine/context_cache/context_cost.h` (M),
  `src/runtime/engine/context_cache/context_cost.cpp` (M),
  `src/runtime/engine/model_instance.cpp` (M, helper moved),
  `bench/context_cost/model_context_fixture.cpp` (M), `tests/runtime/test_context_cost.cpp` (M).
  7 code files changed (+222/−38), 2 new files (106 lines), plus this changelog. CUDA sources touched:
  none, so no `.cu.o` rebuilds and `tools/cuda_stack/cuda_stack_allowlist.json` is unchanged (zero new
  spills, zero eliminated; the ratchet was not `update`d).
- Evidence: `profiles/bench/20261030-t5-prefill-r9.json` (cold protocol),
  `profiles/bench/20261030-t5-prefill-warm-r9.json` (warm protocol), the earlier campaign's
  `profiles/bench/20261029-t5-f4local-prefill-rejected.json`, and the two serve startups
  `profiles/bench/20261030-t5-serve-external-preset.log` /
  `...-t5-serve-unbound.log` (`profiles/` is gitignored, so these are local evidence, not a diff).
- Rebuilds: 4 Ninja invocations (one object probe, two unified target builds, one test-only relink);
  the ~56 s device link ran once per unified build.
- NOT part of T5 and left untouched/unstaged: `tools/generate_tasks.sh` (orchestrator tooling,
  modified in the working tree outside this task).

## 2. Task-premise corrections (each checked against the tree or a measurement)

- **The named files, targets and flags do not exist here.** The live contract is
  `src/runtime/engine/context_cache/context_cost.{h,cpp}` (not `src/runtime/context_prefill_cost.*`),
  `tests/runtime/test_context_cost.cpp` (not `test_context_prefill_cost.cpp`),
  `infernix_context_cost_test` (not `infernix_context_prefill_cost_test`),
  `infernix_context_cost_bench --artifact` (there is no `infernix-bench-prefill-cost`, no `--model`),
  and `--context-cost-presets` (not `--context-cost-table`). There is no `--kv-dtype`: the fixture
  deliberately prices the canonical route (`canonical_kv_cache: bf16`, no speculation — recorded in
  both reports), and `prefill_signature` does not key on the runtime KV dtype, so a per-KV-dtype
  table could not bind even if the flag existed. Measured on the canonical route.
- **No `Program::set_hybrid_cost` was needed.** `models::qwen4_exp::ProgramOptions::prefix_cost` is
  already this architecture's cost seam: `prefix/prefix_program.cpp` installs it into the index at
  construction and `program/program_impl.h:6330` re-rates the restore link from the rate it measures
  at startup. The loader resolves the table *before* it constructs the Program, so it layers the
  resolved coefficients into the options. A post-construction mutator would be a second owner of the
  same state with no caller that needs it. The task's L139–145 mirror pattern is preserved in
  structure (resolve → publish in `LoadSummary` → apply when the cache is on).
- **The "GB10 page-setup term" premise does not survive measurement.** Three runs of the same grid
  (2026-10-29 reps 5; today reps 9 before and reps 9 after a whole-grid warm-up) reproduce the same
  residual structure: +47/+52/+51 ms at `b1024-s512`, `b2048-s256`, `b2048-s1536`, +24..+28 at
  `b4096-s128`, −26 at `b4096-s1024`, −19 at `b0-s128`, against a per-point MAD of 3–5 ms. So it is
  neither noise nor a per-reused-token setup price: fitting `prefix_tokens`, `prefix/chunk` pages, or
  `calls × pages` — unconstrained, i.e. an upper bound on what a non-negative fit could achieve —
  only drags the pair coefficient to zero and leaves both outliers (held-out p95 0.186 → 0.158 on
  yesterday's grid, still rejected). The term is not identified by this grid.
- **What the outliers actually are:** the fixture prices a suffix as `T(b+s) − T(b)` of two full
  roots, and on this engine a continuation call is not the same unit of work as a standalone call of
  that width — a 512-token tail call inside a longer request measures 377 ms where the standalone
  512-token root measures 329 ms (+48), while a 1024-token tail measures ~26 ms *less* than its
  standalone equivalent. No linear term in the contract's features
  (`chunks, suffix_tokens, attention_pairs, vision_*`) expresses a sign change like that.
- **Cold-vs-warm page setup is real but was already excluded.** A length's first visit costs +8..15 ms
  over its warm samples; with 9 repetitions the median is warm. Warming *every* grid length before
  sampling (the fixture change made here, matching the transfer suite's per-point warm-up) moved
  absolute latencies 1–3 % and left every residual where it was. The gate outcome is identical before
  and after: 0.1797 rejected → 0.1808 rejected.

## 3. Subsystem breakdown

- `models/qwen4_exp/` (A): `prefill_signature(const Model&)` — SHA-256 over a facts document
  (`implementation = qwen4_exp-prefill-1`, architecture, widths/layer counts/`public_token_count`,
  the per-layer mixer sequence, attention/rope/gdn/moe/hc/qsa/ple geometry, GGUF-import and lossless
  flags, Vision geometry) plus the sorted `text/`+`vision/` weight inventory (role, shape, activation
  policy, and per part: parent index, span, format, layout, shape, padded columns, group size).
  Derivation is deliberately per-architecture rather than shared with `qwen3_5`: a digest's bits are
  the contract with every table already bound to it, so shared code would let a change for one
  architecture silently rekey the other's bindings. Draft/proposal weights stay out of the key (the
  baseline prices primary reconstruction), as in `qwen3_5`.
- `runtime/engine/qwen4_exp_instance.cpp` (M): computes the signature once the weights are bound and
  resolves `ContextCostIdentity{hardware_class, prefill_signature}` unconditionally (with
  `--context-cost-presets`); publishes `LoadSummary::prefill_signature` and
  `LoadSummary::context_cost`, both previously empty on every Flash-Next artifact — the startup log,
  the request log's `context_cost` object and `run_serve_corpus.py`'s canonical-signature check had
  nothing to report. With the prefix cache on, `layer_hybrid_cost` lays the resolved coefficients
  over the estimates the loader already set, **only over the half the resolution actually sourced**:
  an unbound configuration keeps this architecture's estimates rather than the generic estimate,
  which is one measured 27B groupwise artifact's average and prices Flash-Next's chunk term 3.4× low
  here. A measured *transfer* half still overrides only the transfer terms, and
  `GenericDefault` prefill keeps the artifact's zeroed Vision pricing.
- `runtime/engine/context_cache/` (M): `layer_hybrid_cost(base, model)` holds the ns→s and
  Q32→rate conversion over a caller-configured index cost model (an unmeasured term and any field the
  contract does not carry keep the base), with `hybrid_cache_cost()` as the wrapper over the index's
  own defaults. It moves out of `model_instance.cpp`'s anonymous namespace so both loaders share one
  conversion; Qwen3.5's numbers are unchanged because the wrapper starts from the same default model
  the deleted copy did. `validate_prefill`'s "has no Vision cost" rule became "has a partial Vision
  cost": the old rule refused *any* Text-only artifact's calibration (`--preset-out` threw, and a
  document containing such an entry could not even be parsed) while nothing in the engine prices
  anything with either Vision coefficient. Exactly-one-zero still fails, so a half-measured Vision
  calibration cannot be written.
- `bench/context_cost/` (M): the prefill suite warms every grid length once before any sample is
  taken (previously only the shortest and the longest), and the four Vision cases likewise.
- `tests/runtime/` (M): two new groups — cost layering (an unbound prefill half keeps the
  architecture's constants, a measured half converts to seconds, `chunk_tokens` and
  `call_route_fraction` survive the layering, the default-form conversion equals the layered one) and
  signature binding on `nvidia-gb10-sm121` (an external entry binds its own signature, does not leak
  onto another configuration, survives repeated upserts beside a sibling entry and the machine
  transfer, round-trips a Text-only entry's zero Vision coefficients, and a half-measured Vision cost
  is refused).

## 4. Verification

- `./build-native/tests/infernix_context_cost_test` → `ok` (5 groups; unit-only, no GPU).
- Gates (all re-run on the final tree): `ctest -R infernix_cuda_stack_check` Passed (1.70 s);
  `ctest -R "context|prefix"` **100 % passed, 0 failed out of 12** (22 s; 3 real-artifact registrations
  skip); `ctest -R "request_log|bench_support|startup|qwen4_exp|product"` **100 % passed, 0 failed out
  of 34** (5 s; the `*_real_test` registrations skip without `INFERNIX_QWEN4_ARTIFACT`, so no T5 claim
  below rests on them — the real-artifact evidence is the two §4 runs named explicitly). No CUDA source
  changed, so the spill ratchet is inert by construction.
- Real artifact — `infernix_context_cost_bench --suite prefill --artifact
  Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix
  --prefill-reps 9` (GB10, `hardware_class nvidia-gb10-sm121`, canonical bf16/no-spec route, load
  17.2 s): completes the whole 12-point grid and now reports
  `"prefill_signature": "b99760c8af25d68b35304052663404a0940044322514cc2399921702511dd385"` — empty in
  every prior run of this bench for this artifact (`20261029-t5-f4local-prefill-rejected.json` shows
  `""`). The *fit* is rejected by the harness's own numerical gate (exit 3), which is the gate working,
  not a run failure:

  | run | held-out p95 (limit 0.15) | training p95 | median rel. err | r² (12 pts) | accepted |
  |---|---|---|---|---|---|
  | 2026-10-29, reps 5 | 0.1860 | 0.1328 | 0.0482 | 0.9878 | no |
  | today, reps 9, cold protocol | 0.1797 | 0.1305 | 0.0374 | 0.9895 | no |
  | today, reps 9, warm protocol | 0.1808 | 0.1234 | 0.0508 | 0.9884 | no |

  Fitted coefficients (evidence only, **not bound anywhere**): `chunk_ns` 138'097'163,
  `token_ns_q32` 1'579'929'730'021'409 (367.9 µs/token), `attention_pair_ns_q32` 27'930'392'335
  (6.50 ns/pair), Vision terms zero. Every rejection is the same single held-out point
  (`text-b2048-s256`); the other three held out at 0.025–0.055 across all three runs.
- End-to-end serving proof — `infernix-serve` on the same artifact, `--log-level debug`, prefix cache
  on:

  | run | startup line |
  |---|---|
  | with `--context-cost-presets` keyed to `b99760c8…` | `context cost \| transfer compiled-default \| prefill external \| hardware nvidia-gb10-sm121 \| prefill signature b99760c8af25d68b35304052663404a0940044322514cc2399921702511dd385` |
  | with no table | `context cost \| transfer compiled-default \| prefill generic-default \| hardware nvidia-gb10-sm121 \| prefill signature b99760c8…` |

  Both reach `listening`. The same digest appears in the calibration report, the startup log and the
  request log, so an operator-calibrated table and the calibration bench agree on the key.
- `git diff --check` clean. No commit made.

## 5. Accepted gaps / follow-ups

- **Acceptance criterion 2 is NOT met: no compiled GB10 prefill entry was registered.** Every measured
  fit fails the harness's own gate (held-out p95 0.1797/0.1808 > 0.15; r² 0.9884–0.9895 < 0.99), and
  the earlier campaign's rule stands — a rejected coefficient is not baked. Registering the
  best-failing fit, or relaxing 0.15/0.99 to fit the data, would be evidence gaming, not calibration.
  Criterion 2's first half (the bench runs the real artifact end to end and emits a full report with a
  real signature) is met; criterion 1 (signature populated, matching tables applied via the built-in
  and external routes) and criterion 3 are met.
- Consequence for serving today: unbound Flash-Next configurations keep the loader's startup estimates
  (`qwen4_exp_instance.cpp:280`, chunk 560 ms / token 90 µs / pair 4 ns — fitted 2026-10-05 for the
  5090 host-expert link, hand-scaled for integrated). Priced against today's measurement at whole-root
  shapes, that table is **+209 % / +85 % / +26 % / +26 %** at 128/512/1024/2048 tokens, where the
  generic estimate would be **−62 % / −50 % / −45 % / −44 %**. Both are far worse than the rejected
  table, and neither is honest to bake as a compiled binding. The operator route is now open:
  `--context-cost-presets` binds a measured table to the artifact's digest (proven above), and
  `--preset-out` can record one for a Text-only Flash-Next artifact at all — the Vision-cost rule made
  that impossible before this change.
- The gate has only 4 held-out points, so its p95 *is* its worst point: one structurally-mismatched
  continuation point rejects the whole calibration. That property belongs to the harness, was respected
  here, and is the main reason a 3.7–5.3 % median error still cannot be registered.
- Other Flash-Next artifacts (IQ3_S, Dense8, EXL3, hybrid, W4A16) still cannot be calibrated through
  this fixture (base-KV admission and Vision lending fail as measured 2026-10-29). Their signature keys
  now exist, so a calibration becomes possible the moment the fixture supports offloaded experts.
- The key does not cover everything that changes prefill cost: a substituted `--ngram-volume` (its
  format is resolved after the model is built), `--fast-paths`, or `--prefill-chunk`. A table calibrated
  at one chunk is applied at whatever chunk the route uses — the same limitation the two existing 5090
  entries have.
- Next step toward a registrable GB10 entry: identify the engine's per-call continuation cost (the
  ±50 ms structure in §2) — it needs the actual call sequence of a prefill, which the public
  `GenerationTimings` does not expose today. Alternatively, calibrate on standalone-root points only
  (no `T(b+s) − T(b)` differencing), which removes the confound but needs a wider grid than 12 points to
  keep 4 honest held-out points.
