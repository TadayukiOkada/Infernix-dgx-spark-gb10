# Multirow-drafts A/B on GB10 (dgpp batched-MTP transfer test)

**Question.** Infernix gates `choose_multirow_draft_length` OFF
(`INFERNIX_MULTIROW_DRAFTS=0.0`) on a 5090-era measurement whose own code comment
(`src/models/qwen4_exp/program/program_impl.h`) speculates the call is wrong for GB10:
"bandwidth-starved → verifying draft columns is comparatively cheaper." dgpp's real edge on a
single Spark was aggregate throughput at batch 4 / 4-in-flight. Does turning multirow drafting
ON recover that edge here?

**Harness.** `run_ab.py` — `infernix-serve` on the Dense8 artifact, `--max-concurrency 4`,
greedy (`--temperature 0`), `--spec mtp --draft-tokens 4 --lm-head-draft`, four distinct
reasoning prompts fired concurrently, two legs per arm. Each arm boots its own server
(`kMultiRowDrafts` is process-static). Source of truth: `--request-log-jsonl`.

## Result (artifact `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix`, GB10 sm_121)

| arm | peak avg batch | row_rounds_max | accept_rate | accepted_per_position | aggregate tok/s (cold, warm) | mean |
|-----|----------------|----------------|-------------|-----------------------|------------------------------|------|
| OFF | 4.0 | 380 | 0.885 | {0:12, 1:11, 2:0, 3:0} | 37.58, 73.20 | 55.39 |
| ON  | 4.0 | 376 | 0.808 | {0:13, 1:8, 2:0, 3:0}  | 37.32, 72.81 | 55.07 |

Delta: **-0.6%**, consistently negative in both the cold leg (~37) and the warm/prefix-cached
leg (~73). Acceptance *dropped* under multirow (0.885 → 0.808; position-1 accepts 11 → 8), so
the selector did engage and changed draft widths — this is a genuine wash-to-regression, not a
no-op.

**Bit-exactness gate: PASS.** Greedy transcripts were identical across arms and across legs,
confirming the speculative verify path preserves greedy output regardless of draft count.

## Conclusion — keep `kMultiRowDrafts = 0.0`

The code comment's hypothesis is refuted by measurement. On GB10 the decode bottleneck *is*
bandwidth (Category C linear stream already runs ~89% of the DRAM plateau), so widening drafts
to verify several rows' columns spends **more** verify bandwidth without raising acceptance —
acceptance already saturates near position 1–2 (positions 2–3 never accepted here). Infernix's
existing per-row draft selection already keeps all 4 rows co-decoded (peak avg batch 4.0) in
both arms, so multirow adds no scheduling headroom to exploit. The 5090-era OFF decision holds
on GB10.

This matches the prior `bench/gb10_prof/draft_sweep` finding that draft-width knobs are flat
within noise. No production change made.
