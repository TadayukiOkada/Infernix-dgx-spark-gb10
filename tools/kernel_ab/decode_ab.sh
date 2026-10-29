#!/bin/bash
# Paired, interleaved A/B of two infernix_bench builds on the MTP decode default.
#
# Execution order alternates every round (round 1: BASE then CAND, round 2: CAND then
# BASE, ...), matching the interleaving rule in
# docs/performance/gb10-kernel-optimization-results.md §2.3. A first-run advantage of
# ~0.2 % is normal on this host, so a single ordering is never enough to accept or
# reject a candidate.
#
# Usage: decode_ab.sh BASE_BINARY CAND_BINARY [ROUNDS] [DRAFT_TOKENS]
#   BASE_BINARY, CAND_BINARY  two bench/infernix_bench builds to compare
#   ROUNDS                    paired rounds (default 4)
#   DRAFT_TOKENS              MTP draft tokens (default 3, the measured optimum)
#
# Environment overrides (default to the local Qwen3.8-Flash-Next artifact):
#   INFERNIX_WEIGHTS  artifact path
#   INFERNIX_NGRAM    n-gram table path
#   INFERNIX_CORPUS   benchmark corpus (default bench/fixtures/bench_corpus.ids)
set -u

BASE=$1
CAND=$2
ROUNDS=${3:-4}
DRAFT=${4:-3}

WEIGHTS=${INFERNIX_WEIGHTS:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix}
NGRAM=${INFERNIX_NGRAM:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram}
CORPUS=${INFERNIX_CORPUS:-bench/fixtures/bench_corpus.ids}

cd "$(dirname "$0")/../.." || exit 1
for f in "$BASE" "$CAND"; do
  [ -x "$f" ] || { echo "decode_ab: not executable: $f" >&2; exit 1; }
done

# Prefill and decode throughput of one run: "<pp> <tg> <acceptance> <rounds>".
# The n-gram table is kept in RAM and drafts are disabled so the measurement isolates
# MTP verification rather than n-gram volume reads. One invocation measures both
# phases at the same prompt: pp4096 exercises the prefill kernels, tg128 the
# speculative decode loop.
run_one() {
  "$1" --weights "$WEIGHTS" --ngram-volume "$NGRAM" \
    --ngram-residency ram --ngram-draft-tokens 0 --spec mtp \
    --draft-tokens "$DRAFT" --lm-head-draft --kv-dtype k8v4 \
    --concurrency 1 -p 4096 -n 128 -r 2 --warmup 1 --max-ctx 20480 --corpus "$CORPUS" 2>&1 \
  | awk '/^ *pp4096/{pp=$4} /^ *tg128/{tg=$5; acc=$14; rounds=$15} \
         END{print pp, tg, acc, rounds}'
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then first=$BASE; second=$CAND; label_b=BASE; label_c=CAND
  else                          first=$CAND; second=$BASE; label_b=CAND; label_c=BASE; fi
  rb=$(run_one "$first");  rc=$(run_one "$second")
  # Normalize the labels back to BASE/CAND whichever order this round ran.
  if [ "$label_b" = BASE ]; then b=$rb; c=$rc; else b=$rc; c=$rb; fi
  read -r bpp btg bacc brounds <<< "$b"
  read -r cpp ctg cacc crounds <<< "$c"
  echo "$bpp $cpp" >> "$TMP/pp"
  echo "$btg $ctg" >> "$TMP/tg"
  echo "$bacc" >> "$TMP/bacc"
  echo "$cacc" >> "$TMP/cacc"
  printf 'round %d (order %s first): pp4096 BASE=%s CAND=%s t/s | tg128 BASE=%s CAND=%s t/s\n' \
    "$r" "$label_b" "$bpp" "$cpp" "$btg" "$ctg"
done

# Medians, paired delta and win count for one phase (file of "<base> <cand>" lines).
summarize() {
  awk -v phase="$2" '
    { b[n]=$1; c[n]=$2; n++ }
    END {
      if (n == 0) { printf "%-8s no samples\n", phase; exit }
      asort(b); asort(c)
      mb = (n % 2) ? b[(n+1)/2] : (b[n/2] + b[n/2+1]) / 2
      mc = (n % 2) ? c[(n+1)/2] : (c[n/2] + c[n/2+1]) / 2
      wins = 0
      for (i = 1; i <= n; ++i) { if (c[i] > b[i]) ++wins }
      printf "%-8s BASE median %8.2f t/s   CAND median %8.2f t/s   %+6.2f %%   CAND faster in %d/%d\n",
             phase, mb, mc, (mc/mb - 1) * 100, wins, n
    }
  ' "$1"
}

echo
summarize "$TMP/pp" pp4096
summarize "$TMP/tg" tg128

echo "spec acceptance: BASE $(sort -u "$TMP/bacc" | tr '\n' ' ')CAND $(sort -u "$TMP/cacc" | tr '\n' ' ')"
