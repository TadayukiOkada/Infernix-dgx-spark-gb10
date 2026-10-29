#!/bin/bash
# Sweep --draft-tokens over the MTP decode default and report throughput per count.
#
# On the GB10 host with Qwen3.8-Flash-Next the measured optimum is 3 draft tokens:
#   2 -> 53.3 t/s (acceptance 0.541, 122 rounds), 3 -> 58.8 t/s (0.531, 98 rounds),
#   4 -> 52.8 t/s (0.400, 98 rounds)
# so throughput peaks where acceptance is only slightly lower: extra drafted rows cost
# verification work faster than they add accepted tokens.
#
# Usage: draft_sweep.sh BINARY [COUNTS] [ROUNDS_PER_COUNT]
#   BINARY            bench/infernix_bench build to sweep
#   COUNTS            space-separated draft-token counts (default "2 3 4")
#   ROUNDS_PER_COUNT  repetitions per count (default 3)
#
# Environment overrides as in decode_ab.sh (INFERNIX_WEIGHTS, INFERNIX_NGRAM,
# INFERNIX_CORPUS).
set -u

BINARY=$1
COUNTS=${2:-"2 3 4"}
ROUNDS=${3:-3}

WEIGHTS=${INFERNIX_WEIGHTS:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix}
NGRAM=${INFERNIX_NGRAM:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram}
CORPUS=${INFERNIX_CORPUS:-bench/fixtures/bench_corpus.ids}

cd "$(dirname "$0")/../.." || exit 1
[ -x "$BINARY" ] || { echo "draft_sweep: not executable: $BINARY" >&2; exit 1; }

run_one() {
  "$BINARY" --weights "$WEIGHTS" --ngram-volume "$NGRAM" \
    --ngram-residency ram --ngram-draft-tokens 0 --spec mtp \
    --draft-tokens "$1" --lm-head-draft --kv-dtype k8v4 \
    --concurrency 1 -p 4096 -n 128 -r 2 --warmup 1 --max-ctx 20480 --corpus "$CORPUS" 2>&1 \
  | awk '/^ *pp4096/{pp=$4} /^ *tg128/{tg=$5; acc=$14; rounds=$15} END{print pp, tg, acc, rounds}'
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for d in $COUNTS; do
  best=""
  for r in $(seq 1 "$ROUNDS"); do
    read -r pp tg acc rounds <<< "$(run_one "$d")"
    if [ -z "$pp" ] || [ -z "$tg" ]; then echo "draft_sweep: run failed at --draft-tokens $d" >&2; exit 1; fi
    echo "$pp $tg $acc $rounds" >> "$TMP/$d"
    printf 'draft=%s round %d: pp4096=%s t/s  tg128=%s t/s  acc=%s  rounds=%s\n' "$d" "$r" "$pp" "$tg" "$acc" "$rounds"
  done
done

echo
awk '
  FNR == 1 { d = FILENAME; sub(/.*\//, "", d) }
  { pp[d, ++n[d]] = $1; tg[d, n[d]] = $2; acc[d] = $3; rounds[d] = $4 }
  END {
    for (d in n) {
      for (i = 1; i <= n[d]; ++i) { pbuf[i] = pp[d, i]; tbuf[i] = tg[d, i] }
      asort(pbuf); asort(tbuf)
      h = n[d]
      mpp = (h % 2) ? pbuf[(h+1)/2] : (pbuf[h/2] + pbuf[h/2+1]) / 2
      mtg = (h % 2) ? tbuf[(h+1)/2] : (tbuf[h/2] + tbuf[h/2+1]) / 2
      printf "draft_tokens=%-3s pp4096 median %8.2f t/s   tg128 median %6.2f t/s   acc=%-12s rounds=%s\n",
             d, mpp, mtg, acc[d], rounds[d]
      delete pbuf; delete tbuf
    }
  }
' "$TMP"/* | sort -t= -k2 -n
