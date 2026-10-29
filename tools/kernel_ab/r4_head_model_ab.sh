#!/usr/bin/env bash
# Interleaved same-config A/B for the R4 packed-scale head candidate on the Dense8 artifact.
# Alternates execution order per round and prints pp4096 / tg128 per round plus medians.
set -u

BASE=$1
CAND=$2
ROUNDS=${3:-6}
WEIGHTS=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix
NGRAM=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
CORPUS=bench/fixtures/real_corpus.ids

cd "$(dirname "$0")/../.." || exit 1

run_one() {
  "$1" --weights "$WEIGHTS" --ngram-volume "$NGRAM" --ngram-residency volume \
    --ngram-draft-tokens 0 --spec mtp --draft-tokens 3 --lm-head-draft \
    --kv-dtype int8 --concurrency 1 -p 4096 -n 128 -r 2 --warmup 1 --max-ctx 20480 \
    --corpus "$CORPUS" 2>/dev/null \
  | awk '/^ *pp4096/{pp=$4} /^ *tg128/{tg=$5} END{print pp, tg}'
}

echo "round order base_pp base_tg cand_pp cand_tg"
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    b=$(run_one "$BASE"); c=$(run_one "$CAND"); order="base-first"
  else
    c=$(run_one "$CAND"); b=$(run_one "$BASE"); order="cand-first"
  fi
  echo "$r $order $b $c"
done
