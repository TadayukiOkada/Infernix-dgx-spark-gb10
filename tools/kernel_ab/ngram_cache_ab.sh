#!/usr/bin/env bash
# Interleaved same-binary A/B for NgramVolume::kCacheBits 26 vs 22.
# Two stashed binaries differing only in the constant; alternating order per round; base-base null pair.
set -u
cd /opt/etc/Infernix
OUT=/tmp/kcab; mkdir -p "$OUT"
IQ3=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix
IQ3N=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram
D8=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix
D8N=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix.ngram

run() { # $1=model tag $2=weights $3=ngram $4=arm tag $5=binary $6=round
  "$5" --weights "$2" --ngram-volume "$3" --ngram-residency volume --ngram-draft-tokens 0 \
    --kv-dtype int8 --concurrency 1 -p 4096 -n 128 -r 3 --warmup 1 --max-ctx 20480 \
    --corpus bench/fixtures/bench_corpus.ids -o json \
    --output-file "$OUT/${1}_r$6_$4.json" > "$OUT/${1}_r$6_$4.log" 2>&1
  echo "  $1 r$6 $4 exit=$?"
}

for model in iq3s d8; do
  if [ "$model" = iq3s ]; then W=$IQ3; N=$IQ3N; else W=$D8; N=$D8N; fi
  for round in 1 2 3 4; do
    if [ $((round % 2)) -eq 0 ]; then order="base cand"; else order="cand base"; fi
    echo "== $model round $round order: $order =="
    for arm in $order; do
      if [ "$arm" = base ]; then B=/tmp/bench_kc26; else B=/tmp/bench_kc22; fi
      run "$model" "$W" "$N" "$arm" "$B" "$round"
    done
  done
  # base-base null pair (positional/control check)
  echo "== $model null pair (base, base) =="
  run "$model" "$W" "$N" "nullA" /tmp/bench_kc26 5
  run "$model" "$W" "$N" "nullB" /tmp/bench_kc26 5
done
echo DONE
