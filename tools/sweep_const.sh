#!/bin/bash
# Fast constant sweep for the IQ3_S (ggml) route on GB10.
# Patches one constant, builds ONLY bench/infernix_bench, runs a short prefill+decode
# benchmark and prints one result line. Constants live in headers, so an iteration is
# a single nvcc TU (~47-64 s) plus 3 links instead of 242.
#
# Usage: sweep_const.sh <file> <sed-expression> <label>
set -u
cd /opt/etc/Infernix || exit 1
FILE="$1"; EXPR="$2"; LABEL="$3"

W=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix
N=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram

sed -i "$EXPR" "$FILE" || { echo "$LABEL PATCH_FAIL"; exit 1; }

T0=$(date +%s)
if ! ninja -C build-native -j20 bench/infernix_bench > /tmp/sw_build.log 2>&1; then
  echo "$LABEL BUILD_FAIL"; tail -3 /tmp/sw_build.log; exit 1
fi
BUILD=$(( $(date +%s) - T0 ))

T0=$(date +%s)
timeout 900 ./build-native/bench/infernix_bench --weights "$W" --ngram-volume "$N" \
  --ngram-residency volume --ngram-draft-tokens 0 --kv-dtype int8 --concurrency 1 \
  --prefill-chunk 16384 -p 4096,16384 -n 128 -r 2 --warmup 1 --max-ctx 20480 \
  --corpus bench/fixtures/bench_corpus.ids > /tmp/sw_run.log 2>&1
RC=$?
RUN=$(( $(date +%s) - T0 ))

if [ $RC -ne 0 ]; then
  echo "$LABEL RUN_FAIL(rc=$RC) $(grep -oiE 'exceeds the device.s|out of memory|illegal memory' /tmp/sw_run.log | head -1)"
  exit 1
fi

P4=$(awk '/^pp4096/{print $4}' /tmp/sw_run.log)
P16=$(awk '/^pp16384/{print $4}' /tmp/sw_run.log)
TG=$(awk '/^tg128/{print $5}' /tmp/sw_run.log)
printf "%-34s pp4096=%-10s pp16384=%-10s tg128=%-9s build=%ss run=%ss\n" \
  "$LABEL" "${P4:-NA}" "${P16:-NA}" "${TG:-NA}" "$BUILD" "$RUN"
