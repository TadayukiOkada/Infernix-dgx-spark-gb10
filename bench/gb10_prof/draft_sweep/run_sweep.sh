#!/usr/bin/env bash
# Controlled draft-quality sweep for Qwen3.8-Flash-Next Dense8 on GB10.
# Isolates: (A) INFERNIX_WIDTH_COST, (B) --draft-tokens ceiling,
#           (C) --lm-head-draft on/off, (D) copy drafting on the copy prompt.
# Greedy (temperature 0), fixed binary/artifact/context. Records decode tok/s,
# acceptance length, accepted-by-pos, rounds, fallback.
set -u
BIN=/opt/etc/Infernix/build-native/apps/infernix
DIR=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix
M="$DIR/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix"
N="$DIR/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram"
S=/opt/etc/Infernix/bench/gb10_prof/draft_sweep
REASON=$(cat "$S/prompt_reason.txt")
COPY=$(cat "$S/prompt_copy.txt")
OUT="$S/results.tsv"
: > "$OUT"

run_case () {
  local phase="$1" label="$2" prompt="$3"; shift 3
  # remaining args are extra CLI flags; env vars set inline via INFERNIX_WIDTH_COST
  local tmp; tmp=$(mktemp)
  INFERNIX_WIDTH_COST="${WC:-}" "$BIN" "$M" --ngram-volume "$N" --prompt "$prompt" \
      --max-new 256 --max-context 2048 --temperature 0 --log-level info "$@" > "$tmp" 2>&1
  local ec=$?
  local dec acc_len byp rounds drafted accepted fb
  dec=$(grep -oE "decode speed +[0-9.]+" "$tmp" | grep -oE "[0-9.]+")
  acc_len=$(grep -oE "acceptance length +[0-9.]+" "$tmp" | grep -oE "[0-9.]+$")
  byp=$(grep -oE "accepted by pos +[0-9,]+" "$tmp" | sed -E 's/accepted by pos +//')
  rounds=$(grep -oE "mtp rounds +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  drafted=$(grep -oE "drafted tokens +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  accepted=$(grep -oE "accepted tokens +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  fb=$(grep -oE "fallback steps +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  ngr=$(grep -oE "ngram accepted tokens +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  printf '%s\t%s\t%s\tdec=%s\tacc_len=%s\tby_pos=%s\trounds=%s\tdrafted=%s\taccepted=%s\tfallback=%s\tngram_acc=%s\tec=%s\n' \
     "$phase" "$label" "WC=${WC:-def}" "${dec:-NA}" "${acc_len:-NA}" "${byp:-NA}" \
     "${rounds:-NA}" "${drafted:-NA}" "${accepted:-NA}" "${fb:-NA}" "${ngr:-0}" "$ec" | tee -a "$OUT"
  rm -f "$tmp"
}

echo "== Phase A: reasoning prompt, width cost sweep (draft-tokens=7 non-binding ceiling, lm-head on) =="
for w in 0.15 0.20 0.25 0.30 0.38; do
  WC=$w run_case A "W=$w spec mtp dt7 lmhead" "$REASON" --spec mtp --draft-tokens 7 --lm-head-draft
done

echo "== Phase B: reasoning prompt, draft-tokens ceiling sweep (W=0.25, lm-head on) =="
for d in 1 2 3 4 5 6 7; do
  WC=0.25 run_case B "dt=$d W=0.25 lmhead" "$REASON" --spec mtp --draft-tokens $d --lm-head-draft
done

echo "== Phase C: reasoning prompt, lm-head-draft on/off (W=0.25, dt=4) =="
WC=0.25 run_case C "dt=4 no-lmhead" "$REASON" --spec mtp --draft-tokens 4
WC=0.25 run_case C "dt=4 lmhead" "$REASON" --spec mtp --draft-tokens 4 --lm-head-draft

echo "== Phase D: copy prompt, copy drafting contribution (W=0.25, dt=4 lmhead) =="
WC=0.25 run_case D "copy baseline mtp-only" "$COPY" --spec mtp --draft-tokens 4 --lm-head-draft
WC=0.25 run_case D "copy +ngram dm15 mm8"  "$COPY" --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 8
WC=0.25 run_case D "copy +ngram dm15 mm12" "$COPY" --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12

echo "== DONE: $OUT =="
