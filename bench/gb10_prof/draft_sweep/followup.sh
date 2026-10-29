#!/usr/bin/env bash
# Follow-up: (1) repeat WC 0.15/0.20/0.25 on reasoning x3 to separate signal from noise;
#            (2) does copy drafting hurt the reasoning prompt (window-widening cost)?
set -u
BIN=/opt/etc/Infernix/build-native/apps/infernix
DIR=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix
M="$DIR/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix"
N="$DIR/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram"
S=/opt/etc/Infernix/bench/gb10_prof/draft_sweep
REASON=$(cat "$S/prompt_reason.txt")
OUT="$S/followup.tsv"; : > "$OUT"

run_case () {
  local label="$1" prompt="$2"; shift 2
  local tmp; tmp=$(mktemp)
  INFERNIX_WIDTH_COST="${WC:-}" "$BIN" "$M" --ngram-volume "$N" --prompt "$prompt" \
      --max-new 256 --max-context 2048 --temperature 0 --log-level info "$@" > "$tmp" 2>&1
  local dec al byp fb
  dec=$(grep -oE "decode speed +[0-9.]+" "$tmp" | grep -oE "[0-9.]+")
  al=$(grep -oE "acceptance length +[0-9.]+" "$tmp" | grep -oE "[0-9.]+$")
  byp=$(grep -oE "accepted by pos +[0-9,]+" "$tmp" | sed -E 's/accepted by pos +//')
  fb=$(grep -oE "fallback steps +[0-9]+" "$tmp" | grep -oE "[0-9]+$")
  printf '%s\tWC=%s\tdec=%s\tacc_len=%s\tby_pos=%s\tfallback=%s\n' "$label" "${WC:-def}" "${dec:-NA}" "${al:-NA}" "${byp:-NA}" "${fb:-NA}" | tee -a "$OUT"
  rm -f "$tmp"
}

echo "== Repeat WC on reasoning (dt=5 lmhead), 3 trials each =="
for w in 0.15 0.20 0.25; do
  for t in 1 2 3; do
    WC=$w run_case "reason W=$w trial$t" "$REASON" --spec mtp --draft-tokens 5 --lm-head-draft
  done
done

echo "== Copy drafting on REASONING prompt (expect regression from wider rounds) =="
WC=0.25 run_case "reason mtp-only"   "$REASON" --spec mtp --draft-tokens 4 --lm-head-draft
WC=0.25 run_case "reason +ngram dm15 mm8" "$REASON" --spec mtp --draft-tokens 4 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 8

echo "== DONE: $OUT =="
