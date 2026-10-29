#!/usr/bin/env bash
# Interleaved A/B latency harness for a correctness-preserving kernel change.
#
# Cancels thermal/DVFS drift by alternating baseline and candidate in one
# process timeline, runs both orderings plus a base-to-base null, and reports
# medians and the speedup computed by Python (no mental math). An aggregate
# effect must exceed the configured noise floor (default 0.426%) in BOTH
# orderings to count as a real win.
#
# Usage:
#   ab_bench.sh --base BASE_BIN --cand CAND_BIN \
#               [--args "--bench 1"] [--reps 5] [--noise-pct 0.426]
#
# The binaries must print per-iteration timing to stdout; the harness sums every
# "N.NNN ms" token per invocation into one scalar per run (matches the Infernix
# MoE/ggml bench output). Adjust the extraction in `per_run` for other formats.
set -euo pipefail
set +H

BASE=""; CAND=""; ARGS="--bench 1"; REPS=5; NOISE=0.426
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --cand) CAND="$2"; shift 2;;
    --args) ARGS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --noise-pct) NOISE="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -x "$BASE" && -x "$CAND" ]] || { echo "need executable --base and --cand" >&2; exit 2; }

# Sum every "<float> ms" token in one invocation into a single scalar (ms).
per_run() {
  "$@" $ARGS 2>/dev/null | grep -oE '[0-9]+\.[0-9]+ ms' \
    | awk '{s+=$1} END{printf "%.5f", s}'
}

# Warm the GPU so the first timed reps are not cold.
"$BASE" $ARGS >/dev/null 2>&1 || true
"$CAND" $ARGS >/dev/null 2>&1 || true

FWD_B=(); FWD_C=(); REV_C=(); REV_B=(); NULL1=(); NULL2=()
for ((i=1;i<=REPS;i++)); do
  # forward: baseline then candidate
  b=$(per_run "$BASE"); c=$(per_run "$CAND"); FWD_B+=("$b"); FWD_C+=("$c")
  # reversed: candidate then baseline (kills any positional bias)
  c2=$(per_run "$CAND"); b2=$(per_run "$BASE"); REV_C+=("$c2"); REV_B+=("$b2")
  # base-to-base null: measures same-binary ordering/measurement spread
  n1=$(per_run "$BASE"); n2=$(per_run "$BASE"); NULL1+=("$n1"); NULL2+=("$n2")
  echo "rep$i  fwd B=$b C=$c | rev C=$c2 B=$b2 | null $n1/$n2" >&2
done

joined="$(printf '%s ' "${FWD_B[@]}");joined="${joined};$(printf '%s ' "${FWD_C[@]}");joined="${joined};$(printf '%s ' "${REV_C[@]}");joined="${joined};$(printf '%s ' "${REV_B[@]}");joined="${joined};$(printf '%s ' "${NULL1[@]}");joined="${joined};$(printf '%s ' "${NULL2[@]}")"
python3 - "$NOISE" "$joined" <<'PY'
import sys, statistics as st
noise=float(sys.argv[1])
fb,fc,rc,rb,n1,n2=[[float(v) for v in f.split()] for f in sys.argv[2].split(";")]
base=fb+rb; cand=fc+rc
mb,mc=st.median(base),st.median(cand)
mb_f,mc_f=st.median(fb),st.median(fc)
mb_r,mc_r=st.median(rb),st.median(rc)
null=[abs(a-b) for a,b in zip(n1,n2)]
null_med=st.median(null) if null else 0.0
sp=mb/mc
imp=100*(mb-mc)/mc
imp_f=100*(mb_f-mc_f)/mc_f
imp_r=100*(mb_r-mc_r)/mc_r
print(f"baseline median = {mb:.5f} ms   (fwd {mb_f:.5f} / rev {mb_r:.5f})")
print(f"candidate median= {mc:.5f} ms   (fwd {mc_f:.5f} / rev {mc_r:.5f})")
print(f"speedup base/cand = {sp:.4f}x   improvement = {imp:+.2f}%")
print(f"per-ordering improvement: fwd {imp_f:+.2f}%  rev {imp_r:+.2f}%")
print(f"base-to-base null median |delta| = {null_med*1000:.1f} us ({100*null_med/mb:.2f}%)")
win = imp>noise and imp_f>0 and imp_r>0
lose= imp<-noise
verdict="ACCEPT (win > noise, consistent both orderings)" if win else ("REJECT (regression)" if lose else "NEUTRAL (within noise floor)")
print(f"noise floor = {noise}%   ->  VERDICT: {verdict}")
PY
