#!/bin/bash
# Rebuild infernix_bench and run the two-model KV-dtype benchmark matrix.
#
# Models : Qwen3.8-Flash-Next GSQ-RCO IQ3_S and NVIDIA NVFP4 Dense8 (paths below).
# Matrix : 7 KV dtypes x {plain, MTP+lm-head-draft} x 2 models = 28 runs.
# Shapes : pp4096, tg128, tg256, pp4096+tg128, pp4096+tg256 (one report per run).
# Config : INFERNIX_EXPERTS_ON_GPU=1 (integrated GB10 unified-memory path),
#          IQ3_S n-gram table pinned in RAM, Dense8 table read from its volume
#          (its 52 GiB FP8 table + 77 GiB device experts do not both fit in RAM),
#          INT8-free: the KV dtype is the varied dimension, -r 2 --warmup 1.
#
# Usage:  ./bench/run_kv_matrix_bench.sh [output-dir] [--full]  (default dir: /tmp/kvmatrix)
#         Default gate is the fast 3-run matrix (bf16 x mtp x 3 models). --full runs all
#         42 (7 KV dtypes x {plain,mtp} x 3 models). KV_DTYPES/MODES override both, e.g.
#         KV_DTYPES="vq2 k4v2" MODES="mtp" ./bench/run_kv_matrix_bench.sh /tmp/kvmatrix_kv
# Status: <out>/status.txt gets one "<model> <kv>_<mode> exit=N <time>" line per
#         run and ALL_DONE at the end; per-run stdout/stderr in <out>/<tag>.log.
# Reports: <out>/m_<model>_<kv>_<mode>.json (bench schema v23).
# Other build: BENCH_BIN=/path/to/infernix_bench runs this exact matrix against a prebuilt
#         executable (no rebuild of build-native), e.g. a milestone worktree build; see
#         bench/run_kv_matrix_milestone_bench.sh.

set -u
cd "$(dirname "$0")/.." || exit 1
OUT=/tmp/kvmatrix
rm -rf "$OUT"
FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        *) OUT="$arg" ;;
    esac
done
if [ "$FULL" = 1 ]; then
    DEFAULT_KV='bf16 int8 fp8 nvfp4 k8v4 vq2 k4v2'
    DEFAULT_MODES='mtp'
    DEFAULT_MODELS='nvfp4'
else
    DEFAULT_KV='int8'
    DEFAULT_MODES='mtp'
    DEFAULT_MODELS='nvfp4 iq3s dense8'
fi
KV_DTYPES="${KV_DTYPES:-$DEFAULT_KV}"
MODES="${MODES:-$DEFAULT_MODES}"
MODELS="${MODELS:-$DEFAULT_MODELS}"
# Repetitions/warmup per run; raise them (e.g. REPS=5) to shrink the run-to-run noise floor
# when the delta under test is within ~1%. Defaults preserve the fast gate.
REPS="${REPS:-2}"
WARMUP="${WARMUP:-1}"
mkdir -p "$OUT"
rm -f "$OUT/status.txt"

MI=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix
NI=/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram
MD=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix
ND=/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
M4=/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix
N4=/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4.ngram
DRAFT_PATH=/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
BENCH="${BENCH_BIN:-./build-native/bench/infernix_bench}"

# 1. Rebuild the benchmark executable (incremental; only relinks when stale). Skipped when
#    BENCH_BIN names a prebuilt executable from another build tree.
if [ -z "${BENCH_BIN:-}" ]; then
    cmake --build build-native --target infernix_bench --parallel "$(nproc)" || exit 1
fi

# 2. The matrix. Keep the flags literal (no $var word-splitting: the -pg list is
#    one argument) and the MTP flags in an array.
run() { # run <model-tag> <weights> <ngram> <residency> <kv> <mode>
    local tag="$1" weights="$2" ngram="$3" residency="$4" kv="$5" mode="$6"
    local spec=()
    [ "$mode" = mtp ] && spec=(--spec mtp --draft-tokens auto --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 4)
    local name="${tag}_${kv}_${mode}"
    INFERNIX_EXPERTS_ON_GPU=1 "$BENCH" \
        --weights "$weights" --ngram-volume "$ngram" --ngram-residency "$residency" \
        --corpus bench/fixtures/bench_corpus.ids \
        -p 8192 -n 512 -pg '8192,512' \
        -r "$REPS" --warmup "$WARMUP" --concurrency 1 \
        --kv-dtype "$kv" "${spec[@]}" \
        -o json --output-file "$OUT/m_${name}.json" > "$OUT/m_${name}.log" 2>&1
    echo "$name exit=$? $(date +%T)" >> "$OUT/status.txt"
}

for kv in $KV_DTYPES; do
    for mode in $MODES; do
        for model in $MODELS; do
            case "$model" in
                iq3s)   run iq3s   "$MI" "$NI" ram    "$kv" "$mode" ;;
                dense8) run dense8 "$MD" "$ND" volume "$kv" "$mode" ;;
                nvfp4)  run nvfp4  "$M4" "$N4" ram    "$kv" "$mode" ;;
                *) echo "Unknown model $model" >&2; exit 1 ;;
            esac
        done
    done
done
echo ALL_DONE >> "$OUT/status.txt"

# 3. Summary table: median prefill/decode t/s and MTP accept rate per report.
/opt/etc/Infernix/.venv-convert/bin/python3 - "$OUT" <<'EOF'
import json, sys, glob, os
out = sys.argv[1]
def fmt(x): return f"{x:8.1f}" if isinstance(x, (int, float)) else "       -"
def acc(s): return f"{s['acceptance_rate']:.3f}" if s.get("acceptance_rate") is not None else "  -  "
print(f"{'run':24s} {'test':14s} {'prefill t/s':>11s} {'decode t/s':>11s}  acc")
for f in sorted(glob.glob(os.path.join(out, "m_*.json"))):
    name = os.path.basename(f)[2:-5]
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"{name:24s} MISSING/INVALID: {e}")
        continue
    for t in d["tests"]:
        s = t["speculative"]
        print(f"{name:24s} {t['label']:14s} {fmt(t['prefill_tok_s_median'])} {fmt(t['decode_output_tok_s_median'])}  {acc(s)}")
EOF
