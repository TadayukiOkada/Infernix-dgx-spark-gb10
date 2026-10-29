#!/bin/bash
# Capture a decode or prefill NSYS trace with CUDA-graph node tracing and print the
# ranked kernel table with launch geometry.
#
# CUDA-graph decode rounds are invisible to a plain `nsys profile --trace=cuda`: the
# whole graph collapses into one activity and the report shows 2 - 12 % GPU-busy with
# a ~27 ms "gap" per token, which is an artifact of the trace, not of the engine.
# `--cuda-graph-trace=node` is required before any decode kernel ranking is read.
#
# Usage: profile_decode.sh [MODE] [BINARY] [NSAMPLES] [DRAFT_TOKENS]
#   MODE          decode | prefill (default decode)
#   BINARY        bench/infernix_bench build (default build-native/bench/infernix_bench)
#   NSAMPLES      -n for decode, -p for prefill (default 16 decode / 4096 prefill)
#   DRAFT_TOKENS  MTP draft tokens for decode mode (default 3)
#
# Writes profiles/nsys/kernel_ab_<mode>.nsys-rep and .sqlite, then prints the ranking.
set -u

MODE=${1:-decode}
BINARY=${2:-build-native/bench/infernix_bench}
SAMPLES=${3:-}
DRAFT=${4:-3}

WEIGHTS=${INFERNIX_WEIGHTS:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix}
NGRAM=${INFERNIX_NGRAM:-/opt/etc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram}
CORPUS=${INFERNIX_CORPUS:-bench/fixtures/bench_corpus.ids}

cd "$(dirname "$0")/../.." || exit 1
[ -x "$BINARY" ] || { echo "profile_decode: not executable: $BINARY" >&2; exit 1; }
mkdir -p profiles/nsys

if [ "$MODE" = decode ]; then
  N=${SAMPLES:-16}
  ARGS=(--weights "$WEIGHTS" --ngram-volume "$NGRAM" --ngram-residency ram --ngram-draft-tokens 0
        --spec mtp --draft-tokens "$DRAFT" --lm-head-draft --kv-dtype k8v4
        --concurrency 1 -n "$N" -r 1 --warmup 1 --max-ctx 20480 --corpus "$CORPUS")
elif [ "$MODE" = prefill ]; then
  N=${SAMPLES:-4096}
  ARGS=(--weights "$WEIGHTS" --ngram-volume "$NGRAM" --ngram-residency ram --ngram-draft-tokens 0
        --kv-dtype k8v4 --concurrency 1 -p "$N" -n 8 -r 1 --warmup 1 --max-ctx 20480 --corpus "$CORPUS")
else
  echo "profile_decode: MODE must be decode or prefill" >&2; exit 1
fi

OUT=profiles/nsys/kernel_ab_$MODE
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --output "$OUT" \
  --force-overwrite=true --export=sqlite "$BINARY" "${ARGS[@]}" > /dev/null 2>&1 || {
    echo "profile_decode: nsys run failed" >&2; exit 1
}

python3 "$(dirname "$0")/rank_kernels.py" "$OUT.sqlite"
