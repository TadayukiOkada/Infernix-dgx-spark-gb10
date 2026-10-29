#!/usr/bin/env bash
# nsys_profile.sh — Nsight Systems timeline profiling for ANY CUDA application.
#
# Captures the CPU/GPU pipeline: kernel launches, API calls, memcpys, gaps where
# the GPU is idle. Best for diagnosing launch-overhead and CPU/GPU serialization
# (the low-utilization / low-power failure mode). Exports a .nsys-rep + .sqlite
# and runs the built-in stats reports; use parse_nsys.py for gap/utilization.
#
# Usage:
#   ./nsys_profile.sh [-o OUTBASE] [-t TRACE] [-d DURATION_S] [-- <app> [args...]]
#
# Examples:
#   ./nsys_profile.sh -- ./build/gb10_bench --op layer_pipeline
#   ./nsys_profile.sh -d 5 -- ./serve --model foo            # cap at 5s
#   ./nsys_profile.sh -t cuda,nvtx,osrt -- ./my_app          # add OS/runtime trace
set -euo pipefail

OUTBASE="nsys_profile"
TRACE="cuda,cudnn,cublas,osrt"
DURATION=""                 # empty => run to completion
SAMPLE=""                   # e.g. "cpu" for CPU sampling
EXTRA="${NSYS_EXTRA:-}"

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTBASE="$2"; shift 2 ;;
    -t) TRACE="$2"; shift 2 ;;
    -d) DURATION="$2"; shift 2 ;;
    -s|--sample) SAMPLE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown option '$1' (use -- before the app)" >&2; exit 2 ;;
  esac
done
if [[ $# -lt 1 ]]; then echo "error: no application given (use -- <app>)" >&2; usage; exit 2; fi

mkdir -p "$(dirname "$OUTBASE")" 2>/dev/null || true

PROF_ARGS=(profile -t "$TRACE" --force-overwrite true -o "$OUTBASE")
[[ -n "$DURATION" ]] && PROF_ARGS+=(--duration "$DURATION")
[[ -n "$SAMPLE" ]] && PROF_ARGS+=(--sample "$SAMPLE")
[[ -n "$EXTRA" ]] && PROF_ARGS+=($EXTRA)

echo "[nsys_profile] trace=$TRACE duration=${DURATION:-full} app: $*"
nsys "${PROF_ARGS[@]}" -- "$@" 2>&1 | tail -5

REP="${OUTBASE}.nsys-rep"
SQLITE="${OUTBASE}.sqlite"
echo "[nsys_profile] exporting sqlite..."
nsys export --type sqlite --force-overwrite true -o "$SQLITE" "$REP" >/dev/null 2>&1 || true

# Built-in summaries (CSV) for quick eyeballing.
echo "[nsys_profile] generating stats reports..."
for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_gpu_sum; do
  nsys stats --report "$rep" --format csv --output "${OUTBASE}_${rep}" "$REP" >/dev/null 2>&1 \
    && echo "  wrote ${OUTBASE}_${rep}.csv" || echo "  (report $rep unavailable)"
done

echo "[nsys_profile] wrote: $REP  $SQLITE"
echo "[nsys_profile] gap/util analysis:  python3 parse_nsys.py $SQLITE"
