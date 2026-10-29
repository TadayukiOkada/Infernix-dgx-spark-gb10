#!/usr/bin/env bash
# ncu_profile.sh — Nsight Compute profiling for ANY CUDA application.
#
# Collects per-kernel roofline metrics (Speed-of-Light %, Tensor-Core pipe,
# occupancy, launch config, cache traffic) and exports both a browsable
# .ncu-rep and a machine-readable CSV for parse_ncu.py.
#
# Usage:
#   ./ncu_profile.sh [-o OUTBASE] [-k KERNEL_REGEX] [-s SKIP] [-c COUNT]
#                    [-m METRIC_LIST] [-p PRESET] [-- <app> [app args...]]
#
# Examples:
#   ./ncu_profile.sh -- ./build/gb10_bench --op rmsnorm
#   ./ncu_profile.sh -k 'regex:k_gemm' -c 3 -- ./my_app
#   ./ncu_profile.sh -p full -- ./my_app          # extra stall/branch metrics
#   METRICS="gpu__time_duration.sum" ./ncu_profile.sh -- ./my_app
#
# Notes for GB10 / unified memory:
#   - dram__* counters do NOT exist here; bandwidth is anchored by the
#     lts__throughput / l1tex__throughput SOL percentages and byte counters.
#   - sm__pipe_tensor_* IS available and is the ground truth for Tensor-Core
#     saturation (a GEMM stuck near ~50% means a legacy CUTLASS tile was chosen).
set -euo pipefail

OUTBASE="ncu_profile"
KFILTER=""                 # empty => every kernel
SKIP=0
COUNT=0                    # 0 => all launches after skip
PRESET="roofline"          # roofline | full | bw | tensor | launch
EXTRA="${NCU_EXTRA:-}"
METRICS="${METRICS:-}"

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTBASE="$2"; shift 2 ;;
    -k) KFILTER="$2"; shift 2 ;;
    -s) SKIP="$2"; shift 2 ;;
    -c) COUNT="$2"; shift 2 ;;
    -m) METRICS="$2"; shift 2 ;;
    -p) PRESET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown option '$1' (use -- before the app)" >&2; exit 2 ;;
  esac
done

if [[ $# -lt 1 ]]; then echo "error: no application given (use -- <app>)" >&2; usage; exit 2; fi

# Metric presets. Every name here was validated against GB10 (no dram__*).
case "$PRESET" in
  roofline)
    DEFAULT_METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__grid_size,launch__block_size,\
lts__t_bytes.sum,l1tex__t_bytes.sum" ;;
  tensor)
    DEFAULT_METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active" ;;
  bw)
    DEFAULT_METRICS="gpu__time_duration.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_bytes.sum,l1tex__t_bytes.sum,\
lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct" ;;
  launch)
    DEFAULT_METRICS="gpu__time_duration.sum,\
launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__occupancy_limit_registers,launch__waves_per_multiprocessor,\
sm__warps_active.avg.pct_of_peak_sustained_active" ;;
  full)
    DEFAULT_METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__grid_size,launch__block_size,\
launch__waves_per_multiprocessor,\
smsp__average_warp_latency_per_inst_issued.ratio,\
smsp__stall_reason_not_selected.avg.pct_of_peak_sustained_active,\
smsp__stall_reason_long_scoreboard.avg.pct_of_peak_sustained_active,\
smsp__stall_reason_barrier.avg.pct_of_peak_sustained_active,\
smsp__stall_reason_wait.avg.pct_of_peak_sustained_active" ;;
  *) echo "unknown preset '$PRESET'" >&2; exit 2 ;;
esac
METRICS="${METRICS:-$DEFAULT_METRICS}"

mkdir -p "$(dirname "$OUTBASE")" 2>/dev/null || true

ARGS=(--metrics "$METRICS" --csv --force-overwrite)
[[ -n "$KFILTER" ]] && ARGS+=(-k "$KFILTER")
ARGS+=(--launch-skip "$SKIP")
[[ "$COUNT" -gt 0 ]] && ARGS+=(--launch-count "$COUNT")
[[ -n "$EXTRA" ]] && ARGS+=($EXTRA)

echo "[ncu_profile] preset=$PRESET kernels='${KFILTER:-ALL}' skip=$SKIP count=$COUNT"
echo "[ncu_profile] app: $*"

# CSV to stdout is captured (the app's own stdout is filtered by the parser).
CSV="${OUTBASE}.csv"
REP="${OUTBASE}.ncu-rep"

# Run once producing CSV (fast, machine-readable), and once producing a
# .ncu-rep for interactive inspection in Nsight Compute.
ncu "${ARGS[@]}" -- "$@" > "$CSV" 2> >(tee "${OUTBASE}.log" >&2) || {
  echo "[ncu_profile] ncu run failed; see ${OUTBASE}.log" >&2; exit 1; }

# A separate pass writes the .ncu-rep (sections auto-selected by ncu).
SECT_ARGS=(--set basic --force-overwrite -f -o "$OUTBASE")
[[ -n "$KFILTER" ]] && SECT_ARGS+=(-k "$KFILTER")
SECT_ARGS+=(--launch-skip "$SKIP")
[[ "$COUNT" -gt 0 ]] && SECT_ARGS+=(--launch-count "$COUNT")
ncu "${SECT_ARGS[@]}" -- "$@" > /dev/null 2>>"${OUTBASE}.log" || \
  echo "[ncu_profile] warning: .ncu-rep generation failed (CSV still usable)" >&2

echo "[ncu_profile] wrote: $CSV  $REP"
echo "[ncu_profile] parse with:  python3 parse_ncu.py $CSV"
