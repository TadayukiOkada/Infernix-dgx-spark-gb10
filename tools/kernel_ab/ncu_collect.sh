#!/bin/bash
# Collect Nsight Compute hardware counters for named kernels of a synthetic benchmark,
# exporting to an .ncu-rep report (per-kernel CSV via ncu_collect --import).
#
# Per docs/performance/gb10-kernel-optimization-results.md §3.4: NCU runs strictly on
# synthetic unit/benchmark binaries — never on processes loading real model weights.
#
# Usage: ncu_collect.sh OUT_REP MATCH_REGEX BINARY [ARGS...]
#   OUT_REP        export path prefix (.ncu-rep written next to it)
#   MATCH_REGEX    kernel-name regex against mangled names (--kernel-name-base mangled)
#   BINARY ARGS... the synthetic harness invocation
set -u

OUT=$1; MATCH=$2; shift 2
BINARY=$1; shift

cd "$(dirname "$0")/../.." || exit 1
[ -x "$BINARY" ] || { echo "ncu_collect: not executable: $BINARY" >&2; exit 1; }

ncu --kernel-name-base mangled \
    --kernel-name "regex:${MATCH}" \
    --launch-skip 2 --launch-count 2 \
    --export "$OUT" \
    "$BINARY" "$@" > /dev/null 2>&1
echo "report: ${OUT}.ncu-rep ($(stat -c%s "${OUT}.ncu-rep" 2>/dev/null || echo missing) bytes)"
