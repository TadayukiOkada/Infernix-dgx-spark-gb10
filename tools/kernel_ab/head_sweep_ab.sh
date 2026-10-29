#!/usr/bin/env bash
# Interleaved two-binary A/B for the Qwen3.8-Flash-Next vocabulary head (projection_fp32).
#
# Both binaries are the same bench harness built from the same tree with one compile-time
# constant flipped, so every measured difference is that constant. Rounds alternate execution
# order (candidate-first on odd rounds, base-first on even) and each regime is repeated so a
# base-to-base spread can be computed as the null pair.
#
# Usage: head_sweep_ab.sh <base-bin> <cand-bin> <out-dir> [rounds] [flush-mib]
set -euo pipefail

base=$1
cand=$2
out=$3
rounds=${4:-5}
flush=${5:-256}

mkdir -p "$out"
for ((r = 1; r <= rounds; ++r)); do
  if ((r % 2 == 1)); then
    "$cand" --shape head --warmup 3 --repeat 15 --flush-mib "$flush" --csv "$out/r${r}_cand.csv" >"$out/r${r}_cand.log" 2>&1
    "$base" --shape head --warmup 3 --repeat 15 --flush-mib "$flush" --csv "$out/r${r}_base.csv" >"$out/r${r}_base.log" 2>&1
  else
    "$base" --shape head --warmup 3 --repeat 15 --flush-mib "$flush" --csv "$out/r${r}_base.csv" >"$out/r${r}_base.log" 2>&1
    "$cand" --shape head --warmup 3 --repeat 15 --flush-mib "$flush" --csv "$out/r${r}_cand.csv" >"$out/r${r}_cand.log" 2>&1
  fi
  echo "round $r done (flush=$flush MiB)"
done
