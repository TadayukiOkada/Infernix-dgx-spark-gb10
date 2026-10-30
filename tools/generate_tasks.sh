#!/usr/bin/env bash
set -euo pipefail

# Usage: tools/generate_tasks.sh <topic-slug> <task-description> [tasks-per-cycle]
# Example:
#   tools/generate_tasks.sh gb10-mxfp8-moe-tuning \
#     "Profile and optimize MXFP8 MoE routed expert projections and shared expert fusion on GB10 sm_121a for Qwen3.6-35B-A3B, eliminating remaining stack spills and improving decode tok/s."
#   tools/generate_tasks.sh gb10-mxfp8-moe-tuning "..." 5

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <topic-slug> <task-description> [tasks-per-cycle]" >&2
  exit 1
fi

topics="$1"
task_desc="$2"
tasks_per_cycle="${3:-5}"

/opt/etc/Infernix/.venv-convert/bin/python3 task_orchestrator.py \
  --generate-tasks \
  --plan-only \
  --topics "${topics}" \
  --tasks-per-cycle "${tasks_per_cycle}" \
  -q "${task_desc}"
