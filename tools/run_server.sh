#!/usr/bin/env bash
#
# run_server.sh
#
# Starts an infernix-serve binary for a given milestone build or commit on DGX Spark GB10.
# Automatically sets --model-id to the commit hash so benchmarks (such as tool-eval-bench)
# can target the exact build version.
#
# Usage:
#   ./tools/run_server.sh [options] [commit_or_binary_path]
#
# Examples:
#   ./tools/run_server.sh 76d56902
#   ./tools/run_server.sh dist/milestones/infernix-serve-76d56902
#   ./tools/run_server.sh --port 8000 b9c97634
#
# Options:
#   --port PORT         Listen port (default: 8000)
#   --host HOST         Listen host (default: 0.0.0.0)
#   --model-id ID       Explicit model ID override (default: extracted commit hash)
#   --max-context N     Max context length (default: 262144)
#   --max-concurrency N Max concurrency (default: 4)
#   -h, --help          Show this help message and exit
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default paths
MODEL_PATH="/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix"
NGRAM_PATH="/opt/etc/models/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense/Qwen3.8-Flash-Next-Local-NVFP4.ngram"
DIST_DIR="${REPO_ROOT}/dist/milestones"
DRAFT_PATH="/opt/etc/models/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix"
# DFlash2 vs MTP on GB10 (2026-10-29, pp2048+tg128, fp8 KV, interleaved medians, decode out t/s):
#   c = 1: DFlash2 109.4 (dense8) / 114.8 (F4-Local) vs MTP 81.8 / 91.7 - DFlash2 +25..34 %.
#   c = 2: DFlash2 67.9 on DFlash2Pairs-qualified artifacts (Q8/GGUF/BF16/FP8/MXFP8 dense; the
#          kDFlashPairMaxDrafts default is 8 on GB10) vs MTP 53.8; on the NVFP4-dense F4-Local
#          default artifact below the pair fast path is NOT qualified - DFlash2 there decodes
#          plain (35.6), so MTP below is the right config for this MODEL_PATH at c >= 2.
#   c = 4: MTP 38.8 beats DFlash2's 2-row pair default (20.6) and its INFERNIX_TMP_DFLASH2_ROWS=4
#          probe (37.4): keep this server (MAX_CONCURRENCY=4) on MTP; prefer DFlash2 for c <= 2.
# Server parameters
HOST="0.0.0.0"
PORT="8000"
MAX_CONTEXT="262144"
MAX_CONCURRENCY="4"
EXPLICIT_MODEL_ID=""
TARGET_INPUT=""

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --model-id)
            EXPLICIT_MODEL_ID="$2"
            shift 2
            ;;
        --max-context)
            MAX_CONTEXT="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        -h|--help)
            sed -ne '/^#/!q;s/^# //;2,$p' "$0"
            exit 0
            ;;
        *)
            TARGET_INPUT="$1"
            shift
            ;;
    esac
done

# If no target binary/commit provided, look for the newest in dist/milestones or fall back to build-native
BINARY_PATH=""
COMMIT_HASH=""

if [[ -z "${TARGET_INPUT}" ]]; then
    # Prefer build-native/apps/infernix-serve when newer than dist/milestones, else newest milestone
    LATEST_BIN="$(ls -t "${DIST_DIR}"/infernix-serve-* 2>/dev/null | grep -v '\.tsv$' | head -n 1 || true)"
    if [[ -x "${REPO_ROOT}/build-native/apps/infernix-serve" ]] && { [[ -z "${LATEST_BIN}" ]] || [[ "${REPO_ROOT}/build-native/apps/infernix-serve" -nt "${LATEST_BIN}" ]]; }; then
        BINARY_PATH="${REPO_ROOT}/build-native/apps/infernix-serve"
    elif [[ -n "${LATEST_BIN}" ]]; then
        BINARY_PATH="${LATEST_BIN}"
    else
        echo "Error: No compiled infernix-serve binary found." >&2
        echo "Please specify a commit or binary path, e.g.: ./tools/run_server.sh 76d56902" >&2
        exit 1
    fi
else
    # Check if TARGET_INPUT is an existing file path
    if [[ -x "${TARGET_INPUT}" ]]; then
        BINARY_PATH="${TARGET_INPUT}"
    elif [[ -x "${DIST_DIR}/${TARGET_INPUT}" ]]; then
        BINARY_PATH="${DIST_DIR}/${TARGET_INPUT}"
    elif [[ -x "${DIST_DIR}/infernix-serve-${TARGET_INPUT}" ]]; then
        BINARY_PATH="${DIST_DIR}/infernix-serve-${TARGET_INPUT}"
    elif ls "${DIST_DIR}"/infernix-serve-"${TARGET_INPUT}"* >/dev/null 2>&1; then
        BINARY_PATH="$(ls "${DIST_DIR}"/infernix-serve-"${TARGET_INPUT}"* | head -n 1)"
    elif [[ "${TARGET_INPUT}" == "head" || "${TARGET_INPUT}" == "HEAD" ]]; then
        COMMIT_HASH="$(git -C "${REPO_ROOT}" rev-parse --short=8 HEAD)"
        if [[ -x "${DIST_DIR}/infernix-serve-${COMMIT_HASH}" ]]; then
            BINARY_PATH="${DIST_DIR}/infernix-serve-${COMMIT_HASH}"
        else
            BINARY_PATH="${REPO_ROOT}/build-native/apps/infernix-serve"
        fi
    elif git -C "${REPO_ROOT}" rev-parse --verify "${TARGET_INPUT}^{commit}" >/dev/null 2>&1; then
        RESOLVED_HASH="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${TARGET_INPUT}")"
        echo "Milestone binary for commit ${RESOLVED_HASH} (${TARGET_INPUT}) is not yet compiled in ${DIST_DIR}."
        echo "Automatically compiling milestone binary via tools/build_milestone_binaries.sh..."
        "${REPO_ROOT}/tools/build_milestone_binaries.sh" "${RESOLVED_HASH}"
        if ls "${DIST_DIR}"/infernix-serve-"${RESOLVED_HASH}"* >/dev/null 2>&1; then
            BINARY_PATH="$(ls "${DIST_DIR}"/infernix-serve-"${RESOLVED_HASH}"* | head -n 1)"
        else
            echo "Error: Build completed but binary for ${RESOLVED_HASH} was not found in ${DIST_DIR}" >&2
            exit 1
        fi
    else
        echo "Error: Cannot resolve binary for '${TARGET_INPUT}' in ${DIST_DIR} and not a valid git commit." >&2
        echo "Available compiled binaries in ${DIST_DIR}:" >&2
        ls -1 "${DIST_DIR}"/infernix-serve-* 2>/dev/null | grep -v '\.tsv$' || echo "  (none)" >&2
        exit 1
    fi
fi

# Extract commit hash from binary name or git
BINARY_NAME="$(basename "${BINARY_PATH}")"
if [[ -n "${EXPLICIT_MODEL_ID}" ]]; then
    MODEL_ID="${EXPLICIT_MODEL_ID}"
elif [[ "${BINARY_NAME}" =~ infernix-serve-([0-9a-fA-F]{8}) ]]; then
    MODEL_ID="${BASH_REMATCH[1]}"
elif git -C "${REPO_ROOT}" rev-parse --short=8 HEAD >/dev/null 2>&1; then
    MODEL_ID="$(git -C "${REPO_ROOT}" rev-parse --short=8 HEAD)"
else
    MODEL_ID="infernix-model"
fi

echo "========================================================================================"
echo " INFERNIX SERVER LAUNCHER (GB10 sm_121a)"
echo "========================================================================================"
echo " Binary          : ${BINARY_PATH}"
echo " Model ID (Alias): ${MODEL_ID}"
echo " Model Weights   : ${MODEL_PATH}"
echo " Listen Endpoint : http://${HOST}:${PORT}/v1"
echo " Max Context     : ${MAX_CONTEXT} tokens"
echo " Max Concurrency : ${MAX_CONCURRENCY}"
echo " Speculative     : MTP (auto) + N-gram (15 drafts, min-match 4)"
echo "========================================================================================"
echo

export INFERNIX_EXPERTS_ON_GPU=1
    #--draft-model "$DRAFT_PATH" \
    #--draft-tokens auto \
exec "${BINARY_PATH}" "${MODEL_PATH}" \
    --ngram-volume "${NGRAM_PATH}" \
    --ngram-residency volume \
    --kv-dtype fp8 \
    --spec mtp \
    --draft-tokens auto \
    --lm-head-draft \
    --ngram-draft-tokens 31 \
    --ngram-min-match 4 \
    --max-context "${MAX_CONTEXT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --model-id "${MODEL_ID}" \
    --request-log-jsonl /tmp/serve_requests.jsonl
