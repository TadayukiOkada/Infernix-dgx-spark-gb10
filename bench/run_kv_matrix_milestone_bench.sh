#!/usr/bin/env bash
#
# run_kv_matrix_milestone_bench.sh
#
# Runs the same KV-dtype matrix as bench/run_kv_matrix_bench.sh against an infernix_bench
# executable compiled from a milestone commit, in an isolated git worktree. The working tree
# is never touched, and every target is measured on the identical harness and corpus.
#
# What comes from where: the matrix (models, KV dtypes, modes, shapes, repetitions, report
# schema) and bench/fixtures/bench_corpus.ids always come from the CURRENT tree -- that is what
# keeps two builds comparable. Only the engine inside infernix_bench comes from the measured
# commit. A commit whose bench CLI or artifact container predates the current matrix reports
# non-zero runs in status.txt instead of silently changing the matrix.
#
# Usage:
#   ./bench/run_kv_matrix_milestone_bench.sh [options] <commit_or_ref>...
#
# A target is any git ref, which always builds clean in the worktree, or "head" for the
# current tree through build-native, including uncommitted changes. Multiple targets run in
# sequence (the GPU stays exclusive) and their reports are compared side by side.
#
# Examples:
#   ./bench/run_kv_matrix_milestone_bench.sh 76d56902
#   ./bench/run_kv_matrix_milestone_bench.sh --baseline head head 9e4a6b25 76d56902
#   KV_DTYPES="bf16 fp8" MODES="plain mtp" \
#     ./bench/run_kv_matrix_milestone_bench.sh head 76d56902
#
# Options:
#   --out-dir DIR      Report root, one subdir per target (default: dist/kv_matrix)
#   --worktree DIR     Build worktree (default: /tmp/infernix-kvbench-wt, kept for reuse)
#   --baseline TARGET  Target the deltas are measured against (default: first target);
#                      added as a target automatically when not already listed
#   --jobs N           Ninja compile jobs (default: nproc)
#   --link-jobs N      Concurrent link/device-link jobs (default: 8)
#   --build-only       Compile each target's infernix_bench, skip the matrix runs
#   --skip-build       Reuse each target's existing build, do not compile
#   --compare-only     Only print the comparison for existing report dirs
#   --remove-worktree  Delete the build worktree at exit (it speeds up later runs)
#   --dry-run          Print the plan and exit
#   --list             List curated milestone commits, then exit
#   -h, --help         Show this help message and exit
#
# Matrix gate: the fast 3-run bf16 x mtp matrix by default; the KV_DTYPES and MODES
# environment variables override it exactly as they do for run_kv_matrix_bench.sh.
# Outputs: <out-dir>/<target>/m_<model>_<kv>_<mode>.{json,log} + status.txt per target,
#          <out-dir>/<commit>.build.log, <out-dir>/manifest.tsv, and the comparison
#          table on stdout.
# Cost: a worktree build compiles the engine and ops from scratch on the first target
#       (~10-20 min here, device link included) and relinks only what a commit changed
#       afterwards; each fast-gate target costs ~4 min of GPU time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MATRIX_SCRIPT="${SCRIPT_DIR}/run_kv_matrix_bench.sh"
MILESTONE_BUILDER="${REPO_ROOT}/tools/build_milestone_binaries.sh"
PY="${REPO_ROOT}/.venv-convert/bin/python3"

OUT_DIR="${REPO_ROOT}/dist/kv_matrix"
WORKTREE_DIR="/tmp/infernix-kvbench-wt"
JOBS="$(nproc)"
LINK_JOBS=8
BASELINE_ARG=""
REMOVE_WORKTREE=0
DRY_RUN=0
BUILD_ONLY=0
SKIP_BUILD=0
COMPARE_ONLY=0
TARGET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --worktree) WORKTREE_DIR="$2"; shift 2 ;;
        --baseline) BASELINE_ARG="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --link-jobs) LINK_JOBS="$2"; shift 2 ;;
        --build-only) BUILD_ONLY=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --compare-only) COMPARE_ONLY=1; shift ;;
        --remove-worktree) REMOVE_WORKTREE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --list) "${MILESTONE_BUILDER}" --list; exit 0 ;;
        -h|--help) sed -ne '/^#/!q;s/^# //;2,$p' "$0"; exit 0 ;;
        -*) echo "Error: unknown option '$1' (see --help)." >&2; exit 1 ;;
        *) TARGET_ARGS+=("$1"); shift ;;
    esac
done

BUILD_DIR="${WORKTREE_DIR}/build-kv"
HEAD_HASH="$(git -C "${REPO_ROOT}" rev-parse --short=8 HEAD)"

if [[ ${#TARGET_ARGS[@]} -eq 0 && -z "${BASELINE_ARG}" ]]; then
    echo "Error: give at least one commit or ref to measure." >&2
    echo "Usage: ./bench/run_kv_matrix_milestone_bench.sh [options] <commit_or_ref>..." >&2
    exit 1
fi

# Resolves one argument into R_KIND (tree|worktree), R_HASH, R_DATE, R_SUBJECT.
resolve_target() {
    local arg="$1"
    if [[ "${arg}" == "head" || "${arg}" == "HEAD" ]]; then
        R_KIND="tree"; R_HASH="${HEAD_HASH}"
    elif git -C "${REPO_ROOT}" rev-parse --verify "${arg}^{commit}" >/dev/null 2>&1; then
        R_KIND="worktree"
        R_HASH="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${arg}")"
    else
        echo "Error: cannot resolve '${arg}' to a commit (try --list for milestone hashes)." >&2
        exit 1
    fi
    R_DATE="$(git -C "${REPO_ROOT}" log -1 --format=%cd --date=short "${R_HASH}")"
    R_SUBJECT="$(git -C "${REPO_ROOT}" log -1 --format=%s "${R_HASH}")"
}

declare -a TAGS KINDS HASHES DATES SUBJECTS BINS REPORT_DIRS BUILD_SEC MATRIX_SEC RUN_TOTAL RUN_FAILED

add_target() { # <arg>; dedupes by kind+hash so "head" and its own hash stay separate columns
    resolve_target "$1"
    local j
    for j in "${!KINDS[@]}"; do
        if [[ "${KINDS[j]}" == "${R_KIND}" && "${HASHES[j]}" == "${R_HASH}" ]]; then
            return 0
        fi
    done
    local tag
    if [[ "${R_KIND}" == "tree" ]]; then
        tag="head"
    else
        tag="${R_HASH}"
    fi
    TAGS+=("${tag}")
    KINDS+=("${R_KIND}")
    HASHES+=("${R_HASH}")
    DATES+=("${R_DATE}")
    SUBJECTS+=("${R_SUBJECT}")
    if [[ "${R_KIND}" == "tree" ]]; then
        BINS+=("")                                  # the matrix script builds/uses build-native
    else
        BINS+=("${BUILD_DIR}/bench/infernix_bench")
    fi
    REPORT_DIRS+=("${OUT_DIR}/${tag}")
    BUILD_SEC+=("-"); MATRIX_SEC+=("-"); RUN_TOTAL+=("0"); RUN_FAILED+=("-")
}

# The baseline is a target too; add it first so it is the leftmost comparison column.
BASELINE_TAG=""
if [[ -n "${BASELINE_ARG}" ]]; then
    add_target "${BASELINE_ARG}"
    BASELINE_TAG="${TAGS[$(( ${#TAGS[@]} - 1 ))]}"
fi
for arg in "${TARGET_ARGS[@]}"; do
    add_target "${arg}"
done
[[ -z "${BASELINE_TAG}" ]] && BASELINE_TAG="${TAGS[0]}"

echo "========================================================================================"
echo " INFERNIX KV MATRIX ACROSS BUILT COMMITS (GB10 sm_121a)"
echo "========================================================================================"
printf " Repo           : %s\n" "${REPO_ROOT}"
printf " Worktree/build : %s\n" "${WORKTREE_DIR} (${BUILD_DIR})"
printf " Reports        : %s\n" "${OUT_DIR}"
printf " Matrix         : KV_DTYPES=%s MODES=%s\n" \
    "${KV_DTYPES:-bf16 (fast gate)}" "${MODES:-mtp (fast gate)}"
printf " Build jobs     : %s (link jobs: %s)\n" "${JOBS}" "${LINK_JOBS}"
printf " Baseline       : %s\n" "${BASELINE_TAG}"
printf " Targets        : %s\n" "${#TAGS[@]}"
for i in "${!TAGS[@]}"; do
    printf "   %d. %-9s %-8s %s  %s\n" "$((i + 1))" "${TAGS[$i]}" "${DATES[$i]}" \
        "${KINDS[$i]}" "${SUBJECTS[$i]}"
done
echo "========================================================================================"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[DRY RUN] Per worktree target: checkout in ${WORKTREE_DIR}, configure CMake"
    echo "          (Release, sm_121a, benchmarks ON) if needed, then"
    echo "          ninja -C ${BUILD_DIR} -j ${JOBS} infernix_bench."
    echo "          Then ${MATRIX_SCRIPT} <out-dir>/<target> with BENCH_BIN set to that"
    echo "          executable (BENCH_BIN unset for the head target), then the comparison."
    exit 0
fi

if [[ ! -x "${PY}" ]]; then
    echo "Error: Python interpreter not found at ${PY} (needed for the comparison table)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/manifest.tsv"
if [[ ! -f "${MANIFEST}" ]]; then
    printf "TARGET\tCOMMIT\tDATE\tKIND\tBUILD_SECONDS\tMATRIX_SECONDS\tRUNS\tFAILED_RUNS\tBENCH_SHA256\tREPORT_DIR\tSUBJECT\n" > "${MANIFEST}"
fi

cleanup() {
    if [[ ${REMOVE_WORKTREE} -eq 1 && -e "${WORKTREE_DIR}/.git" ]]; then
        echo
        echo "Removing build worktree at ${WORKTREE_DIR}..."
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_DIR}" >/dev/null 2>&1 \
            || rm -rf "${WORKTREE_DIR}"
        git -C "${REPO_ROOT}" worktree prune >/dev/null 2>&1 || true
    elif [[ -d "${WORKTREE_DIR}" ]]; then
        echo
        echo "Build worktree kept at ${WORKTREE_DIR} (--remove-worktree to delete it)."
    fi
}
trap cleanup EXIT INT TERM

ensure_worktree() { # <hash> <log>; non-zero return means this target cannot be built
    local hash="$1" log="$2"
    if [[ ! -d "${WORKTREE_DIR}" ]]; then
        echo "  Creating build worktree at ${WORKTREE_DIR}..."
        git -C "${REPO_ROOT}" worktree prune >/dev/null 2>&1 || true
        if ! git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${hash}" >"${log}" 2>&1; then
            echo "Error: git worktree add failed; see ${log}." >&2
            return 1
        fi
    elif ! git -C "${WORKTREE_DIR}" checkout --detach "${hash}" >>"${log}" 2>&1; then
        echo "Error: ${WORKTREE_DIR} is not a usable git worktree; pass --worktree DIR." >&2
        return 1
    fi
    if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
        echo "  Configuring CMake (Ninja) in ${BUILD_DIR}..."
        if ! cmake -B "${BUILD_DIR}" -S "${WORKTREE_DIR}" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=121a \
            -DINFERNIX_BUILD_APPS=ON \
            -DINFERNIX_BUILD_BENCHMARKS=ON \
            -DINFERNIX_LINK_JOBS="${LINK_JOBS}" >>"${log}" 2>&1; then
            echo "Error: CMake configure failed; see ${log}." >&2
            return 1
        fi
    fi
    return 0
}

record_manifest() { # <i> <bench path or "->">
    local i="$1" bench="$2" sha="-"
    if [[ -n "${bench}" && -x "${bench}" ]]; then
        sha="$(sha256sum "${bench}" | awk '{print $1}')"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${TAGS[$i]}" "${HASHES[$i]}" "${DATES[$i]}" "${KINDS[$i]}" "${BUILD_SEC[$i]}" \
        "${MATRIX_SEC[$i]}" "${RUN_TOTAL[$i]}" "${RUN_FAILED[$i]}" "${sha}" \
        "${REPORT_DIRS[$i]}" "${SUBJECTS[$i]}" >> "${MANIFEST}"
}

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    hash="${HASHES[$i]}"
    kind="${KINDS[$i]}"
    report_dir="${REPORT_DIRS[$i]}"
    build_log="${OUT_DIR}/${tag}.build.log"

    echo
    echo "----------------------------------------------------------------------------------------"
    echo "[target $((i + 1))/${#TAGS[@]}] ${tag} (${DATES[$i]}) -- ${SUBJECTS[$i]}"
    echo "----------------------------------------------------------------------------------------"

    if [[ ${COMPARE_ONLY} -eq 1 ]]; then
        echo "  --compare-only: not building or running."
        RUN_FAILED[$i]="not run"
        continue
    fi

    if [[ "${kind}" == "tree" ]]; then
        echo "  Using the current tree's build-native executable."
        if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
            echo "  NOTE: the working tree is dirty, so this column measures uncommitted changes."
        fi
        [[ ${SKIP_BUILD} -eq 1 ]] && BUILD_SEC[$i]="0"
    elif [[ ${SKIP_BUILD} -eq 1 ]]; then
        if [[ ! -x "${BINS[$i]}" ]]; then
            echo "Error: --skip-build but ${BINS[$i]} does not exist (build it without --skip-build)." >&2
            RUN_FAILED[$i]="no build"
            record_manifest "$i" ""
            continue
        fi
        BUILD_SEC[$i]="0"
    else
        echo "  Building infernix_bench from ${hash} (log: ${build_log})..."
        t_start="$(date +%s)"
        if ! ensure_worktree "${hash}" "${build_log}"; then
            BUILD_SEC[$i]="failed"
            RUN_FAILED[$i]="configure"
            record_manifest "$i" ""
            continue
        fi
        if ! ninja -C "${BUILD_DIR}" -j "${JOBS}" infernix_bench >>"${build_log}" 2>&1; then
            echo "  ERROR: build failed for ${hash}; see ${build_log}." >&2
            BUILD_SEC[$i]="failed"
            RUN_FAILED[$i]="build"
            record_manifest "$i" ""
            continue
        fi
        BUILD_SEC[$i]="$(( $(date +%s) - t_start ))"
        echo "  Built in ${BUILD_SEC[$i]}s: ${BINS[$i]}"
    fi

    if [[ ${BUILD_ONLY} -eq 1 ]]; then
        echo "  --build-only: skipping the matrix run."
        RUN_FAILED[$i]="build only"
        record_manifest "$i" "${BINS[$i]:-${REPO_ROOT}/build-native/bench/infernix_bench}"
        continue
    fi

    mkdir -p "${report_dir}"
    echo "  Running the KV matrix (reports: ${report_dir})..."
    t_start="$(date +%s)"
    matrix_rc=0
    if [[ -z "${BINS[$i]}" ]]; then
        "${MATRIX_SCRIPT}" "${report_dir}" || matrix_rc=$?
    else
        BENCH_BIN="${BINS[$i]}" "${MATRIX_SCRIPT}" "${report_dir}" || matrix_rc=$?
    fi
    MATRIX_SEC[$i]="$(( $(date +%s) - t_start ))"
    echo "  Matrix finished in ${MATRIX_SEC[$i]}s (script exit ${matrix_rc})."

    # Count the per-run exit codes the matrix recorded; wall time alone proves nothing ran.
    if [[ -f "${report_dir}/status.txt" ]]; then
        counts="$(awk '/exit=/ { n++; if ($2 != "exit=0") f++ }
                       END { printf "%d %d\n", n, f + 0 }' "${report_dir}/status.txt")"
        RUN_TOTAL[$i]="${counts%% *}"
        RUN_FAILED[$i]="${counts##* }"
        if [[ "${RUN_FAILED[$i]}" != "0" ]]; then
            echo "  WARNING: ${RUN_FAILED[$i]} of ${RUN_TOTAL[$i]} run(s) exited non-zero:" >&2
            grep ' exit=' "${report_dir}/status.txt" | grep -v ' exit=0 ' >&2 || true
        fi
    else
        RUN_TOTAL[$i]="0"
        RUN_FAILED[$i]="no status.txt (exit ${matrix_rc})"
        echo "  WARNING: no status.txt in ${report_dir}; the matrix did not complete." >&2
    fi

    record_manifest "$i" "${BINS[$i]:-${REPO_ROOT}/build-native/bench/infernix_bench}"
done

echo
echo "========================================================================================"
echo " CROSS-TARGET COMPARISON (baseline: ${BASELINE_TAG})"
echo "========================================================================================"

compare_pairs=()
for i in "${!TAGS[@]}"; do
    compare_pairs+=("${TAGS[$i]}=${REPORT_DIRS[$i]}")
done

"${PY}" - "${BASELINE_TAG}" "${compare_pairs[@]}" <<'PYEOF'
"""Compare run_kv_matrix_bench reports from several builds on one grid."""
import glob
import json
import os
import sys

baseline_tag = sys.argv[1]
targets = [arg.split("=", 1) for arg in sys.argv[2:]]

rows = {}                       # "run test" -> {tag: {metric: value}}
order = []
failed_runs = []
for tag, path in targets:
    status_file = os.path.join(path, "status.txt")
    if os.path.exists(status_file):
        for line in open(status_file):
            if " exit=" in line and " exit=0 " not in line:
                failed_runs.append(f"{tag}: {line.strip()}")
    for report in sorted(glob.glob(os.path.join(path, "m_*.json"))):
        name = os.path.basename(report)[2:-5]
        try:
            with open(report) as handle:
                doc = json.load(handle)
        except Exception as exc:                # an unreadable report is shown, never guessed
            print(f"{tag} {name}: unreadable report ({exc})")
            continue
        for test in doc["tests"]:
            key = f"{name} {test['label']}"
            if key not in rows:
                rows[key] = {}
                order.append(key)
            spec = test.get("speculative") or {}
            rows[key][tag] = {
                "decode": test.get("decode_output_tok_s_median"),
                "prefill": test.get("prefill_tok_s_median"),
                "accept": spec.get("acceptance_rate"),
            }

if not order:
    print("No reports parsed under the given directories; nothing to compare.")
    sys.exit(0)

NAME_W = min(max(len(key) for key in order), 34) + 2
COL_W = max(18, max(len(tag) for tag, _ in targets) + 9)


def section(title, metric, fmt, delta):
    seen = any(rows[key].get(tag, {}).get(metric) is not None
               for key in order for tag, _ in targets)
    if not seen:
        return
    print(title)
    print(f"{'run / test':<{NAME_W}}" + "".join(f"{tag:>{COL_W}}" for tag, _ in targets))
    for key in order:
        base = rows[key].get(baseline_tag, {}).get(metric)
        line = f"{key[:NAME_W - 2]:<{NAME_W}}"
        for tag, _ in targets:
            value = rows[key].get(tag, {}).get(metric)
            if value is None:
                cell = "-"
            elif base and tag != baseline_tag:
                cell = fmt(value) + delta(value, base)
            else:
                cell = fmt(value)
            line += f"{cell:>{COL_W}}"
        print(line)
    print()


pct = lambda v, b: f" ({(v - b) / b * 100.0:+.1f}%)"
pp = lambda v, b: f" ({(v - b) * 100.0:+.1f}pp)"

section("decode output t/s (median)", "decode", lambda v: f"{v:.1f}", pct)
section("prefill t/s (median)", "prefill", lambda v: f"{v:.1f}", pct)
section("MTP acceptance rate", "accept", lambda v: f"{v:.3f}", pp)

print("Deltas are relative to the baseline column ('-' means that build has no report")
print("for that run, so its engine never produced the number, not that it was slower).")
if failed_runs:
    print()
    print("Runs with a non-zero exit (their metrics above are absent or partial):")
    for line in failed_runs:
        print(f"  {line}")
PYEOF

echo
echo "Reports: ${OUT_DIR}/<target>/   Manifest: ${MANIFEST}"
