#!/usr/bin/env bash
#
# build_milestone_binaries.sh
#
# Clones / checks out each performance milestone commit in an isolated git worktree,
# compiles an infernix-serve binary for GB10 (sm_121a), and stores the resulting binaries
# in an organized distribution directory with a build manifest.
#
# Usage:
#   ./tools/build_milestone_binaries.sh [options] [commit_or_label...]
#
# Options:
#   --out-dir DIR       Output directory for compiled binaries (default: dist/milestones)
#   --worktree DIR      Temporary worktree directory (default: /tmp/infernix-build-wt)
#   --list              List all curated milestone commits and descriptions, then exit
#   --dry-run           Print the execution plan without compiling
#   --keep-worktree     Do not delete the worktree after the build completes
#   --jobs N            Parallel compile jobs for Ninja (default: $(nproc))
#   --link-jobs N       Concurrent link jobs (default: 8)
#   -h, --help          Show this help message and exit
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration defaults
OUT_DIR="${REPO_ROOT}/dist/milestones"
WORKTREE_DIR="/tmp/infernix-build-wt"
KEEP_WORKTREE=0
DRY_RUN=0
JOBS="$(nproc)"
LINK_JOBS=8

# Curated list of milestone commits: HASH:LABEL:DESCRIPTION
MILESTONES=(
    "2d94e830:gb10-spec-cost-0.25:Initial speculation width cost retuned to 0.25 for GB10"
    "68f19150:perf-nvfp4-n6144-k2560:Retune n6144_k2560 dispatch on GB10 (up to 2.10x speedup)"
    "3f9609fa:perf-q4-sliced-mma:Enhance Q4 Linear Operations with Sliced and MMA launch"
    "b26d04f5:perf-fp8-a16-midbands:Retune dominated FP8 a16 mid-bands on GB10 (up to 1.63x)"
    "6a0d8ef9:perf-fp8-tma-11shapes:Optimize GB10 FP8 kernels, 2-block/SM TMA schedules, 11 shapes"
    "fe02652c:perf-q8-spills-cliffs:Eliminate dominated bands, stack spills, and route cliffs on Q8"
    "228ff642:perf-bf16-vision-text:Eliminate dominated bands and route cliffs across BF16 shapes"
    "3fcf5e1b:perf-nvfp4-spill-free:Eliminate stack spills in NVFP4 a4 mma kernels on GB10"
    "c481d25d:perf-nvfp4-all-15shapes:Overhaul all 15 NVFP4 shape dispatches with route sweep"
    "9e4a6b25:refactor-nvfp4-schedules:Refactor NVFP4 Attention and Linear Schedules"
    "8070300d:refactor-q6-q8-linear:Refactor Q6 and Q8 linear operations and memory management"
    "99bc532b:feat-nvfp4-fastpaths:Enhance fast-path configs & forward execution memory mgmt"
    "c90f905b:feat-nvfp4-gdn:Implement NVFP4 support in GDN and related projections"
    "2819b823:feat-moe-2mb-l2-ggml:MoE 2MB L2 prefetch limit + ggml tiled shapes + prefix span"
    "eed450aa:feat-gb10-context-cost:Add NVIDIA GB10 hardware class & context transfer costs"
    "31805702:feat-20thread-prefill:Enhance NgramVolume and PrefillWalk for 20-thread CPU workers"
    "b9c97634:feat-mtp-kvonly-2rowmoe:MTP kv_only slice + 2-row resident_moe (64 reg) + single-pass top_logprobs"
    "76d56902:feat-gb10-roadmap-head:NVIDIA GB10 optimization roadmap and hardware profile (HEAD)"
)

# Parse CLI options
CUSTOM_SELECTION=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --worktree)
            WORKTREE_DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --link-jobs)
            LINK_JOBS="$2"
            shift 2
            ;;
        --keep-worktree)
            KEEP_WORKTREE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --list)
            echo "Curated Infernix Performance Milestones for NVIDIA GB10:"
            echo "----------------------------------------------------------------------------------------"
            printf "%-4s %-10s %-32s %s\n" "#" "COMMIT" "LABEL" "DESCRIPTION"
            echo "----------------------------------------------------------------------------------------"
            idx=1
            for entry in "${MILESTONES[@]}"; do
                IFS=':' read -r chash clabel cdesc <<< "$entry"
                full_hash="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${chash}")"
                printf "%-4d %-10s %-32s %s\n" "$idx" "$full_hash" "$clabel" "$cdesc"
                idx=$((idx + 1))
            done
            exit 0
            ;;
        -h|--help)
            sed -ne '/^#/!q;s/^# //;2,$p' "$0"
            exit 0
            ;;
        *)
            CUSTOM_SELECTION+=("$1")
            shift
            ;;
    esac
done

# Resolve which milestones to build
TARGET_ENTRIES=()
if [[ ${#CUSTOM_SELECTION[@]} -gt 0 ]]; then
    for item in "${CUSTOM_SELECTION[@]}"; do
        found=0
        for entry in "${MILESTONES[@]}"; do
            IFS=':' read -r chash clabel cdesc <<< "$entry"
            if [[ "$item" == "$chash"* || "$item" == "$clabel" ]]; then
                TARGET_ENTRIES+=("$entry")
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            # Treat as raw commit hash or ref
            if git -C "${REPO_ROOT}" rev-parse --verify "${item}^{commit}" >/dev/null 2>&1; then
                full_hash="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${item}")"
                msg="$(git -C "${REPO_ROOT}" log -1 --format=%s "${item}" | tr ':' ' ' | cut -c 1-50)"
                TARGET_ENTRIES+=("${full_hash}:custom-${full_hash}:${msg}")
            else
                echo "Error: Unknown commit or label '${item}'" >&2
                exit 1
            fi
        fi
    done
else
    TARGET_ENTRIES=("${MILESTONES[@]}")
fi

echo "========================================================================================"
echo " INFERNIX MILESTONE BINARY BUILDER (GB10 sm_121a)"
echo "========================================================================================"
echo " Repository   : ${REPO_ROOT}"
echo " Worktree     : ${WORKTREE_DIR}"
echo " Output Dir   : ${OUT_DIR}"
echo " Build Jobs   : ${JOBS} (link jobs: ${LINK_JOBS})"
echo " Total Targets: ${#TARGET_ENTRIES[@]}"
echo "========================================================================================"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] The following binaries would be compiled:"
    idx=1
    for entry in "${TARGET_ENTRIES[@]}"; do
        IFS=':' read -r chash clabel cdesc <<< "$entry"
        full_hash="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${chash}")"
        echo "  ${idx}. ${full_hash} -> infernix-serve-${full_hash}-${clabel} (${cdesc})"
        idx=$((idx + 1))
    done
    exit 0
fi

mkdir -p "${OUT_DIR}"
MANIFEST_FILE="${OUT_DIR}/manifest.tsv"
if [[ ! -f "${MANIFEST_FILE}" ]]; then
    printf "INDEX\tCOMMIT\tDATE\tLABEL\tSIZE_BYTES\tSHA256\tBUILD_SECONDS\tSTATUS\tDESCRIPTION\n" > "${MANIFEST_FILE}"
fi

# Cleanup handler
cleanup() {
    if [[ $KEEP_WORKTREE -eq 0 && -d "${WORKTREE_DIR}" ]]; then
        echo
        echo "Cleaning up build worktree at ${WORKTREE_DIR}..."
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_DIR}" >/dev/null 2>&1 || rm -rf "${WORKTREE_DIR}"
        git -C "${REPO_ROOT}" worktree prune >/dev/null 2>&1 || true
        echo "Worktree cleaned."
    else
        echo
        echo "Preserving worktree at ${WORKTREE_DIR}."
    fi
}
trap cleanup EXIT INT TERM

# Create or reuse worktree
FIRST_HASH="$(cut -d':' -f1 <<< "${TARGET_ENTRIES[0]}")"
if [[ ! -d "${WORKTREE_DIR}" ]]; then
    echo "Creating dedicated build worktree at ${WORKTREE_DIR}..."
    git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${FIRST_HASH}"
else
    echo "Reusing existing worktree at ${WORKTREE_DIR}..."
    git -C "${WORKTREE_DIR}" checkout --detach "${FIRST_HASH}"
fi

BUILD_DIR="${WORKTREE_DIR}/build-wt"

# Build loop
target_idx=1
total_targets="${#TARGET_ENTRIES[@]}"

for entry in "${TARGET_ENTRIES[@]}"; do
    IFS=':' read -r chash clabel cdesc <<< "$entry"
    short_hash="$(git -C "${REPO_ROOT}" rev-parse --short=8 "${chash}")"
    commit_date="$(git -C "${REPO_ROOT}" log -1 --format=%cd --date=short "${chash}")"
    out_bin_name="infernix-serve-${short_hash}-${clabel}"
    out_bin_path="${OUT_DIR}/${out_bin_name}"

    echo "----------------------------------------------------------------------------------------"
    echo "[${target_idx}/${total_targets}] Building: ${out_bin_name}"
    echo "  Commit      : ${short_hash} (${commit_date})"
    echo "  Description : ${cdesc}"
    echo "----------------------------------------------------------------------------------------"

    # Checkout commit in worktree
    git -C "${WORKTREE_DIR}" checkout --detach "${chash}" >/dev/null 2>&1

    # CMake configure (only reconfigures when necessary)
    if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
        echo "  Configuring CMake with Ninja in ${BUILD_DIR}..."
        cmake -B "${BUILD_DIR}" -S "${WORKTREE_DIR}" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=121a \
            -DINFERNIX_BUILD_APPS=ON \
            -DINFERNIX_BUILD_BENCHMARKS=OFF \
            -DINFERNIX_LINK_JOBS="${LINK_JOBS}" \
            >/dev/null
    fi

    # Compile infernix-serve
    t_start="$(date +%s)"
    build_status="SUCCESS"
    if ! ninja -C "${BUILD_DIR}" -j "${JOBS}" infernix-serve; then
        echo "  ERROR: Build failed for ${short_hash}!"
        build_status="FAILED"
        t_end="$(date +%s)"
        t_build=$((t_end - t_start))
        printf "%d\t%s\t%s\t%s\t0\t-\t%d\tFAILED\t%s\n" \
            "$target_idx" "$short_hash" "$commit_date" "$clabel" "$t_build" "$cdesc" >> "${MANIFEST_FILE}"
        target_idx=$((target_idx + 1))
        continue
    fi
    t_end="$(date +%s)"
    t_build=$((t_end - t_start))

    # Copy binary and compute metadata
    src_bin="${BUILD_DIR}/apps/infernix-serve"
    cp "${src_bin}" "${out_bin_path}"
    chmod +x "${out_bin_path}"

    # Also create convenient short-hash symlink
    ln -sf "${out_bin_name}" "${OUT_DIR}/infernix-serve-${short_hash}"

    bin_size="$(stat -c %s "${out_bin_path}")"
    bin_sha="$(sha256sum "${out_bin_path}" | awk '{print $1}')"

    printf "%d\t%s\t%s\t%s\t%s\t%s\t%d\tSUCCESS\t%s\n" \
        "$target_idx" "$short_hash" "$commit_date" "$clabel" "$bin_size" "$bin_sha" "$t_build" "$cdesc" >> "${MANIFEST_FILE}"

    echo "  Completed in ${t_build}s: ${out_bin_name} ($(numfmt --to=iec-i --suffix=B "${bin_size}"))"
    target_idx=$((target_idx + 1))
done

echo
echo "========================================================================================"
echo " BUILD COMPLETE: All milestone binaries saved to ${OUT_DIR}"
echo "========================================================================================"
echo
cat "${MANIFEST_FILE}" | column -t -s $'\t'
echo
