#!/usr/bin/env bash
#
# fetch_corpus.sh
#
# Fetches the four public-domain novels from Project Gutenberg (with fallback URLs/mirrors)
# into tools/niah_gate/corpus/ for the Needle-in-a-Haystack Gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_DIR="${SCRIPT_DIR}/corpus"
mkdir -p "${CORPUS_DIR}"

UA="Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

fetch_book() {
    local id="$1"
    local out_name="$2"
    local dest="${CORPUS_DIR}/${out_name}"

    if [[ -s "${dest}" ]] && grep -qiE '\*\*\* ?START OF|Chapter' "${dest}"; then
        echo "[ok] ${out_name} already exists ($(wc -c < "${dest}") bytes)"
        return 0
    fi

    # Build Gutenberg path components for aleph/mirror layout (e.g. 1342 -> 1/3/4/1342)
    local prefix=""
    local i
    for ((i = 0; i < ${#id} - 1; i++)); do
        prefix+="${id:i:1}/"
    done

    local urls=(
        "https://mirrors.xmission.com/gutenberg/${prefix}${id}/${id}-0.txt"
        "https://mirrors.xmission.com/gutenberg/${prefix}${id}/${id}.txt"
        "https://mirrors.xmission.com/gutenberg/cache/epub/${id}/pg${id}.txt"
        "https://gutenberg.pglaf.org/cache/epub/${id}/pg${id}.txt"
        "https://gutenberg.pglaf.org/${prefix}${id}/${id}-0.txt"
        "https://gutenberg.pglaf.org/${prefix}${id}/${id}.txt"
        "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
        "https://www.gutenberg.org/files/${id}/${id}-0.txt"
        "https://www.gutenberg.org/files/${id}/${id}.txt"
    )

    local tmp="${dest}.tmp"
    for url in "${urls[@]}"; do
        echo "[try] ${out_name} <- ${url}"
        if curl -fsSL --speed-time 20 --speed-limit 100 --max-time 180 -A "${UA}" "${url}" -o "${tmp}"; then
            if [[ -s "${tmp}" ]] && ! grep -qi "<html" <(head -n 5 "${tmp}"); then
                mv "${tmp}" "${dest}"
                echo "[ok] ${out_name} ($(wc -c < "${dest}") bytes)"
                return 0
            fi
        fi
        rm -f "${tmp}"
    done

    echo "ERROR: Failed to fetch ${out_name} (Gutenberg #${id}) from all URLs." >&2
    return 1
}

fetch_book 1342 "pride_and_prejudice.txt"
fetch_book 1661 "sherlock_holmes.txt"
fetch_book 1400 "great_expectations.txt"
fetch_book 2701 "moby_dick.txt"

ls -lh "${CORPUS_DIR}"
