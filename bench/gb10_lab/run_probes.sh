#!/usr/bin/env bash
# run_probes.sh — GB10 feature-support matrix builder.
#
# Compiles each probe translation unit INDEPENDENTLY against the installed GB10
# target (sm_121a), records compile status, runtime status, exit code, and the
# SASS instruction families actually emitted. A failure in one probe cannot
# poison another because each is a separate nvcc invocation.
#
# Emits: results/raw/feature_matrix.jsonl  (one record per candidate)
#        results/profiles/sass/<probe>.sass.txt  (instruction evidence)
#        results/logs/probes.log           (raw stdout/stderr)
set -u

ARCH="${GB10_ARCH:-121a}"
NVCC="${NVCC:-nvcc}"
ROOT="$(cd "$(dirname "$0")" && pwd)"          # bench/gb10_lab (dir holding this script)
REPO="$(cd "$ROOT/../.." && pwd)"              # repo root
OUT_RAW="$REPO/results/raw"
OUT_SASS="$REPO/results/profiles/sass"
OUT_LOG="$REPO/results/logs"
mkdir -p "$OUT_RAW" "$OUT_SASS" "$OUT_LOG"
MATRIX="$OUT_RAW/feature_matrix.jsonl"
PROBELOG="$OUT_LOG/probes.log"
: > "$MATRIX"
: > "$PROBELOG"

TS="$(date -Is)"
echo "run_probes: arch=sm_$ARCH ts=$TS" | tee -a "$PROBELOG"

# Each entry: name|category|file|extra_flags|library_link
CANDIDATES=(
  "dp4a_int|packed_int|probe_dp4a.cu||"
  "dp2a|packed_int|probe_dp2a.cu||"
  "vop4_byte|packed_int|probe_vop4.cu||"
  "pack_sat_wrap|packed_int|probe_pack_sat_wrap.cu||"
  "int4_packed|packed_int|probe_int4.cu||"
  "ternary_dot|ternary|probe_ternary.cu||"
  "lookup_partialsum|lookup|probe_lookup.cu||"
  "popc_xnor_binary|binary|probe_popc.cu||"
  "ptx_dp4a_inline|experimental_ptx|probe_ptx_dp4a.cu||"
  "imma_int8_tc|tensor_core_int8|probe_imma.cu||"
  "cublaslt_int8_gemm|tensor_core_int8|probe_cublaslt_int8.cu|-lcublasLt|"
  "gemm_precisions|tensor_core_float|probe_gemm_precisions.cu|-lcublas -lcublasLt|"
  "cp.async_perthread|data_movement|probe_async_copy.cu||"
  "coop_groups_reduce|reduction|probe_coop_reduce.cu||"
)

classify() { # $1=compile_status $2=exit_code $3=stdout
  if [ "$1" != "ok" ]; then echo "fails_to_compile"; return; fi
  case "$2" in
    0) if echo "$3" | grep -q "OK"; then echo "confirmed_supported"; else echo "runs_unverified"; fi;;
    2) echo "compiles_runtime_fails";;
    3) echo "runs_result_mismatch";;
    4|5) echo "unsupported_by_library_api";;
    6) echo "compiles_runtime_fails";;
    *) echo "runs_nonzero_exit";;
  esac
}

for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r name cat file flags link <<< "$entry"
  src="$ROOT/probes/$file"
  bin="$OUT_LOG/probe_$name"
  sass="$OUT_SASS/$name.sass.txt"
  compile_status="ok"; clog=""
  # Compile (independent). Capture full stderr.
  clog="$( $NVCC -O3 -lineinfo -arch=sm_$ARCH "$src" -o "$bin" $flags 2>&1 )"; ccode=$?
  if [ $ccode -ne 0 ]; then compile_status="fail"; fi
  printf '%s\n---- compile cmd: nvcc -arch=sm_%s %s %s ----\n%s\n' \
    "== probe $name ($cat) ==" "$ARCH" "$file" "$flags" "$clog" >> "$PROBELOG"

  run_status="not_run"; exit_code=""; out=""
  sass_frag="null"
  if [ "$compile_status" = "ok" ]; then
    out="$("$bin" 2>&1)"; exit_code=$?
    printf '%s\n' "$out" >> "$PROBELOG"
    # SASS evidence: instruction families present in the compiled binary.
    $NVCC -O3 -arch=sm_$ARCH -cubin "$src" -o "$OUT_LOG/$name.cubin" $flags 2>/dev/null
    if [ -f "$OUT_LOG/$name.cubin" ]; then
      cuobjdump --dump-sass "$OUT_LOG/$name.cubin" > "$sass" 2>/dev/null
      if [ -s "$sass" ]; then
        fams=$(grep -oE 'IDP\.[0-9A-Z.]+|\b(VABSDIFF4|VADD4|VSUB4|VMNMX|VIMNMX|POPC|IADD3|IMAD|LOP3|FADD|FFMA|LDGSTS|CPASYNC|UTMALDG|UBLKCP|HMMA|QMMA|OMMA|IMMA|BMMA|FMUL2|HADD2)\b' "$sass" | sort -u | paste -sd, -)
        sass_frag="\"$fams\""
      fi
    fi
  fi
  cls="$(classify "$compile_status" "$exit_code" "$out")"
  # Extract every performance line the probe emitted (median_us + ops_per_s or
  # GB_per_s) so the feature matrix shows not just support but measured speed.
  perf_json="[]"
  if [ "$compile_status" = "ok" ]; then
    perf_json=$(printf '%s' "$out" | python3 -c '
import sys,re,json
res=[]
for ln in sys.stdin:
    ln=ln.strip()
    if "median_us=" not in ln: continue
    name=ln.split(" ",1)[0]
    d={"impl":name}
    def num(k):
        m=re.search(k+r"=([0-9.]+)", ln)
        return float(m.group(1)) if m else None
    for k in ("median_us","p10_us","p90_us","ops_per_s","TOPS","GB_per_s"):
        v=num(k)
        if v is not None: d[k.lower()]=v
    res.append(d)
print(json.dumps(res))
' 2>/dev/null || echo "[]")
  fi
  rec="{\"ts\":\"$TS\",\"arch\":\"sm_$ARCH\",\"candidate\":\"$name\",\"category\":\"$cat\",\"source\":\"bench/gb10_lab/probes/$file\",\"compile\":\"$compile_status\",\"exit\":${exit_code:-null},\"classification\":\"$cls\",\"sass_families\":$sass_frag,\"perf\":$perf_json,\"command\":\"nvcc -O3 -arch=sm_$ARCH $src $flags\"}"
  echo "$rec" >> "$MATRIX"
  # Console: show best measured throughput per probe for quick ranking.
  perf_summary=$(printf '%s' "$perf_json" | python3 -c '
import sys,json
arr=json.load(sys.stdin)
parts=[]
for d in arr:
    impl=d.get("impl","?")
    if "tops" in d: parts.append("%s=%.3fTOPS" % (impl, d["tops"]))
    elif "gb_per_s" in d: parts.append("%s=%.0fGB/s" % (impl, d["gb_per_s"]))
print(", ".join(parts) if parts else "-")
' 2>/dev/null || echo "-")
  printf '  %-22s %-24s compile=%-4s exit=%-3s perf=%s\n' "$name" "$cls" "$compile_status" "${exit_code:-NA}" "$perf_summary"
done

echo "Feature matrix written to $MATRIX"
