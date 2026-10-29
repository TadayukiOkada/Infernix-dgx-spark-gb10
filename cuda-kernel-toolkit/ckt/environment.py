"""environment — GPU discovery, CUDA tool detection, environment snapshot.

Nothing in the toolkit hardcodes a GPU model.  The snapshot is assembled from:

1. ``device_probe``  — a tiny native CUDA program (compiled on demand with the
   *detected* arch) reporting every ``cudaDeviceProp`` field plus NVML telemetry.
2. ``nvidia-smi``    — driver-level fallback when the runtime probe is unusable.
3. tool ``--version`` invocations — recorded so results are traceable to a
   toolchain.

The resulting :class:`Environment` object exposes derived limits (register
granularity, theoretical occupancy ceilings, peak bandwidth source) used by the
performance model and the launch-parameter search.
"""

from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from . import spec as _spec
from .util import LOG, ToolError, ensure_dir, read_json, run, stable_hash, to_plain, which, write_json


# --------------------------------------------------------------------------- #
# Tool detection
# --------------------------------------------------------------------------- #

TOOLS = ("nvcc", "ncu", "nsys", "cuobjdump", "nvdisasm", "compute-sanitizer", "nvidia-smi")


@dataclass
class ToolInfo:
    name: str
    path: str | None
    version: str | None = None
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "version": self.version, "available": self.available}


_VERSION_PATTERNS = {
    "nvcc": re.compile(r"release\s+([0-9]+\.[0-9]+)", re.I),
    "ncu": re.compile(r"Version\s+([0-9][0-9.]*)", re.I),
    "nsys": re.compile(r"version\s+([0-9][0-9.]*-[0-9a-f]+)", re.I),
    "cuobjdump": re.compile(r"Copyright[^\n]*", re.I),
    "nvdisasm": re.compile(r"Copyright[^\n]*", re.I),
    "compute-sanitizer": re.compile(r"Version\s+([0-9][0-9.]*)", re.I),
}


def detect_tools() -> dict[str, ToolInfo]:
    """Locate each tool and record a version string where one is parseable."""
    found: dict[str, ToolInfo] = {}
    for name in TOOLS:
        path = which(name)
        info = ToolInfo(name=name, path=path, available=path is not None)
        if path:
            try:
                res = run([path, "--version"] if name != "nvcc" else [path, "--version"],
                          timeout=30, check=False, log=False)
                text = res.stdout + res.stderr
                pat = _VERSION_PATTERNS.get(name)
                if pat:
                    m = pat.search(text)
                    if m:
                        info.version = m.group(1) if m.groups() else m.group(0)
                if not info.version:
                    # Fall back to a CUDA release mention.
                    m = re.search(r"release\s+([0-9]+\.[0-9]+)", text, re.I)
                    if m:
                        info.version = m.group(1)
            except ToolError:
                pass
        found[name] = info
    return found


def nvcc_supported_arches(nvcc: str | None = None) -> list[str]:
    """Return the ``compute_XX`` list the installed nvcc accepts."""
    nvcc = nvcc or which("nvcc")
    if not nvcc:
        return []
    try:
        res = run([nvcc, "--list-gpu-arch"], timeout=30, check=False, log=False)
    except ToolError:
        return []
    return sorted(set(re.findall(r"compute_\d+", res.stdout)))


def nvcc_supported_codes(nvcc: str | None = None) -> list[str]:
    nvcc = nvcc or which("nvcc")
    if not nvcc:
        return []
    try:
        res = run([nvcc, "--list-gpu-code"], timeout=30, check=False, log=False)
    except ToolError:
        return []
    return sorted(set(re.findall(r"sm_\d+", res.stdout)))


# --------------------------------------------------------------------------- #
# Native probe build + parse
# --------------------------------------------------------------------------- #

_NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"


def _parse_records(text: str) -> dict[str, Any]:
    """Parse ``key<TAB>value`` records into a nested dict with numeric coercion."""
    root: dict[str, Any] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        key, _, raw = line.partition("\t")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        value: Any
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        node = root
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - malformed input
                break
        else:
            node[parts[-1]] = value
    return root


def _nvcc_arch_flags(arch: str) -> list[str]:
    """Build nvcc architecture flags preserving architecture-specific 'a'/'f' suffixes in PTX."""
    if arch.startswith("sm_"):
        compute_arch = "compute_" + arch[3:]
        return [f"-gencode=arch={compute_arch},code={arch}"]
    return [f"-arch={arch}"]


def build_native_probe(
    source: Path,
    output: Path,
    arch: str,
    *,
    nvcc: str | None = None,
    extra_flags: list[str] | None = None,
    timeout: float = 300.0,
) -> bool:
    """Compile one native probe; returns False (with a log line) on failure."""
    nvcc = nvcc or which("nvcc")
    if not nvcc:
        LOG.warning("nvcc unavailable; cannot build %s", source.name)
        return False
    ensure_dir(output.parent)
    flags = [*_nvcc_arch_flags(arch), "-O2", "-std=c++17", "-o", str(output), str(source)]
    flags += list(extra_flags or [])
    try:
        run([nvcc, *flags], timeout=timeout, check=True, log=True, what=f"build {source.name}")
    except ToolError as exc:
        LOG.warning("native probe build failed: %s", exc)
        return False
    return True


def run_device_probe(cache_dir: Path, arch: str, *, rebuild: bool = False) -> dict[str, Any] | None:
    """Build (if needed) and run ``device_probe``, returning parsed records."""
    src = _NATIVE_DIR / "device_probe.cu"
    if not src.exists():
        LOG.warning("device probe source missing at %s", src)
        return None
    exe = cache_dir / f"device_probe_{arch}"
    if rebuild or not exe.exists():
        nvml_flags: list[str] = []
        if _nvml_available():
            nvml_flags = ["-DHAVE_NVML", "-lnvidia-ml"]
        if not build_native_probe(src, exe, arch, extra_flags=nvml_flags):
            return None
    try:
        res = run([str(exe)], timeout=60, check=False, log=False)
    except ToolError as exc:
        LOG.warning("device probe execution failed: %s", exc)
        return None
    if res.returncode != 0 and not res.stdout.strip():
        LOG.warning("device probe exited %d: %s", res.returncode, res.stderr.strip()[:200])
        return None
    return _parse_records(res.stdout)


def _nvml_available() -> bool:
    import ctypes

    for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def run_bandwidth_probe(
    cache_dir: Path,
    arch: str,
    *,
    footprint_mb: int = 512,
    reps: int = 20,
    rebuild: bool = False,
    timeout: float = 900.0,
) -> dict[str, Any] | None:
    """Build (if needed) and run ``bandwidth_probe`` for a measured ceiling."""
    src = _NATIVE_DIR / "bandwidth_probe.cu"
    if not src.exists():
        return None
    exe = cache_dir / f"bandwidth_probe_{arch}"
    if rebuild or not exe.exists():
        if not build_native_probe(src, exe, arch):
            return None
    try:
        res = run([str(exe), "--mb", str(footprint_mb), "--reps", str(reps)],
                  timeout=timeout, check=False, log=False)
    except ToolError as exc:
        LOG.warning("bandwidth probe failed: %s", exc)
        return None
    if not res.stdout.strip():
        return None
    return _parse_records(res.stdout)


def run_compute_probe(
    cache_dir: Path,
    arch: str,
    *,
    reps: int = 10,
    rebuild: bool = False,
    timeout: float = 900.0,
) -> dict[str, Any] | None:
    """Build (if needed) and run ``compute_probe`` for measured arithmetic ceilings.

    A vendor peak is the best case for a pure-arithmetic kernel and can be far
    from what the part sustains once clocks and power limits apply; the FP64
    ratio in particular varies by two orders of magnitude across parts and must
    be measured rather than assumed.
    """
    src = _NATIVE_DIR / "compute_probe.cu"
    if not src.exists():
        return None
    exe = cache_dir / f"compute_probe_{arch}"
    if rebuild or not exe.exists():
        if not build_native_probe(src, exe, arch):
            return None
    try:
        res = run([str(exe), "--reps", str(reps)], timeout=timeout, check=False, log=False)
    except ToolError as exc:
        LOG.warning("compute probe failed: %s", exc)
        return None
    if not res.stdout.strip():
        return None
    return _parse_records(res.stdout)


# --------------------------------------------------------------------------- #
# nvidia-smi fallback
# --------------------------------------------------------------------------- #

_SMI_FIELDS = [
    "name",
    "compute_cap",
    "driver_version",
    "memory.total",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.limit",
    "temperature.gpu",
]


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Minimal SMi snapshot; empty dict when the tool is unavailable."""
    exe = which("nvidia-smi")
    if not exe:
        return {}
    try:
        res = run([exe, f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader"],
                  timeout=30, check=False, log=False)
        rows = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            rows.append(dict(zip([f for f in _SMI_FIELDS], parts)))
        # Also capture the driver/toolkit banner.
        info = run([exe], timeout=30, check=False, log=False).stdout
        header = {}
        for key in ("Driver Version", "CUDA Version"):
            m = re.search(rf"{re.escape(key)}\s*:\s*([0-9][0-9.]*)", info)
            if m:
                header[key.lower().replace(" ", "_")] = m.group(1)
        return {"gpus": rows, "banner": header}
    except ToolError:
        return {}


# --------------------------------------------------------------------------- #
# Device / Environment models
# --------------------------------------------------------------------------- #


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "N/A" or value == "[N/A]":
        return None
    try:
        return float(str(value).replace("MHz", "").strip())
    except (TypeError, ValueError):
        return None


@dataclass
class DeviceInfo:
    """Fully-resolved view of one GPU, with derived architecture limits."""

    index: int = 0
    name: str = "unknown"
    compute_capability: str = "0.0"
    cc_major: int = 0
    cc_minor: int = 0
    arch: str = ""
    multi_processor_count: int = 0
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_multiprocessor: int = 2048
    max_blocks_per_multiprocessor: int = 32
    regs_per_multiprocessor: int = 65536
    shared_mem_per_block: int = 49152
    shared_mem_per_multiprocessor: int = 102400
    shared_mem_per_block_optin: int = 49152
    reserved_shared_mem_per_block: int = 0
    l2_cache_size: int = 0
    persisting_l2_cache_max_size: int = 0
    access_policy_max_window_size: int = 0
    total_global_mem: int = 0
    memory_bus_width: int | None = None
    memory_clock_rate_khz: int | None = None
    clock_rate_khz: int | None = None
    integrated_uma: bool = False
    features: dict[str, bool] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # Derived ISA facts.
    arch_facts: _spec.ArchFacts | None = None
    # Measured / configured peaks.  None means genuinely unknown.
    measured_bandwidth_gbps: float | None = None
    measured_read_bandwidth_gbps: float | None = None
    measured_write_bandwidth_gbps: float | None = None
    measured_copy_bandwidth_gbps: float | None = None
    measured_smem_bandwidth_gbps: float | None = None
    naive_bus_bandwidth_gbps: float | None = None
    bandwidth_source: str = "unknown"
    dtype_peaks: dict[str, _spec.DTypePeak] = field(default_factory=dict)
    measured_ratio_notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def reg_alloc_step(self) -> int:
        return self.arch_facts.reg_alloc_step if self.arch_facts else 8

    @property
    def reg_alloc_unit_warp(self) -> int:
        return self.arch_facts.reg_alloc_unit_warp if self.arch_facts else 256

    @property
    def smem_alloc_unit_bytes(self) -> int:
        return self.arch_facts.smem_alloc_unit_bytes if self.arch_facts else 128

    @property
    def max_regs_per_thread_hw(self) -> int:
        """Hardware cap on registers per thread (255 on every modern arch)."""
        return 255

    @property
    def max_warps_per_multiprocessor(self) -> int:
        return max(1, self.max_threads_per_multiprocessor // self.warp_size)

    def round_up_regs(self, regs: int) -> int:
        step = self.reg_alloc_step
        return ((regs + step - 1) // step) * step

    def peak_bandwidth_gbps(self) -> tuple[float | None, str]:
        """Best available DRAM/LPDDR bandwidth estimate and its provenance."""
        if self.measured_bandwidth_gbps:
            return self.measured_bandwidth_gbps, "measured"
        if self.naive_bus_bandwidth_gbps and (self.arch_facts is None or self.arch_facts.dedicated_memory_bus):
            return self.naive_bus_bandwidth_gbps, "derived-bus-arithmetic"
        return None, "unknown"

    def apply_compute_probe(self, probe: dict[str, Any]) -> None:
        """Populate measured arithmetic ceilings from ``compute_probe`` output.

        Only measured values enter ``dtype_peaks``; a dtype the probe did not
        measure keeps no entry, so the model reports the ceiling as unavailable
        instead of substituting a vendor figure.
        """
        if not probe:
            return
        fp32 = _to_float(probe.get("fp32_fma", {}).get("tflops"))
        fp64 = _to_float(probe.get("fp64_fma", {}).get("tflops"))
        fp16_simt = _to_float(probe.get("fp16_simt", {}).get("tflops"))
        bf16_simt = _to_float(probe.get("bf16_simt", {}).get("tflops"))
        tf32_mma = _to_float(probe.get("tf32_mma", {}).get("tflops"))
        fp16_mma = _to_float(probe.get("fp16_mma", {}).get("tflops"))
        fp16_f16acc_mma = _to_float(probe.get("fp16_f16acc_mma", {}).get("tflops"))
        bf16_mma = _to_float(probe.get("bf16_mma", {}).get("tflops"))
        int8_mma = _to_float(probe.get("int8_mma", {}).get("tops"))
        int4_mma = _to_float(probe.get("int4_mma", {}).get("tops"))
        fp8_e4m3_mma = _to_float(probe.get("fp8_e4m3_mma", {}).get("tflops"))
        fp8_e5m2_mma = _to_float(probe.get("fp8_e5m2_mma", {}).get("tflops"))
        mxfp8_mma = _to_float(probe.get("mxfp8_mma", {}).get("tflops"))
        mxfp6_mma = _to_float(probe.get("mxfp6_mma", {}).get("tflops"))
        mxfp4_mma = _to_float(probe.get("mxfp4_mma", {}).get("tflops"))
        nvfp4_mma = _to_float(probe.get("nvfp4_mma", {}).get("tflops"))
        int_add = _to_float(probe.get("int_add", {}).get("gopp_per_s"))
        smem_gbps = _to_float(probe.get("smem_load", {}).get("gbps"))

        if fp32:
            self.dtype_peaks["fp32"] = _spec.DTypePeak("fp32", fp32, "measured")
        if fp64:
            self.dtype_peaks["fp64"] = _spec.DTypePeak("fp64", fp64, "measured")
        if fp16_simt:
            self.dtype_peaks["fp16_simt"] = _spec.DTypePeak("fp16_simt", fp16_simt, "measured")
        if bf16_simt:
            self.dtype_peaks["bf16_simt"] = _spec.DTypePeak("bf16_simt", bf16_simt, "measured")
        if tf32_mma:
            self.dtype_peaks["tf32"] = _spec.DTypePeak("tf32", tf32_mma, "measured")
        if fp16_mma:
            self.dtype_peaks["fp16"] = _spec.DTypePeak("fp16", fp16_mma, "measured")
            self.dtype_peaks["fp16_f32acc"] = _spec.DTypePeak("fp16_f32acc", fp16_mma, "measured")
        if fp16_f16acc_mma:
            self.dtype_peaks["fp16_f16acc"] = _spec.DTypePeak("fp16_f16acc", fp16_f16acc_mma, "measured")
        if bf16_mma:
            self.dtype_peaks["bf16"] = _spec.DTypePeak("bf16", bf16_mma, "measured")
        elif fp16_mma:
            self.dtype_peaks["bf16"] = _spec.DTypePeak("bf16", fp16_mma, "measured")
        if int8_mma:
            self.dtype_peaks["int8"] = _spec.DTypePeak("int8", int8_mma, "measured")
        if int4_mma:
            self.dtype_peaks["int4"] = _spec.DTypePeak("int4", int4_mma, "measured")
        if fp8_e4m3_mma:
            self.dtype_peaks["fp8_e4m3"] = _spec.DTypePeak("fp8_e4m3", fp8_e4m3_mma, "measured")
        if fp8_e5m2_mma:
            self.dtype_peaks["fp8_e5m2"] = _spec.DTypePeak("fp8_e5m2", fp8_e5m2_mma, "measured")
        if mxfp8_mma:
            self.dtype_peaks["mxfp8"] = _spec.DTypePeak("mxfp8", mxfp8_mma, "measured")
            # Primary fp8 ceiling uses block-scaled MXFP8 when faster than plain fp8_e4m3
            self.dtype_peaks["fp8"] = _spec.DTypePeak("fp8", max(mxfp8_mma, fp8_e4m3_mma or 0.0), "measured")
        elif fp8_e4m3_mma:
            self.dtype_peaks["fp8"] = _spec.DTypePeak("fp8", fp8_e4m3_mma, "measured")
        if mxfp6_mma:
            self.dtype_peaks["mxfp6"] = _spec.DTypePeak("mxfp6", mxfp6_mma, "measured")
            self.dtype_peaks["fp6"] = _spec.DTypePeak("fp6", mxfp6_mma, "measured")
        if mxfp4_mma:
            self.dtype_peaks["mxfp4"] = _spec.DTypePeak("mxfp4", mxfp4_mma, "measured")
        if nvfp4_mma:
            self.dtype_peaks["nvfp4"] = _spec.DTypePeak("nvfp4", nvfp4_mma, "measured")
        if mxfp4_mma or nvfp4_mma:
            self.dtype_peaks["fp4"] = _spec.DTypePeak("fp4", max(mxfp4_mma or 0.0, nvfp4_mma or 0.0), "measured")
        if int_add:
            self.dtype_peaks["int32"] = _spec.DTypePeak("int32", int_add / 1000.0, "measured")
        if smem_gbps:
            self.measured_smem_bandwidth_gbps = smem_gbps

        def _add_note(note: str) -> None:
            if note not in self.measured_ratio_notes:
                self.measured_ratio_notes = list(self.measured_ratio_notes) + [note]

        if fp32 and fp64 and fp32 / fp64 > 4.0:
            _add_note(
                f"Measured FP64:FP32 throughput ratio is {fp32 / fp64:.1f}:1. FP64 work on this "
                "part is dramatically slower than FP32 and must be counted as such when "
                "classifying compute-bound kernels."
            )
        if fp16_mma and fp16_f16acc_mma:
            if fp16_f16acc_mma / fp16_mma > 1.3:
                _add_note(
                    f"Measured FP16 Tensor Core with FP16 accumulation ({fp16_f16acc_mma:.1f} TFLOP/s) is "
                    f"{fp16_f16acc_mma / fp16_mma:.2f}x faster than FP32 accumulation ({fp16_mma:.1f} TFLOP/s)."
                )
            else:
                _add_note(
                    f"Measured FP16 Tensor Core has full-rate FP32 accumulation ({fp16_mma:.1f} TFLOP/s vs "
                    f"{fp16_f16acc_mma:.1f} TFLOP/s FP16 accumulate; 1:1 parity on this device)."
                )
        if fp8_e4m3_mma and mxfp8_mma and mxfp8_mma / fp8_e4m3_mma > 1.3:
            _add_note(
                f"Measured block-scaled MXFP8 Tensor Core ({mxfp8_mma:.1f} TFLOP/s) is "
                f"{mxfp8_mma / fp8_e4m3_mma:.2f}x faster than plain FP8-E4M3 with FP32 accumulation "
                f"({fp8_e4m3_mma:.1f} TFLOP/s). Prefer block-scaled mxf8f6f4 MMA with identity UE8M0=0x7f "
                "when accumulating in FP32."
            )
        if int8_mma and int4_mma and int8_mma / int4_mma > 1.5:
            _add_note(
                f"Legacy INT4 (s4) MMA ({int4_mma:.1f} TOP/s) is {int8_mma / int4_mma:.1f}x slower than "
                f"INT8 (s8) MMA ({int8_mma:.1f} TOP/s)"
                + (f" and {(nvfp4_mma or mxfp4_mma or 0.0) / int4_mma:.1f}x slower than native FP4 (e2m1) MMA "
                   f"({max(nvfp4_mma or 0.0, mxfp4_mma or 0.0):.1f} TFLOP/s)" if (nvfp4_mma or mxfp4_mma) else "")
                + "; avoid s4 mma.sync on this architecture and use s8 or e2m1 (mxfp4/nvfp4) instead."
            )

    @property
    def compute_ceiling_tflops(self) -> tuple[float | None, str]:
        """Measured FP32 arithmetic ceiling, if the compute probe has been run."""
        dp = self.dtype_peaks.get("fp32")
        if dp is not None:
            tflops = getattr(dp, "tflops", None) if not isinstance(dp, dict) else dp.get("tflops")
            if tflops:
                return float(tflops), "measured"
        return None, "unknown"

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        d["reg_alloc_step"] = self.reg_alloc_step
        d["reg_alloc_unit_warp"] = self.reg_alloc_unit_warp
        d["smem_alloc_unit_bytes"] = self.smem_alloc_unit_bytes
        d["max_warps_per_multiprocessor"] = self.max_warps_per_multiprocessor
        peak, source = self.peak_bandwidth_gbps()
        d["peak_bandwidth_gbps"] = peak
        d["peak_bandwidth_source"] = source
        return d

    # ------------------------------------------------------------------ #
    @classmethod
    def from_probe(cls, probe: dict[str, Any], index: int = 0) -> "DeviceInfo":
        dev = (probe.get("device") or {}).get(str(index), {})
        if not dev:
            return cls()
        major = _to_int(dev.get("compute_capability_major")) or 0
        minor = _to_int(dev.get("compute_capability_minor")) or 0
        info = cls(
            index=index,
            name=str(dev.get("name", "unknown")),
            compute_capability=str(dev.get("compute_capability", f"{major}.{minor}")),
            cc_major=major,
            cc_minor=minor,
            arch=str(dev.get("arch", f"sm_{major}{minor}")),
            multi_processor_count=_to_int(dev.get("multi_processor_count")) or 0,
            warp_size=_to_int(dev.get("warp_size")) or 32,
            max_threads_per_block=_to_int(dev.get("max_threads_per_block")) or 1024,
            max_threads_per_multiprocessor=_to_int(dev.get("max_threads_per_multiprocessor")) or 2048,
            max_blocks_per_multiprocessor=_to_int(dev.get("max_blocks_per_multiprocessor")) or 32,
            regs_per_multiprocessor=_to_int(dev.get("regs_per_multiprocessor")) or 65536,
            shared_mem_per_block=_to_int(dev.get("shared_mem_per_block")) or 49152,
            shared_mem_per_multiprocessor=_to_int(dev.get("shared_mem_per_multiprocessor")) or 102400,
            shared_mem_per_block_optin=_to_int(dev.get("shared_mem_per_block_optin")) or 49152,
            reserved_shared_mem_per_block=_to_int(dev.get("reserved_shared_mem_per_block")) or 0,
            l2_cache_size=_to_int(dev.get("l2_cache_size")) or 0,
            persisting_l2_cache_max_size=_to_int(dev.get("persisting_l2_cache_max_size")) or 0,
            access_policy_max_window_size=_to_int(dev.get("access_policy_max_window_size")) or 0,
            total_global_mem=_to_int(dev.get("total_global_mem")) or 0,
            memory_bus_width=_to_int(dev.get("memory_bus_width")),
            memory_clock_rate_khz=_to_int(dev.get("memory_clock_rate_khz")),
            clock_rate_khz=_to_int(dev.get("clock_rate_khz")),
            integrated_uma=bool(_to_int(dev.get("integrated_uma"))),
            features={
                "tensor_cores": bool(_to_int(dev.get("supports_tensor_cores"))),
                "fp16": bool(_to_int(dev.get("supports_fp16"))),
                "bf16": bool(_to_int(dev.get("supports_bf16"))),
                "tf32": bool(_to_int(dev.get("supports_tf32"))),
                "int8_mma": bool(_to_int(dev.get("supports_int8_mma"))),
                "cp_async": bool(_to_int(dev.get("supports_cp_async"))),
                "tma": bool(_to_int(dev.get("supports_tma"))),
                "cluster": bool(_to_int(dev.get("supports_cluster"))),
                "fp8": bool(_to_int(dev.get("supports_fp8"))),
                "fp4": bool(_to_int(dev.get("supports_fp4"))),
                "mxfp8": bool(_to_int(dev.get("supports_mxfp8"))),
                "mxfp6": bool(_to_int(dev.get("supports_mxfp6"))),
                "mxfp4": bool(_to_int(dev.get("supports_mxfp4"))),
                "nvfp4": bool(_to_int(dev.get("supports_nvfp4"))),
                "block_scale_mma": bool(_to_int(dev.get("supports_block_scale_mma"))),
                "distributed_shared_mem": bool(_to_int(dev.get("supports_distributed_shared_mem"))),
                "cooperative_launch": bool(_to_int(dev.get("cooperative_launch"))),
                "async_allocator": bool(_to_int(dev.get("supports_async_allocator"))),
            },
            raw=dev,
        )
        info.arch_facts = _spec.arch_facts(major, minor)
        if info.memory_bus_width and info.memory_clock_rate_khz:
            info.naive_bus_bandwidth_gbps = (
                2.0 * info.memory_clock_rate_khz * 1e3 * (info.memory_bus_width / 8.0) / 1e9
            )
        return info

    @classmethod
    def from_smi(cls, index: int = 0) -> "DeviceInfo":
        """Degraded fallback when the CUDA runtime probe cannot run.

        ``nvidia-smi`` reports far fewer fields; everything unavailable stays
        ``None``/0 and is flagged as unknown rather than guessed.  The caller is
        responsible for recording the reduced-confidence warning.
        """
        smi = nvidia_smi_snapshot()
        gpus = smi.get("gpus") or []
        if index >= len(gpus):
            return cls(index=index)
        row = gpus[index]
        cc = str(row.get("compute_cap") or "")
        m = re.match(r"(\d+)\.(\d+)", cc)
        major = int(m.group(1)) if m else 0
        minor = int(m.group(2)) if m else 0
        info = cls(
            index=index,
            name=str(row.get("name") or "unknown"),
            compute_capability=cc or "0.0",
            cc_major=major,
            cc_minor=minor,
            arch=f"sm_{major}{minor}" if m else "",
        )
        # These are ISA/architecture defaults; smem and register fields are NOT
        # reported by nvidia-smi and remain at conservative placeholders flagged
        # by `raw["degraded"]`.
        info.raw = {"degraded": True, "smi_row": row}
        info.arch_facts = _spec.arch_facts(major, minor)
        return info


def detect_profiling_capabilities() -> dict[str, Any]:
    """Inspect Linux kernel performance event paranoid level and pointer restriction."""
    paranoid: int | None = None
    kptr: int | None = None
    try:
        p_path = Path("/proc/sys/kernel/perf_event_paranoid")
        if p_path.exists():
            paranoid = int(p_path.read_text().strip())
    except Exception:
        pass

    try:
        k_path = Path("/proc/sys/kernel/kptr_restrict")
        if k_path.exists():
            kptr = int(k_path.read_text().strip())
    except Exception:
        pass

    import glob

    configured_paranoid: int | None = None
    configured_kptr: int | None = None
    for conf in sorted(glob.glob("/etc/sysctl.d/*.conf") + ["/etc/sysctl.conf"]):
        try:
            content = Path(conf).read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]
                if k == "kernel.perf_event_paranoid":
                    configured_paranoid = int(v)
                elif k == "kernel.kptr_restrict":
                    configured_kptr = int(v)
        except Exception:
            pass

    pending_reload = False
    if configured_paranoid is not None and configured_paranoid != paranoid:
        pending_reload = True
    if configured_kptr is not None and configured_kptr != kptr:
        pending_reload = True

    return {
        "perf_event_paranoid": paranoid,
        "kptr_restrict": kptr,
        "configured_perf_event_paranoid": configured_paranoid,
        "configured_kptr_restrict": configured_kptr,
        "pending_sysctl_reload": pending_reload,
        "cpu_sampling_available": paranoid is not None and paranoid <= 1,
        "pmu_events_available": paranoid is not None and paranoid <= 0,
        "kernel_symbols_unlocked": kptr == 0,
    }


@dataclass
class Environment:
    """Complete, reproducible snapshot of hardware + toolchain."""

    toolkit_version: str = __version__
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    device: DeviceInfo = field(default_factory=DeviceInfo)
    tools: dict[str, ToolInfo] = field(default_factory=dict)
    nvcc_arches: list[str] = field(default_factory=list)
    nvcc_codes: list[str] = field(default_factory=list)
    smi: dict[str, Any] = field(default_factory=dict)
    bandwidth: dict[str, Any] = field(default_factory=dict)
    compute: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    probe_available: bool = False
    profiling: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "toolkit_version": self.toolkit_version,
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "device": self.device.to_dict(),
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "nvcc_arches": self.nvcc_arches,
            "nvcc_codes": self.nvcc_codes,
            "nvidia_smi": self.smi,
            "bandwidth": self.bandwidth,
            "compute": self.compute,
            "probe_available": self.probe_available,
            "profiling": self.profiling,
            "warnings": self.warnings,
        }

    @property
    def fingerprint(self) -> str:
        """Stable hash of the hardware+toolchain identity.

        Used to namespace caches so a measurement is never reused across
        incompatible hardware or tools.
        """
        d = self.device
        return stable_hash(
            {
                "gpu": d.name,
                "cc": d.compute_capability,
                "sms": d.multi_processor_count,
                "l2": d.l2_cache_size,
                "smem": d.shared_mem_per_multiprocessor,
                "regs": d.regs_per_multiprocessor,
                "uma": d.integrated_uma,
                "tools": {k: (v.version or "") for k, v in sorted(self.tools.items())},
                "arches": self.nvcc_arches,
            }
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def discover(
        cls,
        cache_dir: str | Path,
        *,
        device_index: int = 0,
        rebuild_native: bool = False,
        measure_bandwidth: bool = False,
        measure_compute: bool = False,
        bandwidth_mb: int = 512,
    ) -> "Environment":
        """Run the full discovery pipeline."""
        import platform as _platform
        import socket
        import sys

        cache_dir = ensure_dir(cache_dir)
        env = cls(
            hostname=socket.gethostname(),
            platform=_platform.platform(),
            python_version=sys.version.split()[0],
        )
        env.tools = detect_tools()
        env.profiling = detect_profiling_capabilities()

        nvcc_path = env.tools.get("nvcc").path if env.tools.get("nvcc") else None
        env.nvcc_arches = nvcc_supported_arches(nvcc_path)
        env.nvcc_codes = nvcc_supported_codes(nvcc_path)
        # SMi is collected before the native probe because arch selection reads
        # the reported compute capability from it.
        env.smi = nvidia_smi_snapshot()

        # Determine the arch to compile the native probes for.
        arch = _pick_arch(env)
        LOG.debug("native probe arch: %s", arch)
        probe = run_device_probe(cache_dir, arch, rebuild=rebuild_native)
        if probe and "device" in probe:
            env.probe_available = True
            env.device = DeviceInfo.from_probe(probe, device_index)
        else:
            env.warnings.append(
                "CUDA runtime probe unavailable; device limits could not be "
                "measured. Set CUDA_VISIBLE_DEVICES or install the CUDA runtime, "
                "or run `ckt env --smi-only` for a degraded snapshot."
            )
            env.device = DeviceInfo.from_smi(device_index)

        # Bandwidth: measure when asked, otherwise record the naive figure only.
        if measure_bandwidth:
            bw = run_bandwidth_probe(cache_dir, arch, footprint_mb=bandwidth_mb, rebuild=rebuild_native)
            if bw:
                env.bandwidth = bw.get("bandwidth", {})
                meas = _to_float(bw.get("bandwidth", {}).get("sustained_read_gbps"))
                if meas:
                    env.device.measured_bandwidth_gbps = meas
                    env.device.bandwidth_source = "measured"
                env.device.measured_read_bandwidth_gbps = _to_float(bw.get("bandwidth", {}).get("read", {}).get("gbps")) or meas
                env.device.measured_write_bandwidth_gbps = _to_float(bw.get("bandwidth", {}).get("write", {}).get("gbps"))
                env.device.measured_copy_bandwidth_gbps = _to_float(bw.get("bandwidth", {}).get("copy", {}).get("gbps"))

        # Compute ceilings: measured only.  A vendor peak is not substituted,
        # because the achievable figure for the part in situ can differ by a
        # large factor (FP64 in particular).
        if measure_compute:
            cp = run_compute_probe(cache_dir, arch, rebuild=rebuild_native)
            if cp:
                env.compute = cp
                env.device.apply_compute_probe(cp)
            else:
                env.warnings.append(
                    "Compute probe unavailable; arithmetic ceilings will be reported as "
                    "unknown rather than taken from a specification table."
                )
        if env.device.bandwidth_source == "unknown":
            if env.device.naive_bus_bandwidth_gbps and (env.device.arch_facts is None or env.device.arch_facts.dedicated_memory_bus):
                env.device.bandwidth_source = "derived-bus-arithmetic"
            else:
                env.warnings.append(
                    "No measured bandwidth and no trustworthy bus arithmetic "
                    "(unified-memory device). Run with --measure-bandwidth or set "
                    "peak_bandwidth_gbps in the config; roofline efficiency will "
                    "be reported as unknown until then."
                )
        return env

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return read_json(path)


def _pick_arch(env: "Environment") -> str:
    """Choose the CUDA arch string for native probes.

    Prefers an arch-specific form (``sm_121a``) when the toolkit advertises one,
    since arch-specific ISA extensions change which instructions are emitted.
    """
    smi = env.smi.get("gpus") if env.smi else None
    if smi:
        cc = smi[0].get("compute_cap") or ""
        m = re.match(r"(\d+)\.(\d+)", str(cc))
        if m:
            base = f"sm_{m.group(1)}{m.group(2)}"
            for cand in (base + "a", base + "f", base):
                if _arch_compiles(cand, env):
                    return cand
            return base
    return env.nvcc_codes[-1] if env.nvcc_codes else "sm_90"


_ARCH_OK_CACHE: dict[tuple[str, str], bool] = {}


def _arch_compiles(arch: str, env: "Environment") -> bool:
    nvcc = env.tools.get("nvcc").path if env.tools.get("nvcc") else None
    if not nvcc:
        return False
    key = (nvcc, arch)
    if key in _ARCH_OK_CACHE:
        return _ARCH_OK_CACHE[key]
    import tempfile

    src = Path(tempfile.mkdtemp()) / "probe.cu"
    src.write_text("__global__ void k(){}\n")
    try:
        res = run([nvcc, *_nvcc_arch_flags(arch), "-c", str(src), "-o", str(src.with_suffix(".o"))],
                  timeout=120, check=False, log=False)
        ok = res.returncode == 0
    except ToolError:
        ok = False
    _ARCH_OK_CACHE[key] = ok
    return ok


def gpu_contention_warnings(env: "Environment") -> list[str]:
    """Warn about conditions that destabilise measurement."""
    warns: list[str] = []
    nv = (env.to_dict().get("bandwidth") or {})
    dev_raw = env.device.raw
    # Competing workloads: other processes on a shared UMA device.
    if env.device.integrated_uma:
        warns.append(
            "Unified-memory device: CPU and other processes contend for the same "
            "LPDDR bus. Keep host activity quiet during measurement."
        )
    util = dev_raw.get("nvml_gpu_utilization_pct")
    if isinstance(util, int) and util > 5:
        warns.append(f"GPU busy at snapshot time ({util}% util); timings may be noisy.")
    return warns
