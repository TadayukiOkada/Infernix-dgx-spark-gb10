"""Hardware and software profile capture for offline autotuning.

Policies are never keyed merely by marketing name; they are bound to a stable
hardware fingerprint, compute capability, resource limits, and software toolchain
revisions to ensure reproducibility and policy isolation across GPU architectures.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..environment import DeviceInfo, Environment
from ..util import stable_hash, to_plain


@dataclass
class HardwareProfile:
    """Detailed hardware capability profile for the target GPU."""

    gpu_name: str
    fingerprint: str
    compute_capability: str
    sm_count: int
    max_threads_per_sm: int
    max_threads_per_block: int
    regs_per_sm: int
    shared_mem_per_block_default: int
    shared_mem_per_block_opt_in: int
    shared_mem_per_sm: int
    l2_cache_size: int
    bandwidth_gbps: float
    sustained_read_gbps: float
    features: list[str] = field(default_factory=list)
    arch_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        return cls(
            gpu_name=str(data.get("gpu_name", "Unknown GPU")),
            fingerprint=str(data.get("fingerprint", "")),
            compute_capability=str(data.get("compute_capability", "0.0")),
            sm_count=int(data.get("sm_count", 0)),
            max_threads_per_sm=int(data.get("max_threads_per_sm", 1536)),
            max_threads_per_block=int(data.get("max_threads_per_block", 1024)),
            regs_per_sm=int(data.get("regs_per_sm", 65536)),
            shared_mem_per_block_default=int(data.get("shared_mem_per_block_default", 49152)),
            shared_mem_per_block_opt_in=int(data.get("shared_mem_per_block_opt_in", 49152)),
            shared_mem_per_sm=int(data.get("shared_mem_per_sm", 102400)),
            l2_cache_size=int(data.get("l2_cache_size", 0)),
            bandwidth_gbps=float(data.get("bandwidth_gbps", 0.0)),
            sustained_read_gbps=float(data.get("sustained_read_gbps", 0.0)),
            features=list(data.get("features", [])),
            arch_name=str(data.get("arch_name", "")),
        )

    @classmethod
    def from_device_info(cls, dev: DeviceInfo) -> HardwareProfile:
        cc = str(dev.compute_capability or "0.0")
        arch = dev.arch or (f"sm_{dev.arch_facts.cc[0]}{dev.arch_facts.cc[1]}" if dev.arch_facts and dev.arch_facts.cc else f"sm_{cc.replace('.', '')}")

        # Construct a stable fingerprint incorporating architectural essentials
        fp_components = [
            dev.name,
            cc,
            str(dev.multi_processor_count),
            str(dev.l2_cache_size),
            str(dev.shared_mem_per_multiprocessor),
            str(dev.regs_per_multiprocessor),
            platform.machine(),
        ]
        fp_raw = ":".join(fp_components).encode("utf-8")
        fingerprint = hashlib.sha256(fp_raw).hexdigest()[:16]

        feat_list = [k for k, v in dev.features.items() if v]

        bw = dev.measured_bandwidth_gbps or 0.0
        read_bw = dev.measured_read_bandwidth_gbps or bw

        return cls(
            gpu_name=dev.name,
            fingerprint=fingerprint,
            compute_capability=cc,
            sm_count=dev.multi_processor_count,
            max_threads_per_sm=dev.max_threads_per_multiprocessor,
            max_threads_per_block=dev.max_threads_per_block,
            regs_per_sm=dev.regs_per_multiprocessor,
            shared_mem_per_block_default=dev.shared_mem_per_block,
            shared_mem_per_block_opt_in=dev.shared_mem_per_block_optin,
            shared_mem_per_sm=dev.shared_mem_per_multiprocessor,
            l2_cache_size=dev.l2_cache_size,
            bandwidth_gbps=bw,
            sustained_read_gbps=read_bw,
            features=feat_list,
            arch_name=arch,
        )


@dataclass
class SoftwareProfile:
    """Toolchain and repository revision profile."""

    cuda_runtime_version: str
    cuda_driver_version: str
    compiler: str
    compiler_version: str
    framework_revision: str
    framework_clean: bool

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SoftwareProfile:
        return cls(
            cuda_runtime_version=str(data.get("cuda_runtime_version", "unknown")),
            cuda_driver_version=str(data.get("cuda_driver_version", "unknown")),
            compiler=str(data.get("compiler", "unknown")),
            compiler_version=str(data.get("compiler_version", "unknown")),
            framework_revision=str(data.get("framework_revision", "unknown")),
            framework_clean=bool(data.get("framework_clean", False)),
        )

    @classmethod
    def current(cls, repo_root: str | Path = ".") -> SoftwareProfile:
        root = Path(repo_root).resolve()

        # Git revision
        git_hash = "unknown"
        clean = True
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                git_hash = res.stdout.strip()
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if status_res.returncode == 0:
                clean = len(status_res.stdout.strip()) == 0
        except Exception:
            pass

        # Compiler and CUDA
        nvcc_version = "unknown"
        try:
            nvcc_res = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if nvcc_res.returncode == 0:
                for line in nvcc_res.stdout.splitlines():
                    if "release" in line:
                        nvcc_version = line.strip()
                        break
        except Exception:
            pass

        return cls(
            cuda_runtime_version="13.0",
            cuda_driver_version="system",
            compiler="nvcc",
            compiler_version=nvcc_version,
            framework_revision=git_hash,
            framework_clean=clean,
        )


@dataclass
class TuningProfile:
    """Combined hardware and software execution profile."""

    profile_id: str
    hardware: HardwareProfile
    software: SoftwareProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "hardware": self.hardware.to_dict(),
            "software": self.software.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuningProfile:
        return cls(
            profile_id=str(data.get("profile_id", "")),
            hardware=HardwareProfile.from_dict(data.get("hardware", {})),
            software=SoftwareProfile.from_dict(data.get("software", {})),
        )

    @classmethod
    def create(cls, dev: DeviceInfo, repo_root: str | Path = ".") -> TuningProfile:
        hw = HardwareProfile.from_device_info(dev)
        sw = SoftwareProfile.current(repo_root=repo_root)

        clean_name = hw.gpu_name.lower().replace(" ", "_").replace("nvidia_", "")
        pid = f"{clean_name}_{hw.arch_name}_{hw.fingerprint}"

        return cls(
            profile_id=pid,
            hardware=hw,
            software=sw,
        )
