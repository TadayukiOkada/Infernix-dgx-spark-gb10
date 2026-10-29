"""Offline, benchmark-driven CUDA autotuning and dispatch generation.

Quantization-family agnostic autotuning framework for Infernix.
"""

from .database import CorrectnessTestSummary, TuningDatabase, TuningRecord
from .diagnostics import CustomKernelDiagnostician, DiagnosticOutcome
from .discovery import DiscoveredFamily, DiscoveredShape, DiscoveryReport, RepositoryDiscovery
from .generator import DispatchGenerator, DispatchInterval, PolicyCompressor
from .profile import HardwareProfile, SoftwareProfile, TuningProfile
from .registry import CandidateLegality, CandidateRegistry, CandidateSpec
from .reporting import AutotuneReporter
from .runner import BenchmarkMeasurement, BenchmarkRunner, CorrectnessValidator
from .search import AdaptiveRefiner, NoiseAwareSelector, TuningSearch, WinnerSelection
from .validator import PolicyValidationReport, PolicyValidator
from .workload import WorkloadSpec, WorkloadSweep

__all__ = [
    "AdaptiveRefiner",
    "AutotuneReporter",
    "BenchmarkMeasurement",
    "BenchmarkRunner",
    "CandidateLegality",
    "CandidateRegistry",
    "CandidateSpec",
    "CorrectnessTestSummary",
    "CorrectnessValidator",
    "DiagnosticOutcome",
    "CustomKernelDiagnostician",
    "DiscoveredFamily",
    "DiscoveredShape",
    "DiscoveryReport",
    "DispatchGenerator",
    "DispatchInterval",
    "HardwareProfile",
    "NoiseAwareSelector",
    "PolicyCompressor",
    "PolicyValidationReport",
    "PolicyValidator",
    "RepositoryDiscovery",
    "SoftwareProfile",
    "TuningDatabase",
    "TuningProfile",
    "TuningRecord",
    "TuningSearch",
    "WinnerSelection",
    "WorkloadSpec",
    "WorkloadSweep",
]
