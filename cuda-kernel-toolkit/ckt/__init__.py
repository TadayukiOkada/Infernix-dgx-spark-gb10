"""CKT — CUDA Kernel Toolkit.

A lightweight, modular toolkit that orchestrates ``nvcc``, ``ncu``,
``cuobjdump``, ``nvdisasm`` and the CUDA runtime to analyse, model, profile
and tune CUDA kernels.

The package is intentionally dependency-light: the core only needs the Python
standard library.  Optional third-party packages (``yaml``, ``toml``,
``matplotlib``) degrade gracefully when absent.

Module map (see ``docs/architecture.md``)::

    environment          GPU discovery, tool detection, environment snapshot
    source_discovery     kernel discovery, source <-> symbol mapping
    build_manager        compilation, build caching, ptxas resource reports
    binary_inspector     cuobjdump extraction, symbol filtering, nvdisasm
    static_analyzer      source + resource + instruction level analysis
    profiler             baseline benchmarking + ncu metric collection
    performance_model    roofline, ceilings, bottleneck classification
    correctness          reference comparison, runtime validation
    parameter_search     candidate generation, pruning, tuning strategies
    recommendations      evidence-backed optimization suggestions
    reporting            JSON / CSV / Markdown / HTML reports
    experiment_manager   caching, provenance, resumability, storage
    cli                  command line interface and configuration handling
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
