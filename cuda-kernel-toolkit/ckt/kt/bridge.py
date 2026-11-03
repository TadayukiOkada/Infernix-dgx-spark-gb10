"""Kernel Tuner bridge: tune and qualify the *production* CUDA sources in-process.

Kernel Tuner (KT) compiles with NVRTC and launches through the driver API.  Three gaps
between that and the Infernix kernels are closed here, each as a small, explicit extension
of KT's ``nvcuda`` backend rather than a fork of the library:

1. **Launch ABI** - the product kernels take by-value structs (``LinearBf16Output`` holds
   a device pointer) and empty functor structs.  KT only passes pointers and scalars, so a
   :class:`PtrStructArray` is allocated like an array but launched as a pointer struct, and
   :class:`RawBytes` passes empty functors; the parameter block is packed by hand.
2. **Dynamic shared memory above 48 KiB** - KT never calls ``cuFuncSetAttribute``; GB10
   kernels routinely need up to 99 KiB.
3. **Cold L2** - KT times back-to-back launches, which leaves weights resident in L2.  A
   :class:`ColdL2Observer` streams a read-only eviction buffer before every timed launch, the
   same protocol as ``infernix_linear_bench --flush-mib``.

Kernels are compiled for ``compute_121a`` PTX and JIT-assembled by the driver, so SASS can differ
from the nvcc/ptxas build in the repo; use the result to rank and explain candidates, then confirm
winners with the product build (``ckt shapes bench``).
"""

from __future__ import annotations

import ctypes
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

KT_DIR = Path(__file__).resolve().parent
SHIM_DIR = KT_DIR / "shim"
REPO_ROOT = KT_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
ENV_JSON = REPO_ROOT / ".ckt_cache" / "env.json"

MAX_DYNAMIC_SMEM = 99 * 1024  # sm_121a opt-in limit per block (1 KiB reserved by the driver)


# --------------------------------------------------------------------------- #
# Launch arguments
# --------------------------------------------------------------------------- #
class DeviceBuffer(np.ndarray):
    """Array that is allocated, initialised and read back by KT but *not* passed as a kernel parameter.

    It exists only to be referenced from a :class:`StructArg`, e.g. the output tensor behind the
    product's by-value ``LinearBf16Output``.
    """


def device_buffer(array: np.ndarray) -> DeviceBuffer:
    return np.ascontiguousarray(array).view(DeviceBuffer)


@dataclass(frozen=True)
class StructArg:
    """By-value C struct parameter, packed with natural alignment.

    ``fields`` is a list of ``("ptr", arg_index)`` (device pointer of that argument's buffer),
    ``("i32"|"u32"|"i64"|"f32", value)``.
    """

    fields: tuple[tuple[str, Any], ...]

    _FMT = {"ptr": ("<Q", 8), "i32": ("<i", 4), "u32": ("<I", 4), "i64": ("<q", 8), "f32": ("<f", 4)}

    def pack(self, gpu_args: Sequence[Any]) -> bytes:
        import struct

        out = bytearray()
        align_max = 1
        for kind, value in self.fields:
            fmt, size = self._FMT[kind]
            align_max = max(align_max, size)
            out += b"\0" * (-len(out) % size)
            out += struct.pack(fmt, int(gpu_args[value]) if kind == "ptr" else value)
        out += b"\0" * (-len(out) % align_max)
        return bytes(out)


@dataclass(frozen=True)
class RawBytes:
    """By-value parameter passed as raw bytes (e.g. an empty functor struct is 1 byte)."""

    data: bytes


# --------------------------------------------------------------------------- #
# nvcuda backend extensions
# --------------------------------------------------------------------------- #
_PATCHED = False
# Per-configuration side channel.  KT aborts the whole run on a compile error or a failed
# verification; for exploration both are data, so they are recorded here and the run goes on.
CURRENT_PARAMS: dict[str, Any] = {}
VERIFY_LOG: dict[tuple, tuple[bool, dict]] = {}
COMPILE_LOG: dict[tuple, str] = {}
COMPILER = {"name": "nvrtc"}  # "nvrtc" (fast, JIT PTX) or "nvcc" (product flags, ahead-of-time SASS)


def config_key(params: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in params.items()))


def _compile_with_nvcc(dev, kernel_instance):
    """Build the instantiation with the product's nvcc flags and load the cubin (exact product SASS)."""
    import hashlib
    import re
    import shutil
    import subprocess
    import tempfile

    from cuda.bindings import driver

    name = kernel_instance.name
    source = kernel_instance.kernel_string + f'\nextern "C" const void* kt_entry = (const void*)&{name};\n'
    tag = hashlib.sha1(source.encode()).hexdigest()[:16]
    work = Path(tempfile.gettempdir()) / "ckt_kt_nvcc"
    work.mkdir(exist_ok=True)
    cu, cubin = work / f"{tag}.cu", work / f"{tag}.cubin"
    if not cubin.exists():
        cu.write_text(source)
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        cmd = [nvcc, "-x", "cu", "-cubin", "-O3", "-DNDEBUG", "-std=c++20", "-lineinfo",
               "--generate-code=arch=compute_121a,code=sm_121a",
               f"-I{REPO_ROOT / 'include'}", f"-I{SRC_ROOT}", str(cu), "-o", str(cubin)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError("nvcc: " + done.stderr[-1500:])
    usage = subprocess.run(["cuobjdump", "-res-usage", str(cubin)], capture_output=True, text=True).stdout
    names = re.findall(r"Function (\S+):", usage)
    if len(names) != 1:
        raise RuntimeError(f"expected exactly one kernel in cubin, found {len(names)}")
    err, dev.current_module = driver.cuModuleLoadData(np.char.array(cubin.read_bytes()))
    nv_check(err)
    err, dev.func = driver.cuModuleGetFunction(dev.current_module, names[0].encode())
    nv_check(err)
    err, regs = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, dev.func)
    nv_check(err)
    dev.num_regs = regs
    return dev.func


def nv_check(err) -> None:
    from kernel_tuner.backends import nvcuda as nv

    nv.cuda_error_check(err)


def install_nvcuda_extensions() -> None:
    """Idempotently extend ``CudaFunctions`` (see module docstring items 1 and 2)."""
    global _PATCHED
    if _PATCHED:
        return
    from cuda.bindings import driver
    from kernel_tuner.backends import nvcuda as nv

    cls = nv.CudaFunctions
    base_ready, base_compile = cls.ready_argument_list, cls.compile

    def ready_argument_list(self, arguments):
        gpu = base_ready(self, arguments)
        # The CUdeviceptr objects of hidden buffers must not reach the launch parameter block.
        self._kt_hidden = {id(g) for g, a in zip(gpu, arguments) if isinstance(a, DeviceBuffer)}
        return gpu

    def compile(self, kernel_instance):
        CURRENT_PARAMS.clear()
        CURRENT_PARAMS.update(kernel_instance.params)
        try:
            if COMPILER["name"] == "nvcc":
                func = _compile_with_nvcc(self, kernel_instance)
            else:
                func = base_compile(self, kernel_instance)
        except RuntimeError as exc:  # the compiler rejected the instantiation (static_assert, ...)
            COMPILE_LOG[config_key(kernel_instance.params)] = f"nvrtc: {exc}"
            return None
        # Kernels with <= 48 KiB use static shared memory (launch with 0 dynamic bytes); larger ones
        # use `extern __shared__` and need the opt-in.  Static + dynamic may not exceed the limit.
        err, static = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func)
        self.static_smem = int(static) if err == driver.CUresult.CUDA_SUCCESS else 0
        status = driver.cuFuncSetAttribute(
            func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            MAX_DYNAMIC_SMEM - self.static_smem,
        )[0]
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuFuncSetAttribute(max dynamic smem) failed: {status}")
        err, local = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func)
        self.local_bytes = int(local) if err == driver.CUresult.CUDA_SUCCESS else -1
        return func

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        stream = self.stream if stream is None else stream
        holders: list[Any] = []
        for a in gpu_args:
            if isinstance(a, DeviceBuffer) or (isinstance(a, driver.CUdeviceptr) and id(a) in self._kt_hidden):
                continue
            if isinstance(a, StructArg):
                raw = a.pack(gpu_args)
            elif isinstance(a, RawBytes):
                raw = a.data
            elif isinstance(a, driver.CUdeviceptr):
                raw = ctypes.c_uint64(int(a))
                holders.append(raw)
                continue
            else:
                raw = np.ascontiguousarray(a).tobytes()
            holders.append(ctypes.create_string_buffer(raw, len(raw)))
        table = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        err = driver.cuLaunchKernel(
            func, grid[0], grid[1], grid[2], threads[0], threads[1], threads[2],
            self.smem_size, stream, ctypes.addressof(table), 0,
        )
        nv.cuda_error_check(err)

    cls.ready_argument_list = ready_argument_list
    cls.compile = compile
    cls.run_kernel = run_kernel
    _PATCHED = True


# --------------------------------------------------------------------------- #
# Observers
# --------------------------------------------------------------------------- #
def _observer_base():
    from kernel_tuner.observers.observer import BenchmarkObserver

    return BenchmarkObserver


_FLUSH_SRC = r"""
extern "C" __global__ void kt_flush(const uint4* p, unsigned long long n, uint4* sink) {
    uint4 v = make_uint4(0, 0, 0, 0);
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 l = p[i];
        v.x ^= l.x; v.y ^= l.y; v.z ^= l.z; v.w ^= l.w;
    }
    if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9e3779b9u) *sink = v;
}
"""


def make_cold_l2_observer(flush_mib: int):
    """Observer that evicts L2 with a read-only sweep before every timed launch."""
    from cuda.bindings import driver, nvrtc

    base = _observer_base()

    class ColdL2Observer(base):
        def __init__(self):
            self.nbytes = flush_mib << 20
            self.ready = False

        def _setup(self):
            _, prog = nvrtc.nvrtcCreateProgram(_FLUSH_SRC.encode(), b"kt_flush.cu", 0, [], [])
            res = nvrtc.nvrtcCompileProgram(prog, 1, [b"--gpu-architecture=compute_121a"])
            if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError("flush kernel failed to compile")
            _, size = nvrtc.nvrtcGetPTXSize(prog)
            ptx = b" " * size
            nvrtc.nvrtcGetPTX(prog, ptx)
            _, self.module = driver.cuModuleLoadData(np.char.array(ptx))
            _, self.func = driver.cuModuleGetFunction(self.module, b"kt_flush")
            _, self.buf = driver.cuMemAlloc(self.nbytes)
            driver.cuMemsetD8(self.buf, 1, self.nbytes)
            _, self.sink = driver.cuMemAlloc(16)
            self.words = ctypes.c_uint64(self.nbytes // 16)
            self.ready = True

        def register_device(self, dev):
            super().register_device(dev)
            # Evict inside start_event(): the flush is queued *ahead of* the start event, so the host's
            # launch latency for the timed kernel overlaps the flush instead of landing inside the
            # event interval.  (KT's own loop syncs, records, then launches; for a 20 us kernel that
            # adds ~7 us of host latency.)  This is the bench::measure_cold_launch ordering.
            base_start = dev.start_event
            observer = self

            def start_event():
                if not observer.ready:
                    observer._setup()
                observer._evict()
                base_start()

            dev.start_event = start_event

        def _evict(self):
            holders = [ctypes.c_uint64(int(self.buf)), self.words, ctypes.c_uint64(int(self.sink))]
            table = (ctypes.c_void_p * 3)(*[ctypes.addressof(h) for h in holders])
            driver.cuLaunchKernel(self.func, 192, 1, 1, 256, 1, 1, 0, self.dev.stream, ctypes.addressof(table), 0)

        def get_results(self):
            return {}

    return ColdL2Observer()


def make_resource_observer():
    base = _observer_base()

    class ResourceObserver(base):
        """Registers/thread and local-memory (spill) bytes of the compiled kernel."""

        def get_results(self):
            return {"num_regs": int(self.dev.num_regs), "local_bytes": int(getattr(self.dev, "local_bytes", -1))}

    return ResourceObserver()


# --------------------------------------------------------------------------- #
# Problem description and execution
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    """One tunable kernel instance family on one workload."""

    name: str
    kernel_name: str                     # templated name expression; KT_* tokens are substituted
    kernel_source: str
    problem_size: tuple[int, ...]
    grid_div_x: Sequence[str]
    grid_div_y: Sequence[str]
    arguments: list[Any]
    tune_params: dict[str, list[Any]]
    restrictions: list[Any]
    smem_expr: str | None
    answer: list[Any] | None
    check: Callable[[list, list], tuple[bool, dict]] | None   # (answer, results) -> (ok, stats)
    flops: float
    bytes_moved: float
    pipe: str                            # key into env dtype_peaks (e.g. "bf16")
    label: Callable[[dict], str] = lambda p: str(p)
    notes: dict[str, Any] = field(default_factory=dict)


def load_ceilings() -> dict[str, Any]:
    env = json.loads(ENV_JSON.read_text())["device"]
    return {
        "mem_gbps": env["measured_read_bandwidth_gbps"],
        "l2_bytes": env["l2_cache_size"],
        "pipes": {k: v["tflops"] for k, v in env["dtype_peaks"].items()},
    }


def cuda_include_dirs() -> list[str]:
    """Headers of the CUDA toolkit that builds the product (nvcc), not whichever wheel KT finds."""
    import shutil

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    inc = Path(nvcc).resolve().parent.parent / "include"
    return [str(inc), str(inc / "cccl")]


def nvrtc_options() -> list[str]:
    # Order matters: the shim must shadow cuda.h and the host-only repo headers.
    return [
        f"-I{SHIM_DIR}",
        f"-I{SRC_ROOT}",
        *(f"-I{d}" for d in cuda_include_dirs()),
        "--gpu-architecture=compute_121a",
        "--std=c++20",
        "-default-device",  # NVRTC JIT rejects the unannotated host helpers in the repo headers
    ]


def run_tuning(
    problem: Problem,
    *,
    strategy: str = "brute_force",
    strategy_options: dict | None = None,
    iterations: int = 12,
    flush_mib: int = 48,
    cache: str | None = None,
    verbose: bool = False,
    compiler: str = "nvrtc",
) -> tuple[list[dict], dict]:
    """Run Kernel Tuner on ``problem``; return rows with derived TFLOP/s, GB/s, %pipe, %mem."""
    install_nvcuda_extensions()
    COMPILER["name"] = compiler
    from kernel_tuner import tune_kernel

    ceilings = load_ceilings()
    observers = [make_resource_observer()]
    if flush_mib > 0:
        observers.insert(0, make_cold_l2_observer(flush_mib))
    block = ["block_size_x"]
    kwargs: dict[str, Any] = dict(
        grid_div_x=list(problem.grid_div_x),
        grid_div_y=list(problem.grid_div_y),
        restrictions=problem.restrictions,
        lang="nvcuda",
        compiler_options=nvrtc_options(),
        iterations=iterations,
        observers=observers,
        verbose=verbose,
        block_size_names=block,
        strategy=strategy,
        strategy_options=strategy_options or {},
        objective="time",
        objective_higher_is_better=False,
        cache=cache,
        simulation_mode=False,
    )
    if problem.smem_expr:
        kwargs["smem_args"] = {"size": problem.smem_expr}
    VERIFY_LOG.clear()
    COMPILE_LOG.clear()
    if problem.answer is not None:

        def verify(answer, result, atol=None):
            ok, stats = problem.check(answer, result)
            VERIFY_LOG[config_key(CURRENT_PARAMS)] = (ok, stats)
            return True  # never abort the sweep; unverified rows are flagged and excluded

        kwargs["answer"] = problem.answer
        kwargs["verify"] = verify
    results, env = tune_kernel(
        problem.kernel_name, problem.kernel_source, problem.problem_size, problem.arguments,
        problem.tune_params, **{k: v for k, v in kwargs.items() if v is not None},
    )
    rows = []
    for r in results:
        times = [t for t in r.get("times", []) if t is not None]
        if not times or not isinstance(r.get("time"), (int, float)):
            continue
        med_us = statistics.median(times) * 1e3
        row = dict(r)
        row["median_us"] = med_us
        row["min_us"] = min(times) * 1e3
        row["tflops"] = problem.flops / (med_us * 1e-6) / 1e12
        row["gbps"] = problem.bytes_moved / (med_us * 1e-6) / 1e9
        row["pct_pipe"] = 100.0 * row["tflops"] / ceilings["pipes"][problem.pipe]
        row["pct_mem"] = 100.0 * row["gbps"] / ceilings["mem_gbps"]
        row["label"] = problem.label(r)
        ok, stats = VERIFY_LOG.get(config_key({k: r[k] for k in problem.tune_params}), (None, {}))
        row["verified"] = ok
        row.update({f"v_{k}": v for k, v in stats.items()})
        rows.append(row)
    rows.sort(key=lambda x: x["median_us"])
    return rows, env
