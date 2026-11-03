"""Kernel Tuner problems for the production NVFP4 linear kernels.

Each builder returns a :class:`~ckt.kt.bridge.Problem` that tunes the *template parameters of the
product kernel itself* (the schedule structs in ``src/ops/linear/nvfp4``), so a result row names a
schedule that can be pasted into a ``shapes/n<N>_k<K>.cu`` dispatch table.  Operands are synthetic;
the oracle is a float64 decode of the stored NVFP4 representation (FP4 codes, FP8-E4M3 block scales
in the swizzled 128x64 tile layout, global ``alpha``) followed by a float64 GEMM.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .bridge import Problem, RawBytes, StructArg, device_buffer

NS = "infernix::ops::detail::"
FP4_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Synthetic NVFP4 operands and the independent oracle
# --------------------------------------------------------------------------- #
def e4m3_decode(byte: np.ndarray) -> np.ndarray:
    b = byte.astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp, man = (b >> 3) & 0xF, b & 7
    val = np.where(exp == 0, man / 8.0 * 2.0**-6, (1.0 + man / 8.0) * 2.0 ** (exp - 7.0))
    return sign * val


def scale_swizzle(rows: int, k: int) -> np.ndarray:
    """Byte offset of scale (row, group) in the stored 128x64-tile layout (nvfp4_scale_byte_offset)."""
    row = np.arange(rows, dtype=np.int64)[:, None]
    grp = np.arange(k // 16, dtype=np.int64)[None, :]
    return ((row // 128) * (k // 64) + grp // 4) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (grp & 3)


def make_nvfp4_operands(n: int, k: int, tokens: int, seed: int = 7) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    logical_scale = rng.integers(0x30, 0x41, size=(n, k // 16), dtype=np.uint8)  # 0.5 .. 2.0, no NaN
    scales = np.empty(n * (k // 16), dtype=np.uint8)
    scales[scale_swizzle(n, k).ravel()] = logical_scale.ravel()
    x = rng.standard_normal((tokens, k)).astype(np.float32)
    x_bits = (x.view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16; the oracle uses the same bits
    return {"codes": codes, "scales": scales, "logical_scale": logical_scale, "x_bits": x_bits, "alpha": 0.0625}


def bf16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def nvfp4_reference(op: dict, n: int, k: int) -> np.ndarray:
    """float64 y[T, N] = alpha * x @ W^T with W decoded from the stored representation."""
    codes = op["codes"]
    lo, hi = codes & 0xF, codes >> 4          # even k element is the low nibble (cvt.e2m1x2: first = low)
    nib = np.empty((n, k), dtype=np.uint8)
    nib[:, 0::2], nib[:, 1::2] = lo, hi
    w = FP4_VALUES[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    w *= np.repeat(e4m3_decode(op["logical_scale"]), 16, axis=1)
    x = bf16_bits_to_f64(op["x_bits"])
    return op["alpha"] * (x @ w.T)


def make_bf16_checker(shape_tokens_n: tuple[int, int], k: int) -> Callable[[list, list], tuple[bool, dict]]:
    """Criterion: |out - ref| <= 0.5 ulp_bf16(ref) + FP32-accumulation slack.

    The product is exact in FP32 (FP4 x E4M3 x BF16 has <= 16 significant bits) and accumulated in FP32,
    so the only freedom is the final BF16 rounding of a value within ~K*2^-24 of the float64 result.
    """
    tokens, n = shape_tokens_n

    def check(answer: list, results: list) -> tuple[bool, dict]:
        ref = next(a for a in answer if a is not None).reshape(tokens, n)
        out_bits = next(r for a, r in zip(answer, results) if a is not None)
        out = bf16_bits_to_f64(np.asarray(out_bits).reshape(tokens, n))
        rms = float(np.sqrt(np.mean(ref * ref)))
        exp = np.floor(np.log2(np.maximum(np.abs(ref), 1e-30)))
        half_ulp = 0.5 * 2.0 ** (exp - 7)
        tol = half_ulp * 1.001 + rms * k * 2.0**-24 * 4
        excess = np.abs(out - ref) / tol
        stats = {"max_err_over_tol": float(excess.max()), "finite": bool(np.isfinite(out).all())}
        return bool(stats["finite"] and excess.max() <= 1.0), stats

    return check


# --------------------------------------------------------------------------- #
# A16 tensor-core GEMM (prefill widths, T >= 33)
# --------------------------------------------------------------------------- #
_A16_SOURCE = """
#include "ops/linear/nvfp4/nvfp4_a16_mma.cuh"
#include "ops/linear/common/output.cuh"
// KT treats a short string without ';' as a file name.
static_assert(true);
"""

_CACHE = "infernix::ops::Cache::"


def a16_mma_problem(n: int, k: int, tokens: int, *, wide: bool = False) -> Problem:
    """Tune ``nvfp4_a16_mma_kernel<Nvfp4A16MmaSchedule<...>>`` for ``[n, k]`` at ``tokens`` tokens.

    ``wide=False`` spans the schedule integers the dispatch tables already use; ``wide=True`` adds the
    cp.async cache hints and the fragment-pipeline mode, which no dispatch table varies.
    """
    op = make_nvfp4_operands(n, k, tokens)
    ref = nvfp4_reference(op, n, k).reshape(-1)
    name = (
        f"{NS}nvfp4_a16_mma_kernel<{NS}Nvfp4ScheduleInstance<{NS}Nvfp4A16MmaSchedule<"
        f"KT_BM,KT_BN,KT_BK,KT_WM,KT_WN,KT_STAGES,KT_MINB,{_CACHE}KT_WCACHE,{_CACHE}KT_ACACHE,"
        f"{NS}Nvfp4MmaFragmentPipeline::KT_PIPE>,{k}>,KT_FULL,"
        f"{NS}LinearBf16Output,{NS}LinearIdentityEpilogue,{NS}Nvfp4IdentityRows>"
    )
    tune = {
        "KT_BM": [32, 64, 128],
        "KT_BN": [16, 32, 48, 64, 96, 128, 192, 256],
        "KT_BK": [64, 128],
        "KT_WM": [16, 32, 64],
        "KT_WN": [8, 16, 32, 48, 64],
        "KT_STAGES": [1, 2],
        "KT_MINB": [1, 2, 3],
        "KT_WCACHE": ["cg", "ca"] if wide else ["cg"],
        "KT_ACACHE": ["cg", "ca"] if wide else ["cg"],
        "KT_PIPE": ["PingPong", "Serial"] if wide else ["PingPong"],
        "KT_FULL": ["false", "true"],
        "block_size_x": [32 * w for w in (1, 2, 3, 4, 6, 8, 12, 16)],
    }

    def valid(p: dict) -> bool:
        bm, bn, bk, wm, wn = p["KT_BM"], p["KT_BN"], p["KT_BK"], p["KT_WM"], p["KT_WN"]
        if bm % wm or bn % wn or k % bk or n % bm:
            return False
        if p["block_size_x"] != (bm // wm) * (bn // wn) * 32:
            return False
        if (tokens % bn == 0) != (p["KT_FULL"] == "true"):
            return False
        return _a16_smem(p) <= 99 * 1024

    args: list[Any] = [
        op["x_bits"].reshape(-1), op["codes"].reshape(-1), op["scales"], np.float32(op["alpha"]),
        StructArg((("ptr", 11), ("i32", n))),                     # LinearBf16Output {data, rows}
        RawBytes(b"\0"), RawBytes(b"\0"),                         # LinearIdentityEpilogue, Nvfp4IdentityRows
        np.int32(n), np.int32(k), np.int32(0), np.int32(tokens),  # rows, input_rows, token_offset, count
        device_buffer(np.zeros(tokens * n, dtype=np.uint16)),     # hidden output allocation
    ]
    answer: list[Any] = [None] * len(args)
    answer[11] = ref
    return Problem(
        name=f"nvfp4_a16_mma[{n},{k}]T{tokens}",
        kernel_name=name,
        kernel_source=_A16_SOURCE,
        problem_size=(n, tokens),
        grid_div_x=["KT_BM"],
        grid_div_y=["KT_BN"],
        arguments=args,
        tune_params=tune,
        restrictions=[valid],
        smem_expr="(KT_BM*KT_BK*2 + KT_STAGES*KT_BN*KT_BK*2 + KT_STAGES*KT_BM*(KT_BK//2 + KT_BK//16))",
        answer=answer,
        check=make_bf16_checker((tokens, n), k),
        flops=2.0 * n * k * tokens,
        bytes_moved=n * k / 2 + n * k / 16 + tokens * k * 2 + tokens * n * 2,
        pipe="bf16",
        label=lambda p: (
            f"{p['KT_BM']}x{p['KT_BN']}x{p['KT_BK']} w{p['KT_WM']}x{p['KT_WN']} "
            f"s{p['KT_STAGES']} mb{p['KT_MINB']}"
            + (f" {p['KT_WCACHE']}/{p['KT_ACACHE']}/{p['KT_PIPE']}" if wide else "")
        ),
    )


def _a16_smem(p: dict) -> int:
    bm, bn, bk, st = p["KT_BM"], p["KT_BN"], p["KT_BK"], p["KT_STAGES"]
    return bm * bk * 2 + st * bn * bk * 2 + st * bm * (bk // 2 + bk // 16)


# --------------------------------------------------------------------------- #
# A16 sliced-K tensor-core GEMM (decode widths, T <= 64)
# --------------------------------------------------------------------------- #
_SLICED_SOURCE = """
#include "ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh"
#include "ops/linear/common/output.cuh"
static_assert(true);
"""


def a16_sliced_k_problem(n: int, k: int, tokens: int, *, wide: bool = False) -> Problem:
    """Tune ``nvfp4_a16_sliced_k_mma_kernel<Nvfp4A16SlicedKMmaSchedule<...>>``.

    Each CTA owns ``16*RowTiles`` weight rows and reduces K across ``KWarps`` warps, so the grid is
    ``(n / (16*RowTiles), ceil(T / TileTokens))``.  The default space is the tuple the shape tables
    sweep (KWarps, TileTokens, RowTiles, Stages, MinBlocksPerSm); ``wide`` adds the cp.async cache
    hints and the padded-activation staging mode.
    """
    op = make_nvfp4_operands(n, k, tokens)
    ref = nvfp4_reference(op, n, k).reshape(-1)
    name = (
        f"{NS}nvfp4_a16_sliced_k_mma_kernel<{NS}Nvfp4ScheduleInstance<{NS}Nvfp4A16SlicedKMmaSchedule<"
        f"KT_KW,KT_TT,KT_MINB,{_CACHE}KT_ACACHE,{_CACHE}KT_WCACHE,"
        f"{NS}Nvfp4ActivationStage::KT_ASTAGE,KT_STAGES,KT_ROWTILES>,{k}>,"
        f"{NS}LinearBf16Output,{NS}LinearIdentityEpilogue,{NS}Nvfp4IdentityRows>"
    )
    token_tiles = sorted({t for t in (8, 16, 24, 32, 40, 48, 64) if t >= 1} | {((tokens + 7) // 8) * 8})
    tune = {
        "KT_KW": [2, 4, 8, 16],
        "KT_TT": [t for t in token_tiles if t <= 64],
        "KT_ROWTILES": [1, 2],
        "KT_STAGES": [1, 2],
        "KT_MINB": [1, 2, 3, 4],
        "KT_ACACHE": ["ca", "cg"] if wide else ["ca"],
        "KT_WCACHE": ["cg", "ca"] if wide else ["cg"],
        "KT_ASTAGE": ["ActiveOnly", "PaddedZero"] if wide else ["ActiveOnly"],
        "block_size_x": [64, 128, 256, 512],
    }

    def smem(p: dict) -> int:
        kw, tt, st, rt = p["KT_KW"], p["KT_TT"], p["KT_STAGES"], p["KT_ROWTILES"]
        staging = st * (16 * rt * (kw * 64 // 2) + kw * tt * 64 * 2)
        partial = kw * rt * (tt // 8) * 32 * 4 * 4
        return max(staging, partial)

    def valid(p: dict) -> bool:
        if p["block_size_x"] != p["KT_KW"] * 32:
            return False
        if k % (p["KT_KW"] * 64) or n % (16 * p["KT_ROWTILES"]):
            return False
        return smem(p) <= 99 * 1024

    args: list[Any] = [
        StructArg((("ptr", 5), ("ptr", 6), ("ptr", 7), ("i32", n), ("i32", k), ("i32", tokens), ("f32", op["alpha"]))),
        StructArg((("ptr", 8), ("i32", n))),
        RawBytes(b"\0"), RawBytes(b"\0"),
        np.int32(0),
        device_buffer(op["x_bits"].reshape(-1)), device_buffer(op["codes"].reshape(-1)),
        device_buffer(op["scales"]), device_buffer(np.zeros(tokens * n, dtype=np.uint16)),
    ]
    answer: list[Any] = [None] * len(args)
    answer[8] = ref
    return Problem(
        name=f"nvfp4_a16_sliced_k[{n},{k}]T{tokens}",
        kernel_name=name,
        kernel_source=_SLICED_SOURCE,
        problem_size=(n, tokens),
        grid_div_x=["16*KT_ROWTILES"],
        grid_div_y=["KT_TT"],
        arguments=args,
        tune_params=tune,
        restrictions=[valid],
        # nvfp4_prepare_shared: <= 48 KiB is static shared memory (0 dynamic bytes), above it dynamic.
        smem_expr=(
            "(max(KT_STAGES*(16*KT_ROWTILES*(KT_KW*32) + KT_KW*KT_TT*128), KT_KW*KT_ROWTILES*(KT_TT//8)*512)"
            " if max(KT_STAGES*(16*KT_ROWTILES*(KT_KW*32) + KT_KW*KT_TT*128), KT_KW*KT_ROWTILES*(KT_TT//8)*512)"
            " > 49152 else 0)"
        ),
        answer=answer,
        check=make_bf16_checker((tokens, n), k),
        flops=2.0 * n * k * tokens,
        bytes_moved=n * k / 2 + n * k / 16 + tokens * k * 2 + tokens * n * 2,
        pipe="bf16",
        label=lambda p: (
            f"kw{p['KT_KW']} t{p['KT_TT']} rt{p['KT_ROWTILES']} s{p['KT_STAGES']} mb{p['KT_MINB']}"
            + (f" {p['KT_ACACHE']}/{p['KT_WCACHE']}/{p['KT_ASTAGE']}" if wide else "")
        ),
    )


PROBLEMS: dict[str, Callable[..., Problem]] = {
    "nvfp4_a16_mma": a16_mma_problem,
    "nvfp4_a16_sliced_k": a16_sliced_k_problem,
}
