"""Candidate registry and legality/resource pruning.

Defines candidate representations, legality verification against target hardware,
resource bounds (shared memory, register pressure, warp allocation), and parameter search spaces.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .profile import HardwareProfile
from .workload import WorkloadSpec
from ..util import to_plain


@dataclass(frozen=True)
class CandidateSpec:
    """A single executable kernel candidate configuration."""

    id: str
    operation: str
    quantization_family: str
    kernel_family: str  # "gemv", "simt", "sliced_k", "mma", "tma", "pair"
    n: int
    k: int
    template_name: str
    launch_expr: str
    using_decl: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    reduction_signature: str = ""
    tile_rows: int | None = None
    tile_tokens: int | None = None
    tile_k: int | None = None
    threads_per_block: int = 256
    k_slices: int = 1
    shared_mem_bytes: int = 0
    min_tokens: int = 1
    max_tokens: int | None = None
    requires_features: list[str] = field(default_factory=list)
    is_coupled: bool = False
    coupled_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSpec:
        return cls(
            id=str(data.get("id", "")),
            operation=str(data.get("operation", "linear")),
            quantization_family=str(data.get("quantization_family", "")),
            kernel_family=str(data.get("kernel_family", "mma")),
            n=int(data.get("n", 0)),
            k=int(data.get("k", 0)),
            template_name=str(data.get("template_name", "")),
            launch_expr=str(data.get("launch_expr", "")),
            using_decl=str(data.get("using_decl", "")),
            parameters=dict(data.get("parameters", {})),
            reduction_signature=str(data.get("reduction_signature", "")),
            tile_rows=data.get("tile_rows"),
            tile_tokens=data.get("tile_tokens"),
            tile_k=data.get("tile_k"),
            threads_per_block=int(data.get("threads_per_block", 256)),
            k_slices=int(data.get("k_slices", 1)),
            shared_mem_bytes=int(data.get("shared_mem_bytes", 0)),
            min_tokens=int(data.get("min_tokens", 1)),
            max_tokens=data.get("max_tokens"),
            requires_features=list(data.get("requires_features", [])),
            is_coupled=bool(data.get("is_coupled", False)),
            coupled_reason=str(data.get("coupled_reason", "")),
        )


@dataclass
class CandidateLegality:
    """Legality verdict for a candidate on target hardware."""

    legal: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"legal": self.legal, "reasons": self.reasons}


class CandidateRegistry:
    """Manages candidate generation, legality checking, and pruning."""

    def __init__(self, hw_profile: HardwareProfile | None = None):
        self.hw_profile = hw_profile
        self._registered_candidates: dict[str, CandidateSpec] = {}

    def register(self, candidate: CandidateSpec) -> None:
        self._registered_candidates[candidate.id] = candidate

    def get(self, candidate_id: str) -> CandidateSpec | None:
        return self._registered_candidates.get(candidate_id)

    def check_legality(self, candidate: CandidateSpec, hw: HardwareProfile | None = None) -> CandidateLegality:
        """Verify candidate legality against hardware and mathematical rules."""
        profile = hw or self.hw_profile
        reasons: list[str] = []

        # 1. Threads per block
        if candidate.threads_per_block < 32 or candidate.threads_per_block > 1024:
            reasons.append(f"Invalid thread count {candidate.threads_per_block} (must be in 32..1024)")
        elif candidate.threads_per_block % 32 != 0:
            reasons.append(f"Thread count {candidate.threads_per_block} is not a multiple of warp size (32)")

        if profile is not None:
            # 2. Hardware thread limits
            if candidate.threads_per_block > profile.max_threads_per_block:
                reasons.append(
                    f"Threads per block {candidate.threads_per_block} exceeds hardware limit {profile.max_threads_per_block}"
                )

            # 3. Shared memory limits
            max_smem = profile.shared_mem_per_block_opt_in
            if candidate.shared_mem_bytes > max_smem:
                reasons.append(
                    f"Shared memory {candidate.shared_mem_bytes} B exceeds hardware opt-in limit {max_smem} B"
                )

            # 4. Feature requirements
            for feat in candidate.requires_features:
                if feat not in profile.features:
                    reasons.append(f"Required hardware feature '{feat}' not supported by target GPU")

        # 5. Tile dimension divisibility
        if candidate.tile_rows is not None and candidate.tile_rows > 0:
            if candidate.tile_rows > candidate.n and candidate.n % 16 != 0:
                reasons.append(f"Tile rows {candidate.tile_rows} incompatible with N={candidate.n}")

        legal = len(reasons) == 0
        return CandidateLegality(legal=legal, reasons=reasons)

    def generate_candidates_for_shape(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        existing_bands: Sequence[dict[str, Any]] | None = None,
        aliases: dict[str, str] | None = None,
        policy: str = "a16",
    ) -> list[CandidateSpec]:
        """Generate legal candidate set for an operation geometry."""
        candidates: list[CandidateSpec] = []
        aliases = aliases or {}

        # 1. First include existing registered schedules from shape table
        if existing_bands:
            for idx, b in enumerate(existing_bands):
                route = b.get("route_expr", "")
                if not route:
                    continue
                pol = b.get("policy", "a16")
                if pol != policy:
                    continue
                low = route.lower()
                if policy == "a16" and ("_a4_" in route or "a4" in low or "A4" in route):
                    continue
                k_fam = "mma"
                if "gemv" in low:
                    k_fam = "gemv"
                elif "simt" in low:
                    k_fam = "simt"
                elif "sliced" in low:
                    k_fam = "sliced_k"
                elif "tma" in low:
                    k_fam = "tma"

                # Generate a stable candidate ID
                cid = self._stable_candidate_id(operation, qtype, n, k, k_fam, route)

                # Expand alias if any
                using_text = ""
                for alias_name, alias_target in aliases.items():
                    if alias_name in route:
                        using_text = f"using {alias_name} = {alias_target};"
                        break

                cand = CandidateSpec(
                    id=cid,
                    operation=operation,
                    quantization_family=qtype,
                    kernel_family=k_fam,
                    n=n,
                    k=k,
                    template_name=self._extract_template_name(route),
                    launch_expr=route,
                    using_decl=using_text,
                    reduction_signature=b.get("reduction_signature", f"{pol}:{k_fam}:k_slices=1:linear"),
                    tile_rows=b.get("tile_rows"),
                    tile_tokens=b.get("tile_tokens"),
                    tile_k=b.get("tile_k"),
                    threads_per_block=int(b.get("threads_per_block") or 256),
                    k_slices=int(b.get("k_slices") or 1),
                    min_tokens=int(b.get("min_tokens") or 1),
                    max_tokens=1 if k_fam == "gemv" else b.get("max_tokens"),
                    requires_features=["tma"] if "tma" in low else ([]),
                )
                leg = self.check_legality(cand)
                if leg.legal:
                    candidates.append(cand)
                    self.register(cand)

        # 2. Add family-specific candidate schedules if known
        extra_cands = self._generate_family_template_candidates(operation, qtype, n, k)
        for cand in extra_cands:
            leg = self.check_legality(cand)
            if leg.legal and cand.id not in self._registered_candidates:
                candidates.append(cand)
                self.register(cand)

        return candidates

    def _stable_candidate_id(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
        kernel_family: str,
        route_text: str,
    ) -> str:
        clean_route = re.sub(r"[^A-Za-z0-9_]+", "_", route_text).strip("_")
        sig = hashlib.sha256(f"{operation}:{qtype}:{n}:{k}:{route_text}".encode()).hexdigest()[:8]
        return f"{operation}_{qtype}_n{n}_k{k}_{kernel_family}_{clean_route[:32]}_{sig}"

    def _extract_template_name(self, route_text: str) -> str:
        m = re.search(r"([A-Za-z0-9_]+Schedule|[A-Za-z0-9_]+Instance)", route_text)
        return m.group(1) if m else "StandardSchedule"

    def _generate_family_template_candidates(
        self,
        operation: str,
        qtype: str,
        n: int,
        k: int,
    ) -> list[CandidateSpec]:
        """Synthesize legal schedule configurations for a family and geometry."""
        results: list[CandidateSpec] = []
        if operation != "linear":
            return results

        # Common tile grids for GEMV, SIMT, Sliced-K, MMA
        if qtype == "nvfp4":
            # NVFP4 legal parameter candidates
            # Sliced-K candidates: Tokens in {8, 16, 24, 32, 48, 64}, Warps in {2, 4, 8}, Stages in {1, 2}
            for tok in [8, 16, 24, 32, 48, 64]:
                for w in [2, 4, 8]:
                    for stg in [1, 2]:
                        # Physical shared memory check: kSharedBytes <= 99 KiB (nvfp4_schedule.cuh:256)
                        block_k = w * 64
                        staging = stg * (16 * (block_k // 2 + block_k // 16) + w * tok * 64 * 2)
                        partial = w * (tok // 8) * 32 * 4 * 4
                        if max(staging, partial) > 99 * 1024:
                            continue
                        route = f"nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<{tok}, {w}, {stg}>>"
                        cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", route)
                        results.append(
                            CandidateSpec(
                                id=cid,
                                operation=operation,
                                quantization_family=qtype,
                                kernel_family="sliced_k",
                                n=n,
                                k=k,
                                template_name="Nvfp4SlicedInstance",
                                launch_expr=route,
                                parameters={"tokens": tok, "warps": w, "stages": stg},
                                reduction_signature=f"a16:sliced_mma:k_slices={w}:linear",
                                tile_rows=tok,
                                tile_tokens=tok,
                                tile_k=None,
                                threads_per_block=w * 32,
                                k_slices=w,
                                min_tokens=1,
                                max_tokens=tok,
                            )
                        )
            # MMA candidates: BlockRows x BlockTokens in {(32, 32), (32, 64), (64, 64), (64, 80), (64, 96), (64, 112), (64, 128)}
            mma_configs = [
                (32, 32, 128, 16, 16, 2, 2),
                (32, 64, 128, 16, 16, 1, 3),
                (32, 64, 128, 32, 16, 2, 2),
                (64, 64, 64, 32, 16, 2, 2),
                (64, 64, 128, 32, 16, 1, 2),
                (64, 64, 128, 32, 16, 2, 1),
                (64, 80, 128, 32, 16, 1, 2),
                (64, 96, 128, 32, 16, 1, 2),
                (64, 96, 128, 64, 16, 1, 2),
                (64, 112, 128, 32, 16, 1, 2),
                (64, 128, 64, 64, 16, 2, 2),
                (64, 128, 128, 32, 16, 1, 1),
                (64, 128, 128, 32, 16, 2, 1),
            ]
            for br, bt, bk, wr, wt, stg, min_sm in mma_configs:
                route = f"nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<{br}, {bt}, {bk}, {wr}, {wt}, {stg}, {min_sm}>>"
                cid = self._stable_candidate_id(operation, qtype, n, k, "mma", route)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="mma",
                        n=n,
                        k=k,
                        template_name="Nvfp4A16MmaSchedule",
                        launch_expr=route,
                        parameters={"br": br, "bt": bt, "bk": bk, "wr": wr, "wt": wt, "stages": stg},
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=br,
                        tile_tokens=bt,
                        tile_k=bk,
                        threads_per_block=(br // wr) * (bt // wt) * 32 if wr > 0 and wt > 0 else 256,
                        k_slices=1,
                        min_tokens=max(16, bt // 2),
                        max_tokens=None,
                    )
                )

        elif qtype == "fp8":
            # FP8 legal candidates, mirroring the infernix_fp8_flash_next_sweep grid
            # (bench/ops/fp8_flash_next_sweep.cu). Sliced-K: Fp8SlicedInstance<Tokens,
            # Warps, Stages> always uses RowTiles = 1 (BlockRows = 16); RowTiles = 2
            # (BlockRows = 32) halves activation L2 re-reads per weight row and is
            # emitted through the explicit Fp8A16SlicedKMmaSchedule template.
            def fp8_sliced_min_blocks(warps: int, tok: int, stg: int, row_tiles: int) -> int:
                block_rows = 16 * row_tiles
                staging = stg * (block_rows * warps * 64 + warps * tok * 128)
                return 1 if staging > 48 * 1024 else 2

            # (Tokens, Warps, Stages, RowTiles) validated against the sweep grid.
            sliced_configs = [
                (8, 4, 1, 1), (16, 4, 1, 1), (16, 4, 2, 1), (16, 8, 1, 1), (16, 8, 2, 1),
                (32, 4, 1, 1), (32, 4, 2, 1), (32, 8, 1, 1), (32, 8, 2, 1),
                (64, 2, 1, 1), (64, 2, 2, 1), (64, 4, 1, 1), (64, 8, 1, 1),
                # RowTiles = 2 (BlockRows = 32): share one CTA's staged activations and
                # B fragments across two 16-row MMA tiles for T = 9..32 bands.
                (16, 4, 2, 2), (16, 8, 2, 2), (32, 4, 1, 2), (32, 4, 2, 2), (32, 8, 2, 2),
            ]
            for tok, w, stg, row_tiles in sliced_configs:
                mbps = fp8_sliced_min_blocks(w, tok, stg, row_tiles)
                if row_tiles == 1:
                    sched = f"Fp8SlicedInstance<{tok}, {w}, {stg}>"
                else:
                    sched = f"Fp8SlicedR2Instance<{tok}, {w}, {stg}>"
                route = f"fp8_linear_a16_sliced_k<Geometry, {sched}>(x, weight, out, stream)"
                cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", route)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="sliced_k",
                        n=n,
                        k=k,
                        template_name="Fp8A16SlicedKMmaSchedule",
                        launch_expr=route,
                        parameters={"tokens": tok, "warps": w, "stages": stg,
                                    "row_tiles": row_tiles, "min_blocks_per_sm": mbps},
                        reduction_signature=f"a16:sliced_mma:k_slices={w}:linear",
                        tile_rows=16 * row_tiles,
                        tile_tokens=tok,
                        tile_k=None,
                        threads_per_block=w * 32,
                        k_slices=w,
                        min_tokens=1,
                        max_tokens=tok,
                    )
                )
            # TMA schedules: Fp8A16TmaMmaSchedule<BlockTokens, BlockRows, WarpsTokens,
            # WarpsRows, Stages, MinBlocksPerSm[, BlockK[, Raster]]>. The full sweep
            # grid (BlockTokens 32/64/96/128, BlockRows 64/128/256, BlockK 64/128 and
            # the RowFast raster) so the autotuner's proposed routes match the
            # sweep-measured candidates and existing shape dispatches.
            tma_configs = [
                (32, 64, 1, 4, 6, 1),       # Fp8A16TmaT32R64
                (32, 64, 1, 4, 3, 2),       # Fp8A16TmaT32R64S3
                (32, 128, 1, 8, 4, 1),      # Fp8A16TmaT32R128
                (64, 128, 2, 4, 3, 1),      # Fp8A16TmaT64R128
                (64, 64, 2, 2, 4, 1),       # TmaT64R64S4
                (64, 128, 2, 4, 4, 1, 64),  # TmaT64R128S4K64
                (96, 128, 2, 4, 3, 1, 64),  # TmaT96R128S3K64
                (64, 256, 2, 8, 2, 1),      # TmaT64R256
                (64, 256, 2, 8, 3, 1, 64),  # TmaT64R256S3K64
                (128, 128, 4, 4, 2, 1),     # TmaT128R128
                (32, 256, 1, 8, 4, 1, 64),   # TmaT32R256S4K64
                (32, 128, 1, 4, 8, 1, 64),   # TmaT32R128S8K64
                (32, 128, 1, 4, 4, 2, 64),   # Fp8A16TmaT32R128S4B2K64 (2 blocks/SM on GB10)
                (64, 128, 2, 4, 3, 2, 64),   # Fp8A16TmaT64R128S3B2K64 (2 blocks/SM on GB10)
                (128, 128, 2, 4, 2, 2, 64),  # Fp8A16TmaT128R128S2B2K64 (2 blocks/SM on GB10)
            ]
            for cfg in tma_configs:
                cfg_s = ", ".join(str(x) for x in cfg)
                route = f"fp8_linear_a16_tma<Geometry, Fp8A16TmaMmaSchedule<{cfg_s}>>(x, weight, out, stream)"
                cid = self._stable_candidate_id(operation, qtype, n, k, "tma", route)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="tma",
                        n=n,
                        k=k,
                        template_name="Fp8A16TmaMmaSchedule",
                        launch_expr=route,
                        parameters={"cfg": list(cfg)},
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=cfg[1],
                        tile_tokens=cfg[0],
                        threads_per_block=32 + cfg[2] * cfg[3] * 32,
                        k_slices=1,
                        requires_features=["tma"],
                    )
                )
            # RowFast and Grouped L2-swizzled raster variants for GB10's 24 MiB L2 cache.
            raster_tma_configs = [
                ("Fp8A16TmaT64R256RF",  [64, 256, 2, 8, 2, 1, 128], "RowFast", 1),
                ("Fp8A16TmaT128R128RF", [128, 128, 4, 4, 2, 1, 128], "RowFast", 1),
                ("Fp8A16TmaT128R128G8", [128, 128, 4, 4, 2, 1, 128], "Grouped", 8),
                ("Fp8A16TmaT64R256G8",  [64, 256, 2, 8, 2, 1, 128], "Grouped", 8),
            ]
            for alias, cfg, raster_name, group_rows in raster_tma_configs:
                r_route = f"fp8_linear_a16_tma<Geometry, {alias}>(x, weight, out, stream)"
                results.append(
                    CandidateSpec(
                        id=self._stable_candidate_id(operation, qtype, n, k, "tma", r_route),
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="tma",
                        n=n,
                        k=k,
                        template_name=alias,
                        launch_expr=r_route,
                        parameters={"cfg": cfg, "raster": raster_name, "group_rows": group_rows},
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=cfg[1],
                        tile_tokens=cfg[0],
                        threads_per_block=32 + cfg[2] * cfg[3] * 32,
                        k_slices=1,
                        requires_features=["tma"],
                    )
                )

        elif qtype == "q8":
            # Q8 legal candidates
            q8_simt_routes = [
                ("launch_q8_a16_simt_r8_t4", 8, 4),
                ("launch_q8_a16_simt_r8_t8", 8, 8),
                ("launch_q8_a16_simt_r4_t4_w2_g16_s2", 4, 4),
            ]
            for rname, r_rows, r_toks in q8_simt_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "simt", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="simt",
                        n=n,
                        k=k,
                        template_name="Q8A16SimtSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:simt:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=128,
                        k_slices=1,
                    )
                )

            q8_sliced_routes = [
                ("launch_q8_a16_sliced_r16_t16_w8_s2", 16, 16),
                ("launch_q8_a16_sliced_r16_t16_w4_s2", 16, 16),
                ("launch_q8_a16_sliced_r16_t32_w4_s2", 16, 32),
            ]
            for rname, r_rows, r_toks in q8_sliced_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="sliced_k",
                        n=n,
                        k=k,
                        template_name="Q8A16SlicedKMmaSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:sliced_mma:k_slices=4:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=4,
                    )
                )

            q8_mma_routes = [
                ("launch_q8_a16_mma_r32_t64", 32, 64),
                ("launch_q8_a16_mma_r32_t96", 32, 96),
                ("launch_q8_a16_mma_r32_t128", 32, 128),
                ("launch_q8_a16_mma_r48_t64", 48, 64),
                ("launch_q8_a16_mma_r64_t96", 64, 96),
                ("launch_q8_a16_mma_r64_t128", 64, 128),
                ("launch_q8_a16_mma_r96_t96", 96, 96),
                ("launch_q8_a16_mma_r128_t64", 128, 64),
                ("launch_q8_a16_mma_r128_t80", 128, 80),
                ("launch_q8_a16_mma_r64x16_t48_k128_a1", 64, 48),
                ("launch_q8_a16_mma_r64x32_t64_k128_a1", 64, 64),
            ]
            for rname, r_rows, r_toks in q8_mma_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "mma", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="mma",
                        n=n,
                        k=k,
                        template_name="Q8A16MmaSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        min_tokens=16,
                    )
                )

        elif qtype == "q4":
            q4_gemv_routes: list[tuple[str, int, int, int]] = [
                ("launch_q4_a16_gemv_r4_w1_direct", 4, 1, 0),
                ("launch_q4_a16_simt_r4_t1_w2_g8_s2", 4, 1, 0),
            ]
            if k == 5120:
                q4_gemv_routes.append(("launch_q4_a16_gemv_r1_w8_direct", 1, 1, 5120))
            elif k == 6144:
                q4_gemv_routes.append(("launch_q4_a16_gemv_r1_w8_k6144", 1, 1, 6144))

            for rname, r_rows, r_toks, _ in q4_gemv_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "gemv", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="gemv",
                        n=n,
                        k=k,
                        template_name="Q4A16GemvSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:gemv:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        min_tokens=1,
                        max_tokens=1,
                    )
                )

            q4_simt_routes = [
                ("launch_q4_a16_simt_r4_t4_w2_g8_s2", 4, 4, 1, 64),
            ]
            for rname, r_rows, r_toks, min_t, max_t in q4_simt_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "simt", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="simt",
                        n=n,
                        k=k,
                        template_name="Q4A16SimtSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:simt:k_slices=2:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=2,
                        min_tokens=min_t,
                        max_tokens=max_t,
                    )
                )

            q4_sliced_routes: list[tuple[str, int, int, int, int, int]] = [
                ("launch_q4_a16_sliced_r16_t8_capacity4", 16, 4, 4, 1, 4),
                ("launch_q4_a16_sliced_r16_t8_w4_s2", 16, 8, 4, 1, 64),
                ("launch_q4_a16_sliced_r16_t16_w4_s2", 16, 16, 4, 1, 128),
                ("launch_q4_a16_sliced_r16_t16_w2_s2", 16, 16, 2, 1, 128),
                ("launch_q4_a16_sliced_r16_t32_w4_s2", 16, 32, 4, 1, 256),
                ("launch_q4_a16_sliced_r32_t8_w4_s2", 32, 8, 4, 1, 64),
                ("launch_q4_a16_sliced_r32_t16_w4_s2", 32, 16, 4, 1, 128),
                ("launch_q4_a16_sliced_r32_t16_w4_s1", 32, 16, 4, 1, 128),
                ("launch_q4_a16_sliced_r32_t24_w4_s2", 32, 24, 4, 1, 192),
                ("launch_q4_a16_sliced_r32_t32_w4_s1", 32, 32, 4, 1, 576),
                ("launch_q4_a16_sliced_r32_t32_w4_s2", 32, 32, 4, 1, 576),
                ("launch_q4_a16_sliced_r32_t32_w2_s2", 32, 32, 2, 1, 576),
                ("launch_q4_a16_sliced_r32_t64_w2_s1", 32, 64, 2, 1, 576),
            ]
            if k == 2048:
                q4_sliced_routes.append(("launch_q4_a16_sliced_k2048_t4", 16, 4, 8, 1, 4))
            elif k == 5120:
                q4_sliced_routes.extend([
                    ("launch_q4_a16_sliced_k5120_t4", 16, 4, 8, 1, 4),
                    ("launch_q4_a16_sliced_k5120_t8", 16, 8, 8, 1, 8),
                    ("launch_q4_a16_sliced_k5120_t16", 16, 16, 8, 1, 16),
                    ("launch_q4_a16_sliced_k5120_t24", 16, 24, 8, 1, 24),
                ])

            for rname, r_rows, r_toks, k_warps, min_t, max_t in q4_sliced_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="sliced_k",
                        n=n,
                        k=k,
                        template_name="Q4A16SlicedKMmaSchedule",
                        launch_expr=rname,
                        reduction_signature=f"a16:sliced_mma:k_slices={k_warps}:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=k_warps * 32,
                        k_slices=k_warps,
                        min_tokens=min_t,
                        max_tokens=max_t,
                    )
                )

            q4_mma_routes = [
                ("launch_q4_a16_mma_r32_t32_k128_s2_a2", 32, 32, 16),
                ("launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2", 32, 32, 16),
                ("launch_q4_a16_mma_r32_t64_k128_s2_a1", 32, 64, 16),
                ("launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2", 32, 64, 16),
                ("launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2", 32, 64, 16),
                ("launch_q4_a16_mma_r32_t128_k64_s2_a2", 32, 128, 32),
                ("launch_q4_a16_mma_r64_t48", 64, 48, 16),
                ("launch_q4_a16_mma_r64_t64_k128_s2_a1", 64, 64, 16),
                ("launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2", 64, 64, 16),
                ("launch_q4_a16_mma_r64_t72", 64, 72, 24),
                ("launch_q4_a16_mma_r64_t80", 64, 80, 24),
                ("launch_q4_a16_mma_r64_t96", 64, 96, 32),
                ("launch_q4_a16_mma_r64_t112", 64, 112, 32),
                ("launch_q4_a16_mma_r64_t120", 64, 120, 32),
                ("launch_q4_a16_mma_r64_t128", 64, 128, 32),
            ]
            for rname, r_rows, r_toks, min_t in q4_mma_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "mma", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="mma",
                        n=n,
                        k=k,
                        template_name="Q4A16MmaSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        min_tokens=min_t,
                        max_tokens=None,
                    )
                )

        elif qtype == "q5":
            q5_direct_routes: list[tuple[str, int, int, int, int, int]] = []
            if n != 1152:
                if k == 5120:
                    q5_direct_routes.extend([
                        ("launch_q5_a16_direct_r1_t1_w4_k5120", 1, 1, 1, 1, 5120),
                        ("launch_q5_a16_direct_r1_t2_w4_k5120", 1, 2, 2, 2, 5120),
                        ("launch_q5_a16_direct_r1_t3_w4_k5120", 1, 3, 3, 3, 5120),
                    ])
                elif k == 6144:
                    q5_direct_routes.extend([
                        ("launch_q5_a16_direct_r1_t1_w4_k6144", 1, 1, 1, 1, 6144),
                        ("launch_q5_a16_direct_r1_t2_w2_k6144", 1, 2, 2, 2, 6144),
                        ("launch_q5_a16_direct_r1_t3_w2_k6144", 1, 3, 3, 3, 6144),
                    ])
                elif k == 17408:
                    q5_direct_routes.extend([
                        ("launch_q5_a16_direct_r1_t1_w4_k17408", 1, 1, 1, 1, 17408),
                        ("launch_q5_a16_direct_r1_t2_w2_k17408", 1, 2, 2, 2, 17408),
                        ("launch_q5_a16_direct_r1_t3_w2_k17408", 1, 3, 3, 3, 17408),
                        ("launch_q5_a16_direct_r1_t4_w2_k17408", 1, 4, 4, 4, 17408),
                    ])
            q5_direct_routes.extend([
                ("launch_q5_a16_direct_r2_t4_w2_g8_b4", 2, 4, 1, 32, 0),
                ("launch_q5_a16_direct_r2_t4_w4_g4_b4", 2, 4, 1, 32, 0),
            ])
            for rname, r_rows, r_toks, min_t, max_t, _ in q5_direct_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "simt", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="simt",
                        n=n,
                        k=k,
                        template_name="Q5A16DirectSimtSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:simt:k_slices=2:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=128,
                        k_slices=2,
                        min_tokens=min_t,
                        max_tokens=max_t,
                    )
                )

            q5_sliced_routes: list[tuple[str, int, int, int, int, int]] = [
                ("launch_q5_a16_sliced_r16_t8_capacity4", 16, 4, 4, 1, 4),
                ("launch_q5_a16_sliced_r16_t8_w4_s2", 16, 8, 4, 1, 64),
                ("launch_q5_a16_sliced_r16_t16_w4_s2", 16, 16, 4, 1, 128),
                ("launch_q5_a16_sliced_r16_t16_w2_s2", 16, 16, 2, 1, 128),
                ("launch_q5_a16_sliced_r16_t24_w4_s2", 16, 24, 4, 1, 192),
                ("launch_q5_a16_sliced_r16_t32_w4_s2", 16, 32, 4, 1, 256),
                ("launch_q5_a16_sliced_r32_t16_w4_s2", 32, 16, 4, 1, 128),
                ("launch_q5_a16_sliced_r32_t24_w4_s2_pairwise", 32, 24, 4, 1, 192),
                ("launch_q5_a16_sliced_r32_t32_w2_s2", 32, 32, 2, 1, 576),
                ("launch_q5_a16_sliced_r32_t32_w4_s1", 32, 32, 4, 1, 576),
                ("launch_q5_a16_sliced_r32_t32_w4_s2", 32, 32, 4, 1, 576),
                ("launch_q5_a16_sliced_r32_t64_w2_s1", 32, 64, 2, 1, 576),
                ("launch_q5_a16_sliced_r32_t64_w2_s2", 32, 64, 2, 1, 576),
            ]
            for rname, r_rows, r_toks, k_warps, min_t, max_t in q5_sliced_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="sliced_k",
                        n=n,
                        k=k,
                        template_name="Q5A16SlicedKMmaSchedule",
                        launch_expr=rname,
                        reduction_signature=f"a16:sliced_mma:k_slices={k_warps}:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=k_warps * 32,
                        k_slices=k_warps,
                        min_tokens=min_t,
                        max_tokens=max_t,
                    )
                )

            q5_mma_routes = [
                ("launch_q5_a16_mma_r32_t32_k128_s2_a2", 32, 32, 16),
                ("launch_q5_a16_mma_r32_t64_k128_s2_a1", 32, 64, 16),
                ("launch_q5_a16_mma_r32_t128", 32, 128, 32),
                ("launch_q5_a16_mma_r32_t128_k128_s2_a1", 32, 128, 32),
                ("launch_q5_a16_mma_r64_t64_k128_s1_a1", 64, 64, 16),
                ("launch_q5_a16_mma_r64_t64_k128_s2_a1", 64, 64, 16),
                ("launch_q5_a16_mma_r64_t80_k128_s1_a1", 64, 80, 24),
                ("launch_q5_a16_mma_r64_t96_k128_s1_a1", 64, 96, 32),
                ("launch_q5_a16_mma_r64_t128", 64, 128, 32),
                ("launch_q5_a16_mma_r64_t128_k128_s1_a1", 64, 128, 32),
            ]
            for rname, r_rows, r_toks, min_t in q5_mma_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "mma", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="mma",
                        n=n,
                        k=k,
                        template_name="Q5A16MmaSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        min_tokens=min_t,
                        max_tokens=None,
                    )
                )

        elif qtype == "q6":
            if n != 1152:
                cid = self._stable_candidate_id(operation, qtype, n, k, "gemv", "launch_q6_a16_gemv_r4_w2_g16")
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="gemv",
                        n=n,
                        k=k,
                        template_name="Q6A16GemvSchedule",
                        launch_expr="launch_q6_a16_gemv_r4_w2_g16",
                        reduction_signature="a16:gemv:k_slices=2:linear",
                        tile_rows=4,
                        tile_tokens=1,
                        threads_per_block=256,
                        k_slices=2,
                        min_tokens=1,
                        max_tokens=1,
                    )
                )

            cid_simt = self._stable_candidate_id(operation, qtype, n, k, "simt", "launch_q6_a16_simt_r8_t4")
            results.append(
                CandidateSpec(
                    id=cid_simt,
                    operation=operation,
                    quantization_family=qtype,
                    kernel_family="simt",
                    n=n,
                    k=k,
                    template_name="Q6A16SimtSchedule",
                    launch_expr="launch_q6_a16_simt_r8_t4",
                    reduction_signature="a16:simt:k_slices=1:serial",
                    tile_rows=8,
                    tile_tokens=4,
                    threads_per_block=256,
                    k_slices=1,
                    min_tokens=1,
                    max_tokens=32,
                )
            )

            q6_sliced_routes: list[tuple[str, int, int, int, int, int]] = [
                ("launch_q6_a16_sliced_r16_t8_w4_s2", 16, 8, 4, 1, 64),
                ("launch_q6_a16_sliced_r16_t24_w4_s2", 16, 24, 4, 1, 192),
                ("launch_q6_a16_sliced_r16_t32_w4_s2", 16, 32, 4, 1, 256),
                ("launch_q6_a16_sliced_r32_t16_w4_s2", 32, 16, 4, 1, 128),
                ("launch_q6_a16_sliced_r32_t32_w4_s1", 32, 32, 4, 1, 576),
                ("launch_q6_a16_sliced_r32_t32_w4_s2", 32, 32, 4, 1, 576),
                ("launch_q6_a16_sliced_r32_t64_w2_s1", 32, 64, 2, 1, 1152),
            ]
            for rname, r_rows, r_toks, k_warps, min_t, max_t in q6_sliced_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "sliced_k", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="sliced_k",
                        n=n,
                        k=k,
                        template_name="Q6A16SlicedKMmaSchedule",
                        launch_expr=rname,
                        reduction_signature=f"a16:sliced_mma:k_slices={k_warps}:linear",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=k_warps * 32,
                        k_slices=k_warps,
                        min_tokens=min_t,
                        max_tokens=max_t,
                    )
                )

            q6_mma_routes = [
                ("launch_q6_a16_mma_r32_t64_k128_s2_a1", 32, 64, 16),
                ("launch_q6_a16_mma_r32_t128_k64_s2_a2", 32, 128, 32),
                ("launch_q6_a16_mma_r32_t128_k128_s2_a1", 32, 128, 32),
                ("launch_q6_a16_mma_r64_t40_k128", 64, 40, 16),
                ("launch_q6_a16_mma_r64_t48_k128", 64, 48, 16),
                ("launch_q6_a16_mma_r64_t56_k128", 64, 56, 16),
                ("launch_q6_a16_mma_r64_t64_k128", 64, 64, 16),
                ("launch_q6_a16_mma_r64_t72_k128", 64, 72, 24),
                ("launch_q6_a16_mma_r64_t80", 64, 80, 24),
                ("launch_q6_a16_mma_r64_t96", 64, 96, 32),
                ("launch_q6_a16_mma_r64_t112", 64, 112, 32),
                ("launch_q6_a16_mma_r64_t128", 64, 128, 32),
                ("launch_q6_a16_mma_r64_t128_k128_s1_a1", 64, 128, 32),
            ]
            for rname, r_rows, r_toks, min_t in q6_mma_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, "mma", rname)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family="mma",
                        n=n,
                        k=k,
                        template_name="Q6A16MmaSchedule",
                        launch_expr=rname,
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        min_tokens=min_t,
                        max_tokens=None,
                    )
                )

        elif qtype == "bf16":
            # BF16 candidates matching bf16_flash_next_sweep
            bf16_routes = [
                ("launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>", "sliced_r32t16w4", "sliced_k", 32, 16),
                ("launch_bf16_mma<Bf16A16MmaR32T32K128S3>", "mma_r32t32k128s3", "mma", 32, 32),
                ("launch_bf16_mma<Bf16A16MmaR32T32K256S3>", "mma_r32t32k256s3", "mma", 32, 32),
                ("launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>", "tail_r64t32s3", "tma", 64, 32),
                ("launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>", "tail_r64t64s3", "tma", 64, 64),
                ("launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>", "tail_r64t128s2_rows", "tma", 64, 128),
                ("launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>", "tail_r128t64s4_rows", "tma", 128, 64),
                ("launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 256, 32, 64, 2, 1, Bf16MmaRaster::RowFast>>", "tail_r64t256s2_rows", "tma", 64, 256),
                ("launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>", "tma_r64t128k64s2_rows", "tma", 64, 128),
                ("launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>>", "tma_r128t64k64s4_rows", "tma", 128, 64),
                ("launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>", "tma_r128t128k64s3_w64x32_rows", "tma", 128, 128),
                ("launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 256, 64, 32, 64, 2, 1, Bf16MmaRaster::RowFast>>", "tma_r64t256k64s2_rows", "tma", 64, 256),
            ]
            for r_expr, r_name, k_fam, r_rows, r_toks in bf16_routes:
                cid = self._stable_candidate_id(operation, qtype, n, k, k_fam, r_name)
                results.append(
                    CandidateSpec(
                        id=cid,
                        operation=operation,
                        quantization_family=qtype,
                        kernel_family=k_fam,
                        n=n,
                        k=k,
                        template_name=r_name,
                        launch_expr=r_expr,
                        reduction_signature="a16:mma:k_slices=1:serial",
                        tile_rows=r_rows,
                        tile_tokens=r_toks,
                        threads_per_block=256,
                        k_slices=1,
                        requires_features=["tma"] if k_fam == "tma" else [],
                    )
                )

        return results
