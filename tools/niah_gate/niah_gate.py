#!/usr/bin/env python3
"""Needle-in-a-Haystack Gate: test generation, calibration, streaming grading, and pass-rate reporting."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
CORPUS_FILES = (
    "pride_and_prejudice.txt",
    "sherlock_holmes.txt",
    "great_expectations.txt",
    "moby_dick.txt",
)

MIN_PARAGRAPH_CHARS = 200
MAX_NEEDLE_REGION_TOKENS = 512 * 1024
TOKENS_PER_LINE_BUDGET = 40
MIN_TOKENS_PER_LINE = 12
PROMPT_HEADROOM_TOKENS = 512
MIN_DIAGNOSTIC_NEEDLES = 256
MAX_CALIBRATION_ITERS = 5

BASE_WORDS = (
    "acorn", "agate", "alder", "amber", "anchor", "anise", "apple", "apricot",
    "arch", "aspen", "aster", "atlas", "azalea", "badger", "bamboo", "banyan",
    "barley", "basalt", "beacon", "beaver", "beech", "berry", "birch", "bison",
    "blossom", "bobcat", "boulder", "bramble", "breeze", "briar", "bridge", "brook",
    "buckeye", "cairn", "camphor", "candle", "canyon", "caribou", "cascade", "cedar",
    "chalk", "clover", "cobalt", "cobble", "condor", "copper", "coral", "cougar",
    "coyote", "crane", "creek", "crest", "cricket", "crocus", "crystal", "cypress",
    "dahlia", "daisy", "dapple", "delta", "desert", "dipper", "dolphin", "dove",
    "drift", "dune", "eagle", "ebony", "eddy", "egret", "elder", "elm",
    "ember", "emerald", "ermine", "falcon", "fennel", "fern", "finch", "fir",
    "fjord", "flint", "flora", "flute", "fog", "forest", "fossil", "fox",
    "frost", "garnet", "gazelle", "gecko", "geyser", "glacier", "glade", "gorge",
    "granite", "gravel", "grouse", "harbor", "hare", "hawk", "hazel", "heather",
    "hemlock", "heron", "hickory", "hill", "hollow", "holly", "horizon", "hornet",
    "ibis", "indigo", "iris", "island", "ivory", "ivy", "jackal", "jade",
    "jaguar", "jasper", "jay", "juniper", "kestrel", "kinglet", "knoll", "lagoon",
    "larch", "lark", "laurel", "lava", "lichen", "lilac", "lily", "linden",
    "linnet", "lizard", "locust", "lotus", "lupine", "lynx", "magpie", "mahogany",
    "mallow", "mantis", "maple", "marble", "marmot", "marsh", "martin", "meadow",
    "mercury", "mesa", "mica", "minnow", "mist", "monarch", "moose", "moss",
    "moth", "mulberry", "myrtle", "nebula", "nectar", "nettle", "newt", "nickel",
    "nutmeg", "oak", "oasis", "obsidian", "ocelot", "ochre", "olive", "onyx",
    "opal", "oriole", "osprey", "otter", "owl", "pebble", "pelican", "peony",
    "pepper", "perch", "petal", "pewter", "pheasant", "pine", "plover", "plum",
    "poplar", "poppy", "prairie", "puffin", "puma", "pyrite", "quail", "quartz",
    "quince", "rabbit", "rapids", "raven", "redwood", "reed", "ridge", "ripple",
    "river", "robin", "rowan", "ruby", "sage", "salmon", "sand", "sapphire",
    "scarlet", "sedge", "sequoia", "serval", "shale", "shell", "shrike", "silver",
    "slate", "sorrel", "sparrow", "spruce", "starling", "stone", "stork", "stream",
    "summit", "sundew", "swallow", "sycamore", "tamarack", "tanager", "tarn", "teal",
    "thistle", "thrush", "thyme", "tiger", "timber", "topaz", "torrent", "trillium",
    "trout", "tulip", "tundra", "turtle", "umbra", "valley", "verbena", "vermilion",
    "vetch", "violet", "vireo", "walnut", "warbler", "weasel", "willow", "wren",
    "yarrow", "yew", "zephyr", "zircon",
)

TAGS = tuple("abcdefghijklmnopqrstuvwxyz")

PLACE_WORDS = (
    "alcove", "arch", "bank", "basin", "bay", "bend", "bluff", "bog",
    "brook", "cairn", "canyon", "cape", "cave", "chasm", "clearing", "cliff",
    "coast", "copse", "cove", "crag", "creek", "crest", "crossing", "dale",
    "dell", "delta", "divide", "downs", "drift", "dune", "edge", "estuary",
    "falls", "fen", "field", "fjord", "flat", "ford", "forest", "fork",
    "gap", "garden", "gate", "glade", "glen", "gorge", "gravel", "grove",
    "gully", "harbor", "haven", "head", "heath", "heights", "hill", "hollow",
    "inlet", "island", "isle", "knoll", "lagoon", "lake", "ledge", "loch",
    "marsh", "meadow", "mesa", "moor", "mound", "narrows", "notch", "oasis",
    "pass", "pasture", "peak", "peninsula", "plain", "plateau", "point", "pond",
    "pool", "prairie", "rapids", "reach", "reef", "ridge", "rill", "rim",
    "rise", "river", "rock", "run", "saddle", "sandbar", "scarp", "shelf",
    "shoal", "shore", "slope", "sound", "spire", "spring", "spur", "steppe",
    "strait", "strand", "stream", "summit", "swale", "tarn", "terrace", "thicket",
    "tor", "trail", "trench", "tundra", "vale", "valley", "verge", "wash",
    "watch", "water", "weir", "well", "woods",
)

ANSWER_LINE_RE = re.compile(r"([a-z]+-[a-z0-9]+-[a-z]+)\s*:\s*(\d{5})\b")

LEGACY_FILLER_SENTENCES = (
    "The archival ledger records standard shipping manifests across northern ports during winter.",
    "Inspectors reviewed every wooden crate before transferring the cargo onto river barges.",
    "Clerks verified ink stamps, wax seals, and warehouse inventory schedules throughout the afternoon.",
    "Surveyors measured stone embankments along the canal and noted steady water levels.",
    "Messengers delivered sealed dispatches between coastal stations before sunset.",
)


@dataclass(frozen=True)
class Needle:
    index: int
    codeword: str
    number: str
    sentence: str


@dataclass(frozen=True)
class PlacementStats:
    count: int
    haystack_chars: int
    needle_region_chars: int
    first_char: int
    last_char: int
    gap_mean: float
    gap_max: float
    expected_gap: float
    valid: bool
    reason: str


@dataclass
class GradeResult:
    unique_hits: int
    total_needles: int
    duplicates: int
    wrong_numbers: int
    missing: list[str] = field(default_factory=list)
    wrong_pairs: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class RunResult:
    run_index: int
    passed: bool
    void_reasons: list[str]
    failures: list[str]
    grade: GradeResult
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    finish_reason: str
    elapsed_s: float
    output_budget: int
    placement: PlacementStats
    calibration_iters: int
    calibrated: bool
    http_status: int | None = None
    error: str | None = None

    def summary_line(self) -> str:
        return (
            f"+{self.grade.unique_hits} new unique={self.grade.unique_hits}/{self.grade.total_needles} "
            f"(dup {self.grade.duplicates}) in={self.prompt_tokens} out={self.completion_tokens} "
            f"reas={self.reasoning_tokens} cached={self.cached_tokens} finish={self.finish_reason}"
        )


def strip_gutenberg(text: str) -> str:
    """Strip Project Gutenberg header and footer boilerplate."""
    start_m = re.search(r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*\*\*", text, re.IGNORECASE)
    if start_m:
        text = text[start_m.end():]
    end_m = re.search(r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*\*\*", text, re.IGNORECASE)
    if end_m:
        text = text[:end_m.start()]
    return text.strip()


def load_corpus_paragraphs(corpus_dir: Path, min_chars: int = MIN_PARAGRAPH_CHARS) -> list[str]:
    """Load public-domain novel paragraphs >= min_chars after stripping Gutenberg headers/footers."""
    paragraphs: list[str] = []
    for name in CORPUS_FILES:
        path = corpus_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing corpus file {path}. Run tools/niah_gate/fetch_corpus.sh first."
            )
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = strip_gutenberg(raw)
        # Split on blank lines (normalize CRLF first)
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        for para in re.split(r"\n\s*\n", normalized):
            cleaned = re.sub(r"\s+", " ", para).strip()
            if len(cleaned) >= min_chars:
                paragraphs.append(cleaned)
    if not paragraphs:
        raise ValueError(f"No paragraphs >= {min_chars} chars found in {corpus_dir}")
    return paragraphs


def default_needle_count(max_context: int) -> int:
    return 1024 if max_context <= 65536 else 2048


def target_needle_region_tokens(max_context: int, depth: float | None) -> int:
    if depth is not None:
        if not (0.0 < depth <= 1.0):
            raise ValueError(f"--depth must be a fraction in (0, 1], got {depth} (never a token count)")
        return int(max_context * depth)
    return min((max_context * 2) // 3, MAX_NEEDLE_REGION_TOKENS)


def validate_run_args(max_context: int, needles: int, depth: float | None, max_turns: int, allow_small_needles: bool) -> None:
    if max_turns != 1:
        raise ValueError(f"--max-turns must be 1 (single inference only), got {max_turns}")
    if depth is not None and not (0.0 < depth <= 1.0):
        raise ValueError(f"--depth must be a fraction of max_context in (0, 1], got {depth}")
    if not allow_small_needles and needles < MIN_DIAGNOSTIC_NEEDLES:
        raise ValueError(
            f"--needles {needles} is degenerate (< {MIN_DIAGNOSTIC_NEEDLES} minimum diagnostic arm); "
            "aborting loudly"
        )
    if needles <= 0:
        raise ValueError(f"--needles must be > 0, got {needles}")
    if max_context <= PROMPT_HEADROOM_TOKENS + needles * MIN_TOKENS_PER_LINE:
        raise ValueError(
            f"max_context={max_context} cannot fit {needles} needles "
            f"(need > {PROMPT_HEADROOM_TOKENS + needles * MIN_TOKENS_PER_LINE} tokens)"
        )


def compute_output_budget(max_context: int, prompt_tokens: int, needles: int) -> int:
    available = max_context - prompt_tokens - PROMPT_HEADROOM_TOKENS
    budget = min(needles * TOKENS_PER_LINE_BUDGET, available)
    min_required = needles * MIN_TOKENS_PER_LINE
    if budget < min_required:
        raise RuntimeError(
            f"ABORT: output budget {budget} cannot fit {needles} answer lines "
            f"(need >= {min_required} tokens; max_context={max_context}, prompt_tokens={prompt_tokens})"
        )
    return budget


def generate_needles(count: int, seed: int, prefix_collisions: bool = False) -> list[Needle]:
    """Generate unique needles with pinned RNG seed."""
    rng = random.Random(seed)
    numbers = rng.sample(range(10000, 100000), count)

    codewords: list[str] = []
    if prefix_collisions:
        # Legacy sibling families sharing <base> (and <base>-<place>-<tag> collisions)
        bases = list(BASE_WORDS)
        places = list(PLACE_WORDS)
        rng.shuffle(bases)
        rng.shuffle(places)
        seen: set[str] = set()
        b_idx = 0
        while len(codewords) < count:
            base = bases[b_idx % len(bases)]
            b_idx += 1
            for place in places:
                for tag in TAGS:
                    word = f"{base}-{place}-{tag}"
                    if word not in seen:
                        seen.add(word)
                        codewords.append(word)
                        if len(codewords) == count:
                            break
                if len(codewords) == count:
                    break
    else:
        # Globally unique <base>-<tag> per needle
        pairs = [(b, t) for b in BASE_WORDS for t in TAGS]
        if count > len(pairs):
            raise ValueError(f"Requested {count} needles exceeds max unique <base>-<tag> pairs ({len(pairs)})")
        chosen_pairs = rng.sample(pairs, count)
        for b, t in chosen_pairs:
            p = rng.choice(PLACE_WORDS)
            codewords.append(f"{b}-{t}-{p}")

    needles: list[Needle] = []
    for i, (word, num) in enumerate(zip(codewords, numbers)):
        num_str = f"{num:05d}"
        sentence = f"The secret codeword for {word} is the number {num_str}."
        needles.append(Needle(index=i, codeword=word, number=num_str, sentence=sentence))
    return needles


def build_background_text(paragraphs: list[str] | None, target_chars: int, story_on: bool, seed: int) -> str:
    """Build coherent background text of at least target_chars characters."""
    if target_chars <= 0:
        return ""
    chunks: list[str] = []
    total = 0
    if story_on:
        if not paragraphs:
            raise ValueError("Story mode requires loaded corpus paragraphs")
        # Rotate deterministically by seed, cycling cleanly if target_chars exceeds one pass
        rng = random.Random(seed ^ 0x9E3779B9)
        start_idx = rng.randrange(len(paragraphs))
        idx = start_idx
        while total < target_chars:
            p = paragraphs[idx % len(paragraphs)]
            chunks.append(p)
            total += len(p) + 2
            idx += 1
    else:
        idx = 0
        while total < target_chars:
            s = LEGACY_FILLER_SENTENCES[idx % len(LEGACY_FILLER_SENTENCES)]
            chunks.append(s)
            total += len(s) + 1
            idx += 1
    joined = "\n\n".join(chunks) if story_on else " ".join(chunks)
    if len(joined) > target_chars:
        # Trim cleanly at whitespace boundary
        cut = joined.rfind(" ", 0, target_chars)
        joined = joined[: cut if cut > target_chars // 2 else target_chars]
    return joined


def evaluate_placement(positions: list[int], region_chars: int, haystack_chars: int) -> PlacementStats:
    n = len(positions)
    if n == 0:
        return PlacementStats(0, haystack_chars, region_chars, 0, 0, 0.0, 0.0, 0.0, False, "empty")
    first = positions[0]
    last = positions[-1]
    expected_gap = region_chars / float(n)
    if n == 1:
        return PlacementStats(1, haystack_chars, region_chars, first, last, 0.0, 0.0, expected_gap, True, "ok")
    gaps = [positions[i + 1] - positions[i] for i in range(n - 1)]
    gap_mean = sum(gaps) / float(len(gaps))
    gap_max = float(max(gaps))

    # Stratified-jitter invariants: first in/near stratum 0, last in/near stratum n-1,
    # mean gap near region/n, max gap <= ~2.5 * expected_gap (for n >= 4).
    reasons: list[str] = []
    if first > 2.5 * expected_gap:
        reasons.append(f"first_char {first} > 2.5*expected_gap ({2.5 * expected_gap:.1f})")
    if last < region_chars - 2.5 * expected_gap:
        reasons.append(f"last_char {last} < region-2.5*expected_gap ({region_chars - 2.5 * expected_gap:.1f})")
    ratio = gap_mean / expected_gap if expected_gap > 0 else 0.0
    if not (0.75 <= ratio <= 1.25):
        reasons.append(f"gap_mean ratio {ratio:.3f} outside [0.75, 1.25]")
    if n >= 4 and gap_max > 2.6 * expected_gap:
        reasons.append(f"gap_max {gap_max:.1f} > 2.6*expected_gap ({2.6 * expected_gap:.1f})")

    valid = len(reasons) == 0
    return PlacementStats(
        count=n,
        haystack_chars=haystack_chars,
        needle_region_chars=region_chars,
        first_char=first,
        last_char=last,
        gap_mean=gap_mean,
        gap_max=gap_max,
        expected_gap=expected_gap,
        valid=valid,
        reason="ok" if valid else "; ".join(reasons),
    )


def embed_needles_stratified_jitter(
    background: str, needles: list[Needle], region_chars: int, seed: int
) -> tuple[str, PlacementStats]:
    """Embed one needle per stratum [i*L/n, (i+1)*L/n) with uniform jitter within the stratum."""
    n = len(needles)
    if region_chars <= 0 or region_chars > len(background):
        region_chars = len(background)
    rng = random.Random(seed ^ 0x517CC1B7)

    # Pick stratified-jitter raw insertion offsets in [0, region_chars], snapping to space boundaries
    raw_offsets: list[int] = []
    for i in range(n):
        lo = (i * region_chars) / n
        hi = ((i + 1) * region_chars) / n
        pos = int(rng.uniform(lo, hi))
        if pos > 0:
            sp = background.find(" ", pos)
            if sp != -1 and sp < int(hi) + 128 and sp <= len(background):
                pos = sp + 1
        pos = max(0, min(pos, len(background)))
        if raw_offsets and pos < raw_offsets[-1]:
            pos = raw_offsets[-1]
        raw_offsets.append(pos)

    # Assemble haystack with embedded sentences and record exact final character offsets of needles
    parts: list[str] = []
    final_positions: list[int] = []
    cursor = 0
    out_len = 0
    for offset, needle in zip(raw_offsets, needles):
        seg = background[cursor:offset]
        parts.append(seg)
        out_len += len(seg)
        if out_len > 0 and not seg.endswith((" ", "\n")):
            parts.append(" ")
            out_len += 1
        final_positions.append(out_len)
        parts.append(needle.sentence)
        out_len += len(needle.sentence)
        parts.append(" ")
        out_len += 1
        cursor = offset
    tail = background[cursor:]
    parts.append(tail)
    out_len += len(tail)

    haystack = "".join(parts)
    # Approximate equivalent region length in the final haystack (background region + inserted needles)
    final_region = out_len - len(tail)
    stats = evaluate_placement(final_positions, max(1, final_region), out_len)
    return haystack, stats


def build_prompt(needles: list[Needle], haystack: str) -> str:
    codeword_list = ", ".join(n.codeword for n in needles)
    return (
        "Below is a long story archive. Follow ONLY the final instruction. "
        "For each of these codewords, state its secret number as '<word>: <number>' "
        f"one per line, nothing else. Codewords: {codeword_list} ARCHIVE: {haystack}"
    )


def grade_response(answer_text: str, needles: list[Needle]) -> GradeResult:
    """Single regex pass over answer_text: exact match for known codeword + correct 5-digit number."""
    expected = {n.codeword: n.number for n in needles}
    seen_correct: set[str] = set()
    duplicates = 0
    wrong_pairs: list[tuple[str, str, str]] = []

    for m in ANSWER_LINE_RE.finditer(answer_text):
        word = m.group(1)
        num = m.group(2)
        exp_num = expected.get(word)
        if exp_num is None:
            continue
        if num == exp_num:
            if word in seen_correct:
                duplicates += 1
            else:
                seen_correct.add(word)
        else:
            wrong_pairs.append((word, num, exp_num))

    missing = [n.codeword for n in needles if n.codeword not in seen_correct]
    return GradeResult(
        unique_hits=len(seen_correct),
        total_needles=len(needles),
        duplicates=duplicates,
        wrong_numbers=len(wrong_pairs),
        missing=missing,
        wrong_pairs=wrong_pairs,
    )


def classify_verdict(
    grade: GradeResult,
    finish_reason: str,
    completion_tokens: int,
    output_budget: int,
    reasoning_tokens: int,
    cached_tokens: int,
    calibrated: bool,
    placement: PlacementStats,
    require_thinking: bool,
    require_cold: bool,
    http_status: int | None,
    error: str | None,
) -> tuple[bool, list[str], list[str]]:
    void_reasons: list[str] = []
    failures: list[str] = []

    if not calibrated:
        void_reasons.append("void_calibration_not_converged")
    if not placement.valid:
        void_reasons.append(f"void_skewed_placement({placement.reason})")
    if require_cold and cached_tokens > 0:
        void_reasons.append(f"void_warm_cache(cached={cached_tokens})")

    if http_status == 500:
        failures.append("non_finite_logits_http_500")
    elif error:
        if "stall" in error.lower() or "timed out" in error.lower():
            failures.append("idle_stall")
        else:
            failures.append(f"transport_error({error})")

    if grade.wrong_numbers > 0:
        failures.append(f"wrong_numbers({grade.wrong_numbers})")
    if grade.duplicates > 0:
        failures.append(f"duplicate_loop(dup={grade.duplicates})")

    if finish_reason == "length":
        failures.append("budget_length_clipped(fix_budget_never_report_as_recall)")
    elif finish_reason == "stop" and grade.unique_hits < grade.total_needles:
        if completion_tokens < output_budget * 0.5:
            failures.append("premature_eos_mid_enumeration")
        else:
            failures.append("incomplete_recall")
    elif grade.unique_hits < grade.total_needles and not failures:
        failures.append("incomplete_recall")

    if require_thinking and reasoning_tokens <= 0:
        failures.append("thinking_off(reas=0)")

    passed = (
        len(void_reasons) == 0
        and len(failures) == 0
        and grade.unique_hits == grade.total_needles
        and grade.wrong_numbers == 0
        and finish_reason == "stop"
    )
    return passed, void_reasons, failures


def parse_base_url(base_url: str) -> tuple[str, int, bool]:
    parsed = urllib.parse.urlparse(base_url)
    use_tls = parsed.scheme == "https"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if use_tls else 80)
    return host, port, use_tls


def probe_prompt_tokens(
    base_url: str, model: str, prompt: str, seed: int, reasoning_effort: str | None = None
) -> int:
    """Measure exact prompt_tokens via a non-streaming max_tokens=1 call."""
    host, port, use_tls = parse_base_url(base_url)
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode("utf-8")
    conn = (
        http.client.HTTPSConnection(host, port, timeout=180.0)
        if use_tls
        else http.client.HTTPConnection(host, port, timeout=180.0)
    )
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        if resp.status != 200:
            raise RuntimeError(f"Calibration probe failed with HTTP {resp.status}: {raw[:400]}")
        obj = json.loads(raw)
        usage = obj.get("usage") or {}
        pt = usage.get("prompt_tokens")
        if pt is None:
            raise RuntimeError(f"Calibration probe response missing usage.prompt_tokens: {raw[:400]}")
        return int(pt)
    finally:
        conn.close()


def calibrate_haystack(
    paragraphs: list[str] | None,
    needles: list[Needle],
    target_tokens: int,
    story_on: bool,
    seed: int,
    probe_fn: Callable[[str], int],
    max_iters: int = MAX_CALIBRATION_ITERS,
) -> tuple[str, str, PlacementStats, int, int, bool]:
    """Iteratively rebuild haystack until prompt_tokens ≈ target_tokens (<= max_iters)."""
    # Wrapper overhead (instruction + codeword list)
    wrapper_tokens = probe_fn(build_prompt(needles, ""))
    needles_chars = sum(len(n.sentence) + 1 for n in needles)
    bg_target_tokens = max(64, target_tokens - wrapper_tokens)

    # Initial estimate ~3.8 chars/token minus embedded needle sentences
    chars_per_token = 3.8
    bg_chars = max(256, int(bg_target_tokens * chars_per_token) - needles_chars)

    tol = max(256, int(target_tokens * 0.03))
    best_prompt = ""
    best_haystack = ""
    best_placement = PlacementStats(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, False, "uncalibrated")
    measured = 0
    converged = False
    iters_used = 0

    for it in range(1, max_iters + 1):
        iters_used = it
        bg = build_background_text(paragraphs, bg_chars, story_on, seed)
        haystack, placement = embed_needles_stratified_jitter(bg, needles, len(bg), seed)
        prompt = build_prompt(needles, haystack)
        measured = probe_fn(prompt)
        best_prompt = prompt
        best_haystack = haystack
        best_placement = placement

        err = measured - target_tokens
        if abs(err) <= tol:
            converged = True
            break

        # Update chars_per_token using measured haystack tokens
        haystack_tokens = max(1, measured - wrapper_tokens)
        chars_per_token = len(haystack) / float(haystack_tokens)
        desired_total_haystack_chars = int(bg_target_tokens * chars_per_token)
        bg_chars = max(256, desired_total_haystack_chars - needles_chars)

    return best_prompt, best_haystack, best_placement, measured, iters_used, converged


def qc_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    idle_timeout_s: float = 60.0,
    reasoning_effort: str | None = None,
) -> dict:
    """Stream /v1/chat/completions with idle-stall detection and collect text + usage."""
    host, port, use_tls = parse_base_url(base_url)
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    body = json.dumps(payload).encode("utf-8")
    conn = (
        http.client.HTTPSConnection(host, port, timeout=idle_timeout_s)
        if use_tls
        else http.client.HTTPConnection(host, port, timeout=idle_timeout_s)
    )

    t0 = time.monotonic()
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = "unknown"
    usage: dict = {}
    status: int | None = None
    error: str | None = None

    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")[:500]
            return {
                "status": status,
                "error": f"HTTP {status}: {err_body}",
                "answer": "",
                "reasoning": "",
                "finish_reason": "error",
                "usage": {},
                "elapsed_s": time.monotonic() - t0,
            }

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                ch = choices[0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    answer_parts.append(delta["content"])
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    reasoning_parts.append(rc)
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
    except TimeoutError as exc:
        error = f"idle_stall_timeout({idle_timeout_s}s): {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()

    return {
        "status": status,
        "error": error,
        "answer": "".join(answer_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_s": time.monotonic() - t0,
    }


def extract_usage_counts(usage: dict, reasoning_text: str, answer_text: str, prompt_tokens_fallback: int) -> tuple[int, int, int, int]:
    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens_fallback)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    pt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(pt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    ct_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(ct_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    if reasoning_tokens == 0 and reasoning_text:
        # Fallback estimate if server doesn't populate completion_tokens_details.reasoning_tokens
        reasoning_tokens = max(1, len(reasoning_text) // 4)
    if completion_tokens == 0 and (answer_text or reasoning_text):
        completion_tokens = max(1, (len(answer_text) + len(reasoning_text)) // 4)
    return prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens


def format_pass_rate_table(results: list[RunResult]) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pct = (100.0 * passed / total) if total else 0.0
    lines = [
        "=== NIAH Gate Pass-Rate Table ===",
        f"Runs: {passed}/{total} passed ({pct:.1f}%)",
        "run | pass | unique | dup | wrong | in | out | reas | cached | finish | notes",
        "----|------|--------|-----|-------|----|-----|------|--------|--------|------",
    ]
    for r in results:
        notes = ",".join(r.void_reasons + r.failures) if (r.void_reasons or r.failures) else "ok"
        lines.append(
            f"{r.run_index:3d} | {'PASS' if r.passed else 'FAIL':4s} | "
            f"{r.grade.unique_hits}/{r.grade.total_needles} | {r.grade.duplicates:3d} | "
            f"{r.grade.wrong_numbers:5d} | {r.prompt_tokens} | {r.completion_tokens} | "
            f"{r.reasoning_tokens} | {r.cached_tokens} | {r.finish_reason} | {notes}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack Gate harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--needles", type=int, default=None)
    parser.add_argument("--depth", type=float, default=None, help="Fraction of max_context in (0, 1]")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--story", choices=["on", "off"], default="on")
    parser.add_argument("--prefix-collisions", action="store_true")
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--no-require-thinking", action="store_true")
    parser.add_argument("--require-cold", action="store_true")
    parser.add_argument("--allow-small-needles", action="store_true")
    parser.add_argument("--dump-dir", type=Path, default=None, help="Directory to dump raw reasoning/answer/meta per run")
    args = parser.parse_args(argv)

    needles_count = args.needles if args.needles is not None else default_needle_count(args.max_context)
    validate_run_args(args.max_context, needles_count, args.depth, args.max_turns, args.allow_small_needles)

    story_on = args.story == "on"
    paragraphs = load_corpus_paragraphs(args.corpus_dir) if story_on else None
    needles = generate_needles(needles_count, args.seed, prefix_collisions=args.prefix_collisions)
    target_tokens = target_needle_region_tokens(args.max_context, args.depth)

    print(
        f"[setup] model={args.model} ctx={args.max_context} needles={needles_count} "
        f"target_region_tokens={target_tokens} seed={args.seed} story={args.story} "
        f"prefix_collisions={args.prefix_collisions}",
        flush=True,
    )

    prompt, _, placement, calibrated_pt, cal_iters, converged = calibrate_haystack(
        paragraphs=paragraphs,
        needles=needles,
        target_tokens=target_tokens,
        story_on=story_on,
        seed=args.seed,
        probe_fn=lambda p: probe_prompt_tokens(args.base_url, args.model, p, args.seed, args.reasoning_effort),
    )

    print(
        f"[calibration] iters={cal_iters}/{MAX_CALIBRATION_ITERS} converged={converged} "
        f"measured_prompt_tokens={calibrated_pt} target={target_tokens}",
        flush=True,
    )
    print(
        f"[distribution] n={placement.count} haystack_chars={placement.haystack_chars} "
        f"region_chars={placement.needle_region_chars} first={placement.first_char} "
        f"last={placement.last_char} gap_mean={placement.gap_mean:.1f} "
        f"gap_max={placement.gap_max:.1f} expected_gap={placement.expected_gap:.1f} "
        f"valid={placement.valid} ({placement.reason})",
        flush=True,
    )

    output_budget = compute_output_budget(args.max_context, calibrated_pt, needles_count)
    print(f"[budget] output_budget={output_budget} ({output_budget / needles_count:.1f} tok/line)", flush=True)

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for run_idx in range(1, args.runs + 1):
        stream_res = qc_stream(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=output_budget,
            seed=args.seed,
            idle_timeout_s=args.idle_timeout,
            reasoning_effort=args.reasoning_effort,
        )
        grade = grade_response(stream_res["answer"], needles)
        in_tok, out_tok, reas_tok, cached_tok = extract_usage_counts(
            stream_res["usage"], stream_res["reasoning"], stream_res["answer"], calibrated_pt
        )
        passed, void_reasons, failures = classify_verdict(
            grade=grade,
            finish_reason=stream_res["finish_reason"],
            completion_tokens=out_tok,
            output_budget=output_budget,
            reasoning_tokens=reas_tok,
            cached_tokens=cached_tok,
            calibrated=converged,
            placement=placement,
            require_thinking=not args.no_require_thinking,
            require_cold=args.require_cold,
            http_status=stream_res["status"],
            error=stream_res["error"],
        )
        rr = RunResult(
            run_index=run_idx,
            passed=passed,
            void_reasons=void_reasons,
            failures=failures,
            grade=grade,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            reasoning_tokens=reas_tok,
            cached_tokens=cached_tok,
            finish_reason=stream_res["finish_reason"],
            elapsed_s=stream_res["elapsed_s"],
            output_budget=output_budget,
            placement=placement,
            calibration_iters=cal_iters,
            calibrated=converged,
            http_status=stream_res["status"],
            error=stream_res["error"],
        )
        results.append(rr)
        if args.dump_dir is not None:
            (args.dump_dir / f"run_{run_idx}_reasoning.txt").write_text(stream_res["reasoning"], encoding="utf-8")
            (args.dump_dir / f"run_{run_idx}_answer.txt").write_text(stream_res["answer"], encoding="utf-8")
            meta = {
                "run_index": run_idx,
                "passed": passed,
                "void_reasons": void_reasons,
                "failures": failures,
                "summary": rr.summary_line(),
                "grade": {
                    "unique_hits": grade.unique_hits,
                    "total_needles": grade.total_needles,
                    "duplicates": grade.duplicates,
                    "wrong_numbers": grade.wrong_numbers,
                    "missing_count": len(grade.missing),
                    "wrong_pairs": grade.wrong_pairs,
                },
                "usage": stream_res["usage"],
                "finish_reason": stream_res["finish_reason"],
                "elapsed_s": stream_res["elapsed_s"],
                "http_status": stream_res["status"],
                "error": stream_res["error"],
            }
            (args.dump_dir / f"run_{run_idx}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(rr.summary_line(), flush=True)
        if void_reasons or failures:
            print(f"[verdict run={run_idx}] FAIL void={void_reasons} failures={failures}", flush=True)
        else:
            print(f"[verdict run={run_idx}] PASS", flush=True)

    print(format_pass_rate_table(results), flush=True)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
