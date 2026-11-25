#!/usr/bin/env python3
"""Unit tests for every requirement in the Needle-in-a-Haystack Gate specification."""

from __future__ import annotations

import http.server
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "niah_gate"))

import niah_gate  # noqa: E402


def _make_sample_corpus(root: Path) -> Path:
    long_para_a = (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        "must be in want of a wife. However little known the feelings or views of such a man may be "
        "on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families."
    )
    long_para_b = (
        "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. "
        "In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion "
        "akin to love for Irene Adler, all emotions were abhorrent to his cold precise mind."
    )
    short_para = "Chapter I. Short line."
    for idx, fname in enumerate(niah_gate.CORPUS_FILES):
        content = (
            "The Project Gutenberg eBook header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
            f"{short_para}\n\n"
            f"{long_para_a} Book {idx}.\n\n"
            f"{long_para_b} Section {idx}.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Project Gutenberg footer license text.\n"
        )
        (root / fname).write_text(content, encoding="utf-8")
    return root


class TestCorpusAndBackground(unittest.TestCase):
    def test_strip_gutenberg_and_min_paragraph_length(self):
        with tempfile.TemporaryDirectory() as td:
            corpus_dir = _make_sample_corpus(Path(td))
            paras = niah_gate.load_corpus_paragraphs(corpus_dir)
            self.assertEqual(len(paras), 8)
            for p in paras:
                self.assertGreaterEqual(len(p), niah_gate.MIN_PARAGRAPH_CHARS)
                self.assertNotIn("START OF THE PROJECT GUTENBERG", p)
                self.assertNotIn("END OF THE PROJECT GUTENBERG", p)
                self.assertNotIn("Short line", p)

    def test_story_on_vs_legacy_filler(self):
        with tempfile.TemporaryDirectory() as td:
            paras = niah_gate.load_corpus_paragraphs(_make_sample_corpus(Path(td)))
            bg_story = niah_gate.build_background_text(paras, 2500, story_on=True, seed=1234)
            bg_filler = niah_gate.build_background_text(None, 2500, story_on=False, seed=1234)
            self.assertIn("Sherlock Holmes", bg_story)
            self.assertIn("archival ledger", bg_filler)
            self.assertGreaterEqual(len(bg_story), 1500)
            self.assertGreaterEqual(len(bg_filler), 1500)


class TestNeedleGeneration(unittest.TestCase):
    def test_unique_base_tag_and_unique_5digit_numbers_and_seed_determinism(self):
        n1 = niah_gate.generate_needles(2048, seed=1234, prefix_collisions=False)
        n2 = niah_gate.generate_needles(2048, seed=1234, prefix_collisions=False)
        self.assertEqual(n1, n2)
        self.assertEqual(len(n1), 2048)

        base_tags = set()
        numbers = set()
        for item in n1:
            parts = item.codeword.split("-")
            self.assertEqual(len(parts), 3)
            base_tag = f"{parts[0]}-{parts[1]}"
            self.assertNotIn(base_tag, base_tags)
            base_tags.add(base_tag)

            self.assertEqual(len(item.number), 5)
            self.assertTrue(item.number.isdigit())
            self.assertGreaterEqual(int(item.number), 10000)
            self.assertNotIn(item.number, numbers)
            numbers.add(item.number)

            self.assertEqual(
                item.sentence,
                f"The secret codeword for {item.codeword} is the number {item.number}.",
            )

    def test_prefix_collisions_legacy_mode_produces_sibling_families(self):
        needles = niah_gate.generate_needles(256, seed=1234, prefix_collisions=True)
        bases = [n.codeword.split("-")[0] for n in needles]
        self.assertLess(len(set(bases)), len(bases))
        self.assertEqual(len({n.codeword for n in needles}), 256)
        self.assertEqual(len({n.number for n in needles}), 256)


class TestPlacementAndSizing(unittest.TestCase):
    def test_stratified_jitter_uniform_distribution_and_skew_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            paras = niah_gate.load_corpus_paragraphs(_make_sample_corpus(Path(td)))
            bg = niah_gate.build_background_text(paras, 200_000, story_on=True, seed=1234)
            needles = niah_gate.generate_needles(256, seed=1234)
            haystack1, stats1 = niah_gate.embed_needles_stratified_jitter(bg, needles, len(bg), seed=1234)
            haystack2, stats2 = niah_gate.embed_needles_stratified_jitter(bg, needles, len(bg), seed=1234)
            self.assertEqual(haystack1, haystack2)
            self.assertEqual(stats1, stats2)
            self.assertTrue(stats1.valid, stats1.reason)
            self.assertLess(stats1.gap_max, 2.5 * stats1.expected_gap)

            # Skewed placement must be marked invalid
            skewed = niah_gate.evaluate_placement([10, 20, 30, 40, 50], region_chars=100_000, haystack_chars=100_000)
            self.assertFalse(skewed.valid)

    def test_prompt_format_rule_5(self):
        needles = niah_gate.generate_needles(4, seed=1234)
        prompt = niah_gate.build_prompt(needles, "BODY_TEXT")
        self.assertTrue(prompt.startswith("Below is a long story archive. Follow ONLY the final instruction."))
        self.assertIn("state its secret number as '<word>: <number>' one per line, nothing else.", prompt)
        self.assertIn("Codewords: " + ", ".join(n.codeword for n in needles), prompt)
        self.assertTrue(prompt.endswith("ARCHIVE: BODY_TEXT"))

    def test_sizing_defaults_and_invalid_run_shapes(self):
        self.assertEqual(niah_gate.default_needle_count(65536), 1024)
        self.assertEqual(niah_gate.default_needle_count(262144), 2048)

        self.assertEqual(niah_gate.target_needle_region_tokens(65536, None), 43690)
        self.assertEqual(niah_gate.target_needle_region_tokens(1048576, None), 512 * 1024)

        # depth > 1.0 (e.g. token count 8192 passed as depth) must abort loudly
        with self.assertRaises(ValueError):
            niah_gate.validate_run_args(262144, 2048, depth=8192.0, max_turns=1, allow_small_needles=False)

        # --needles 1 is degenerate and must abort loudly
        with self.assertRaises(ValueError):
            niah_gate.validate_run_args(262144, 1, depth=None, max_turns=1, allow_small_needles=False)

        # --max-turns != 1 must abort loudly
        with self.assertRaises(ValueError):
            niah_gate.validate_run_args(262144, 256, depth=None, max_turns=2, allow_small_needles=False)

    def test_output_budget_reservation_and_abort(self):
        # 2048 * 40 = 81920 < 262144 - 174762 - 512 = 86870
        self.assertEqual(niah_gate.compute_output_budget(262144, 174762, 2048), 81920)
        # Cannot fit 2048 lines when available room < 2048 * 12
        with self.assertRaises(RuntimeError):
            niah_gate.compute_output_budget(262144, 250000, 2048)


class TestCalibrationAndGrading(unittest.TestCase):
    def test_calibration_converges_within_5_iters(self):
        with tempfile.TemporaryDirectory() as td:
            paras = niah_gate.load_corpus_paragraphs(_make_sample_corpus(Path(td)))
            needles = niah_gate.generate_needles(256, seed=1234)
            # Simulate a 3.2 chars/token tokenizer
            def fake_probe(text: str) -> int:
                return max(1, int(len(text) / 3.2))

            _, _, placement, measured, iters, converged = niah_gate.calibrate_haystack(
                paragraphs=paras,
                needles=needles,
                target_tokens=43690,
                story_on=True,
                seed=1234,
                probe_fn=fake_probe,
            )
            self.assertTrue(converged)
            self.assertLessEqual(iters, 5)
            self.assertLessEqual(abs(measured - 43690), int(43690 * 0.03))
            self.assertTrue(placement.valid)

    def test_grading_exact_duplicates_wrong_numbers_and_verdicts(self):
        needles = niah_gate.generate_needles(3, seed=1234)
        answer = (
            f"{needles[0].codeword}: {needles[0].number}\n"
            f"{needles[0].codeword}: {needles[0].number}\n"  # duplicate should not credit twice
            f"{needles[1].codeword}: 99999\n"               # wrong number
            f"{needles[2].codeword}: {needles[2].number}\n"
        )
        if needles[1].number == "99999":
            answer = answer.replace("99999", "10000")

        grade = niah_gate.grade_response(answer, needles)
        self.assertEqual(grade.unique_hits, 2)
        self.assertEqual(grade.duplicates, 1)
        self.assertEqual(grade.wrong_numbers, 1)
        self.assertEqual(grade.missing, [needles[1].codeword])

        placement_ok = niah_gate.PlacementStats(3, 1000, 900, 10, 890, 440.0, 440.0, 300.0, True, "ok")

        # Premature EOS
        passed, _, failures = niah_gate.classify_verdict(
            grade=niah_gate.GradeResult(2, 3, 0, 0),
            finish_reason="stop",
            completion_tokens=100,
            output_budget=1000,
            reasoning_tokens=50,
            cached_tokens=0,
            calibrated=True,
            placement=placement_ok,
            require_thinking=True,
            require_cold=True,
            http_status=200,
            error=None,
        )
        self.assertFalse(passed)
        self.assertIn("premature_eos_mid_enumeration", failures)

        # Length clipping
        passed, _, failures = niah_gate.classify_verdict(
            grade=niah_gate.GradeResult(2, 3, 0, 0),
            finish_reason="length",
            completion_tokens=1000,
            output_budget=1000,
            reasoning_tokens=50,
            cached_tokens=0,
            calibrated=True,
            placement=placement_ok,
            require_thinking=True,
            require_cold=False,
            http_status=200,
            error=None,
        )
        self.assertFalse(passed)
        self.assertTrue(any("budget_length_clipped" in f for f in failures))

        # Thinking off (reas=0) fails for Qwen3.8
        passed, _, failures = niah_gate.classify_verdict(
            grade=niah_gate.GradeResult(3, 3, 0, 0),
            finish_reason="stop",
            completion_tokens=60,
            output_budget=1000,
            reasoning_tokens=0,
            cached_tokens=0,
            calibrated=True,
            placement=placement_ok,
            require_thinking=True,
            require_cold=False,
            http_status=200,
            error=None,
        )
        self.assertFalse(passed)
        self.assertIn("thinking_off(reas=0)", failures)

        # HTTP 500 non-finite logits
        passed, _, failures = niah_gate.classify_verdict(
            grade=niah_gate.GradeResult(0, 3, 0, 0),
            finish_reason="error",
            completion_tokens=0,
            output_budget=1000,
            reasoning_tokens=0,
            cached_tokens=0,
            calibrated=True,
            placement=placement_ok,
            require_thinking=True,
            require_cold=False,
            http_status=500,
            error="HTTP 500: non-finite logits",
        )
        self.assertFalse(passed)
        self.assertIn("non_finite_logits_http_500", failures)

        # True PASS
        passed, voids, failures = niah_gate.classify_verdict(
            grade=niah_gate.GradeResult(3, 3, 0, 0),
            finish_reason="stop",
            completion_tokens=120,
            output_budget=1000,
            reasoning_tokens=40,
            cached_tokens=0,
            calibrated=True,
            placement=placement_ok,
            require_thinking=True,
            require_cold=True,
            http_status=200,
            error=None,
        )
        self.assertTrue(passed)
        self.assertEqual(voids, [])
        self.assertEqual(failures, [])

    def test_qc_stream_mock_sse_pass_and_idle_stall(self):
        needles = niah_gate.generate_needles(2, seed=1234)
        answer_lines = "".join(f"{n.codeword}: {n.number}\n" for n in needles)

        class MockHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if body.get("max_tokens") == 1 and not body.get("stream"):
                    resp = json.dumps({"usage": {"prompt_tokens": 100}}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"choices": [{"delta": {"reasoning_content": "checking archive..."}}]},
                    {"choices": [{"delta": {"content": answer_lines}, "finish_reason": "stop"}],
                     "usage": {
                         "prompt_tokens": 100,
                         "completion_tokens": 40,
                         "prompt_tokens_details": {"cached_tokens": 0},
                         "completion_tokens_details": {"reasoning_tokens": 12},
                     }},
                ]
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            res = niah_gate.qc_stream(url, "mock-model", "prompt", max_tokens=80, seed=1234, idle_timeout_s=5.0)
            self.assertEqual(res["status"], 200)
            self.assertIsNone(res["error"])
            self.assertEqual(res["finish_reason"], "stop")
            grade = niah_gate.grade_response(res["answer"], needles)
            self.assertEqual(grade.unique_hits, 2)
            in_tok, out_tok, reas_tok, cached_tok = niah_gate.extract_usage_counts(
                res["usage"], res["reasoning"], res["answer"], 100
            )
            self.assertEqual((in_tok, out_tok, reas_tok, cached_tok), (100, 40, 12, 0))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
