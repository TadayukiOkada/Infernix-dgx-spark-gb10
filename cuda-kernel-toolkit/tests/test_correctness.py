"""Correctness-gate tests.

These protect the numerical-comparison semantics that decide whether any
performance conclusion may be drawn.  The cases chosen here are the ones that
silently give wrong answers if handled naively: NaN propagation, signed zero
under exact comparison, length mismatch, and tolerance boundaries.
"""

from __future__ import annotations

import unittest

from ckt.correctness import compare_arrays, compare_scalars


class TestCompareScalars(unittest.TestCase):
    def test_exact_int(self):
        self.assertTrue(compare_scalars(3, 3, dtype="int32").passed)
        self.assertFalse(compare_scalars(3, 4, dtype="int32").passed)

    def test_float_within_tolerance(self):
        r = compare_scalars(1.0 + 1e-6, 1.0, dtype="fp32")
        self.assertTrue(r.passed)

    def test_float_outside_tolerance(self):
        r = compare_scalars(1.5, 1.0, dtype="fp32")
        self.assertFalse(r.passed)

    def test_nan_candidate_fails(self):
        # A NaN output is never correct, even though `NaN > atol` is False in
        # Python and would slip through a naive comparison.
        r = compare_scalars(float("nan"), 1.0, dtype="fp32")
        self.assertFalse(r.passed)

    def test_nan_reference_with_equal_flag_passes(self):
        # When both are NaN they represent the same value for our purposes.
        r = compare_scalars(float("nan"), float("nan"), dtype="fp32")
        self.assertTrue(r.passed)

    def test_signed_zero_under_exact_comparison(self):
        # -0.0 == 0.0 numerically but differs bit-wise; exact mode catches it.
        self.assertFalse(compare_arrays([-0.0], [0.0], dtype="fp32", exact=True).passed)
        # And without exact mode it passes, since they compare equal.
        self.assertTrue(compare_arrays([-0.0], [0.0], dtype="fp32").passed)


class TestCompareArrays(unittest.TestCase):
    def test_identical(self):
        r = compare_arrays([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], dtype="fp32")
        self.assertTrue(r.passed)

    def test_one_element_off(self):
        r = compare_arrays([1.0, 2.0, 3.0], [1.0, 2.0, 99.0], dtype="fp32")
        self.assertFalse(r.passed)
        self.assertEqual(r.first_mismatch_index, 2)

    def test_length_mismatch_not_ok(self):
        r = compare_arrays([1.0, 2.0], [1.0, 2.0, 3.0], dtype="fp32")
        self.assertFalse(r.passed)

    def test_nan_in_output_fails(self):
        r = compare_arrays([1.0, float("nan")], [1.0, 2.0], dtype="fp32")
        self.assertFalse(r.passed)

    def test_exact_integer(self):
        r = compare_arrays([1, 2, 3], [1, 2, 3], dtype="int32", exact=True)
        self.assertTrue(r.passed)
        r = compare_arrays([1, 2, 3], [1, 2, 4], dtype="int32", exact=True)
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main()
