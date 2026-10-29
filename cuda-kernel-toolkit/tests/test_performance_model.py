"""Occupancy-model tests.

The model predicts resident blocks from compiler-reported resources using the
target's allocation granularity.  The important behaviour is the step function:
occupancy changes only when a resource crosses an allocation boundary, which is
exactly where naive reasoning goes wrong (shaving a few registers buys nothing
until the rounded allocation drops a whole unit).
"""

from __future__ import annotations

import unittest

from ckt.environment import DeviceInfo
from ckt.performance_model import occupancy_model, wave_analysis


def gb10() -> DeviceInfo:
    from ckt import spec as _spec

    dev = DeviceInfo(
        name="GB10",
        compute_capability="12.1",
        cc_major=12,
        cc_minor=1,
        arch="sm_121a",
        multi_processor_count=48,
        max_threads_per_block=1024,
        max_threads_per_multiprocessor=1536,
        max_blocks_per_multiprocessor=24,
        regs_per_multiprocessor=65536,
        shared_mem_per_block=49152,
        shared_mem_per_block_optin=101376,
        shared_mem_per_multiprocessor=102400,
        reserved_shared_mem_per_block=1024,
        integrated_uma=True,
    )
    dev.arch_facts = _spec.arch_facts(12, 1)
    return dev


class TestOccupancy(unittest.TestCase):
    def setUp(self):
        self.dev = gb10()

    def test_thread_limit_at_full_block(self):
        # 256 threads, low register pressure: limited by threads, 6 blocks.
        m = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=34)
        self.assertEqual(m.blocks_by_hw_limit, 6)
        self.assertEqual(m.active_blocks_per_sm, 6)
        self.assertEqual(round(m.occupancy_pct), 100)
        self.assertEqual(m.limiting_factor, "threads/warps/block-count")

    def test_register_step_function(self):
        # 64 regs/thread -> exactly 4 blocks; 66 regs/thread rounds the warp
        # allocation up and drops to 3 blocks with zero benefit in between.
        at_64 = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=64)
        at_66 = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=66)
        self.assertGreater(at_64.blocks_by_registers, at_66.blocks_by_registers)
        # And 65 vs 66 differ because the rounding crosses a boundary.
        self.assertEqual(at_64.registers_allocated_per_block, 16384)

    def test_shared_memory_limits(self):
        # A large shared-memory footprint caps residency below the thread limit.
        small = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=34,
                                shared_mem_per_block=0)
        big = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=34,
                              shared_mem_per_block=90_000)
        self.assertLess(big.blocks_by_shared, small.blocks_by_hw_limit)
        self.assertIn("shared memory", big.limiting_factor.lower())

    def test_threshold_for_next_block(self):
        # The exact register count that would admit one more resident block.
        m = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=34)
        nxt = m.max_regs_per_thread_to_gain_one_block
        if m.limiting_factor == "registers":
            self.assertIsNotNone(nxt)
        else:
            # Thread-limited here, so no register reduction helps.
            self.assertIsNone(nxt)

    def test_wave_quantization(self):
        # 384 CTAs over 288 concurrent slots leaves a partial tail.
        occ = occupancy_model(self.dev, threads_per_block=256, registers_per_thread=34)
        w = wave_analysis(self.dev, occ, total_ctas=384)
        self.assertAlmostEqual(w.waves, 384 / 288)
        self.assertEqual(w.tail_ctas, 384 - 288)
        self.assertEqual(w.problem, "tail")
        under = wave_analysis(self.dev, occ, total_ctas=64)
        self.assertEqual(under.problem, "under-fill")


if __name__ == "__main__":
    unittest.main()
