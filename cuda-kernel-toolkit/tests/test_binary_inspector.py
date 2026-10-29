"""SASS parser tests.

Width classification is subtle on this toolchain: ``LDG.E`` with no explicit
width suffix means 32 bits, while ``LDG.E.128`` means 128.  Getting this wrong
makes every coalescing conclusion wrong, so the cases here mirror instructions
actually observed in the sample-suite binaries.
"""

from __future__ import annotations

import unittest

from ckt.binary_inspector import parse_sass


SAMPLE = """\
        /*0000*/                   IMAD.MOV R1, RZ, RZ, R0 ;
        /*0010*/                   LDG.E.CONSTANT R2, [R1] ;
        /*0020*/                   FADD.F32 R2, R2, R3 ;
        /*0030*/                   STG.E R2, [R1] ;
        /*0040*/                   LDG.E.128 R4, R5, R6, R7, [R8] ;
        /*0050*/                   STG.E.128 R9, R10, R11, R12, [R8] ;
        /*0060*/                   LDS R13, [R14] ;
        /*0070*/                   STS [R14], R13 ;
        /*0080*/                   STL R15, [R16] ;
        /*0090*/                   LDL R15, [R16] ;
        /*00a0*/                   HMMA.16816.F32.F16 R16, R16, R17, R18 ;
        /*00b0*/                   BAR.SYNC.DEFER_BLOCKING 0 ;
        /*00c0*/                   BRA !PT, 0x0 ;
        /*00d0*/                   NOP ;
        /*00e0*/                   EXIT ;
"""


class TestSassParsing(unittest.TestCase):
    def setUp(self):
        self.instrs = parse_sass(SAMPLE)

    def test_width_classification(self):
        ldg_plain = next(i for i in self.instrs if i.opcode.startswith("LDG") and ".128" not in i.opcode and "CONSTANT" in i.raw)
        self.assertEqual(ldg_plain.width_bytes, 4)
        ldg_vec = next(i for i in self.instrs if ".128" in i.opcode and i.opcode.startswith("LDG"))
        self.assertEqual(ldg_vec.width_bytes, 16)
        stg_plain = next(i for i in self.instrs if i.opcode == "STG.E")
        self.assertEqual(stg_plain.width_bytes, 4)
        non_mem = next(i for i in self.instrs if i.opcode.startswith("FADD"))
        self.assertIsNone(non_mem.width_bytes)

    def test_local_memory_detection(self):
        locals_ = [i for i in self.instrs if i.is_local]
        self.assertTrue(any(i.opcode.startswith("STL") for i in locals_))
        self.assertTrue(any(i.opcode.startswith("LDL") for i in locals_))

    def test_shared_and_tensor_detection(self):
        shared = [i for i in self.instrs if i.is_shared_load or i.is_shared_store]
        self.assertGreaterEqual(len(shared), 2)
        tensor = [i for i in self.instrs if i.is_tensor_core]
        self.assertEqual(len(tensor), 1)

    def test_profile_counts(self):
        prof = _profile(self.instrs)
        self.assertGreater(prof.global_loads, 0)
        self.assertGreater(prof.global_stores, 0)
        self.assertGreater(prof.local_loads, 0)
        self.assertGreater(prof.local_stores, 0)
        self.assertEqual(prof.tensor_core_ops, 1)
        # Widest load should be the 128-bit one.
        self.assertEqual(prof.max_load_width, 16)


def _profile(instrs):
    from ckt.binary_inspector import Disassembly

    d = Disassembly(symbol="k_test", arch="sm_121", instructions=instrs, raw=SAMPLE)
    return d.profile


if __name__ == "__main__":
    unittest.main()
