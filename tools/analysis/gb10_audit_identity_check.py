#!/usr/bin/env python3
"""Recompute every ratio the GB10 ggml audit reports, from raw measurements.

Written because §13.2 of exp_logs/2026-10-19-gb10-ggml-constant-audit.md mixed
MAC/s with ops/s when forming the machine-balance ratio, which halved the reported
arithmetic-intensity margin, and because several ratios in §13 were derived from
weight bytes over elapsed time and then described as "bandwidth".

Conventions enforced here (each was a real error in the first draft):

  * ops/s != MAC/s. One MAC is one multiply and one add, so TOPS counts a MAC as
    two operations. A machine balance quoted in TOPS/GBps is therefore twice the
    same balance quoted in MAC/byte.
  * GB = 1e9 bytes, GiB = 2**30 bytes. The audit mixes both ("24 MiB L2" is
    correct; "MB" for weight traffic is decimal).
  * weight_bytes/elapsed is a *lower bound* on traffic, not total bandwidth. Only
    L2/DRAM counters establish bandwidth; the audit's 1.4-2.1 GB/s figures are
    weight bytes over time and must never be presented as DRAM utilization.

Run:  python3 tools/analysis/gb10_audit_identity_check.py
Exit: 0 if every identity holds, 1 otherwise.
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Raw measurements quoted in the audit. Each is either a synthetic peak measured
# on this machine or a bench result recorded in the log; nothing here is derived.

INT8_TC_TOPS = 253.2        # /tmp/int8_tc_peak: 16*8*32*2 FLOP per MMA => ops/s
DRAM_READ_GBPS = 234.5      # /tmp/dram_peak, streamed read
DRAM_COPY_GBPS = 218.1      # /tmp/dram_peak, read+write combined

# tile_kernel Q6_K, NCU, synthetic bench (infernix_ggml_linear_test --bench 4096)
TILE_DUR_S = 1.19e-3
TILE_L2_BYTES = 1.28e9      # lts__t_bytes.sum
TILE_L2_HIT = 0.9345        # L2 Hit Rate
TILE_WAVES = 10.67
TILE_OCC_ACHIEVED = 0.3261  # Achieved Occupancy
TILE_OCC_THEORETICAL = 0.3333
TILE_SM_THROUGHPUT = 0.516  # Compute (SM) Throughput
TILE_MEM_THROUGHPUT = 0.566 # Memory Throughput (L2, see note in check_balance)
TILE_REGS = 128
TILE_SMEM_KB = 42.24
SM_COUNT = 48
REGS_PER_SM = 65536
SMEM_PER_SM_KB = 102.4
MAX_WARPS_PER_SM = 48
WARPS_PER_CTA = 8           # 256 threads

# bench geometry for the dominant projection ("gdn q/k/v/z", T=4096)
PROJ_ROWS = 2048 + 2048 + 6144 + 6144
PROJ_K = 2560
PROJ_T = 4096
PROJ_MED_US = 14537.79      # infernix_ggml_linear_test --bench 4096 rows
PROJ_Q6K_ROWS = 2048 + 2048 + 6144
PROJ_IQ4XS_ROWS = 6144
Q6K_BLOCK_BYTES, IQ4XS_BLOCK_BYTES, BLOCK_ELEMS = 210, 136, 256

# correctness of the reported pp ratios
PP4096_IQ3S, PP4096_DENSE8 = 935, 2409
PAYLOAD_IQ3S_GB, PAYLOAD_DENSE8_GB = 56.75, 76.87

failures: list[str] = []


def check(label: str, got: float, want: float, tol: float = 1e-6) -> None:
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<52} got {got:,.3f}  want {want:,.3f}")
    if not ok:
        failures.append(label)


def check_balance() -> None:
    """The identity that was wrong: MAC/byte against a TOPS-derived balance."""
    print("machine balance")
    ops_per_byte = INT8_TC_TOPS * 1e12 / (DRAM_READ_GBPS * 1e9)
    mac_per_byte = ops_per_byte / 2.0
    check("ops/byte (TOPS / GBps)", ops_per_byte, 1079.7, 1e-3)
    check("MAC/byte (half of ops/byte)", mac_per_byte, 539.9, 1e-3)

    weight_bytes = (
        PROJ_Q6K_ROWS * PROJ_K * Q6K_BLOCK_BYTES / BLOCK_ELEMS
        + PROJ_IQ4XS_ROWS * PROJ_K * IQ4XS_BLOCK_BYTES / BLOCK_ELEMS
    )
    macs = PROJ_ROWS * PROJ_K * PROJ_T
    ai_mac_per_byte = macs / weight_bytes
    check("projection MAC/byte", ai_mac_per_byte, 5754.0, 1e-3)
    # The audit claimed 5.3x by dividing MAC/byte by an ops/byte balance.
    check("margin vs MAC balance", ai_mac_per_byte / mac_per_byte, 10.657, 1e-3)
    # That, not 5.3x, is the correct multiple. The first draft computed
    # ai_mac_per_byte / ops_per_byte = 5.33 and labelled it "MAC/byte margin".


def check_traffic() -> None:
    """weight_bytes/elapsed is not bandwidth; L2 misses bound DRAM."""
    print("tile_kernel traffic (NCU)")
    weight_bytes = PROJ_ROWS * PROJ_K * Q6K_BLOCK_BYTES / BLOCK_ELEMS  # upper bound, all Q6_K
    weight_gbps = weight_bytes / TILE_DUR_S / 1e9
    print(f"        weight-bytes/elapsed = {weight_gbps:.2f} GB/s (a lower bound on traffic, "
          f"{weight_bytes / TILE_L2_BYTES * 100:.1f}% of the L2 traffic actually requested)")

    dram_bytes = TILE_L2_BYTES * (1.0 - TILE_L2_HIT)
    dram_gbps = dram_bytes / TILE_DUR_S / 1e9
    # This is the number that may be compared with DRAM_READ_GBPS; the audit's
    # 1.4-2.1 GB/s figure is ~40x smaller and was never a DRAM measurement.
    check("DRAM GB/s from L2 misses", dram_gbps, 70.5, 1e-2)
    check("DRAM % of ceiling", dram_gbps / DRAM_READ_GBPS * 100.0, 30.1, 1e-2)
    l2_gbps = TILE_L2_BYTES / TILE_DUR_S / 1e9
    print(f"        L2 traffic {TILE_L2_BYTES / 1e9:.2f} GB over {TILE_DUR_S * 1e3:.2f} ms "
          f"= {l2_gbps:,.0f} GB/s")


DRIVER_SMEM_KB = 1.02       # NCU: "Driver Shared Memory Per Block"


def check_occupancy() -> None:
    """Both ceilings sit at 2 CTAs/SM, so relieving one alone cannot help.

    Capacity is a floor of whole CTAs, not a quotient: 102.4 / 42.24 = 2.42 allows
    two CTAs, not 2.42 of them. The audit wrote the quotient where the capacity
    belonged ("shared memory allows 2"), which happened to give the right answer
    here but is not an identity that survives a different block size.
    """
    print("occupancy ceilings")
    ctas_reg = int(REGS_PER_SM / (TILE_REGS * 32 * WARPS_PER_CTA))
    smem_used_kb = TILE_SMEM_KB + DRIVER_SMEM_KB
    ctas_smem = int(SMEM_PER_SM_KB / smem_used_kb)
    check("CTAs/SM allowed by registers", ctas_reg, 2.0)
    check("smem per block incl. driver overhead (KB)", smem_used_kb, 43.26, 1e-3)
    check("CTAs/SM allowed by shared memory", ctas_smem, 2.0)
    # The binding capacity is the smaller of the two; a third CTA needs both to rise.
    check("binding CTAs/SM", min(ctas_reg, ctas_smem), 2.0)
    check("threads/SM in use", ctas_reg * 32 * WARPS_PER_CTA, 512.0)
    check("warps/SM in use", ctas_reg * WARPS_PER_CTA, 16.0)
    check("theoretical occupancy", ctas_reg * WARPS_PER_CTA / MAX_WARPS_PER_SM, TILE_OCC_THEORETICAL, 1e-2)
    check("achieved / theoretical", TILE_OCC_ACHIEVED / TILE_OCC_THEORETICAL, 0.979, 1e-2)


def check_pp_ratios() -> None:
    print("reported pp ratios")
    check("Dense8/IQ3_S at pp4096", PP4096_DENSE8 / PP4096_IQ3S, 2.576, 1e-3)
    check("payload ratio", PAYLOAD_DENSE8_GB / PAYLOAD_IQ3S_GB, 1.355, 1e-3)
    # The audit's §1 says "2.44x slower" for this pair; 2.409x is the §11.2
    # 994-vs-2395 figure, i.e. a different run quoted in the wrong place.


def check_rates() -> None:
    print("achieved rates")
    macs = PROJ_ROWS * PROJ_K * PROJ_T
    secs = PROJ_MED_US * 1e-6
    tops = 2.0 * macs / secs / 1e12
    check("projection TOPS", tops, 23.6, 1e-2)
    check("projection % of INT8 TC peak", tops / INT8_TC_TOPS * 100.0, 9.3, 1e-2)


def main() -> int:
    print(f"peaks: {INT8_TC_TOPS} TOPS, {DRAM_READ_GBPS} GB/s read, {DRAM_COPY_GBPS} GB/s copy, "
          f"{SM_COUNT} SMs\n")
    check_balance()
    check_traffic()
    check_occupancy()
    check_pp_ratios()
    check_rates()
    print()
    if failures:
        print(f"FAILED: {len(failures)} identit{'y' if len(failures) == 1 else 'ies'}: {failures}")
        return 1
    print("all identities hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
