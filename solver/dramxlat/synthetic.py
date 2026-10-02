"""Synthetic noisy truth table — validate the whole Phase 3 pipeline without
any hardware.

We pick a *known* set of GF(2) bank-selecting functions, plus which bits are
row bits, then draw random physical-address pairs and label each:

    conflict (slow)  <=>  same bank  AND  different row
    otherwise        <=>  fast (different bank, or same row)

Latencies are drawn from two Gaussians with a configurable gap, plus a fraction
of page-fault/TLB-miss spikes that are far above the conflict mode (and so must
be trimmed, not mistaken for conflicts). A solver that recovers the original
functions from this data is doing the real job.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from . import gf2


@dataclass
class DramModel:
    """A ground-truth address map for testing."""
    functions: List[int]        # bank/rank/BG/channel selecting XOR masks
    row_bits: List[int]         # bits that index the row within a bank
    addr_bits: List[int]        # all physical bits that vary in the experiment
    name: str = "synthetic"

    @property
    def row_mask(self) -> int:
        m = 0
        for b in self.row_bits:
            m |= (1 << b)
        return m

    def same_bank(self, a: int, b: int) -> bool:
        return all(gf2.dot(f, a) == gf2.dot(f, b) for f in self.functions)


def example_model() -> DramModel:
    """A plausible DDR4-ish single-channel map (bits chosen for clarity):
    four bank-group/bank functions and one rank function, each a 2-bit XOR;
    row bits high; cache-line/column bits low and free."""
    functions = [
        (1 << 13) | (1 << 17),   # BG0
        (1 << 14) | (1 << 18),   # BG1
        (1 << 15) | (1 << 19),   # bank0
        (1 << 16) | (1 << 20),   # bank1
        (1 << 21) | (1 << 22),   # rank
    ]
    row_bits = list(range(23, 33))
    addr_bits = list(range(6, 33))
    return DramModel(functions, row_bits, addr_bits, name="ddr4-example")


@dataclass
class Dataset:
    addr_a: np.ndarray          # uint64[N]
    addr_b: np.ndarray          # uint64[N]
    latency: np.ndarray         # float[N]
    truth_conflict: np.ndarray  # bool[N] ground-truth labels
    model: DramModel = field(default=None)


def _parity_masked(vals: np.ndarray, mask: int) -> np.ndarray:
    """Vectorised GF(2) parity of (vals & mask) for uint64 arrays."""
    x = (vals & np.uint64(mask)).copy()
    x = x ^ (x >> np.uint64(32))
    x = x ^ (x >> np.uint64(16))
    x = x ^ (x >> np.uint64(8))
    x = x ^ (x >> np.uint64(4))
    x = x ^ (x >> np.uint64(2))
    x = x ^ (x >> np.uint64(1))
    return (x & np.uint64(1)).astype(np.uint8)


def generate(model: DramModel, n: int = 200_000,
             mu_hit: float = 210.0, mu_conflict: float = 360.0,
             sigma: float = 18.0, outlier_frac: float = 0.02,
             fault_floor: float = 700.0, outlier_boost: float = 800.0,
             seed: int = 0) -> Dataset:
    """Draw `n` random address pairs and time them under `model`.

    outlier_frac injects page-fault/TLB-miss spikes: microsecond-scale, far
    above a row conflict, so each adds `fault_floor + Exp(outlier_boost)` cycles
    and lands in a distinct high tail the classifier must trim.
    """
    rng = np.random.default_rng(seed)
    bit_arr = np.array(model.addr_bits, dtype=np.uint64)

    def rand_addrs(k):
        choice = rng.integers(0, 2, size=(k, bit_arr.size), dtype=np.uint8)
        vals = np.zeros(k, dtype=np.uint64)
        for j, b in enumerate(bit_arr):
            vals |= (choice[:, j].astype(np.uint64) << b)
        return vals

    a = rand_addrs(n)
    b = rand_addrs(n)

    same_bank = np.ones(n, dtype=bool)
    for f in model.functions:
        same_bank &= (_parity_masked(a, f) == _parity_masked(b, f))
    diff_row = ((a ^ b) & np.uint64(model.row_mask)) != 0
    conflict = same_bank & diff_row

    latency = np.where(conflict,
                       rng.normal(mu_conflict, sigma, n),
                       rng.normal(mu_hit, sigma, n))

    # one-sided outliers on random rows (both classes), far above the conflict
    # mode (fault_floor guarantees clear separation)
    n_out = int(outlier_frac * n)
    if n_out:
        idx = rng.choice(n, size=n_out, replace=False)
        latency[idx] += fault_floor + rng.exponential(outlier_boost, n_out)

    return Dataset(a.astype(np.uint64), b.astype(np.uint64),
                   latency.astype(float), conflict, model)


def to_csv(ds: Dataset, path: str) -> None:
    """Write a probe-compatible dataset.csv (addr_a,addr_b,latency)."""
    with open(path, "w") as f:
        f.write("addr_a,addr_b,latency\n")
        for a, b, l in zip(ds.addr_a, ds.addr_b, ds.latency):
            f.write(f"{int(a)},{int(b)},{int(round(l))}\n")
