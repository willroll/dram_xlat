"""dramxlat — recover DRAM address-translation XOR functions from noisy
row-buffer-conflict timing data.

Modules
-------
gf2       GF(2) linear algebra over integer bit-vectors (basis, null space).
classify  turn a noisy latency distribution into conflict/hit labels.
genetic   noise-tolerant evolutionary search over candidate XOR masks.
symbolic  Z3-backed formal cross-check / small-instance symbolic solve.
synthetic generate a known noisy truth table for validation.
pipeline  end-to-end: dataset -> candidate bank/rank XOR functions.
report    pretty-printing of masks as bit lists.
"""
from . import gf2, classify, genetic, symbolic, synthetic, report, pipeline  # noqa: F401

__all__ = ["gf2", "classify", "genetic", "symbolic", "synthetic",
           "report", "pipeline"]
