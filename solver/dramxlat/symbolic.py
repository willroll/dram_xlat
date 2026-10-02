"""Symbolic (Z3) cross-check and small-instance solve — the "symbolic logic
solver" half of the neuro-symbolic pipeline.

The GF(2) null-space and genetic stages produce candidate functions
numerically. This module uses Z3 to *formally* reason about them:

  * `consistency(basis, dconf)` — a pure arithmetic check that every basis mask
    is invariant on the conflict deltas (no solver needed; kept here so the
    whole "is this model self-consistent?" question lives in one place).

  * `solve_functions(dconf, dnon, k, cols)` — encode the discovery problem as a
    SAT instance and let Z3 find k XOR functions such that all conflict deltas
    have zero parity under every function and every non-conflict delta has
    parity 1 under at least one function. Exponential, so meant for a *small,
    high-confidence subset* — a formal confirmation that a consistent map of
    the claimed size exists, independent of the numeric path.

Z3 is optional; if it is not installed these degrade gracefully.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np


def consistency(basis: Sequence[int], delta_conflict: np.ndarray) -> float:
    """Fraction of conflict deltas on which *every* basis mask is invariant."""
    if delta_conflict.size == 0 or not basis:
        return 0.0
    ok = np.ones(delta_conflict.size, dtype=bool)
    for m in basis:
        x = (delta_conflict & np.uint64(m)).copy()
        for sh in (32, 16, 8, 4, 2, 1):
            x ^= x >> np.uint64(sh)
        ok &= ((x & np.uint64(1)) == 0)
    return float(ok.mean())


def solve_functions(delta_conflict: Sequence[int],
                    delta_nonconflict: Sequence[int],
                    k: int, cols: Sequence[int], *,
                    max_constraints: int = 400,
                    timeout_ms: int = 20000) -> Optional[List[int]]:
    """Ask Z3 for k XOR functions consistent with a (small) labeled subset.

    Returns a list of k masks, or None if unsatisfiable / Z3 unavailable.
    Intended as a formal sanity check on a clean subset, not the full noisy set.
    """
    try:
        import z3
    except Exception:
        return None

    cols = list(cols)
    dconf = list(delta_conflict)[:max_constraints]
    dnon = list(delta_nonconflict)[:max_constraints]

    s = z3.Solver()
    s.set("timeout", timeout_ms)

    # bit variables: f[j][c] == function j uses address bit c
    f = [[z3.Bool(f"f_{j}_{c}") for c in cols] for j in range(k)]

    def parity_expr(j, delta):
        terms = [f[j][i] for i, c in enumerate(cols) if (delta >> c) & 1]
        if not terms:
            return z3.BoolVal(False)
        acc = terms[0]
        for t in terms[1:]:
            acc = z3.Xor(acc, t)
        return acc

    # conflict deltas: every function has even parity (invariant)
    for d in dconf:
        for j in range(k):
            s.add(z3.Not(parity_expr(j, d)))

    # non-conflict deltas: at least one function has odd parity (separates)
    for d in dnon:
        s.add(z3.Or([parity_expr(j, d) for j in range(k)]))

    # each function is non-trivial
    for j in range(k):
        s.add(z3.Or(f[j]))

    # symmetry breaking: order functions by their bitmask value
    def mask_int(j):
        return z3.Sum([z3.If(f[j][i], z3.IntVal(1 << c), z3.IntVal(0))
                       for i, c in enumerate(cols)])
    for j in range(k - 1):
        s.add(mask_int(j) < mask_int(j + 1))

    if s.check() != z3.sat:
        return None

    model = s.model()
    out = []
    for j in range(k):
        m = 0
        for i, c in enumerate(cols):
            if z3.is_true(model.eval(f[j][i], model_completion=True)):
                m |= (1 << c)
        out.append(m)
    return out
