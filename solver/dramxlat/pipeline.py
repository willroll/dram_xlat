"""End-to-end discovery: noisy timing dataset -> candidate XOR functions.

Stages (neuro-symbolic + evolutionary, as per the plan):

  1. classify   robust latency model -> conflict/hit + confidence + fault trim.
  2. deltas     conflict pairs -> difference vectors a^b (uint64).
  3. solve      three independent routes to the bank-selecting functions:
                  - exact GF(2) null space on high-confidence conflict deltas
                  - RANSAC-robust null space (tolerates mislabeled pairs)
                  - genetic search (fully noise-tolerant)
  4. combine    union every route's masks, keep the ones individually invariant
                on (nearly) all conflict deltas AND informative, reduce to a
                GF(2) basis. Robust to any one route returning an incomplete
                basis; a spurious mask is dropped on its own merits.
  5. confirm    optional Z3 formal check that a consistent map of that size
                exists on a small high-confidence subset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from . import gf2, classify, genetic, symbolic, report
from .dataio import Timing


# ---------------------------------------------------------------- scoring

def _split_mean(masks: Sequence[int], dnon: np.ndarray) -> float:
    """Mean fraction of non-conflict pairs each mask distinguishes."""
    if not masks or dnon.size == 0:
        return 0.0
    return float(np.mean([genetic._parity_batch(dnon, m).mean() for m in masks]))


@dataclass
class BasisScore:
    masks: List[int]
    consistency: float          # invariant on this fraction of conflict deltas
    separation: float           # mean non-conflict split across masks
    dim: int
    score: float


def evaluate_basis(masks: Sequence[int],
                   dconf: np.ndarray, dnon: np.ndarray) -> BasisScore:
    masks = gf2.rref([m for m in masks if m])
    cons = symbolic.consistency(masks, dconf) if masks else 0.0
    sep = _split_mean(masks, dnon)
    dim = len(masks)
    # A good basis is highly consistent AND informative. Empty/degenerate bases
    # (sep ~ 0) are punished so a trivial "everything invariant" answer never
    # wins.
    score = cons + 0.5 * sep - (0.0 if sep > 0.01 else 1.0)
    return BasisScore(masks, cons, sep, dim, score)


# ---------------------------------------------------------------- robust NS

def robust_nullspace(dconf: np.ndarray, dnon: np.ndarray, cols: Sequence[int],
                     *, iters: int = 150, sample: Optional[int] = None,
                     split_floor: float = 0.03, seed: int = 0) -> List[int]:
    """RANSAC over GF(2) null spaces: sample clean-ish subsets of conflict
    deltas, solve each exactly, keep the basis that best explains ALL conflict
    deltas. A subset free of mislabeled pairs yields the true bank space, which
    then scores highest on the full set."""
    n = int(dconf.size)
    if n == 0:
        return []
    dconf_ints = [int(x) for x in dconf]
    if sample is None:
        sample = min(n, max(len(cols) + 8, 40))
    rng = np.random.default_rng(seed)

    best: List[int] = []
    best_score = -1e9
    for _ in range(iters):
        idx = rng.choice(n, size=sample, replace=False)
        subset = [dconf_ints[i] for i in idx]
        null = gf2.nullspace(subset, cols)
        basis = [m for m in null if genetic._parity_batch(dnon, m).mean() >= split_floor] \
            if dnon.size else null
        if not basis:
            continue
        bs = evaluate_basis(basis, dconf, dnon)
        if bs.score > best_score:
            best_score, best = bs.score, bs.masks
    return best


# ---------------------------------------------------------------- driver

@dataclass
class DiscoveryResult:
    functions: List[int]                 # chosen basis (bank-selecting masks)
    method: str                          # which route(s) produced it
    classification: classify.Classification
    candidates: Dict[str, BasisScore] = field(default_factory=dict)
    cols: List[int] = field(default_factory=list)
    n_conflict: int = 0
    n_total: int = 0
    z3_confirmed: Optional[List[int]] = None

    def to_dict(self) -> dict:
        c = self.classification
        return {
            "functions": report.functions_json(self.functions),
            "method": self.method,
            "num_functions": len(self.functions),
            "classification": {
                "method": c.method,
                "threshold_cycles": round(c.threshold, 2),
                "mu_hit": round(c.mu_hit, 2),
                "mu_conflict": round(c.mu_conflict, 2),
                "separation_sigmas": round(c.separation, 2),
                "n_fault_discarded": c.n_fault,
            },
            "candidates": {
                name: {
                    "dim": bs.dim,
                    "consistency": round(bs.consistency, 4),
                    "separation": round(bs.separation, 4),
                    "score": round(bs.score, 4),
                    "functions": [report.mask_to_str(m) for m in bs.masks],
                } for name, bs in self.candidates.items()
            },
            "cols": self.cols,
            "n_conflict": self.n_conflict,
            "n_total": self.n_total,
            "z3_confirmed": (None if self.z3_confirmed is None
                             else [report.mask_to_str(m) for m in self.z3_confirmed]),
        }


def discover(timing: Timing, *, bit_lo: int = 6, bit_hi: Optional[int] = None,
             method: str = "auto", hi_conf: float = 0.9,
             use_genetic: bool = True, use_z3: bool = False,
             z3_functions: Optional[int] = None, seed: int = 0) -> DiscoveryResult:
    lat = timing.latency
    delta = timing.delta
    n_total = int(delta.size)

    cls = classify.classify_latencies(lat, method=method)
    conflict = cls.is_conflict
    dconf = delta[conflict]
    dnon = delta[~conflict]

    # bits that actually vary, within the requested window
    or_all = int(np.bitwise_or.reduce(delta)) if delta.size else 0
    hi = bit_hi if bit_hi is not None else 47
    cols = [b for b in range(bit_lo, hi + 1) if (or_all >> b) & 1]

    # high-confidence conflict subset for the exact solver
    hc = conflict & (cls.confidence >= hi_conf)
    dconf_hc = delta[hc]

    # Every stage below is statistical; capping the working sets keeps the
    # search fast without weakening the signal.
    rng = np.random.default_rng(seed)

    def cap(arr: np.ndarray, k: int) -> np.ndarray:
        if arr.size <= k:
            return arr
        return arr[rng.choice(arr.size, size=k, replace=False)]

    dconf_eval = cap(dconf, 20000)
    dnon_eval = cap(dnon, 20000)
    dconf_hc_s = cap(dconf_hc, 4000)

    candidates: Dict[str, BasisScore] = {}

    exact = gf2.nullspace([int(x) for x in dconf_hc_s], cols)
    exact = [m for m in exact if genetic._parity_batch(dnon_eval, m).mean() >= 0.01] \
        if dnon_eval.size else exact
    candidates["exact-nullspace"] = evaluate_basis(exact, dconf_eval, dnon_eval)

    robust = robust_nullspace(dconf_eval, dnon_eval, cols, seed=seed)
    candidates["robust-nullspace"] = evaluate_basis(robust, dconf_eval, dnon_eval)

    if use_genetic:
        ga = [ms.mask for ms in genetic.search(dconf_eval, dnon_eval, cols, seed=seed)]
        candidates["genetic"] = evaluate_basis(ga, dconf_eval, dnon_eval)

    # Final answer = the maximal invariant+informative subspace: union every
    # route's masks, keep ones individually invariant on (nearly) all conflict
    # deltas and that separate non-conflict pairs, reduce to a GF(2) basis.
    cons_floor = 0.85
    union_masks: List[int] = []
    for bs in candidates.values():
        union_masks.extend(bs.masks)

    good: List[int] = []
    for m in set(union_masks):
        if m == 0:
            continue
        cons = 1.0 - genetic._parity_batch(dconf_eval, m).mean() if dconf_eval.size else 0.0
        split = genetic._parity_batch(dnon_eval, m).mean() if dnon_eval.size else 0.0
        if cons >= cons_floor and split >= 0.02:
            good.append(m)
    final = gf2.rref(gf2.span_basis(good))
    candidates["combined"] = evaluate_basis(final, dconf_eval, dnon_eval)

    contributors = [name for name, bs in candidates.items()
                    if name != "combined" and final
                    and gf2.rank(bs.masks) == len(final)
                    and gf2.same_span(bs.masks, final)]
    method_name = contributors[0] if contributors else "combined"

    z3_conf = None
    if use_z3 and final:
        k = z3_functions or len(final)
        z3_conf = symbolic.solve_functions(
            [int(x) for x in dconf_hc_s[:400]],
            [int(x) for x in dnon_eval[:400]],
            k, cols)

    return DiscoveryResult(
        functions=final,
        method=method_name,
        classification=cls,
        candidates=candidates,
        cols=cols,
        n_conflict=int(conflict.sum()),
        n_total=n_total,
        z3_confirmed=z3_conf,
    )
