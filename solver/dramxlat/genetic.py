"""Evolutionary search over candidate XOR masks — the noise-tolerant path.

The exact GF(2) null-space solver is fast and correct, but assumes *every*
conflict label is right: a single mislabeled pair adds a bad row and can
collapse the null space. When the data is noisy we instead search the space of
masks directly for ones that are *invariant on conflict pairs* —
parity(m & (a^b)) == 0 for a high fraction of them — and informative on
non-conflict pairs.

A bank-selecting function (or any XOR of them) has:
    conflict-agreement  ~= 1 - c/2   (c = label contamination; >0.5 always)
    non-conflict-split   > 0          (it distinguishes some different banks)
Random masks sit near 0.5 agreement, so the floor sits between them. The search
is seeded with an exhaustive enumeration of low-popcount masks, because real
DRAM functions XOR only a handful of bits.

`search()` returns a reduced GF(2) basis of the best invariant masks found.
Population evaluation is vectorised over the dataset with numpy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from . import gf2


@dataclass
class MaskScore:
    mask: int
    conflict_agreement: float   # P(parity(m&Δ)=0 | conflict)  -> want ~1
    noncon_split: float         # P(parity(m&Δ)=1 | non-conflict) -> want >0
    fitness: float


def _parity_batch(delta: np.ndarray, mask: int) -> np.ndarray:
    """parity(mask & delta) for a uint64 array; returns uint8 array."""
    x = (delta & np.uint64(mask)).copy()
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return (x & np.uint64(1)).astype(np.uint8)


def score_mask(mask: int, dconf: np.ndarray, dnon: np.ndarray) -> MaskScore:
    if mask == 0:
        return MaskScore(0, 1.0, 0.0, -1.0)
    agree = 1.0 - _parity_batch(dconf, mask).mean() if dconf.size else 0.0
    split = _parity_batch(dnon, mask).mean() if dnon.size else 0.0
    pc = bin(mask).count("1")
    # reward conflict-invariance first, then informativeness, and gently prefer
    # sparse masks (real functions XOR few bits).
    fitness = agree + 0.15 * split - 0.005 * pc
    return MaskScore(mask, float(agree), float(split), float(fitness))


def _random_mask(rng, cols: Sequence[int], max_pc: int) -> int:
    pc = rng.integers(1, max_pc + 1)
    chosen = rng.choice(len(cols), size=min(pc, len(cols)), replace=False)
    m = 0
    for i in chosen:
        m |= (1 << cols[i])
    return m


def _mutate(rng, mask: int, cols: Sequence[int]) -> int:
    c = cols[rng.integers(len(cols))]
    return mask ^ (1 << c)


def _enumerate_low_popcount(cols: Sequence[int], max_pairs: int) -> List[int]:
    """All 1-bit and 2-bit masks over `cols` (2-bit capped at max_pairs).

    Real DRAM addressing functions XOR only a handful of bits, so the true
    2-bit functions are almost always in this tiny set — enumerating it makes
    the search reliable instead of hoping mutation stumbles onto the pair.
    """
    seeds = [1 << c for c in cols]
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((1 << cols[i]) | (1 << cols[j]))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return seeds + pairs


def search(delta_conflict: np.ndarray, delta_nonconflict: np.ndarray,
           cols: Sequence[int], *,
           pop: int = 200, generations: int = 30, max_popcount: int = 6,
           elite_frac: float = 0.25, agreement_floor: float = 0.62,
           split_floor: float = 0.05, enum_pairs: int = 4000,
           seed: int = 0) -> List[MaskScore]:
    """Run the search; return a reduced basis of high-quality invariant masks.

    A candidate must be invariant on conflict pairs (agreement >=
    agreement_floor) and separate some non-conflict pairs (split >=
    split_floor). Survivors are reduced to a GF(2) basis so we report
    independent functions, not their XOR combinations.
    """
    rng = np.random.default_rng(seed)
    cols = list(cols)

    # Fitness is statistical; a few thousand deltas is ample and keeps each
    # generation cheap. Cap the working set.
    cap = 4000
    if delta_conflict.size > cap:
        delta_conflict = delta_conflict[rng.choice(delta_conflict.size, cap, replace=False)]
    if delta_nonconflict.size > cap:
        delta_nonconflict = delta_nonconflict[rng.choice(delta_nonconflict.size, cap, replace=False)]

    n_elite = max(2, int(elite_frac * pop))
    seen_good: dict[int, MaskScore] = {}

    def consider(masks):
        for m in masks:
            s = score_mask(m, delta_conflict, delta_nonconflict)
            if (s.conflict_agreement >= agreement_floor
                    and s.noncon_split >= split_floor):
                prev = seen_good.get(m)
                if prev is None or s.fitness > prev.fitness:
                    seen_good[m] = s

    # Seed: exhaustive low-popcount enumeration finds the sparse functions.
    seeds = _enumerate_low_popcount(cols, enum_pairs)
    consider(seeds)

    seed_scores = sorted(
        (score_mask(m, delta_conflict, delta_nonconflict) for m in seeds),
        key=lambda s: s.fitness, reverse=True)
    population = [s.mask for s in seed_scores[:pop]]
    while len(population) < pop:
        population.append(_random_mask(rng, cols, max_popcount))

    for _ in range(generations):
        scored = [score_mask(m, delta_conflict, delta_nonconflict)
                  for m in population]
        scored.sort(key=lambda s: s.fitness, reverse=True)
        consider(m for m in population)

        elites = [s.mask for s in scored[:n_elite]]
        nxt = list(elites)
        while len(nxt) < pop:
            p = elites[rng.integers(len(elites))]
            child = _mutate(rng, p, cols)
            if rng.random() < 0.3:                 # crossover with another elite
                q = elites[rng.integers(len(elites))]
                child ^= (q & int(rng.integers(0, 1 << 30)))
            if rng.random() < 0.15:                # occasional fresh blood
                child = _random_mask(rng, cols, max_popcount)
            nxt.append(child)
        population = nxt

    # Reduce the good masks to an independent basis, preferring sparse/strong
    # representatives, then rescore the basis for reporting.
    goods = sorted(seen_good.values(),
                   key=lambda s: (s.fitness, -bin(s.mask).count("1")),
                   reverse=True)
    basis: List[int] = []
    for s in goods:
        trial = gf2.span_basis(basis + [s.mask])
        if len(trial) > len(basis):
            basis.append(s.mask)
    return [score_mask(m, delta_conflict, delta_nonconflict) for m in basis]
