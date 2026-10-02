"""GF(2) linear algebra on integer bit-vectors.

A "vector" is a Python int: bit *i* set means coordinate *i* is 1. XOR is
vector addition; the GF(2) dot product of two vectors is the parity of their
AND. Physical-address masks and address-difference vectors are both just ints,
which makes the whole solver a few popcounts and XORs.

The one theorem the solver rests on:

    two addresses map to the same bank  <=>  every bank-selecting XOR function
    m has equal parity on both  <=>  m . (a XOR b) == 0.

So the bank-selecting functions are exactly the vectors orthogonal to every
difference vector `a XOR b` over same-bank ("conflict") pairs — the null space
of the matrix whose rows are those difference vectors.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence


def parity(x: int) -> int:
    """GF(2) sum of the bits of x (0 or 1)."""
    return bin(x).count("1") & 1


def dot(a: int, b: int) -> int:
    """GF(2) dot product: parity(a & b)."""
    return bin(a & b).count("1") & 1


def lowest_set_bit(x: int) -> int:
    return (x & -x).bit_length() - 1


def bits_of(x: int) -> List[int]:
    """Sorted list of set-bit positions."""
    out = []
    i = 0
    while x:
        if x & 1:
            out.append(i)
        x >>= 1
        i += 1
    return out


def span_basis(vectors: Iterable[int]) -> List[int]:
    """A basis of the span of `vectors` (echelon form, highest-bit pivots).

    Dependent vectors are dropped; the result's length is the rank.
    """
    pivots: dict[int, int] = {}      # leading-bit -> reduced vector
    for v in vectors:
        cur = v
        while cur:
            hb = cur.bit_length() - 1
            if hb in pivots:
                cur ^= pivots[hb]
            else:
                pivots[hb] = cur
                break
    return sorted(pivots.values(), reverse=True)


def rank(vectors: Iterable[int]) -> int:
    return len(span_basis(vectors))


def rref(vectors: Iterable[int]) -> List[int]:
    """Reduced row-echelon basis (each pivot bit appears in exactly one row).

    Canonical: two sets of vectors span the same subspace iff their rref lists
    are equal. Used to compare a recovered function set against ground truth
    regardless of basis choice.
    """
    rows: List[int] = []
    piv_cols: List[int] = []
    for v in vectors:
        cur = v
        for i, pc in enumerate(piv_cols):
            if (cur >> pc) & 1:
                cur ^= rows[i]
        if cur == 0:
            continue
        pc = lowest_set_bit(cur)
        for i in range(len(rows)):
            if (rows[i] >> pc) & 1:
                rows[i] ^= cur
        rows.append(cur)
        piv_cols.append(pc)
    return sorted(rows, reverse=True)


def same_span(a: Iterable[int], b: Iterable[int]) -> bool:
    """True iff `a` and `b` span the same GF(2) subspace."""
    return rref(a) == rref(b)


def nullspace(rows: Sequence[int], cols: Sequence[int]) -> List[int]:
    """Basis of { x supported on `cols` : row . x == 0 for every row }.

    `cols` is the list of bit positions the solution may use (the bits that
    actually vary in the data). Bits outside `cols` are held at 0, so the
    returned functions never reference constant/irrelevant address bits.
    """
    col_mask = 0
    for c in cols:
        col_mask |= (1 << c)

    # RREF of the rows, restricted to `cols`.
    basis: List[int] = []
    piv_cols: List[int] = []
    for v in rows:
        cur = v & col_mask
        for i, pc in enumerate(piv_cols):
            if (cur >> pc) & 1:
                cur ^= basis[i]
        if cur == 0:
            continue
        pc = lowest_set_bit(cur)
        for i in range(len(basis)):
            if (basis[i] >> pc) & 1:
                basis[i] ^= cur
        basis.append(cur)
        piv_cols.append(pc)

    piv_set = set(piv_cols)
    free_cols = [c for c in cols if c not in piv_set]

    null: List[int] = []
    for fc in free_cols:
        x = 1 << fc
        for i, pc in enumerate(piv_cols):
            # each RREF row has exactly one pivot column among piv_cols (pc);
            # row . x == x_pc + row_fc, so x_pc = row_fc keeps it zero.
            if (basis[i] >> fc) & 1:
                x |= (1 << pc)
        null.append(x)
    return sorted(null, reverse=True)


def varying_bits(vectors: Iterable[int], max_bit: int = 48) -> List[int]:
    """Bit positions set in at least one of `vectors` (up to max_bit)."""
    acc = 0
    for v in vectors:
        acc |= int(v)
    return [b for b in range(max_bit) if (acc >> b) & 1]
