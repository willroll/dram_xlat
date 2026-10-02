"""Human-readable rendering of recovered XOR functions."""
from __future__ import annotations
from typing import List, Sequence
from . import gf2


def mask_to_str(mask: int) -> str:
    """e.g. 0x24000 -> 'b14 ^ b17'."""
    bits = gf2.bits_of(mask)
    if not bits:
        return "0"
    return " ^ ".join(f"b{b}" for b in bits)


def format_functions(masks: Sequence[int]) -> str:
    lines = []
    for i, m in enumerate(gf2.rref(masks)):
        lines.append(f"  f{i}: {mask_to_str(m):<28} (0x{m:X})")
    return "\n".join(lines) if lines else "  (none)"


def functions_json(masks: Sequence[int]) -> List[dict]:
    out = []
    for i, m in enumerate(gf2.rref(masks)):
        out.append({
            "index": i,
            "mask_hex": f"0x{m:X}",
            "mask_int": m,
            "bits": gf2.bits_of(m),
            "expr": mask_to_str(m),
        })
    return out
