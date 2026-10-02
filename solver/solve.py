#!/usr/bin/env python3
"""Phase 3 CLI: timing dataset -> candidate DRAM XOR functions.

    python3 solver/solve.py dataset.csv --out functions.json

Reads the probe's 'addr_a,addr_b,latency' CSV, classifies the latency modes,
recovers the bank/rank-selecting GF(2) functions, and writes them as JSON (also
consumed by verify/verify.py in Phase 4).
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dramxlat import dataio, pipeline, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="probe CSV: addr_a,addr_b,latency")
    ap.add_argument("--out", default=None, help="write functions to this JSON")
    ap.add_argument("--bit-lo", type=int, default=6)
    ap.add_argument("--bit-hi", type=int, default=None)
    ap.add_argument("--classify", default="auto",
                    choices=["auto", "robust", "gmm", "quantile"])
    ap.add_argument("--no-genetic", action="store_true")
    ap.add_argument("--z3", action="store_true",
                    help="formally confirm with Z3 on a high-confidence subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    timing = dataio.load_csv(args.dataset)
    if len(timing) == 0:
        sys.exit(f"no usable rows in {args.dataset}")
    print(f"[*] loaded {len(timing)} pairs from {args.dataset}", file=sys.stderr)

    res = pipeline.discover(
        timing, bit_lo=args.bit_lo, bit_hi=args.bit_hi,
        method=args.classify, use_genetic=not args.no_genetic,
        use_z3=args.z3, seed=args.seed)

    c = res.classification
    print(f"[*] latency modes: hit≈{c.mu_hit:.0f}  conflict≈{c.mu_conflict:.0f} "
          f"cyc  (separation {c.separation:.1f}σ, threshold {c.threshold:.0f}, "
          f"{c.n_fault} faults trimmed)", file=sys.stderr)
    print(f"[*] {res.n_conflict}/{res.n_total} pairs labeled conflict",
          file=sys.stderr)
    if c.separation < 2.0:
        print("[!] weak bimodal separation (<2σ): increase --reps, pin an "
              "isolated core, disable prefetchers.", file=sys.stderr)

    print("\n[*] candidate bases (winner marked *):", file=sys.stderr)
    for name, bs in res.candidates.items():
        star = "*" if name == res.method or (res.method == "combined" and name == "combined") else " "
        print(f"  {star} {name:<18} dim={bs.dim} "
              f"consistency={bs.consistency:.3f} separation={bs.separation:.3f} "
              f"score={bs.score:.3f}", file=sys.stderr)

    print(f"\n[+] recovered {len(res.functions)} bank/rank-selecting functions "
          f"(via {res.method}):", file=sys.stderr)
    print(report.format_functions(res.functions), file=sys.stderr)
    if res.z3_confirmed is not None:
        print(f"[+] Z3 confirmed a consistent {len(res.z3_confirmed)}-function "
              f"map exists on the clean subset.", file=sys.stderr)

    payload = res.to_dict()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[+] wrote {args.out}", file=sys.stderr)
        print(f"[+] next: python3 verify/verify.py {args.out} "
              f"--probe ./probe/dram_probe", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
