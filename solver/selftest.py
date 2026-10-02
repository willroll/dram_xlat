#!/usr/bin/env python3
"""Validate the Phase 3 pipeline end-to-end WITHOUT any hardware.

Synthesise a known DRAM map, generate a noisy timing truth table from it, run
the full discovery pipeline, and assert the recovered GF(2) functions span
exactly the same subspace as the ground-truth bank-selecting functions.

The recovered functions are only defined up to a change of basis (the hardware's
choice of "which XOR is bank bit 0" is arbitrary), so correctness is "same
span", checked via reduced row-echelon form.

Run:  python3 solver/selftest.py      (exit 0 = all cases recovered the map)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dramxlat import synthetic, pipeline, gf2, report  # noqa: E402
from dramxlat.dataio import Timing  # noqa: E402


def run_case(name, *, noise, sep_gap, n=120_000, use_z3=False):
    model = synthetic.example_model()
    mu_hit, mu_conf = 210.0, 210.0 + sep_gap
    ds = synthetic.generate(model, n=n, mu_hit=mu_hit, mu_conflict=mu_conf,
                            sigma=18.0, outlier_frac=noise, seed=7)
    timing = Timing(ds.addr_a, ds.addr_b, ds.latency)

    res = pipeline.discover(timing, bit_lo=6, bit_hi=32,
                            use_genetic=True, use_z3=use_z3, seed=1)

    truth = model.functions
    ok = gf2.same_span(res.functions, truth)

    print(f"\n=== case: {name} (outlier_frac={noise}, gap={sep_gap}cyc) ===")
    c = res.classification
    print(f"  classifier: hit≈{c.mu_hit:.0f} conflict≈{c.mu_conflict:.0f} "
          f"sep={c.separation:.1f}σ  faults_trimmed={c.n_fault}  winner={res.method}")
    for cname, bs in res.candidates.items():
        print(f"    {cname:<18} dim={bs.dim} cons={bs.consistency:.3f} "
              f"sep={bs.separation:.3f} score={bs.score:.3f}")
    print("  recovered:")
    print(report.format_functions(res.functions))
    if res.z3_confirmed is not None:
        print(f"  z3: consistent {len(res.z3_confirmed)}-fn map exists")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (same span as ground truth = {ok})")
    return ok


def main():
    cases = [
        ("clean",        dict(noise=0.0,  sep_gap=150)),
        ("light-noise",  dict(noise=0.02, sep_gap=150)),
        ("heavy-noise",  dict(noise=0.06, sep_gap=130)),
        ("tight-modes",  dict(noise=0.03, sep_gap=90)),
        ("z3-confirm",   dict(noise=0.0,  sep_gap=150, n=20_000, use_z3=True)),
    ]
    results = []
    for name, kw in cases:
        try:
            results.append((name, run_case(name, **kw)))
        except Exception as e:  # pragma: no cover
            print(f"\n=== case {name}: ERROR {e!r} ===")
            results.append((name, False))

    print("\n" + "=" * 52)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}")
    print(f"  {passed}/{len(results)} cases recovered the true map")
    print("=" * 52)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
