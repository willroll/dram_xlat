#!/usr/bin/env python3
"""Phase 4: formally verify a derived DRAM map against the hardware.

Given the GF(2) functions from Phase 3, synthesise two address sets:

  * SAME-BANK / DIFFERENT-ROW pairs — every bank function has equal parity on
    the two addresses (shared bank), and they differ in a high "row" bit
    (different row). If the map is correct, ALL must show the slow row-conflict
    latency.

  * DIFFERENT-BANK pairs (control) — at least one bank function differs, so the
    accesses go to different banks and are served in parallel (fast).

Time both sets on the target, derive the conflict threshold from the two
populations themselves (self-calibrating), and report the fraction of same-bank
pairs that hit the conflict mode. A correct map yields ~100%.

Modes:
  --probe PATH     time on real hardware via the dram_probe binary
  --simulate       time under a known ground-truth model (offline validation)
  --emit-only      just write the two pair files for a manual probe run
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solver"))
from dramxlat import gf2, synthetic  # noqa: E402
import numpy as np  # noqa: E402


# ----------------------------------------------------------- pair synthesis

def load_masks(path):
    with open(path) as f:
        data = json.load(f)
    return [int(fn["mask_int"]) for fn in data["functions"]]


def buffer_bits(size_bytes):
    """Highest usable address bit for a buffer of this size."""
    return max(6, (size_bytes.bit_length() - 1))


def synth_pairs(masks, size_bytes, num_pairs, align, seed):
    """Return (same_bank_pairs, diff_bank_pairs) as lists of (off_a, off_b).

    same-bank/diff-row: XOR difference lies in the complement of the bank space
    AND flips a bit above the bank region (a row bit).
    diff-bank: flip a single bit that participates in some bank function.
    """
    rng = np.random.default_rng(seed)
    hi = buffer_bits(size_bytes)
    align_bits = max(6, align.bit_length() - 1)

    used = set()
    for m in masks:
        used |= set(gf2.bits_of(m))
    max_fbit = max(used) if used else align_bits

    row_bits = [b for b in range(max_fbit + 1, hi) if b not in used]
    bank_bits = sorted(used)

    if not row_bits:
        raise SystemExit("no row bits available above the bank region; use a "
                         "larger --hugepage/--size so high address bits vary.")
    if not bank_bits:
        raise SystemExit("no bank-function bits in the model; nothing to verify.")

    size_mask = (size_bytes - 1) & ~(align - 1)

    def rand_base():
        return int(rng.integers(0, size_bytes)) & size_mask

    same, diff = [], []
    for _ in range(num_pairs):
        a = rand_base()
        k = 1 + int(rng.integers(0, 2))
        d = 0
        for b in rng.choice(row_bits, size=min(k, len(row_bits)), replace=False):
            d |= (1 << int(b))
        b1 = (a ^ d) & size_mask
        if b1 != a:
            same.append((a, b1))

        a2 = rand_base()
        bb = 1 << int(rng.choice(bank_bits))
        b2 = (a2 ^ bb) & size_mask
        if b2 != a2:
            diff.append((a2, b2))

    return same, diff


def check_pairs(masks, same, diff):
    """Sanity-check the synthesised pairs against the masks (offline)."""
    def is_same_bank(a, b):
        return all(gf2.dot(m, a ^ b) == 0 for m in masks)
    ok_same = all(is_same_bank(a, b) for a, b in same)
    ok_diff = all(not is_same_bank(a, b) for a, b in diff)
    return ok_same, ok_diff


# ------------------------------------------------------------ timing backends

def write_pairs(path, pairs):
    with open(path, "w") as f:
        f.write("addr_a,addr_b\n")
        for a, b in pairs:
            f.write(f"{a},{b}\n")


def time_with_probe(probe, pairs_path, args):
    cmd = [probe, "verify", "--offsets", "--pairs-file", pairs_path,
           "--hugepage", args.hugepage, "--reps", str(args.reps),
           "--align", str(args.align), "--cpu", str(args.cpu), "--stat", args.stat]
    if args.size:
        cmd += ["--size", str(args.size)]
    if args.sudo:
        cmd = ["sudo"] + cmd
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    lat = []
    for line in out.stdout.splitlines():
        if line.startswith("addr_a") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) == 3:
            try:
                lat.append(float(parts[2]))
            except ValueError:
                pass
    return np.array(lat, dtype=float)


def time_with_sim(pairs, model, seed):
    """Emulate the probe using a known ground-truth model (offline)."""
    rng = np.random.default_rng(seed)
    a = np.array([p[0] for p in pairs], dtype=np.uint64)
    b = np.array([p[1] for p in pairs], dtype=np.uint64)
    same_bank = np.ones(len(pairs), dtype=bool)
    for f in model.functions:
        same_bank &= (synthetic._parity_masked(a, f) == synthetic._parity_masked(b, f))
    diff_row = ((a ^ b) & np.uint64(model.row_mask)) != 0
    conflict = same_bank & diff_row
    return np.where(conflict, rng.normal(360, 18, len(pairs)),
                    rng.normal(210, 18, len(pairs)))


# ------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("functions", help="functions.json from Phase 3")
    ap.add_argument("--probe", default=None, help="path to dram_probe binary")
    ap.add_argument("--simulate", action="store_true",
                    help="offline: time under the built-in ground-truth model")
    ap.add_argument("--emit-only", action="store_true",
                    help="only write the two pair files, do not time")
    ap.add_argument("--hugepage", default="1G", choices=["0", "2M", "1G"])
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--num-pairs", type=int, default=2000)
    ap.add_argument("--align", type=int, default=64)
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--stat", default="median", choices=["min", "median"])
    ap.add_argument("--sudo", action="store_true")
    ap.add_argument("--pass-rate", type=float, default=0.95,
                    help="min same-bank conflict fraction to PASS")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--outdir", default=".", help="where to write pair files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    masks = load_masks(args.functions)
    print(f"[*] verifying {len(masks)} functions: "
          f"{[gf2.bits_of(m) for m in masks]}", file=sys.stderr)

    size_bytes = args.size or {"0": 256 * 1024 * 1024, "2M": 2 * 1024 * 1024,
                               "1G": 1024 * 1024 * 1024}[args.hugepage]

    same, diff = synth_pairs(masks, size_bytes, args.num_pairs, args.align, args.seed)
    ok_same, ok_diff = check_pairs(masks, same, diff)
    print(f"[*] synthesised {len(same)} same-bank/diff-row and {len(diff)} "
          f"different-bank pairs", file=sys.stderr)
    print(f"[*] offline pair sanity: same-bank_ok={ok_same} diff-bank_ok={ok_diff}",
          file=sys.stderr)

    same_path = os.path.join(args.outdir, "verify_same_bank.csv")
    diff_path = os.path.join(args.outdir, "verify_diff_bank.csv")
    write_pairs(same_path, same)
    write_pairs(diff_path, diff)
    print(f"[*] wrote {same_path} and {diff_path}", file=sys.stderr)

    if args.emit_only:
        print("[*] --emit-only: run the probe yourself, e.g.:", file=sys.stderr)
        print(f"      ./dram_probe verify --offsets --pairs-file {same_path} "
              f"--hugepage {args.hugepage}", file=sys.stderr)
        return

    if args.simulate:
        model = synthetic.example_model()
        lat_same = time_with_sim(same, model, args.seed)
        lat_diff = time_with_sim(diff, model, args.seed + 1)
    elif args.probe:
        lat_same = time_with_probe(args.probe, same_path, args)
        lat_diff = time_with_probe(args.probe, diff_path, args)
    else:
        sys.exit("provide --probe PATH (hardware) or --simulate (offline)")

    if lat_same.size == 0 or lat_diff.size == 0:
        sys.exit("no latencies measured; check the probe / hugepage setup")

    med_same = float(np.median(lat_same))     # should be the slow (conflict) mode
    med_diff = float(np.median(lat_diff))     # should be the fast mode
    threshold = 0.5 * (med_same + med_diff)

    conflict_rate = float((lat_same >= threshold).mean())
    control_fast_rate = float((lat_diff < threshold).mean())
    passed = conflict_rate >= args.pass_rate and ok_same and ok_diff

    print("\n=== Phase 4 verification ===", file=sys.stderr)
    print(f"  same-bank median latency : {med_same:.0f} cyc", file=sys.stderr)
    print(f"  diff-bank median latency : {med_diff:.0f} cyc", file=sys.stderr)
    print(f"  conflict threshold       : {threshold:.0f} cyc", file=sys.stderr)
    print(f"  same-bank pairs at conflict latency : {conflict_rate*100:.1f}%",
          file=sys.stderr)
    print(f"  diff-bank pairs at fast latency     : {control_fast_rate*100:.1f}%",
          file=sys.stderr)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} "
          f"(need >= {args.pass_rate*100:.0f}% same-bank conflicts)",
          file=sys.stderr)

    report = {
        "functions": [gf2.bits_of(m) for m in masks],
        "n_same_bank": len(same), "n_diff_bank": len(diff),
        "pair_sanity": {"same_bank_ok": ok_same, "diff_bank_ok": ok_diff},
        "median_same_bank": round(med_same, 1),
        "median_diff_bank": round(med_diff, 1),
        "threshold": round(threshold, 1),
        "same_bank_conflict_rate": round(conflict_rate, 4),
        "diff_bank_fast_rate": round(control_fast_rate, 4),
        "passed": passed,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[+] wrote {args.out}", file=sys.stderr)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
