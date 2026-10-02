#!/usr/bin/env python3
"""Phase 2 orchestration: drive the C probe to build a solver dataset.

The probe does the heavy inner loop (millions of timed pairs) in C. This
orchestrator runs it across acquisition passes, concatenates the CSV output into
one dataset, and records a manifest so a run is reproducible.

    python3 orchestrator/sweep.py \\
        --probe ./probe/dram_probe --hugepage 1G \\
        --reps 64 --pairs 2000000 --out dataset.csv
"""
import argparse
import json
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="path to dram_probe binary")
    ap.add_argument("--out", default="dataset.csv", help="aggregated CSV output")
    ap.add_argument("--hugepage", default="1G", choices=["0", "2M", "1G"])
    ap.add_argument("--size", type=str, default=None, help="mapping size in bytes")
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--align", type=int, default=64)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--stat", default="median", choices=["min", "median"])
    ap.add_argument("--rt", action="store_true", help="request SCHED_FIFO")
    ap.add_argument("--disable-prefetch", action="store_true")
    ap.add_argument("--pairs", type=int, default=1_000_000,
                    help="random pairs per acquisition pass")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="one acquisition pass per seed (independent samples)")
    ap.add_argument("--sweep-first", action="store_true",
                    help="also run a diagnostic single-bit sweep pass")
    ap.add_argument("--bit-lo", type=int, default=6)
    ap.add_argument("--bit-hi", type=int, default=30)
    ap.add_argument("--sudo", action="store_true",
                    help="prefix probe invocations with sudo")
    args = ap.parse_args()

    common = [
        "--hugepage", args.hugepage, "--reps", str(args.reps),
        "--align", str(args.align), "--cpu", str(args.cpu), "--stat", args.stat,
    ]
    if args.size:
        common += ["--size", args.size]
    if args.rt:
        common += ["--rt"]
    if args.disable_prefetch:
        common += ["--disable-prefetch"]

    def probe_cmd(mode, extra):
        prefix = ["sudo"] if args.sudo else []
        return prefix + [args.probe, mode] + common + extra

    log_path = args.out + ".log"
    manifest = {
        "probe": args.probe, "hugepage": args.hugepage, "reps": args.reps,
        "align": args.align, "stat": args.stat, "pairs_per_pass": args.pairs,
        "seeds": args.seeds, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "passes": [],
    }

    header_written = False
    t0 = time.time()
    with open(args.out, "w") as out_fh, open(log_path, "w") as log_fh:
        def stream(mode, extra):
            nonlocal header_written
            cmd = probe_cmd(mode, extra)
            print("  $ " + " ".join(cmd), file=sys.stderr)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=log_fh, text=True)
            for line in proc.stdout:
                if line.startswith("addr_a"):
                    if header_written:
                        continue
                    header_written = True
                out_fh.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise SystemExit(f"probe exited {proc.returncode} ({mode})")

        if args.sweep_first:
            print("[*] diagnostic single-bit sweep", file=sys.stderr)
            stream("sweep", ["--bit-lo", str(args.bit_lo), "--bit-hi", str(args.bit_hi)])
            manifest["passes"].append({"mode": "sweep", "bit_lo": args.bit_lo,
                                       "bit_hi": args.bit_hi})

        for seed in args.seeds:
            print(f"[*] random-pair pass seed={seed} ({args.pairs} pairs)",
                  file=sys.stderr)
            stream("pairs", ["--num-pairs", str(args.pairs), "--seed", str(seed)])
            manifest["passes"].append({"mode": "pairs", "seed": seed,
                                       "num_pairs": args.pairs})

    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    manifest["out"] = args.out
    with open(args.out + ".manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)

    n = sum(1 for _ in open(args.out)) - 1
    print(f"[+] wrote {n} rows -> {args.out}", file=sys.stderr)
    print(f"[+] diagnostics -> {log_path}", file=sys.stderr)
    print(f"[+] next: python3 solver/solve.py {args.out} --out functions.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
