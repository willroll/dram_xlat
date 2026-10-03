# dram_xlat — Automated DRAM Address-Translation Discovery

A toolkit that reverse-engineers the *undocumented* physical-address → DRAM-coordinate
translation a memory controller uses, purely from software, via **row-buffer conflict
timing** as a side channel.

The goal is to recover the GF(2) (XOR) functions the Unified Memory Controller (UMC)
applies to map a physical address to `(Channel, Rank, Bank Group, Bank, Row, Column)`.
Once recovered, those functions predict, for any two physical addresses, whether they
collide in the same bank — the foundational primitive for open hardware analysis,
memory-performance characterization, DRAM-topology diagnostics, and defensive
(Rowhammer-mitigation) research.

This is a *measurement / characterization* tool. It observes access latency; it does
not hammer memory, flip bits, or exploit anything. The method is the published
technique from **Pessl, Gruss, Maurice, Schwarz, Mangard — "DRAMA: Exploiting DRAM
Addressing for Cross-CPU Attacks", USENIX Security 2016**, generalized here to a
noise-tolerant solver.

> ### Scope & responsible use
> DRAM addressing functions are hardware facts about the machine you run this on. The
> toolkit only *reads* local timing; it needs no special target and moves no data
> off-box. Run it on hardware you own or are authorized to characterize. The optional
> MSR prefetcher-control path writes a model-specific register and is **off by
> default** — read `scripts/prepare_target.sh` before enabling it.

---

## Architecture

The pipeline mirrors the four phases of the execution plan:

```
   Phase 1 + 2   probe/ (C)      bare-metal timing harness: rdtscp + fences +
   acquisition                   clflush, hugepage-backed contiguous memory, core
                                 pinning. Emits CSV:  addr_a,addr_b,latency_cycles
                                        │
   Phase 2       orchestrator/sweep.py  drives the probe over strided pair sweeps,
   orchestration                 aggregates millions of pairs into one dataset.
                                        │
   Phase 3       solver/ (Python)       classify latency → conflict/hit, recover XOR
   discovery                     functions via GF(2) null-space + genetic search,
                                 cross-check with Z3. Emits a basis of functions.
                                        │
   Phase 4       verify/verify.py       synthesize addresses that MUST collide under
   verification                  the model, measure them, report the hit rate.
```

### Why it works (one paragraph)

Two physical addresses that map to the **same bank but different rows** force the
controller to close one row and open another on every alternating access — a
*row-buffer conflict* — measurably slower than two addresses in different banks
(served in parallel). "Same bank" means every bank-selecting XOR function agrees on
both addresses. Writing that agreement as `parity(m · a) == parity(m · b)`, i.e.
`m · (a ⊕ b) == 0`, turns the problem into linear algebra over GF(2): the
bank-selecting functions are exactly the **null space** of the matrix of difference
vectors `a ⊕ b` over all *conflict* (slow) pairs. Timing noise mislabels some pairs,
so the exact null-space solver is backed by a genetic search tolerant of a
configurable error rate.

---

## Layout

| Path | Phase | What it is |
|------|-------|-----------|
| `probe/`               | 1 & 2 | C timing harness (`dram_probe`) + build files |
| `orchestrator/sweep.py`| 2     | drives the probe over pair sweeps, builds the dataset |
| `solver/dramxlat/`     | 3     | Python package: classify, genetic, symbolic, synthetic, pipeline |
| `solver/solve.py`      | 3     | CLI: dataset.csv → functions.json |
| `solver/selftest.py`   | 3     | end-to-end recovery test on noisy synthetic data (no hardware) |
| `verify/verify.py`     | 4     | targeted-pair generation + hit-rate report |
| `docs/index.html`      | —     | self-contained dashboard visualizing a solver run (open in a browser) |
| `scripts/prepare_target.sh` | — | hugepages / core isolation / prefetcher notes for the target |
| `Makefile`             | —     | `make build` / `make test` / `make clean` convenience |
| `requirements.txt`     | —     | Python deps (numpy, scikit-learn, z3-solver) |

---

## Quick start

```sh
# 0. prepare the target (once, as root): reserve hugepages, print isolation guidance
sudo scripts/prepare_target.sh

# 1. build the probe
cd probe && make            # or: cmake -B build && cmake --build build

# 2. acquire a dataset on the Zen target
sudo ./probe/dram_probe pairs --hugepage 1G --reps 64 --num-pairs 2000000 > dataset.csv
#    (or let the orchestrator manage the sweeps:)
#    python3 orchestrator/sweep.py --probe ./probe/dram_probe --out dataset.csv

# 3. discover the functions (anywhere) — and get a visual report of YOUR run
pip install -r requirements.txt
python3 solver/solve.py dataset.csv --out functions.json \
        --report report.html --label "my-machine"

# 4. verify against the hardware
python3 verify/verify.py functions.json --probe ./probe/dram_probe --hugepage 1G
```

### Report — graphics for your own RAM

Pass `--report report.html` to `solve.py` and it writes a **self-contained HTML
report of your run** (no build step, no network, no view-time dependencies) —
open it in any browser. It shows the headline numbers, your measured bimodal
latency distribution, how your physical address decodes (which bits select a
bank vs. row/column), the recovered XOR functions and their structure, and
whether the three independent solver routes agree (a confidence check). Add a
single-bit row sweep panel by also collecting one:

```sh
sudo ./probe/dram_probe sweep --hugepage 1G --reps 64 > sweep.csv
python3 solver/solve.py dataset.csv --report report.html --sweep sweep.csv
```

Unlike the demo below, the report carries no ground-truth labels: on real
hardware you recover the bank-selecting XOR functions, but naming them (rank vs
bank group) needs correlating with your DIMM topology.

### Dashboard (synthetic demo)

`docs/index.html` is a self-contained page that visualizes a solver run on a
*known* synthetic map — the same panels as the report, plus a robustness sweep
showing how each solver route holds up as the timing modes get harder to
separate. Open it directly in a browser, or serve `docs/` with GitHub Pages.

### Validate the analysis without hardware

The solver is fully testable on any machine — it recovers a *known* DRAM function from
a noisy synthetic truth table:

```sh
make test        # or: python3 solver/selftest.py
```

---

## Iterative workflow

Per the execution plan this runs as a loop: build & run Phase 1 on the target, feed
the latency logs back, refine the timing loop or solver parameters from the empirical
distribution. The probe prints a latency **histogram summary** to stderr on every run
so you can confirm the bimodal hit/conflict separation before trusting the solver; if
the two modes are not cleanly separated, increase `--reps`, pin to an isolated core,
and disable prefetchers (see `scripts/prepare_target.sh`).

---

## Targets: Intel vs AMD

The probe is x86-64 generic (`rdtscp` + `mfence`/`lfence` + `clflush`); the solver and
verifier are architecture-agnostic (they consume the CSV). The only vendor-specific
piece is prefetcher control, which is hardcoded for Intel and env-overridable for AMD.

### Intel (incl. Xeon E5-1650 v4, Broadwell-EP) — the better-supported path

- **Prefetcher disable works natively.** `--disable-prefetch` writes `MSR 0x1A4`
  (MISC_FEATURE_CONTROL), bits `[3:0]` = L2 stream, L2 adjacent-line, L1 DCU, and
  DCU-IP prefetchers. Correct for Nehalem through at least Skylake/Broadwell.
- **No `clflushopt` on Broadwell** (Skylake+ only) — `timing.h` `#ifdef`-guards it and
  falls back to `clflush`, so it just works. Build on the box with `-march=native`, or
  cross-build with `make CFLAGS="-O2 -march=broadwell"`.
- **Invariant TSC** (`constant_tsc`/`nonstop_tsc`): `rdtscp` counts are stable across
  frequency changes; still fix the frequency (disable Turbo) for a clean split.
- **Quad-channel → you also recover a *channel* hash.** Intel interleaves channels with
  an XOR of several physical bits; the solver returns it as extra functions in the
  basis. What you recover depends on how many DIMMs/channels you populate and the BIOS
  interleave settings. Single socket (E5-1650 v4) keeps it to one controller's map.
- **Use 1 GiB hugepages, reserved at boot.** Runtime 1 GiB reservation usually fails
  from fragmentation, so add to the kernel cmdline and reboot:
  `default_hugepagesz=1G hugepagesz=1G hugepages=8`. (2 MiB only covers bits 0–20 — too
  low for channel/rank bits.)
- **Ground truth exists:** DRAMA's published Intel results are from this Core/Xeon-E5
  era, so you can sanity-check the recovered functions.

Recommended run on an E5-1650 v4:

```sh
cd probe && make                               # -march=native → broadwell, clflush fallback
sudo ../scripts/prepare_target.sh              # reserve hugepages, print isolation steps
sudo ./dram_probe pairs --hugepage 1G --reps 64 --num-pairs 2000000 \
     --cpu 3 --rt --disable-prefetch > dataset.csv
python3 ../solver/solve.py dataset.csv --out functions.json
python3 ../verify/verify.py functions.json --probe ./dram_probe --hugepage 1G
```

Not in scope: Intel's LLC **cache-slice** hash is a *different* undocumented function
(cache, not DRAM banks) — same timing-methodology family, but this tool targets DRAM.

### AMD (Zen 17h+)

Same flow, except prefetcher control is model-specific. Disable prefetchers in firmware,
or pass the register from your model's PPR:
`DRAM_PREFETCH_MSR=0x... DRAM_PREFETCH_MASK=0x... ./dram_probe ... --disable-prefetch`.
Disable Core Performance Boost in firmware (no `intel_pstate/no_turbo` knob).
