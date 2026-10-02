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

# 3. discover the functions (anywhere)
pip install -r requirements.txt
python3 solver/solve.py dataset.csv --out functions.json

# 4. verify against the hardware
python3 verify/verify.py functions.json --probe ./probe/dram_probe --hugepage 1G
```

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
