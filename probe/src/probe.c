/* dram_probe — the acquisition harness (Plan phases 1 & 2).
 *
 * Three sub-commands, all emitting the same 3-column CSV on stdout:
 *
 *     addr_a,addr_b,latency
 *
 * where addr_* are physical addresses (or in-page offsets if pagemap is not
 * available) and latency is the row-buffer measurement in reference cycles.
 *
 *   sweep   diagnostic: single-bit toggles + a linear stride scan, so you can
 *           eyeball the bimodal hit/conflict structure.
 *   pairs   the workhorse: N random address pairs -> the solver's dataset.
 *   verify  read a list of address pairs and report how many hit the slow
 *           (conflict) mode — used by Phase 4 to check a derived model.
 *
 * Every run prints a latency histogram summary to stderr so you can confirm
 * the two modes are cleanly separated before trusting the data.
 */
#define _GNU_SOURCE
#include "timing.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* --------------------------------------------------------------- options */

typedef struct {
    const char *cmd;
    size_t   hugepage;      /* 0, HP_2M, HP_1G                  */
    size_t   size;          /* mapping size (0 => derive)       */
    size_t   reps;          /* measurements per pair            */
    size_t   align;         /* address granularity (bytes)      */
    int      cpu;
    int      use_min;       /* 1 => min statistic, 0 => median  */
    int      realtime;      /* SCHED_FIFO                       */
    int      no_prefetch;   /* disable HW prefetchers           */
    unsigned seed;
    int      bit_lo, bit_hi;/* sweep single-bit range           */
    size_t   num_pairs;     /* pairs / scan length              */
    uint64_t threshold;     /* verify: conflict cutoff (cycles) */
    const char *pairs_file; /* verify: input pairs              */
    int      offsets;       /* verify: inputs are buffer offsets */
} opts;

static void usage(const char *p) {
    fprintf(stderr,
      "usage: %s <sweep|pairs|verify> [options]\n"
      "  --hugepage {0|2M|1G}   backing page size (default 1G)\n"
      "  --size BYTES           mapping size (default = hugepage or 256M)\n"
      "  --reps N               measurements per pair (default 64)\n"
      "  --align BYTES          address granularity (default 64)\n"
      "  --cpu N                pin to this core (default 0)\n"
      "  --stat {min|median}    per-pair reduction (default median)\n"
      "  --rt                   request SCHED_FIFO priority\n"
      "  --disable-prefetch     toggle HW prefetchers off (see platform.c)\n"
      "  --seed N               RNG seed (default 1)\n"
      "  --bit-lo N --bit-hi N  sweep: single-bit toggle range (default 6..30)\n"
      "  --num-pairs N          pairs: dataset size / sweep scan length\n"
      "  --threshold CYC        verify: conflict cutoff cycles\n"
      "  --pairs-file PATH      verify: 'addr_a,addr_b' per line ('-'=stdin)\n"
      "  --offsets              verify: inputs are buffer offsets, not phys addrs\n",
      p);
}

static size_t parse_hp(const char *s) {
    if (!strcmp(s, "1G")) return HP_1G;
    if (!strcmp(s, "2M")) return HP_2M;
    return 0;
}

static int parse_opts(int argc, char **argv, opts *o) {
    memset(o, 0, sizeof(*o));
    o->hugepage = HP_1G;
    o->reps = 64;
    o->align = 64;
    o->cpu = 0;
    o->use_min = 0;
    o->seed = 1;
    o->bit_lo = 6;
    o->bit_hi = 30;
    o->num_pairs = 0;
    o->threshold = 0;
    o->pairs_file = NULL;

    if (argc < 2) { usage(argv[0]); return -1; }
    o->cmd = argv[1];

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : (usage(argv[0]), exit(2), ""))
        if      (!strcmp(a, "--hugepage")) o->hugepage = parse_hp(NEXT());
        else if (!strcmp(a, "--size"))     o->size = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--reps"))     o->reps = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--align"))    o->align = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--cpu"))      o->cpu = atoi(NEXT());
        else if (!strcmp(a, "--stat"))     o->use_min = !strcmp(NEXT(), "min");
        else if (!strcmp(a, "--rt"))       o->realtime = 1;
        else if (!strcmp(a, "--disable-prefetch")) o->no_prefetch = 1;
        else if (!strcmp(a, "--seed"))     o->seed = (unsigned)strtoul(NEXT(), NULL, 0);
        else if (!strcmp(a, "--bit-lo"))   o->bit_lo = atoi(NEXT());
        else if (!strcmp(a, "--bit-hi"))   o->bit_hi = atoi(NEXT());
        else if (!strcmp(a, "--num-pairs"))o->num_pairs = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--threshold"))o->threshold = strtoull(NEXT(), NULL, 0);
        else if (!strcmp(a, "--pairs-file"))o->pairs_file = NEXT();
        else if (!strcmp(a, "--offsets"))  o->offsets = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return -1; }
        else { fprintf(stderr, "unknown option: %s\n", a); usage(argv[0]); return -1; }
        #undef NEXT
    }
    if (o->size == 0)
        o->size = o->hugepage ? o->hugepage : (256u * 1024 * 1024);
    if (o->reps == 0) o->reps = 1;
    return 0;
}

/* --------------------------------------------------------- shared helpers */

static uint64_t *g_samples;   /* scratch, sized to reps */

static uint64_t measure(const mem_region *r, size_t off_a, size_t off_b,
                        const opts *o) {
    const volatile char *a = r->base + off_a;
    const volatile char *b = r->base + off_b;
    time_pair_samples(a, b, g_samples, o->reps);
    return o->use_min ? reduce_min(g_samples, o->reps)
                      : reduce_median(g_samples, o->reps);
}

/* Rolling latency histogram, printed to stderr as a sanity check. */
typedef struct { uint64_t lo, hi, n, sum; } hist;
static void hist_init(hist *h) { memset(h, 0, sizeof(*h)); h->lo = ~0ULL; }
static void hist_add(hist *h, uint64_t v) {
    if (v < h->lo) h->lo = v;
    if (v > h->hi) h->hi = v;
    h->n++; h->sum += v;
}
static void hist_report(hist *h, const char *label) {
    if (h->n == 0) { fprintf(stderr, "%s: no samples\n", label); return; }
    fprintf(stderr, "%s: n=%llu  min=%llu  max=%llu  mean=%.1f\n",
            label, (unsigned long long)h->n, (unsigned long long)h->lo,
            (unsigned long long)h->hi, (double)h->sum / (double)h->n);
    fprintf(stderr, "  (feed the CSV to the solver; a clean run shows two "
                    "well-separated latency modes)\n");
}

/* ------------------------------------------------------------- sub-commands */

static void emit(uint64_t a, uint64_t b, uint64_t lat) {
    printf("%llu,%llu,%llu\n", (unsigned long long)a,
           (unsigned long long)b, (unsigned long long)lat);
}

static int cmd_sweep(const mem_region *r, const opts *o) {
    printf("addr_a,addr_b,latency\n");
    hist h; hist_init(&h);

    size_t base = (r->size / 2) & ~(o->align - 1);

    /* Part A: single-bit toggles. A pure row bit toggled alone keeps the bank
     * fixed and changes the row -> conflict (slow). Bank/column bits -> fast.
     * This alone separates row bits from the rest. */
    for (int bit = o->bit_lo; bit <= o->bit_hi; bit++) {
        size_t delta = (size_t)1 << bit;
        if (delta >= r->size) break;
        size_t off_b = base ^ delta;
        if (off_b >= r->size) continue;
        uint64_t lat = measure(r, base, off_b, o);
        hist_add(&h, lat);
        emit(mem_phys(r, base), mem_phys(r, off_b), lat);
    }

    /* Part B: linear stride scan across the buffer. */
    size_t n = o->num_pairs ? o->num_pairs : 4096;
    for (size_t i = 1; i <= n; i++) {
        size_t off_b = (base + i * o->align);
        if (off_b >= r->size) break;
        uint64_t lat = measure(r, base, off_b, o);
        hist_add(&h, lat);
        emit(mem_phys(r, base), mem_phys(r, off_b), lat);
    }

    fflush(stdout);
    hist_report(&h, "sweep");
    return 0;
}

static int cmd_pairs(const mem_region *r, const opts *o) {
    printf("addr_a,addr_b,latency\n");
    hist h; hist_init(&h);

    size_t n = o->num_pairs ? o->num_pairs : 1000000;
    size_t slots = r->size / o->align;
    unsigned seed = o->seed;

    for (size_t i = 0; i < n; i++) {
        size_t ia = (size_t)rand_r(&seed) % slots;
        size_t ib = (size_t)rand_r(&seed) % slots;
        if (ia == ib) { ib = (ib + 1) % slots; }
        size_t off_a = ia * o->align;
        size_t off_b = ib * o->align;
        uint64_t lat = measure(r, off_a, off_b, o);
        hist_add(&h, lat);
        emit(mem_phys(r, off_a), mem_phys(r, off_b), lat);
        if ((i & 0x3FFFF) == 0x3FFFF)
            fprintf(stderr, "  ... %zu / %zu pairs\n", i + 1, n);
    }
    fflush(stdout);
    hist_report(&h, "pairs");
    return 0;
}

/* verify: input pairs are PHYSICAL addresses (or offsets with --offsets) that
 * the model predicts collide. We map them back into the buffer. */
static int cmd_verify(const mem_region *r, const opts *o) {
    if (!o->pairs_file) { fprintf(stderr, "verify: --pairs-file required\n"); return -1; }
    FILE *f = (!strcmp(o->pairs_file, "-")) ? stdin : fopen(o->pairs_file, "r");
    if (!f) { fprintf(stderr, "verify: cannot open %s\n", o->pairs_file); return -1; }

    printf("addr_a,addr_b,latency\n");
    hist h; hist_init(&h);
    /* inputs are either physical addresses (subtract the mapping's phys base)
     * or raw buffer offsets when --offsets is given. */
    uint64_t phys_base = (o->offsets || !r->phys_known) ? 0 : r->phys_base;
    size_t hits = 0, total = 0;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == 'a') continue; /* skip header/comments */
        unsigned long long pa, pb;
        if (sscanf(line, "%llu,%llu", &pa, &pb) != 2) continue;
        uint64_t off_a = (uint64_t)pa - phys_base;
        uint64_t off_b = (uint64_t)pb - phys_base;
        if (off_a >= r->size || off_b >= r->size) {
            fprintf(stderr, "verify: pair out of range (%llu,%llu) — physical "
                    "base mismatch?\n", pa, pb);
            continue;
        }
        uint64_t lat = measure(r, off_a & ~(o->align - 1), off_b & ~(o->align - 1), o);
        hist_add(&h, lat);
        emit(pa, pb, lat);
        total++;
        if (o->threshold && lat >= o->threshold) hits++;
    }
    if (f != stdin) fclose(f);
    fflush(stdout);
    hist_report(&h, "verify");
    if (o->threshold && total)
        fprintf(stderr, "verify: %zu/%zu (%.1f%%) at/above conflict threshold "
                "%llu cyc\n", hits, total, 100.0 * hits / total,
                (unsigned long long)o->threshold);
    return 0;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv) {
    opts o;
    if (parse_opts(argc, argv, &o) != 0) return 1;

    if (pin_to_cpu(o.cpu) != 0)
        fprintf(stderr, "warning: could not pin to cpu %d\n", o.cpu);
    if (o.realtime) raise_realtime_priority();
    if (o.no_prefetch) set_prefetchers(o.cpu, 0);

    mem_region r;
    if (mem_alloc(&r, o.size, o.hugepage) != 0) {
        if (o.hugepage) {
            fprintf(stderr, "retrying with 4K pages (reserve hugepages with "
                            "scripts/prepare_target.sh for best results)\n");
            if (mem_alloc(&r, o.size, 0) != 0) return 1;
        } else return 1;
    }
    fprintf(stderr, "mapping: %zu bytes, hugepage=%zu, phys_base=%s0x%llx\n",
            r.size, r.hugepage, r.phys_known ? "" : "(offset) ",
            (unsigned long long)r.phys_base);

    g_samples = malloc(o.reps * sizeof(uint64_t));
    if (!g_samples) { perror("malloc"); mem_free(&r); return 1; }

    int rc;
    if      (!strcmp(o.cmd, "sweep"))  rc = cmd_sweep(&r, &o);
    else if (!strcmp(o.cmd, "pairs"))  rc = cmd_pairs(&r, &o);
    else if (!strcmp(o.cmd, "verify")) rc = cmd_verify(&r, &o);
    else { fprintf(stderr, "unknown command: %s\n", o.cmd); usage(argv[0]); rc = -1; }

    if (o.no_prefetch) set_prefetchers(o.cpu, 1); /* restore */
    free(g_samples);
    mem_free(&r);
    return rc == 0 ? 0 : 1;
}
