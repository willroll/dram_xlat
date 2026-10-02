/* timing.h — nanosecond-precision access-latency primitives (x86-64).
 *
 * Everything here is header-only and force-inlined so the compiler cannot
 * reorder our fences across a call boundary. The measurement of a single
 * address *pair* is the atom the whole toolkit is built on:
 *
 *   flush a, flush b  -> guarantee both come from DRAM, not cache
 *   mfence; lfence    -> drain the store buffer, serialize
 *   t0 = rdtscp       -> read the cycle counter (waits for prior loads)
 *   lfence            -> stop later loads from starting before t0 is latched
 *   load a; load b    -> the two accesses under test
 *   lfence            -> ensure both loads retire before we stop the clock
 *   t1 = rdtscp       -> stop
 *
 * If a and b sit in the same bank on different rows the controller must close
 * a's row and open b's row, so (t1 - t0) is a row-buffer *conflict* and
 * measurably larger than two accesses that hit different banks.
 *
 * We follow Intel's "How to Benchmark Code Execution Times" guidance
 * (rdtsc/rdtscp bracketed by lfence) rather than cpuid-serialization, to keep
 * the measured window as small as possible.
 */
#ifndef DRAM_XLAT_TIMING_H
#define DRAM_XLAT_TIMING_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#else
#error "dram_probe timing primitives require x86 (rdtscp/clflush)."
#endif

/* --- fences ------------------------------------------------------------- */

static inline void fence_full(void) { _mm_mfence(); _mm_lfence(); }
static inline void fence_load(void) { _mm_lfence(); }

/* --- cache eviction ----------------------------------------------------- */

static inline void flush_line(const volatile void *p) {
    _mm_clflush((const void *)p);
}

/* clflushopt is weakly ordered and faster when flushing many lines; callers
 * must fence afterwards. Falls back to clflush when unavailable. */
static inline void flush_line_opt(const volatile void *p) {
#ifdef __CLFLUSHOPT__
    _mm_clflushopt((void *)p);
#else
    _mm_clflush((const void *)p);
#endif
}

/* --- cycle counter ------------------------------------------------------ */

static inline uint64_t rdtscp_now(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

/* --- the core measurement ---------------------------------------------- *
 * Access a then b, from DRAM, and return the elapsed reference cycles. The
 * two addresses are read through volatile pointers so the loads are never
 * elided or hoisted. */
static inline uint64_t time_pair_once(const volatile char *a,
                                      const volatile char *b) {
    uint64_t t0, t1;

    flush_line(a);
    flush_line(b);
    fence_full();

    t0 = rdtscp_now();
    fence_load();

    (void)*a;
    (void)*b;

    fence_load();
    t1 = rdtscp_now();

    return t1 - t0;
}

/* --- robust statistic over many repetitions ---------------------------- *
 * Row-buffer state, scheduling, refresh (tRFC) and TLB effects all add
 * one-sided noise on top of the true DRAM latency. The *minimum* over many
 * repetitions is the cleanest estimator of the structural latency (it is the
 * run least perturbed by noise); the median is a robust middle ground. We
 * expose both and let the caller choose. */

static inline int cmp_u64(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}

/* Fill `samples` with `reps` measurements of the (a,b) pair. Caller owns the
 * buffer (>= reps entries). A few warm-up iterations prime the TLB and branch
 * predictors so they do not bias early samples. */
static inline void time_pair_samples(const volatile char *a,
                                     const volatile char *b,
                                     uint64_t *samples, size_t reps) {
    for (int w = 0; w < 4; w++) (void)time_pair_once(a, b);
    for (size_t i = 0; i < reps; i++) samples[i] = time_pair_once(a, b);
}

static inline uint64_t reduce_min(uint64_t *s, size_t n) {
    uint64_t m = s[0];
    for (size_t i = 1; i < n; i++) if (s[i] < m) m = s[i];
    return m;
}

static inline uint64_t reduce_median(uint64_t *s, size_t n) {
    qsort(s, n, sizeof(uint64_t), cmp_u64);
    return s[n / 2];
}

#endif /* DRAM_XLAT_TIMING_H */
