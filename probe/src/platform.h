/* platform.h — host-control + physical-memory acquisition for the probe.
 *
 * The solver needs to reason about *physical* address bits, but user space
 * hands out virtual addresses. Two mechanisms bridge the gap:
 *
 *   1. Huge pages. A 1 GiB (or 2 MiB) huge page is physically contiguous, so
 *      the low 30 (or 21) bits of an offset inside the page ARE the low bits
 *      of the physical address. Since DRAM bank/rank/channel functions live in
 *      roughly bits 6..30, one 1 GiB page is enough to observe every relevant
 *      bit with zero kernel help.
 *
 *   2. /proc/self/pagemap. When we want the *absolute* physical base of the
 *      page (to report true physical addresses rather than in-page offsets),
 *      we translate the page's virtual base once. This needs CAP_SYS_ADMIN on
 *      modern kernels (unprivileged readers get a zeroed PFN).
 *
 * Also here: pin to one core, optionally raise scheduling priority, and an
 * optional model-specific prefetcher disable (off by default — see notes).
 */
#ifndef DRAM_XLAT_PLATFORM_H
#define DRAM_XLAT_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

#define PAGE_4K   ((size_t)4096)
#define HP_2M     ((size_t)(2u  * 1024 * 1024))
#define HP_1G     ((size_t)(1024u * 1024 * 1024))

typedef struct {
    char    *base;          /* virtual base of the mapping                */
    size_t   size;          /* mapping size in bytes                      */
    size_t   hugepage;      /* 0 = 4K pages, HP_2M, or HP_1G              */
    uint64_t phys_base;     /* physical base if known, else 0 (unknown)   */
    int      phys_known;    /* 1 if phys_base came from pagemap           */
} mem_region;

/* Allocate `size` bytes. `hugepage` selects the backing page size (0, HP_2M,
 * HP_1G). Returns 0 on success. On success the region is faulted in and, if
 * possible, its physical base resolved via pagemap. */
int  mem_alloc(mem_region *r, size_t size, size_t hugepage);
void mem_free(mem_region *r);

/* Physical address of an in-region offset. If the physical base is unknown
 * this returns the offset itself (which still carries the correct *relative*
 * low bits inside a single huge page — sufficient for the solver). */
uint64_t mem_phys(const mem_region *r, size_t offset);

/* Pin the calling thread to `cpu`. Returns 0 on success. */
int  pin_to_cpu(int cpu);

/* Best-effort SCHED_FIFO real-time priority to cut scheduler noise. Returns 0
 * on success, negative if it could not be set (non-fatal). */
int  raise_realtime_priority(void);

/* Optional, model-specific, OFF BY DEFAULT. Toggle hardware prefetchers via
 * MSR on `cpu`. `enable`=0 disables. Returns 0 on success, negative on
 * failure (no per-cpu msr device, not root, unknown vendor). The exact MSR
 * and bit layout are validated against the CPU vendor; see platform.c. */
int  set_prefetchers(int cpu, int enable);

#endif /* DRAM_XLAT_PLATFORM_H */
