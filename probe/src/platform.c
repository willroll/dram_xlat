/* platform.c — see platform.h for the contract. */
#define _GNU_SOURCE
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/types.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

/* Linux hugepage size selectors (may be missing on older headers). */
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

/* ---------------------------------------------------------------- pagemap */

/* Translate one virtual address to physical via /proc/self/pagemap. Returns 0
 * and sets *phys on success; negative if the PFN is not readable
 * (unprivileged process, page not present). */
static int pagemap_translate(uintptr_t vaddr, uint64_t *phys) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return -1;

    uint64_t entry = 0;
    off_t pos = (off_t)(vaddr / PAGE_4K) * (off_t)sizeof(uint64_t);
    ssize_t n = pread(fd, &entry, sizeof(entry), pos);
    close(fd);
    if (n != (ssize_t)sizeof(entry)) return -1;

    const uint64_t PRESENT = 1ULL << 63;
    const uint64_t PFN_MASK = (1ULL << 55) - 1;
    if (!(entry & PRESENT)) return -1;

    uint64_t pfn = entry & PFN_MASK;
    if (pfn == 0) return -1; /* kernel scrubbed it: we lack CAP_SYS_ADMIN */

    *phys = pfn * PAGE_4K + (vaddr % PAGE_4K);
    return 0;
}

/* ------------------------------------------------------------------ memory */

int mem_alloc(mem_region *r, size_t size, size_t hugepage) {
    memset(r, 0, sizeof(*r));

    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
    if (hugepage == HP_2M) flags |= MAP_HUGETLB | MAP_HUGE_2MB;
    else if (hugepage == HP_1G) flags |= MAP_HUGETLB | MAP_HUGE_1GB;

    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "mmap(size=%zu, hugepage=%zu) failed: %s\n",
                size, hugepage, strerror(errno));
        return -1;
    }

    r->base = (char *)p;
    r->size = size;
    r->hugepage = hugepage;

    /* Fault every 4 KiB in so the mapping is fully resident before timing. */
    for (size_t off = 0; off < size; off += PAGE_4K)
        r->base[off] = (char)off;

    /* Try to learn the true physical base (best effort). */
    uint64_t phys = 0;
    if (pagemap_translate((uintptr_t)r->base, &phys) == 0) {
        r->phys_base = phys;
        r->phys_known = 1;
    } else {
        r->phys_base = 0;
        r->phys_known = 0;
        fprintf(stderr,
            "note: physical base unknown (need root for pagemap). Reporting "
            "in-page offsets; low %d bits are still exact for a single huge "
            "page.\n",
            hugepage == HP_1G ? 30 : hugepage == HP_2M ? 21 : 12);
    }
    return 0;
}

void mem_free(mem_region *r) {
    if (r->base) munmap(r->base, r->size);
    memset(r, 0, sizeof(*r));
}

uint64_t mem_phys(const mem_region *r, size_t offset) {
    return r->phys_known ? (r->phys_base + offset) : (uint64_t)offset;
}

/* ----------------------------------------------------------- host control */

int pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity(cpu=%d) failed: %s\n",
                cpu, strerror(errno));
        return -1;
    }
    return 0;
}

int raise_realtime_priority(void) {
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        fprintf(stderr, "note: SCHED_FIFO unavailable (%s); continuing at "
                        "normal priority.\n", strerror(errno));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------- prefetcher control
 *
 * Disabling hardware prefetchers sharpens the bimodal latency separation. The
 * MSR and bit layout are vendor/model specific:
 *
 *   Intel  MSR 0x1A4 (MISC_FEATURE_CONTROL): bits [3:0]=1 disable the four
 *          prefetchers. Documented and stable across Core/Xeon.
 *
 *   AMD    Family 17h+ prefetch control is NOT uniform across models and a
 *          wrong write can hang the core. We deliberately do NOT hardcode a
 *          guess. Supply the exact register from your model's PPR via the env
 *          vars below, or disable prefetchers in firmware instead.
 *
 * Override for any vendor:
 *      DRAM_PREFETCH_MSR   e.g. 0x1A4   (hex or decimal)
 *      DRAM_PREFETCH_MASK  bits to SET to DISABLE, e.g. 0xF
 */

static int vendor_is_intel(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned a, b, c, d;
    if (!__get_cpuid(0, &a, &b, &c, &d)) return 0;
    /* "GenuineIntel": ebx=0x756e6547, edx=0x49656e69, ecx=0x6c65746e */
    return b == 0x756e6547u && d == 0x49656e69u && c == 0x6c65746eu;
#else
    return 0;
#endif
}

static int msr_read(int cpu, uint32_t reg, uint64_t *val) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int rc = (pread(fd, val, sizeof(*val), reg) == (ssize_t)sizeof(*val)) ? 0 : -1;
    close(fd);
    return rc;
}

static int msr_write(int cpu, uint32_t reg, uint64_t val) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -1;
    int rc = (pwrite(fd, &val, sizeof(val), reg) == (ssize_t)sizeof(val)) ? 0 : -1;
    close(fd);
    return rc;
}

int set_prefetchers(int cpu, int enable) {
    uint32_t reg;
    uint64_t mask;

    const char *env_reg = getenv("DRAM_PREFETCH_MSR");
    const char *env_mask = getenv("DRAM_PREFETCH_MASK");

    if (env_reg && env_mask) {
        reg  = (uint32_t)strtoul(env_reg, NULL, 0);
        mask = (uint64_t)strtoull(env_mask, NULL, 0);
    } else if (vendor_is_intel()) {
        reg  = 0x1A4;   /* MSR_MISC_FEATURE_CONTROL */
        mask = 0xF;     /* disable all four prefetchers */
    } else {
        fprintf(stderr,
            "prefetcher control: non-Intel CPU and no override given. Set "
            "DRAM_PREFETCH_MSR / DRAM_PREFETCH_MASK from your model's PPR, or "
            "disable prefetchers in firmware. Skipping.\n");
        return -2;
    }

    uint64_t cur = 0;
    if (msr_read(cpu, reg, &cur) != 0) {
        fprintf(stderr, "prefetcher control: cannot read MSR 0x%X on cpu %d "
                        "(need root + `modprobe msr`). Skipping.\n", reg, cpu);
        return -1;
    }

    uint64_t next = enable ? (cur & ~mask) : (cur | mask);
    if (msr_write(cpu, reg, next) != 0) {
        fprintf(stderr, "prefetcher control: MSR 0x%X write failed. Skipping.\n",
                reg);
        return -1;
    }
    fprintf(stderr, "prefetcher control: MSR 0x%X 0x%llX -> 0x%llX (%s)\n",
            reg, (unsigned long long)cur, (unsigned long long)next,
            enable ? "enabled" : "disabled");
    return 0;
}
