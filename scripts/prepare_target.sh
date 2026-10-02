#!/bin/sh
# prepare_target.sh — get a Zen (or any x86-64) host into a low-noise state for
# DRAM row-conflict timing. Run as root. Everything here is reversible and
# read-mostly; nothing is written to persistent config.
#
#   sudo scripts/prepare_target.sh [--hugepages-1g N] [--hugepages-2m N]
#
# It (1) reserves huge pages, (2) prints the isolation / prefetcher / governor
# steps you should apply, and (3) reports what is currently in effect.

set -eu

NR_1G=4
NR_2M=512

while [ $# -gt 0 ]; do
  case "$1" in
    --hugepages-1g) NR_1G="$2"; shift 2 ;;
    --hugepages-2m) NR_2M="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" != "0" ]; then
  echo "run as root (needed for hugepage reservation and MSR access)" >&2
  exit 1
fi

echo "== 1. Huge pages =="
# 1 GiB pages give physical contiguity across bits 0..29 — enough to observe
# every bank/rank/channel bit with no pagemap help.
if [ -d /sys/kernel/mm/hugepages/hugepages-1048576kB ]; then
  echo "$NR_1G" > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages || true
  have1g=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages)
  echo "  1G hugepages reserved: $have1g (requested $NR_1G)"
else
  echo "  1G hugepages not supported by kernel; falling back to 2M"
fi
if [ -d /sys/kernel/mm/hugepages/hugepages-2048kB ]; then
  echo "$NR_2M" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages || true
  have2m=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
  echo "  2M hugepages reserved: $have2m (requested $NR_2M)"
fi

echo
echo "== 2. MSR access (for optional --disable-prefetch) =="
modprobe msr 2>/dev/null && echo "  msr module loaded" || echo "  msr module unavailable (skip prefetcher control)"

echo
echo "== 3. Manual steps for the cleanest bimodal signal =="
cat <<'EOF'
  * Isolate a core from the scheduler (kernel cmdline, then reboot):
        isolcpus=3 nohz_full=3 rcu_nocbs=3
    then run the probe with --cpu 3 --rt.

  * Pin the CPU to a fixed frequency (disable turbo / set performance governor):
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance > "$g"; done
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo   # Intel
    On AMD, disable Core Performance Boost in firmware or via cpupower.

  * Prefetchers: on Intel the probe handles MSR 0x1A4 with --disable-prefetch.
    On AMD Zen the prefetch-control MSR is model-specific — disable prefetchers
    in firmware, OR pass the register explicitly from your model's PPR:
        DRAM_PREFETCH_MSR=0x... DRAM_PREFETCH_MASK=0x... ./dram_probe ... --disable-prefetch

  * Disable NUMA balancing and THP migration noise:
        echo 0 > /proc/sys/kernel/numa_balancing
EOF

echo
echo "== 4. Current state =="
echo "  governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "  numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"
echo "done. Build the probe:  (cd probe && make)"
