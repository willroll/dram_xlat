"""Turn a noisy 1-D latency distribution into binary conflict/hit labels.

Row-buffer conflicts and non-conflicts form a bimodal latency distribution; on
top of that sit rare, *much* slower events — page faults, TLB misses, context
switches — that are microsecond-scale, an order of magnitude above a row
conflict. Those are not conflicts and must not pollute the solver, so the
classifier:

  1. locates the fast (hit) mode robustly (it dominates): median + MAD.
  2. separates the slow tail into the conflict cluster vs the fault cluster.
     The conflict cluster is *dense* (narrow sigma); faults are *sparse*
     (smeared over thousands of cycles). So the conflict mode is the
     highest-count histogram bin of the slow tail, and the fault cutoff is the
     first near-empty bin above it — robust even when faults outnumber
     conflicts.
  3. labels pairs between the two modes' midpoint and the fault cutoff as
     conflicts; above the cutoff = discarded fault; below the midpoint = hit.

A Gaussian-mixture variant (method='gmm') is offered for well-separated,
higher-prevalence data; the robust method is the default because conflict pairs
are typically a small minority (P(same bank) is small).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Classification:
    is_conflict: np.ndarray     # bool[N]: True = row-buffer conflict (slow)
    confidence: np.ndarray      # float[N] in [0,1]
    threshold: float            # midpoint latency separating the two modes
    mu_hit: float               # fast-mode location
    mu_conflict: float          # slow-mode location
    separation: float           # (mu_conflict - mu_hit) / sigma_hit
    method: str
    fault_cutoff: float = float("inf")   # above this = discarded fault
    n_fault: int = 0            # pairs discarded as faults


def _mad_sigma(x: np.ndarray, center: float) -> float:
    """Robust std estimate via median absolute deviation."""
    mad = float(np.median(np.abs(x - center)))
    return max(1.4826 * mad, 1e-9)


def _slow_split(sv: np.ndarray, sigma_hit: float,
                fault_sigmas: float) -> tuple[float, float, float]:
    """Given the slow-tail latencies, return (mu_conflict, sigma_conflict,
    fault_cutoff).

    The slow tail holds two populations with very different densities: a tight
    conflict cluster just above the hit mode, and sparse faults far higher. The
    conflict mode is therefore the highest-count histogram bin; the fault
    cutoff is the first near-empty bin above it.
    """
    bw = max(0.5 * sigma_hit, 1.0)
    top = float(sv.max())
    lo = float(sv.min())
    edges = np.arange(lo, top + 2 * bw, bw)
    if edges.size < 3:
        med_s = float(np.median(sv))
        s = _mad_sigma(sv, med_s)
        return med_s, s, med_s + fault_sigmas * s

    hist, edges = np.histogram(sv, bins=edges)
    peak = int(np.argmax(hist))
    peak_count = float(hist[peak])
    mu_conflict = 0.5 * (edges[peak] + edges[peak + 1])

    near = sv[(sv >= mu_conflict - 3 * bw) & (sv <= mu_conflict + 3 * bw)]
    sigma_conflict = max(float(near.std()) if near.size > 2 else sigma_hit, 1.0)

    # valley: first bin above the peak whose count collapses -> fault boundary
    floor = max(2.0, 0.02 * peak_count)
    fault_cutoff = top + bw
    for k in range(peak + 1, len(hist)):
        if hist[k] <= floor:
            fault_cutoff = float(edges[k])
            break
    fault_cutoff = max(fault_cutoff, mu_conflict + fault_sigmas * sigma_conflict)
    return mu_conflict, sigma_conflict, fault_cutoff


def _robust(latency: np.ndarray, slow_sigmas: float,
            fault_sigmas: float, min_slow: int) -> Classification:
    med = float(np.median(latency))
    sigma_hit = _mad_sigma(latency, med)

    slow0 = latency >= med + slow_sigmas * sigma_hit
    if int(slow0.sum()) < min_slow:
        return _quantile(latency, 0.05)

    sv = latency[slow0]
    mu_conflict, sigma_conflict, fault_cutoff = _slow_split(sv, sigma_hit,
                                                            fault_sigmas)

    # decision threshold = midpoint between the two modes (near-optimal for
    # roughly equal-variance Gaussians)
    threshold = 0.5 * (med + mu_conflict)

    is_conflict = (latency >= threshold) & (latency <= fault_cutoff)
    n_fault = int((latency > fault_cutoff).sum())

    conf = np.zeros_like(latency, dtype=float)
    span_hi = max(mu_conflict - threshold, 1e-9)
    span_lo = max(threshold - med, 1e-9)
    conf[is_conflict] = np.clip((latency[is_conflict] - threshold) / span_hi, 0, 1)
    hit = latency < threshold
    conf[hit] = np.clip((threshold - latency[hit]) / span_lo, 0, 1)

    sep = (mu_conflict - med) / sigma_hit
    return Classification(is_conflict, conf, threshold, med, mu_conflict,
                          sep, "robust", fault_cutoff, n_fault)


def _gmm(latency: np.ndarray) -> Optional[Classification]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None
    x = latency.reshape(-1, 1).astype(float)
    gm = GaussianMixture(n_components=2, covariance_type="full",
                         n_init=4, random_state=0).fit(x)
    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    slow = int(np.argmax(means))
    fast = 1 - slow
    post = gm.predict_proba(x)
    p_conf = post[:, slow]
    is_conflict = p_conf >= 0.5
    conf = np.where(is_conflict, p_conf, 1.0 - p_conf)
    grid = np.linspace(latency.min(), latency.max(), 4096).reshape(-1, 1)
    gp = gm.predict_proba(grid)[:, slow]
    cross = float(grid[np.argmin(np.abs(gp - 0.5))].item())
    pooled = float(np.sqrt((stds[slow] ** 2 + stds[fast] ** 2) / 2.0)) or 1.0
    sep = (means[slow] - means[fast]) / pooled
    return Classification(is_conflict, conf, cross, float(means[fast]),
                          float(means[slow]), float(sep), "gmm")


def _quantile(latency: np.ndarray, hi_frac: float) -> Classification:
    thr = float(np.quantile(latency, 1.0 - hi_frac))
    is_conflict = latency >= thr
    mu_hit = float(latency[~is_conflict].mean()) if (~is_conflict).any() else 0.0
    mu_con = float(latency[is_conflict].mean()) if is_conflict.any() else 0.0
    pooled = float(latency.std()) or 1.0
    conf = np.clip(np.abs(latency - thr) / pooled, 0, 1)
    return Classification(is_conflict, conf, thr, mu_hit, mu_con,
                          (mu_con - mu_hit) / pooled, "quantile")


def classify_latencies(latency: np.ndarray, method: str = "auto", *,
                       slow_sigmas: float = 3.5, fault_sigmas: float = 5.0,
                       min_slow: int = 30, hi_frac: float = 0.35) -> Classification:
    """Label each pair conflict (slow) vs hit (fast).

    method: 'auto'/'robust' (default, MAD + density-peak), 'gmm', or 'quantile'.
    """
    latency = np.asarray(latency, dtype=float)
    if method in ("auto", "robust"):
        return _robust(latency, slow_sigmas, fault_sigmas, min_slow)
    if method == "gmm":
        c = _gmm(latency)
        if c is None:
            raise RuntimeError("scikit-learn required for method='gmm'")
        return c
    return _quantile(latency, hi_frac)
