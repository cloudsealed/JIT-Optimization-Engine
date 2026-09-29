"""
GCP-calibrated realism benchmark (companion to masking_benchmark.py).

The three scenarios in masking_benchmark.py use hand-chosen synthetic
parameters to isolate each design decision. This benchmark instead shapes the
synthetic bill from a REAL Google Cloud service mix (gcp_calibrated_profile.json:
service-level proportions only, no account/project identifiers or amounts), so
the structure -- a dominant always-on database plus smaller deploy/traffic-
driven services -- is real rather than invented.

Honesty boundary: only the *structure* (service mix, steady-vs-variable split)
comes from real data. The weekly seasonality and the labeled anomalies are
INJECTED synthetically, because the real export contains neither; that is what
keeps the ground truth exact and the result reproducible. Nothing here is
presented as a real anomaly found in real spend.

A realistic consequence of the real mix: the weekend dip applies only to the
~21% of spend that is deploy/traffic driven; the ~79% database floor stays on
all week. That makes the seasonality subtler (and the benchmark harder) than a
bill modeled as uniformly seasonal.

Extra deps (same as sota_comparison.py): pip install -e ".[bench]"

Run:
    python benchmarks/gcp_calibrated_benchmark.py            # N=300
    python benchmarks/gcp_calibrated_benchmark.py --runs 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np

from benchmarks.masking_benchmark import (
    score, detect_textbook_flat_std, detect_shipped, Series,
)
from benchmarks.sota_comparison import (
    detect_isolation_forest, detect_stl_3sigma, detect_shesd,
)

_PROFILE = os.path.join(os.path.dirname(__file__), "gcp_calibrated_profile.json")
with open(_PROFILE, encoding="utf-8") as _fh:
    _prof = json.load(_fh)
P_STEADY = sum(v for k, v in _prof["steady_services"].items() if not k.startswith("_"))
P_VARIABLE = sum(v for k, v in _prof["variable_services"].items() if not k.startswith("_"))

METHODS = ["Textbook", "Isolation Forest", "STL + 3-sigma",
           "Seasonal-Hybrid ESD", "Proposed"]


def make_calibrated(seed, n_days=90, base=1000.0, weekend_drop=0.55,
                    trend=4.0, spikes=None):
    """Daily cost = steady floor (flat, low noise) + variable top (weekday
    cycle, higher noise), with the steady/variable split taken from the real
    GCP profile. `base` scales magnitude only (F1 is scale-invariant)."""
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 5)  # a Monday
    days = [start + timedelta(days=i) for i in range(n_days)]
    costs = np.empty(n_days)
    for i, d in enumerate(days):
        level = base + trend * i
        steady = level * P_STEADY * (1 + rng.normal(0, 0.02))
        wk = 1.0 if d.weekday() < 5 else (1.0 - weekend_drop)
        variable = level * P_VARIABLE * wk * (1 + rng.normal(0, 0.08))
        costs[i] = max(steady + variable, 0.0)
    true = set()
    for idx, mult in (spikes or []):
        costs[idx] *= mult
        true.add(idx)
    return Series(days=days, costs=costs, true_anomalies=true)


def _rand_positions(rng, n, k, margin=5, gap=4):
    while True:
        p = sorted(rng.choice(range(margin, n - margin), size=k, replace=False))
        if all(p[j + 1] - p[j] >= gap for j in range(len(p) - 1)):
            return [int(x) for x in p]


def _f1s(seed):
    rng = np.random.default_rng(seed)
    pos = _rand_positions(rng, 90, 6)
    mags = [rng.uniform(2.5, 6.5) for _ in range(6)]
    s = make_calibrated(seed, spikes=list(zip(pos, mags)))
    t = s.true_anomalies
    return {
        "Textbook":            score(detect_textbook_flat_std(s.costs), t).f1,
        "Isolation Forest":    score(detect_isolation_forest(s.costs), t).f1,
        "STL + 3-sigma":       score(detect_stl_3sigma(s.costs), t).f1,
        "Seasonal-Hybrid ESD": score(detect_shesd(s.costs), t).f1,
        "Proposed":            score(detect_shipped(s), t).f1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)

    print("cloudsealed-jit — GCP-calibrated end-to-end benchmark")
    print(f"(steady floor {P_STEADY:.1%} of spend, deploy/traffic-driven "
          f"{P_VARIABLE:.1%}; seasonality + anomalies injected)")
    print("=" * 78)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, 10_000_000, size=args.runs)
    acc = {m: [] for m in METHODS}
    for sd in seeds:
        r = _f1s(int(sd))
        for m in METHODS:
            acc[m].append(r[m])

    print(f"\nEnd-to-end (N={args.runs} randomized runs)")
    print(f"  {'method':<22}{'mean F1':>9}{'std':>8}{'p2.5':>8}{'p97.5':>8}")
    for m in METHODS:
        a = np.array(acc[m])
        print(f"  {m:<22}{a.mean():>9.3f}{a.std():>8.3f}"
              f"{np.percentile(a, 2.5):>8.3f}{np.percentile(a, 97.5):>8.3f}")
    prop = np.array(acc["Proposed"])
    others = np.vstack([np.array(acc[m]) for m in METHODS if m != "Proposed"])
    print(f"  -> Proposed >= all alternatives in "
          f"{np.mean(prop >= others.max(axis=0)) * 100:.1f}% of runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
