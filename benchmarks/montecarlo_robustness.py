"""
Monte Carlo robustness study (companion to sota_comparison.py).

sota_comparison.py runs each detector once per scenario. A reviewer will
rightly ask whether a single seed is representative. This script re-runs the
three scenarios over N independent randomizations, random spike positions,
jittered magnitudes, and fresh noise on every run, and reports mean F1,
standard deviation, and a 95% interval for every detector, plus the fraction
of runs in which the shipped method matches or beats every alternative.

Extra dependencies (same as sota_comparison.py):
    pip install scikit-learn statsmodels scipy

Run:
    python benchmarks/montecarlo_robustness.py            # default N=300
    python benchmarks/montecarlo_robustness.py --runs 500
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from benchmarks.masking_benchmark import (
    make_series, score, detect_textbook_flat_std, detect_shipped,
)
from benchmarks.sota_comparison import (
    detect_isolation_forest, detect_stl_3sigma, detect_shesd,
)

METHODS = ["Textbook", "Isolation Forest", "STL + 3-sigma",
           "Seasonal-Hybrid ESD", "Proposed"]


def _rand_positions(rng, n_days, k, margin=5, min_gap=4):
    while True:
        pos = sorted(rng.choice(range(margin, n_days - margin), size=k, replace=False))
        if all(pos[i + 1] - pos[i] >= min_gap for i in range(len(pos) - 1)):
            return [int(p) for p in pos]


def _f1s(s):
    truth = s.true_anomalies
    return {
        "Textbook":            score(detect_textbook_flat_std(s.costs), truth).f1,
        "Isolation Forest":    score(detect_isolation_forest(s.costs), truth).f1,
        "STL + 3-sigma":       score(detect_stl_3sigma(s.costs), truth).f1,
        "Seasonal-Hybrid ESD": score(detect_shesd(s.costs), truth).f1,
        "Proposed":            score(detect_shipped(s), truth).f1,
    }


def _gen_masking(seed):
    rng = np.random.default_rng(seed)
    pos = _rand_positions(rng, 90, 6)
    mags = [rng.uniform(7.0, 9.0) for _ in range(3)] + \
           [rng.uniform(2.0, 2.6) for _ in range(3)]
    rng.shuffle(mags)
    return make_series(seed=seed, n_days=90, base=1000.0, weekly_amplitude=0.0,
                       daily_trend=0.0, noise_frac=0.02, spikes=list(zip(pos, mags)))


def _gen_seasonality(seed):
    rng = np.random.default_rng(seed)
    pos = _rand_positions(rng, 84, 2)
    mags = [rng.uniform(3.2, 4.2) for _ in range(2)]
    return make_series(seed=seed, n_days=84, base=1000.0, weekly_amplitude=0.45,
                       daily_trend=0.0, noise_frac=0.03, spikes=list(zip(pos, mags)))


def _gen_e2e(seed):
    rng = np.random.default_rng(seed)
    pos = _rand_positions(rng, 90, 6)
    mags = [rng.uniform(2.5, 6.5) for _ in range(6)]
    return make_series(seed=seed, n_days=90, base=800.0, weekly_amplitude=0.35,
                       daily_trend=6.0, noise_frac=0.04, spikes=list(zip(pos, mags)))


GENS = {"Masking": _gen_masking, "Seasonality": _gen_seasonality,
        "End-to-end": _gen_e2e}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, 10_000_000, size=args.runs)

    print(f"cloudsealed-jit — Monte Carlo robustness (N={args.runs} runs)")
    print("=" * 78)
    for scen, gen in GENS.items():
        acc = {m: [] for m in METHODS}
        for sd in seeds:
            res = _f1s(gen(int(sd)))
            for m in METHODS:
                acc[m].append(res[m])
        print(f"\n{scen}")
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
