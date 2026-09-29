"""
State-of-the-art comparison benchmark (companion to masking_benchmark.py).

masking_benchmark.py isolates each design decision against the textbook
mean/standard-deviation detector. This script instead pits the *full shipped
method* against three established time-series anomaly detectors, on the exact
same synthetic scenarios (same seeds, same ground truth, same day-index
scoring), so the paper's comparison table can be reproduced end to end:

  * Isolation Forest        Liu, Ting & Zhou, ICDM 2008        (scikit-learn)
  * STL residual + 3-sigma  Cleveland et al., J. Off. Stat. 1990
  * Seasonal-Hybrid ESD     Hochenbaum, Vallis & Kejariwal, 2017 (Twitter);
                            STL + Rosner's Generalized ESD, Technometrics 1983

Every detector is run with library-default / standard parameters (no per-
scenario tuning). The point of the comparison is not that these methods cannot
be tuned for a single scenario, but that one fixed configuration robust across
all three billing-specific failure modes is what the shipped method provides.

Extra dependencies (kept out of the core package on purpose):
    pip install scikit-learn statsmodels scipy

Run:
    python benchmarks/sota_comparison.py            # print the table
    python benchmarks/sota_comparison.py --json      # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from statsmodels.tsa.seasonal import STL
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        "sota_comparison.py needs extra deps: "
        "pip install scikit-learn statsmodels scipy\n"
        f"(import failed: {exc})"
    )

from benchmarks.masking_benchmark import (
    make_series, score, detect_textbook_flat_std, detect_shipped, Series,
)


# --------------------------------------------------------------------------
# SOTA detectors : each returns the set of day indices it flags
# --------------------------------------------------------------------------

def detect_isolation_forest(costs: np.ndarray, seed: int = 0) -> set[int]:
    """Isolation Forest on the univariate daily-cost series,
    contamination='auto' (not tuned to the known anomaly count)."""
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=seed)
    pred = clf.fit_predict(costs.reshape(-1, 1))  # -1 == anomaly
    return {i for i in range(costs.size) if pred[i] == -1}


def _generalized_esd(x: np.ndarray, max_out: int, alpha: float = 0.05) -> set[int]:
    """Rosner's Generalized ESD many-outlier test (Technometrics, 1983)."""
    work_x = x.astype(float).copy()
    work_idx = np.arange(x.size)
    R, lam, cand = [], [], []
    for _ in range(max_out):
        if work_x.size < 3:
            break
        sd = work_x.std(ddof=1)
        if sd == 0:
            break
        resid = np.abs(work_x - work_x.mean())
        j = int(np.argmax(resid))
        nn = work_x.size
        R.append(resid[j] / sd)
        cand.append(int(work_idx[j]))
        t = stats.t.ppf(1 - alpha / (2 * nn), nn - 2)
        lam.append((nn - 1) * t / np.sqrt((nn - 2 + t**2) * nn))
        work_x = np.delete(work_x, j)
        work_idx = np.delete(work_idx, j)
    n_anom = 0
    for i in range(len(R)):
        if R[i] > lam[i]:
            n_anom = i + 1
    return set(cand[:n_anom])


def detect_shesd(costs: np.ndarray, period: int = 7, max_frac: float = 0.1) -> set[int]:
    """Seasonal-Hybrid ESD: robust STL to strip trend + seasonality, then
    Generalized ESD on the residual."""
    max_out = max(1, int(costs.size * max_frac))
    if costs.size < 2 * period:
        return _generalized_esd(costs, max_out)
    stl = STL(costs, period=period, robust=True).fit()
    resid = costs - stl.seasonal - np.median(stl.trend)
    return _generalized_esd(resid, max_out)


def detect_stl_3sigma(costs: np.ndarray, period: int = 7) -> set[int]:
    """STL decomposition, 3-sigma z-score on the residual."""
    if costs.size < 2 * period:
        return set()
    resid = STL(costs, period=period, robust=True).fit().resid
    sd = resid.std(ddof=1)
    if sd == 0:
        return set()
    z = (resid - resid.mean()) / sd
    return {i for i in range(costs.size) if abs(z[i]) >= 3.0}


# --------------------------------------------------------------------------
# The same three scenarios as masking_benchmark.py
# --------------------------------------------------------------------------

def _series_masking() -> Series:
    return make_series(seed=1, n_days=90, base=1000.0, weekly_amplitude=0.0,
                       daily_trend=0.0, noise_frac=0.02,
                       spikes=[(12, 8.0), (37, 7.5), (66, 8.5),
                               (23, 2.2), (51, 2.3), (80, 2.1)])

def _series_seasonality() -> Series:
    return make_series(seed=2, n_days=84, base=1000.0, weekly_amplitude=0.45,
                       daily_trend=0.0, noise_frac=0.03,
                       spikes=[(30, 4.0), (61, 3.5)])

def _series_e2e() -> Series:
    return make_series(seed=3, n_days=90, base=800.0, weekly_amplitude=0.35,
                       daily_trend=6.0, noise_frac=0.04,
                       spikes=[(15, 6.0), (28, 3.2), (44, 5.0),
                               (59, 2.8), (73, 4.5), (85, 3.0)])

SCENARIOS = {
    "masking": _series_masking,
    "seasonality": _series_seasonality,
    "end_to_end": _series_e2e,
}


def _detectors(s: Series) -> dict:
    return {
        "Textbook (flat mean + stddev)":       detect_textbook_flat_std(s.costs),
        "Isolation Forest":                    detect_isolation_forest(s.costs),
        "STL residual + 3-sigma":              detect_stl_3sigma(s.costs),
        "Seasonal-Hybrid ESD":                 detect_shesd(s.costs),
        "Proposed (rolling median + DOW + MAD)": detect_shipped(s),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = {
        name: {m: vars(score(pred, make().true_anomalies))
               for m, pred in _detectors(make()).items()}
        for name, make in SCENARIOS.items()
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print("cloudsealed-jit — full method vs established anomaly detectors")
    print("=" * 78)
    for name, make in SCENARIOS.items():
        s = make()
        print(f"\n{name}  (ground-truth anomalies: {len(s.true_anomalies)})")
        print(f"  {'method':<40}{'P':>8}{'R':>8}{'F1':>8}")
        for m, pred in _detectors(s).items():
            sc = score(pred, s.true_anomalies)
            print(f"  {m:<40}{sc.precision:>8.3f}{sc.recall:>8.3f}{sc.f1:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
