"""
Reproducible benchmark: why robust statistics beat the textbook z-score on
cloud billing data.

The claim this package makes is that anomaly detection on cloud spend must use
(a) a rolling-median, day-of-week-aware *baseline* and (b) a MAD-based *scale*,
not a flat mean and a standard deviation. This script quantifies both halves of
that claim against synthetic series whose true anomalies are known by
construction, and reports precision / recall / F1 for each method.

It is fully reproducible: every series is drawn from a fixed seed, there is no
private data, and it runs in a couple of seconds. `--check` re-runs the
headline scenario and exits non-zero if the robust method ever stops beating
the textbook method by the documented margin — so the claim in METHODOLOGY.md
can never silently rot.

Run:
    python benchmarks/masking_benchmark.py            # print the tables
    python benchmarks/masking_benchmark.py --json     # machine-readable
    python benchmarks/masking_benchmark.py --check     # CI regression gate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from cloudsealed_jit.analysis import Z_THRESHOLD, analyze
from cloudsealed_jit.kernels import modified_zscores, rolling_median
from cloudsealed_jit.parsing import BillingSeries

# A robust improvement worth documenting must clear this F1 margin on the
# end-to-end scenario, or `--check` fails. Chosen well below the observed
# gap so ordinary noise never trips it, but high enough to catch a real
# regression that reintroduced the masking bug.
HEADLINE_F1_MARGIN = 0.25


# --------------------------------------------------------------------------
# Synthetic billing generator (ground truth known by construction)
# --------------------------------------------------------------------------


@dataclass
class Series:
    days: list[date]
    costs: np.ndarray
    true_anomalies: set[int]  # indices deliberately spiked


def make_series(
    *,
    seed: int,
    n_days: int,
    base: float,
    weekly_amplitude: float,
    daily_trend: float,
    noise_frac: float,
    spikes: list[tuple[int, float]],
) -> Series:
    """Build a daily cost series with a known set of injected anomalies.

    Args:
        base: level of daily spend before seasonality and trend.
        weekly_amplitude: fraction by which weekdays exceed weekends (the
            weekday/weekend cycle that dominates real cloud bills).
        daily_trend: additive growth per day (organic growth / adoption).
        noise_frac: gaussian noise as a fraction of the local level.
        spikes: (day_index, multiplier) pairs — the ground-truth anomalies.
    """
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 5)  # a Monday, so weekday math is legible
    days = [start + timedelta(days=i) for i in range(n_days)]

    costs = np.empty(n_days, dtype=np.float64)
    for i, day in enumerate(days):
        level = base + daily_trend * i
        # Weekdays cost more than weekends; Sat/Sun sit below the level.
        weekday_mult = 1.0 if day.weekday() < 5 else (1.0 - weekly_amplitude)
        clean = level * weekday_mult
        costs[i] = clean + rng.normal(0.0, noise_frac * clean)

    true = set()
    for idx, mult in spikes:
        costs[idx] *= mult
        true.add(idx)

    costs = np.maximum(costs, 0.0)
    return Series(days=days, costs=costs, true_anomalies=true)


# --------------------------------------------------------------------------
# Detectors — each returns the set of day indices it flags as anomalous
# --------------------------------------------------------------------------


def detect_textbook_flat_std(costs: np.ndarray) -> set[int]:
    """The method this package argues against: flat mean, standard-deviation
    z-score. No baseline, no seasonality."""
    mean = float(np.mean(costs))
    std = float(np.std(costs))
    if std == 0.0:
        return set()
    z = (costs - mean) / std
    return {i for i in range(costs.size) if abs(z[i]) >= Z_THRESHOLD}


def detect_median_baseline_std(costs: np.ndarray) -> set[int]:
    """Isolates the *scale* estimator: same rolling-median baseline as the
    robust method, but scored with a standard deviation instead of the MAD.
    The gap between this and the robust detector is purely the masking effect."""
    baseline = rolling_median(costs, window=7)
    residuals = costs - baseline
    std = float(np.std(residuals))
    if std == 0.0:
        return set()
    z = residuals / std
    return {i for i in range(costs.size) if abs(z[i]) >= Z_THRESHOLD}


def detect_median_baseline_mad(costs: np.ndarray) -> set[int]:
    """Isolates the scale estimator, robust half: rolling-median baseline
    scored with the MAD-based modified z-score."""
    baseline = rolling_median(costs, window=7)
    z = modified_zscores(costs, baseline)
    return {i for i in range(costs.size) if abs(z[i]) >= Z_THRESHOLD}


def detect_shipped(series: Series) -> set[int]:
    """The full method exactly as shipped: parse-free path straight into
    analyze() with the day-of-week baseline and MAD scale."""
    bs = BillingSeries(days=series.days, costs=[float(c) for c in series.costs])
    result = analyze(bs, max_anomalies=len(series.days))
    index = {d.isoformat(): i for i, d in enumerate(series.days)}
    return {index[a.date] for a in result.anomalies}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def score(predicted: set[int], truth: set[int]) -> Score:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not truth else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Score(round(precision, 3), round(recall, 3), round(f1, 3), tp, fp, fn)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def scenario_masking() -> dict:
    """Graded spikes — three very large, three moderate — on an otherwise
    clean series. Same rolling-median baseline for both detectors, so the only
    variable is the scale estimator (MAD vs standard deviation). The large
    spikes inflate the standard deviation enough that its 3.5-sigma threshold
    climbs above the moderate spikes and hides them: the classic masking
    effect. The MAD is unmoved by the large spikes and still catches all six."""
    s = make_series(
        seed=1,
        n_days=90,
        base=1000.0,
        weekly_amplitude=0.0,       # no seasonality here: isolate scale only
        daily_trend=0.0,
        noise_frac=0.02,
        spikes=[(12, 8.0), (37, 7.5), (66, 8.5),   # very large (~+7000)
                (23, 2.2), (51, 2.3), (80, 2.1)],  # moderate (~+1200)
    )
    return {
        "name": "Masking — scale estimator (MAD vs stddev), same baseline",
        "true_anomalies": len(s.true_anomalies),
        "results": {
            "textbook stddev scale": score(detect_median_baseline_std(s.costs), s.true_anomalies),
            "robust MAD scale": score(detect_median_baseline_mad(s.costs), s.true_anomalies),
        },
    }


def scenario_seasonality() -> dict:
    """Strong weekday/weekend cycle plus two genuine spikes. Isolates the
    baseline: a flat mean flags every normal weekend low and Monday high as an
    anomaly (false positives) while missing nothing it should catch; the
    day-of-week baseline does not."""
    s = make_series(
        seed=2,
        n_days=84,
        base=1000.0,
        weekly_amplitude=0.45,      # weekends 45% cheaper than weekdays
        daily_trend=0.0,
        noise_frac=0.03,
        spikes=[(30, 4.0), (61, 3.5)],
    )
    return {
        "name": "Seasonality — baseline (day-of-week vs flat mean)",
        "true_anomalies": len(s.true_anomalies),
        "results": {
            "textbook flat mean + stddev": score(detect_textbook_flat_std(s.costs), s.true_anomalies),
            "robust shipped (DOW + MAD)": score(detect_shipped(s), s.true_anomalies),
        },
    }


def scenario_end_to_end() -> dict:
    """A realistic bill: organic growth, a weekly cycle, noise, and a mix of
    large and moderate spikes. The full shipped method against the full
    textbook method. This is the headline number."""
    s = make_series(
        seed=3,
        n_days=90,
        base=800.0,
        weekly_amplitude=0.35,
        daily_trend=6.0,            # ~+68% over the quarter, organic
        noise_frac=0.04,
        spikes=[(15, 6.0), (28, 3.2), (44, 5.0), (59, 2.8), (73, 4.5), (85, 3.0)],
    )
    return {
        "name": "End-to-end — full method vs full textbook",
        "true_anomalies": len(s.true_anomalies),
        "results": {
            "textbook flat mean + stddev": score(detect_textbook_flat_std(s.costs), s.true_anomalies),
            "robust shipped (rolling median + DOW + MAD)": score(detect_shipped(s), s.true_anomalies),
        },
    }


SCENARIOS = (scenario_masking, scenario_seasonality, scenario_end_to_end)


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _print_table(scenario: dict) -> None:
    print(f"\n{scenario['name']}")
    print(f"  ground-truth anomalies: {scenario['true_anomalies']}")
    print(f"  {'method':<48}{'precision':>10}{'recall':>9}{'F1':>7}"
          f"{'TP':>5}{'FP':>5}{'FN':>5}")
    for method, sc in scenario["results"].items():
        print(f"  {method:<48}{sc.precision:>10.3f}{sc.recall:>9.3f}"
              f"{sc.f1:>7.3f}{sc.tp:>5}{sc.fp:>5}{sc.fn:>5}")


def _headline_delta() -> float:
    e2e = scenario_end_to_end()
    results = list(e2e["results"].values())
    textbook_f1, robust_f1 = results[0].f1, results[1].f1
    return robust_f1 - textbook_f1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--check", action="store_true",
                        help="CI gate: fail if the robust method regresses")
    args = parser.parse_args(argv)

    scenarios = [fn() for fn in SCENARIOS]

    if args.check:
        delta = _headline_delta()
        ok = delta >= HEADLINE_F1_MARGIN
        print(f"end-to-end F1 advantage of robust over textbook: {delta:+.3f} "
              f"(required >= {HEADLINE_F1_MARGIN:+.3f}) -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.json:
        payload = [
            {
                "name": s["name"],
                "true_anomalies": s["true_anomalies"],
                "results": {m: vars(sc) for m, sc in s["results"].items()},
            }
            for s in scenarios
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print("cloudsealed-jit — robust vs textbook anomaly detection on cloud billing")
    print("=" * 78)
    for s in scenarios:
        _print_table(s)
    print(f"\nHeadline: robust method's end-to-end F1 exceeds the textbook "
          f"method's by {_headline_delta():+.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
