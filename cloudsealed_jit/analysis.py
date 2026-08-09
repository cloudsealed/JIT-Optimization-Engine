"""
Waste analysis over a daily billing series.

Method
------

**Baseline.** Expected spend for a day is the product of a level term and a
weekday term:

    expected[i] = rolling_median(cost, 7)[i] * dow_factor[weekday(i)]

The rolling median tracks growth and step changes without being dragged by
spikes. The weekday factor is the median ratio of observed spend to the level
term for that weekday, which captures the weekday/weekend cycle that dominates
most cloud bills. It is only estimated when at least two full weeks are
available; below that the factor is 1.0 for every day.

**Anomalies.** Residuals against the baseline are scored with a modified
z-score (median absolute deviation, see :mod:`cloudsealed_jit.kernels`). Days
scoring at or above ``Z_THRESHOLD`` are reported. The threshold of 3.5 is the
value recommended by Iglewicz & Hoaglin for the modified z-score.

**Waste.** Only *positive* excess counts as waste: spending less than expected
is not an opportunity. Waste percentage is the share of total spend that sits
above the baseline on anomalous days, which makes it directly convertible to
currency rather than being a count of unusual days.

**Recommendations.** Every recommendation carries a saving derived from the
series itself, normalised to 30 days, and states the assumption behind it in
its description. Estimates that depend on facts the analyser cannot observe --
whether a workload is production, whether a commitment is acceptable -- are
labelled as conditional rather than presented as findings.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np

from .kernels import modified_zscores, rolling_median
from .parsing import BillingSeries

__all__ = [
    "Anomaly",
    "Metrics",
    "Recommendation",
    "Forecast",
    "AnalysisResult",
    "analyze",
    "Z_THRESHOLD",
]

AnalysisType = Literal["waste-audit", "cost-forecast", "efficiency"]

#: Modified z-score at which a day is reported as anomalous.
Z_THRESHOLD = 3.5

#: Minimum days required before the weekday seasonality term is estimated.
MIN_DAYS_FOR_SEASONALITY = 14

#: Assumed discount for a one-year commitment against the sustained baseline.
#: Conservative relative to published AWS/GCP/Azure commitment discounts.
COMMITMENT_DISCOUNT = 0.25

#: A service whose weekend spend is at least this fraction of its weekday
#: spend is running through the weekend at full cost.
WEEKEND_RATIO_FLAG = 0.80

# Severity combines statistical strength with financial size. Either alone is
# misleading: a very tight baseline turns a 5x spike into a moderate z-score,
# while a noisy series can produce a large z-score over a trivial amount.
# Each tier is (min |z|, min |deviation %|, label) and matches on either.
_SEVERITY_TIERS = (
    (10.0, 200.0, "CRITICAL"),
    (7.0, 100.0, "HIGH"),
    (5.0, 25.0, "MEDIUM"),
)


@dataclass
class Anomaly:
    date: str
    expectedCost: float
    actualCost: float
    deviation: float  # percent against expected
    zScore: float
    severity: str
    description: str


@dataclass
class Metrics:
    averageDailyCost: float
    stdDeviation: float
    sharpeRatio: float  # spend stability: mean / stddev of daily cost
    wastePercentage: float


@dataclass
class Recommendation:
    title: str
    description: str
    potentialSavings: float  # per 30 days, in the export's currency
    effort: str


@dataclass
class Forecast:
    """A trend-aware projection of future spend.

    Mechanical extrapolation of the observed level (rolling median) and weekly
    seasonality — not a probabilistic prediction. It answers "if the current
    trend holds, what will we spend, and when do we cross a budget?", which is
    proactive where anomaly detection alone is reactive.
    """

    horizonDays: int
    projectedSpend: float          # trend + weekday-seasonality projection over the horizon
    runRateSpend: float            # flat mean * horizon, for comparison
    dailyTrend: float              # slope of the level term: +/- currency per day
    budget: float | None = None    # echoed if the caller supplied one
    budgetBreachDay: int | None = None  # 1-indexed horizon day cumulative spend crosses budget
    method: str = "rolling-median level trend + weekday seasonality"


@dataclass
class AnalysisResult:
    anomalies: list[Anomaly] = field(default_factory=list)
    metrics: Metrics = field(
        default_factory=lambda: Metrics(0.0, 0.0, 0.0, 0.0)
    )
    recommendations: list[Recommendation] = field(default_factory=list)
    summary: str = ""
    forecast: Forecast | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def _weekday_factors(costs: np.ndarray, level: np.ndarray,
                     days: list[date]) -> np.ndarray:
    """Median ratio of observed spend to the level term, per weekday."""
    factors = np.ones(7, dtype=np.float64)
    if len(days) < MIN_DAYS_FOR_SEASONALITY:
        return factors

    buckets: dict[int, list[float]] = {i: [] for i in range(7)}
    for i, day in enumerate(days):
        if level[i] > 0:
            buckets[day.weekday()].append(costs[i] / level[i])

    for weekday, ratios in buckets.items():
        # Two observations is not enough to separate a pattern from noise.
        if len(ratios) >= 2:
            factors[weekday] = float(statistics.median(ratios))

    # Renormalise so the factors do not shift the overall level.
    mean_factor = float(np.mean(factors))
    if mean_factor > 0:
        factors /= mean_factor
    return factors


def _baseline(costs: np.ndarray, days: list[date]) -> np.ndarray:
    level = rolling_median(costs, window=7)
    factors = _weekday_factors(costs, level, days)
    expected = np.array(
        [level[i] * factors[day.weekday()] for i, day in enumerate(days)],
        dtype=np.float64,
    )
    return np.maximum(expected, 0.0)


def _severity(z: float, deviation_pct: float) -> str:
    magnitude = abs(z)
    deviation = abs(deviation_pct)
    for min_z, min_deviation, label in _SEVERITY_TIERS:
        if magnitude >= min_z or deviation >= min_deviation:
            return label
    return "LOW"


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------


def _weekend_recommendations(series: BillingSeries,
                             currency: str) -> list[Recommendation]:
    """Flag services that cost the same on weekends as on weekdays."""
    out: list[Recommendation] = []
    days = series.days
    if len(days) < MIN_DAYS_FOR_SEASONALITY:
        return out

    weekend_idx = [i for i, d in enumerate(days) if d.weekday() >= 5]
    weekday_idx = [i for i, d in enumerate(days) if d.weekday() < 5]
    if len(weekend_idx) < 4 or len(weekday_idx) < 8:
        return out

    span = len(days)
    for name, values in series.by_service.items():
        weekend = statistics.median(values[i] for i in weekend_idx)
        weekday = statistics.median(values[i] for i in weekday_idx)
        if weekday <= 0:
            continue
        ratio = weekend / weekday
        if ratio < WEEKEND_RATIO_FLAG:
            continue

        service_total = sum(values)
        if service_total <= 0:
            continue

        # Upper bound: the weekend spend itself, normalised to 30 days.
        weekend_spend = sum(values[i] for i in weekend_idx)
        monthly = weekend_spend / span * 30
        if monthly < 1.0:
            continue

        out.append(
            Recommendation(
                title=f"Idle weekend spend in {name}",
                description=(
                    f"{name} costs {ratio:.0%} as much on weekends as on weekdays "
                    f"(weekday median {currency} {weekday:,.2f}/day, weekend median "
                    f"{currency} {weekend:,.2f}/day). If this workload is not "
                    f"production, stopping it outside business days removes the "
                    f"weekend spend entirely. The figure is the observed weekend "
                    f"spend normalised to 30 days and assumes the workload can be "
                    f"stopped; verify before acting."
                ),
                potentialSavings=round(monthly, 2),
                effort="LOW",
            )
        )
    return out


def _commitment_recommendation(costs: np.ndarray,
                               currency: str) -> Recommendation | None:
    """Estimate the saving available on the always-on portion of spend."""
    if costs.size < MIN_DAYS_FOR_SEASONALITY:
        return None
    floor = float(np.percentile(costs, 10))
    if floor <= 0:
        return None

    monthly = floor * 30 * COMMITMENT_DISCOUNT
    if monthly < 1.0:
        return None

    return Recommendation(
        title="Sustained baseline eligible for commitment pricing",
        description=(
            f"Daily spend never fell below {currency} {floor:,.2f} (10th percentile) "
            f"over the period, so that portion is always-on. Committed-use discounts "
            f"or savings plans typically price this tier {COMMITMENT_DISCOUNT:.0%} "
            f"below on-demand. The figure applies that rate to the observed floor "
            f"over 30 days and assumes a commitment is commercially acceptable."
        ),
        potentialSavings=round(monthly, 2),
        effort="MEDIUM",
    )


def _excess_recommendation(excess_total: float, span: int,
                           currency: str, count: int) -> Recommendation | None:
    if excess_total <= 0 or count == 0:
        return None
    monthly = excess_total / span * 30
    if monthly < 1.0:
        return None
    return Recommendation(
        title="Investigate spend above the modelled baseline",
        description=(
            f"{count} day(s) spent {currency} {excess_total:,.2f} more than the "
            f"day-of-week baseline predicted. This is unplanned spend rather than "
            f"growth: the baseline already tracks trend and weekly seasonality. "
            f"The figure is that excess normalised to 30 days."
        ),
        potentialSavings=round(monthly, 2),
        effort="MEDIUM",
    )


def _concentration_recommendation(series: BillingSeries,
                                  currency: str) -> Recommendation | None:
    total = series.total_cost
    if total <= 0 or not series.by_service:
        return None
    name, values = max(series.by_service.items(), key=lambda kv: sum(kv[1]))
    share = sum(values) / total
    if share < 0.40:
        return None
    return Recommendation(
        title=f"Spend concentrated in {name}",
        description=(
            f"{name} accounts for {share:.0%} of total spend ({currency} "
            f"{sum(values):,.2f} of {currency} {total:,.2f}). Optimisation effort "
            f"applied here has the highest leverage; a 10% reduction on this "
            f"service alone yields the figure shown."
        ),
        potentialSavings=round(sum(values) / max(len(series.days), 1) * 30 * 0.10, 2),
        effort="MEDIUM",
    )


def _volatility_recommendation(stability: float, mean: float,
                               std: float, currency: str) -> Recommendation | None:
    if stability >= 2.0 or mean <= 0:
        return None
    return Recommendation(
        title="Daily spend is highly volatile",
        description=(
            f"Daily cost varies with a standard deviation of {currency} {std:,.2f} "
            f"against a mean of {currency} {mean:,.2f} (stability ratio "
            f"{stability:.2f}). Spend this irregular cannot be budgeted or alerted "
            f"on reliably. Establishing budget alerts and per-environment cost "
            f"attribution is a prerequisite for any further optimisation. No direct "
            f"saving is claimed."
        ),
        potentialSavings=0.0,
        effort="LOW",
    )


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------

#: Default projection horizon, matching the 30-day normalisation used for
#: recommendations elsewhere in the module.
FORECAST_HORIZON_DAYS = 30

#: Recent window used to estimate the level trend. Long enough to smooth noise,
#: short enough to reflect the current trajectory rather than the whole history.
_TREND_WINDOW = 28


def _forecast(series: BillingSeries, costs: np.ndarray,
              horizon: int = FORECAST_HORIZON_DAYS,
              budget: float | None = None) -> Forecast:
    """Project spend forward from the observed level trend and weekday shape.

    The level (rolling median) already tracks growth and step changes; its
    slope over a recent window is the daily trend. Each future day is that
    projected level times the weekday factor for that calendar day, so the
    projection carries the same weekly seasonality the baseline uses.
    """
    days = series.days
    level = rolling_median(costs, window=7)
    factors = _weekday_factors(costs, level, days)

    # Trend of the level term over the recent window (currency per day).
    window = min(len(level), _TREND_WINDOW)
    recent = level[-window:]
    x = np.arange(window, dtype=np.float64)
    slope = float(np.polyfit(x, recent, 1)[0]) if window >= 2 else 0.0
    last_level = float(level[-1])
    last_day = days[-1]

    projected_daily: list[float] = []
    for k in range(1, horizon + 1):
        future_level = max(0.0, last_level + slope * k)
        weekday = (last_day + timedelta(days=k)).weekday()
        projected_daily.append(future_level * factors[weekday])

    projected_spend = float(sum(projected_daily))
    mean = float(np.mean(costs))
    run_rate = mean * horizon

    breach_day: int | None = None
    if budget is not None and budget > 0:
        cumulative = 0.0
        for k, day_cost in enumerate(projected_daily, start=1):
            cumulative += day_cost
            if cumulative > budget:
                breach_day = k
                break

    return Forecast(
        horizonDays=horizon,
        projectedSpend=round(projected_spend, 2),
        runRateSpend=round(run_rate, 2),
        dailyTrend=round(slope, 2),
        budget=round(budget, 2) if budget is not None else None,
        budgetBreachDay=breach_day,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def analyze(series: BillingSeries,
            analysis_type: AnalysisType = "waste-audit",
            *, max_anomalies: int = 25,
            budget: float | None = None) -> AnalysisResult:
    """Run waste analysis over a parsed billing series."""
    currency = series.currency or "USD"
    span = series.span_days

    if span < 3:
        return AnalysisResult(
            summary=(
                f"Only {span} day(s) of billing data. At least 3 days are needed "
                f"to model a baseline, and 14 for weekday seasonality."
            )
        )

    costs = np.asarray(series.costs, dtype=np.float64)
    expected = _baseline(costs, series.days)
    zscores = modified_zscores(costs, expected)

    mean = float(np.mean(costs))
    std = float(np.std(costs, ddof=1)) if span > 1 else 0.0
    stability = float(mean / std) if std > 0 else 0.0

    anomalies: list[Anomaly] = []
    excess_total = 0.0
    for i in range(span):
        z = float(zscores[i])
        if abs(z) < Z_THRESHOLD:
            continue
        exp = float(expected[i])
        act = float(costs[i])
        deviation = ((act - exp) / exp * 100.0) if exp > 0 else 0.0
        if act > exp:
            excess_total += act - exp

        anomalies.append(
            Anomaly(
                date=series.days[i].isoformat(),
                expectedCost=round(exp, 2),
                actualCost=round(act, 2),
                deviation=round(deviation, 2),
                zScore=round(z, 2),
                severity=_severity(z, deviation),
                description=(
                    f"Spend {'above' if act > exp else 'below'} the day-of-week "
                    f"baseline by {currency} {abs(act - exp):,.2f} "
                    f"({abs(deviation):.1f}%)."
                ),
            )
        )

    total = series.total_cost
    waste_pct = (excess_total / total * 100.0) if total > 0 else 0.0

    metrics = Metrics(
        averageDailyCost=round(mean, 2),
        stdDeviation=round(std, 2),
        sharpeRatio=round(stability, 2),
        wastePercentage=round(waste_pct, 2),
    )

    recommendations: list[Recommendation] = []
    if analysis_type in ("waste-audit", "cost-forecast"):
        over = sum(1 for a in anomalies if a.actualCost > a.expectedCost)
        excess_rec = _excess_recommendation(excess_total, span, currency, over)
        if excess_rec:
            recommendations.append(excess_rec)
        recommendations.extend(_weekend_recommendations(series, currency))
        commitment = _commitment_recommendation(costs, currency)
        if commitment:
            recommendations.append(commitment)
        concentration = _concentration_recommendation(series, currency)
        if concentration:
            recommendations.append(concentration)

    volatility = _volatility_recommendation(stability, mean, std, currency)
    if volatility:
        recommendations.append(volatility)

    recommendations.sort(key=lambda r: r.potentialSavings, reverse=True)

    identified = sum(r.potentialSavings for r in recommendations)
    summary = (
        f"Analysed {span} day(s) of billing ({series.rows_parsed:,} line items) "
        f"totalling {currency} {total:,.2f}. "
        f"Mean daily spend {currency} {mean:,.2f}, stability ratio {stability:.2f}. "
        f"{len(anomalies)} anomalous day(s) detected; {waste_pct:.1f}% of total "
        f"spend sits above the modelled baseline. "
        f"Identified up to {currency} {identified:,.2f} in addressable monthly spend."
    )
    if series.rows_skipped:
        summary += f" {series.rows_skipped:,} row(s) could not be parsed."

    # Forecast is computed for the cost-forecast type, or whenever a budget is
    # given (so a budget-breach question works regardless of analysis type).
    forecast: Forecast | None = None
    if analysis_type == "cost-forecast" or budget is not None:
        forecast = _forecast(series, costs, budget=budget)
        trend_word = "rising" if forecast.dailyTrend > 0 else "falling" if forecast.dailyTrend < 0 else "flat"
        summary += (
            f" Projected {forecast.horizonDays}-day spend {currency} "
            f"{forecast.projectedSpend:,.2f} (trend {trend_word}, "
            f"{currency} {forecast.dailyTrend:+,.2f}/day)."
        )
        if forecast.budgetBreachDay is not None:
            summary += (
                f" At this trend the {currency} {forecast.budget:,.2f} budget is "
                f"crossed on day {forecast.budgetBreachDay} of the horizon."
            )
        elif forecast.budget is not None:
            summary += (
                f" The {currency} {forecast.budget:,.2f} budget is not projected to "
                f"be crossed within {forecast.horizonDays} days."
            )

    anomalies.sort(key=lambda a: abs(a.zScore), reverse=True)

    return AnalysisResult(
        anomalies=anomalies[:max_anomalies],
        metrics=metrics,
        recommendations=recommendations,
        summary=summary,
        forecast=forecast,
    )
