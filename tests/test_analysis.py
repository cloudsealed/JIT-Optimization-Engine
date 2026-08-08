import numpy as np
import pytest

from cloudsealed_jit.analysis import Z_THRESHOLD, analyze
from cloudsealed_jit.kernels import modified_zscores, rolling_median
from cloudsealed_jit.parsing import parse_billing_csv


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


def test_rolling_median_ignores_isolated_spike():
    values = np.array([10.0, 10.0, 10.0, 900.0, 10.0, 10.0, 10.0])
    baseline = rolling_median(values, window=5)
    # The median at the spike must stay at the level of the surrounding days.
    assert baseline[3] == pytest.approx(10.0)


def test_modified_zscore_does_not_mask_spikes():
    """A standard-deviation z-score hides spikes that inflate sigma."""
    values = np.array([10.0] * 20 + [400.0])
    baseline = np.full(values.shape, 10.0)

    classic = (values - values.mean()) / values.std()
    robust = modified_zscores(values, baseline)

    # The classic score cannot clear the usual 3-sigma bar; the robust one does.
    assert abs(classic[-1]) < 5.0
    assert abs(robust[-1]) > Z_THRESHOLD


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def test_stable_series_reports_no_anomalies(stable_billing):
    result = analyze(parse_billing_csv(stable_billing))

    assert result.anomalies == []
    assert result.metrics.wastePercentage == pytest.approx(0.0, abs=0.5)
    assert result.metrics.averageDailyCost == pytest.approx(100.0, abs=1.0)


def test_injected_spike_is_detected_with_correct_day(spiked_billing):
    result = analyze(parse_billing_csv(spiked_billing))

    assert result.anomalies, "the 5x spike must be reported"
    top = result.anomalies[0]
    assert top.date == "2026-01-31"  # start 2026-01-01 + 30 days
    assert top.actualCost == pytest.approx(500.0)
    assert top.severity == "CRITICAL"
    assert top.deviation > 300


def test_waste_percentage_is_share_of_spend_not_count_of_days(spiked_billing):
    result = analyze(parse_billing_csv(spiked_billing))

    # One anomalous day out of 60 would be 1.7% if it counted days.
    # The excess is ~400 on a total of ~6400, i.e. ~6%.
    assert result.metrics.wastePercentage > 4.0
    assert result.metrics.wastePercentage < 10.0


def test_excess_recommendation_is_derived_from_the_excess(spiked_billing):
    result = analyze(parse_billing_csv(spiked_billing))
    rec = next(
        r for r in result.recommendations
        if r.title == "Investigate spend above the modelled baseline"
    )

    # ~400 excess over 60 days, normalised to 30 days -> ~200.
    assert rec.potentialSavings == pytest.approx(200.0, rel=0.25)


def test_weekend_idle_service_is_flagged_and_production_is_not(weekend_idle_billing):
    result = analyze(parse_billing_csv(weekend_idle_billing))
    flagged = [
        r.title for r in result.recommendations
        if r.title.startswith("Idle weekend spend")
    ]

    assert flagged == ["Idle weekend spend in staging"], result.recommendations


def test_commitment_recommendation_uses_the_observed_floor(stable_billing):
    result = analyze(parse_billing_csv(stable_billing))
    rec = next(r for r in result.recommendations if "commitment" in r.title.lower())

    # floor ~99 * 30 days * 25% -> ~742
    assert rec.potentialSavings == pytest.approx(742.0, rel=0.1)


def test_short_series_refuses_to_model_a_baseline():
    csv_text = "date,cost\n2026-01-01,10\n2026-01-02,10\n"
    result = analyze(parse_billing_csv(csv_text))

    assert result.anomalies == []
    assert result.recommendations == []
    assert "At least 3 days" in result.summary


def test_efficiency_mode_omits_saving_recommendations(spiked_billing):
    result = analyze(parse_billing_csv(spiked_billing), "efficiency")

    assert all(r.potentialSavings == 0.0 for r in result.recommendations)


def test_forecast_mode_projects_run_rate(stable_billing):
    result = analyze(parse_billing_csv(stable_billing), "cost-forecast")
    assert "Projected 30-day spend" in result.summary


def test_zero_spend_series_does_not_divide_by_zero():
    rows = "\n".join(f"2026-01-{d:02d},0.0" for d in range(1, 21))
    result = analyze(parse_billing_csv(f"date,cost\n{rows}\n"))

    assert result.metrics.wastePercentage == 0.0
    assert result.metrics.sharpeRatio == 0.0
