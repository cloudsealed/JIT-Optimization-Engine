# Methodology

How `cloudsealed-jit` decides that a day of cloud spend is anomalous, why the
method is built the way it is, and a reproducible benchmark quantifying its
advantage over the textbook approach.

## The problem with the textbook approach

Cost anomaly detection is usually done by comparing each day against the period
mean and flagging anything beyond two or three standard deviations. On cloud
billing data that method fails in two specific, measurable ways.

**Standard deviation is inflated by the very spikes you are looking for.** A
handful of large anomalies raises σ enough to pull themselves back inside the
threshold and to hide every smaller anomaly with them. This is the *masking
effect*, and it gets worse as the anomalies get bigger — exactly the regime
that matters for cost control.

**A flat mean ignores the weekly cycle.** Most cloud bills have a pronounced
weekday/weekend shape. Measured against a flat mean, ordinary Mondays look like
overspend and ordinary Sundays look like savings, burying real signals under
false positives.

## The method

### Baseline

Expected spend for a day is a *level* term times a *weekday* term:

```
expected[i] = rolling_median(cost, 7)[i] × dow_factor[weekday(i)]
```

The centred 7-day rolling median tracks organic growth and step changes without
being dragged by spikes. The weekday factor is the median ratio of observed
spend to the level term for that weekday; it is only estimated once at least
two full weeks are present (below that every factor is 1.0). Factors are
renormalised so seasonality reshapes the baseline without shifting its overall
level. (`analysis.py:_baseline`, `_weekday_factors`)

### Scoring

Residuals against the baseline are scored with a modified z-score built on the
median absolute deviation (MAD):

```
z = 0.6745 × (x − baseline) / MAD
```

The 0.6745 constant makes the MAD a consistent estimator of σ for normally
distributed data, so the score keeps the familiar "number of deviations"
reading while tolerating contamination in up to ~50% of the sample. Days at or
above |z| = 3.5 are reported — the threshold recommended by Iglewicz & Hoaglin
(1993) for the modified z-score. (`kernels.py:modified_zscores`,
`analysis.py:Z_THRESHOLD`)

Where the MAD is exactly zero (a series that is constant apart from a few
spikes), the scale falls back to a standard deviation; if both are zero the
series is flat and every score is zero.

### Waste

Only *positive* excess counts. Waste percentage is the share of total spend
sitting above the baseline on anomalous days, which converts directly to
currency instead of being a count of unusual days. (`analysis.py:analyze`)

### Recommendations

Every recommendation carries a figure derived from the series itself,
normalised to 30 days, and states its assumption in the description. Estimates
that depend on facts the analyser cannot observe — whether a workload is
production, whether a commitment is acceptable — are labelled conditional
rather than presented as findings.

## Benchmark

`benchmarks/masking_benchmark.py` builds synthetic billing series whose true
anomalies are known by construction (fixed seed, no private data) and reports
precision / recall / F1 for each method. Reproduce with:

```
python benchmarks/masking_benchmark.py
```

### Result 1 — the scale estimator (masking)

Six spikes on an otherwise clean 90-day series: three very large (~+7000) and
three moderate (~+1200). Both detectors use the *same* rolling-median baseline,
so the only variable is MAD vs standard deviation.

| scale estimator | precision | recall | F1 |
|---|---|---|---|
| textbook standard deviation | 1.000 | 0.500 | 0.667 |
| **robust MAD** | 0.857 | **1.000** | **0.923** |

The three large spikes inflate the standard deviation until its 3.5-σ threshold
climbs *above* the three moderate spikes, masking them (recall 0.500). The MAD
is unmoved by the large spikes and recovers all six (recall 1.000).

### Result 2 — the baseline (seasonality)

A strong weekday/weekend cycle (weekends 45% cheaper) with two genuine spikes,
comparing a flat mean against the day-of-week baseline.

| baseline | precision | recall | F1 |
|---|---|---|---|
| textbook flat mean + stddev | 1.000 | 0.500 | 0.667 |
| **robust day-of-week + MAD** | 1.000 | **1.000** | **1.000** |

### Result 3 — end to end (headline)

A realistic bill: organic growth (~+68% over the quarter), a weekly cycle,
noise, and six spikes of graded size. Full shipped method vs full textbook
method.

| method | precision | recall | F1 |
|---|---|---|---|
| textbook flat mean + stddev | 1.000 | 0.500 | 0.667 |
| **robust (rolling median + DOW + MAD)** | 1.000 | **1.000** | **1.000** |

**The full method's F1 exceeds the textbook method's by 0.333** on this
scenario. `benchmarks/masking_benchmark.py --check` re-runs this comparison and
fails CI if the advantage ever drops below 0.25, so the claim cannot silently
regress.

## Reference

Iglewicz, B. and Hoaglin, D. C. (1993). *How to Detect and Handle Outliers.*
ASQC Quality Press. (Origin of the modified z-score and the |z| = 3.5
threshold.)
