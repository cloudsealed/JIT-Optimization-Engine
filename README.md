# cloudsealed-jit

Detects structural waste in cloud billing exports.

Given a billing export from AWS, GCP or Azure, it models what each day *should*
have cost, reports the days that did not match, and turns the excess into a
monthly figure. It is a library, a CLI and an HTTP service.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## The problem

Cloud cost anomaly detection is usually done by comparing each day against the
period average and flagging anything beyond two or three standard deviations.
On billing data that method fails in two specific ways.

**Standard deviation is inflated by the very spikes you are looking for.** A
handful of large anomalies raises σ enough to pull themselves back inside the
threshold, and to hide every smaller anomaly with them. This is the masking
effect, and it gets worse as the anomalies get bigger.

**A flat average ignores the weekly cycle.** Most cloud bills have a pronounced
weekday/weekend shape. Measured against a flat mean, ordinary Mondays look like
overspend and ordinary Sundays look like savings.

## The method

**Baseline.** Expected spend for a day is a level term times a weekday term:

```
expected[i] = rolling_median(cost, 7)[i] × dow_factor[weekday(i)]
```

The rolling median follows growth and step changes without being dragged by
spikes. The weekday factor is the median ratio of observed spend to the level
term for that weekday. It is only estimated with at least two full weeks of
data; below that every factor is 1.0.

**Scoring.** Residuals are scored with a modified z-score built on the median
absolute deviation:

```
z = 0.6745 × (x − baseline) / MAD
```

The 0.6745 constant makes MAD a consistent estimator of σ for normal data, so
the score keeps the familiar "number of deviations" reading while tolerating
contamination in roughly half the sample. Days at or above |z| = 3.5 are
reported — the threshold recommended by Iglewicz & Hoaglin (1993).

**Waste.** Only positive excess counts. Waste percentage is the share of total
spend sitting above the baseline on anomalous days, which converts directly to
currency instead of being a count of unusual days.

**Recommendations.** Each carries a figure derived from the series itself,
normalised to 30 days, and states its assumption in the description. Estimates
that depend on facts the analyser cannot observe — whether a workload is
production, whether a commitment is acceptable — are labelled conditional
rather than presented as findings.

## Does it actually work better?

Yes, and it is measured, not asserted. `benchmarks/masking_benchmark.py` builds
synthetic bills whose anomalies are known by construction and scores this
method against the textbook mean+standard-deviation approach:

| scenario | textbook F1 | this method F1 |
|---|---|---|
| masking (scale estimator) | 0.667 | **0.923** |
| seasonality (baseline) | 0.667 | **1.000** |
| end-to-end | 0.667 | **1.000** |

Full derivation and reproduction steps in [METHODOLOGY.md](METHODOLOGY.md); the
design of the codebase is in [architecture.md](architecture.md). The benchmark
runs in CI (`--check`) and fails the build if the advantage ever regresses.

## Install

```bash
pip install cloudsealed-jit              # library + CLI
pip install "cloudsealed-jit[jit]"       # + numba-compiled kernels
pip install "cloudsealed-jit[jit,api]"   # + HTTP service
```

`numba` is optional. Without it the kernels run on pure NumPy and the results
are identical; only large inputs get slower.

## GitHub Action

The fastest way to use this: run the audit in CI and get the findings as a pull
request comment, without installing anything locally.

```yaml
- uses: cloudsealed/JIT-Optimization-Engine@main
  with:
    billing-csv: billing/latest-export.csv
    fail-on-severity: CRITICAL   # optional: fail the check on CRITICAL anomalies
```

Re-runs on the same PR edit the existing comment instead of piling up new ones.
See [action.yml](action.yml) for all inputs/outputs and
[.github/workflows/example-usage.yml](.github/workflows/example-usage.yml) for
a working example (this repository dogfoods its own action against
[examples/sample-billing.csv](examples/sample-billing.csv) on every push).

## Use

### CLI

```bash
cloudsealed-jit billing-export.csv
cloudsealed-jit billing-export.csv --json > findings.json
cat export.csv | cloudsealed-jit - --type cost-forecast
```

### Library

```python
from cloudsealed_jit import parse_billing_csv, analyze

series = parse_billing_csv(open("export.csv").read())
result = analyze(series)

print(result.metrics.wastePercentage)
for r in result.recommendations:
    print(r.title, r.potentialSavings)
```

### HTTP service

```bash
docker run -p 8091:8091 cloudsealed/jit-optimization-engine
```

```
GET  /health
POST /v1/analyze-billing
```

```bash
curl -X POST localhost:8091/v1/analyze-billing \
  -H 'Content-Type: application/json' \
  -d '{"companyName":"Acme","csvContent":"date,cost\n2026-01-01,100\n..."}'
```

Set `JIT_OPTIMIZATION_API_KEY` to require an `X-Api-Key` header. Set
`JIT_MAX_CSV_BYTES` to change the 64 MB upload ceiling.

Response shape:

```jsonc
{
  "anomalies": [
    { "date": "2026-01-31", "expectedCost": 99.0, "actualCost": 500.0,
      "deviation": 405.05, "zScore": 7.82, "severity": "CRITICAL",
      "description": "Spend above the day-of-week baseline by USD 401.00 (405.1%)." }
  ],
  "metrics": {
    "averageDailyCost": 106.32,
    "stdDeviation": 51.69,
    "sharpeRatio": 2.06,       // spend stability: mean / stddev of daily cost
    "wastePercentage": 6.29    // share of total spend above the baseline
  },
  "recommendations": [
    { "title": "...", "description": "...", "potentialSavings": 200.5, "effort": "MEDIUM" }
  ],
  "summary": "..."
}
```

`sharpeRatio` is a **spend stability ratio** — mean daily cost divided by its
standard deviation, the reciprocal of the coefficient of variation. Higher
means more predictable spend. It is named for the field in the consuming API
contract; it is not a risk-adjusted return.

## Supported exports

| Provider | Date column | Cost column |
|---|---|---|
| AWS Cost and Usage Report | `lineItem/UsageStartDate` | `lineItem/UnblendedCost` |
| GCP billing export | `usage_start_time` | `cost` |
| Azure cost export | `Date`, `UsageDateTime` | `Cost`, `CostInBillingCurrency` |
| Generic | heuristic | heuristic |

Line items are aggregated to calendar days. Days with no line items are
inserted as zero-spend days rather than skipped. Rows that cannot be parsed are
counted and reported in the summary rather than dropped silently.

## Development

```bash
pip install -e ".[jit,api,dev]"
pytest
```

The test suite builds synthetic exports whose correct answer is known in
advance — a known spike at a known date, a known weekend-idle service, a stable
series that must produce no findings — so the assertions test behaviour rather
than the current output.

## License

MIT. See [LICENSE](LICENSE).
