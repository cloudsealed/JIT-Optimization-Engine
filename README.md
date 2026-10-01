# 🛡️ CloudSealed JIT-Optimization-Engine

[![PyPI version](https://img.shields.io/pypi/v/cloudsealed-jit.svg)](https://pypi.org/project/cloudsealed-jit/)
[![PyPI downloads](https://img.shields.io/pypi/dm/cloudsealed-jit.svg)](https://pypi.org/project/cloudsealed-jit/)
[![GitHub CI](https://github.com/cloudsealed/JIT-Optimization-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/cloudsealed/JIT-Optimization-Engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cloudsealed/JIT-Optimization-Engine/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/cloudsealed/JIT-Optimization-Engine?style=flat)](https://github.com/cloudsealed/JIT-Optimization-Engine/stargazers)

**⚡️ Fast, ergonomic JIT compilation for Python + FinOps‑aware anomaly detection**

* Zero‑boilerplate JIT (`@jit`, `@jitdataclass`) with kwargs & f‑strings support.
* AI Cost Firewall GitHub Action that blocks costly PRs and posts viral comments.
* Optional Numba – works out‑of‑the‑box without it.
* Extensible – plug in custom risk rules or SIMD C# kernels.

**Stop AWS Bill Shocks BEFORE they happen.**

CloudSealed JIT is the only FinOps engine that runs **inside your CI/CD pipeline in real-time**. Using Numba-accelerated Streaming MAD algorithms (Machine Learning), it intercepts and blocks anomalous cloud deployments in microseconds, before they hit production and drain your budget.

> *"It's like a firewall, but for your AWS/GCP Invoice."*

[![CI](https://github.com/cloudsealed/JIT-Optimization-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/cloudsealed/JIT-Optimization-Engine/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cloudsealed-jit.svg)](https://pypi.org/project/cloudsealed-jit/)
[![PyPI downloads](https://img.shields.io/pypi/dm/cloudsealed-jit.svg)](https://pypi.org/project/cloudsealed-jit/)
[![Docker pulls](https://img.shields.io/docker/pulls/cloudsealed/jit-optimization-engine.svg)](https://hub.docker.com/r/cloudsealed/jit-optimization-engine)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## 🔥 Why CloudSealed is taking over FinOps:
1. **Zero-Friction GitHub Action**: Add 2 lines to your `.yml` and your pipeline is financially protected. No credit cards, no SaaS dashboards, no 3-month integrations.
2. **Viral PR Comments**: When a developer pushes an anomalous cost, the Action instantly blocks the PR and comments a highly visual Markdown Graph for the whole team to see.
3. **Microsecond Latency**: Built with `numba` `@njit(fastmath=True)`, it calculates complex streaming anomalies in 13.4 microseconds per event. It does not slow down your pipeline.

---

## 🧠 The CloudSealed Compiler

While this tool is known for FinOps, its core is an open-source **ergonomic JIT compiler** that bridges the gap between idiomatic Python and Numba's nopython mode. If you're building numerical engines and tired of Numba's limitations, you can use it directly.

### The problem with raw Numba

```python
# ✗ Raw Numba — this fails at compile time
@njit
def process(data: np.ndarray, label: str) -> float:
    result = [x * 2 for x in data]   # ✗ list comprehensions not supported
    print(f"Processing {label}")      # ✗ f-strings not supported
    return result[0]

# ✓ CloudSealed @jit — idiomatic Python, LLVM speed
from cloudsealed_jit import jit, jitdataclass

@jit()
def process(data: np.ndarray, label: str) -> float:
    result = [x * 2 for x in data]   # ✓ rewritten to explicit loop at AST level
    print(f"Processing {label}")      # ✓ rewritten to str() + concatenation
    return result[0]
```

### What the compiler does

The `@jit` decorator applies four transforms before handing code to Numba's LLVM backend:

| Transform | Problem solved | How |
|---|---|---|
| **F-string rewriter** | `f"{x:.2f}"` crashes in nopython | AST: `JoinedStr` → `str(x) + …` |
| **List comp rewriter** | `[x for x in arr]` not allowed | AST: `ListComp` → `init + for loop + append` |
| **PEP-484 type mapper** | Manual string signatures (`"float64[:](float64)"`) | Reads `__annotations__`, maps to Numba IR types |
| **Kwargs unroller** | `func(a=1, b=2)` not supported in nopython | Wraps compiled fn with `inspect.Signature.bind` |

Type mapping supports: `int`, `float`, `bool`, `str`, numpy scalars, `np.ndarray`, `Optional[T]`, `List[T]`, `Tuple[T, ...]`, `Dict[K, V]`, and any `@jitdataclass`.

### @jitdataclass — Python dataclasses as Numba C-structs

```python
from cloudsealed_jit import jit, jitdataclass

@jitdataclass
class Vec2:
    x: float
    y: float

@jit()
def magnitude(v: Vec2) -> float:
    return (v.x ** 2 + v.y ** 2) ** 0.5

v = Vec2(3.0, 4.0)
magnitude(v)  # → 5.0, fully compiled to machine code
```

`@jitdataclass` converts a standard Python dataclass into a Numba `@jitclass` (contiguous memory C-struct), preserving default values, compiling user-defined methods, and resolving nested `@jitdataclass` fields.

### Benchmarks

Measured on x86-64 Linux, Python 3.11 (Numba optional — pure-Python fallback is automatic):

| Operation | Time |
|---|---|
| F-string AST rewrite | ~680 µs (p50) |
| List comprehension AST rewrite | ~680 µs (p50) |
| Kwargs dispatch overhead | ~0.05 µs/call |

Reproduce: `python benchmarks/compiler_benchmark.py`

### Human-readable errors

When Numba rejects a function, `CloudSealedCompileError` replaces the raw LLVM traceback with a plain-English message:

```
CloudSealed compilation failed for 'my_func'.
Numba error: ...
Possible causes:
  → Sets are not supported in Numba nopython mode. Use a typed List or array instead.
Tip: run with NUMBA_DISABLE_JIT=1 to bypass compilation and debug in pure Python.
```

### Compilation stats

```python
from cloudsealed_jit import compilation_stats

stats = compilation_stats()
# [{'name': 'my_module.my_func', 'compile_time_us': 18.3,
#   'from_cache': True, 'signature': 'float64(float64, float64)'}]
```

### Why not Cython / mypyc / raw Numba?

| | cloudsealed @jit | raw Numba @njit | Cython | mypyc |
|---|---|---|---|---|
| Type hints drive compilation | ✓ | ✗ (manual strings) | ✓ (pxd files) | ✓ |
| f-strings | ✓ (AST rewrite) | ✗ | ✓ | ✓ |
| List comprehensions | ✓ (AST rewrite) | ✗ | ✓ | ✓ |
| Kwargs support | ✓ | ✗ | ✓ | ✓ |
| Pure Python fallback | ✓ | ✗ | ✗ | ✗ |
| Compilation build step | ✗ | ✗ | ✓ | ✓ |
| Targets LLVM / SIMD | ✓ | ✓ | ✗ | ✗ |

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

## Alerts

Send the result to Slack (or any generic webhook listener) when an anomaly
reaches a severity threshold, without standing up a dashboard:

```bash
cloudsealed-jit billing-export.csv --webhook-url "$SLACK_WEBHOOK_URL"
cloudsealed-jit billing-export.csv --webhook-url "$SLACK_WEBHOOK_URL" --webhook-min-severity CRITICAL
```

A Slack incoming-webhook URL (`hooks.slack.com`) is auto-detected and rendered
as a formatted message; any other URL receives the full JSON result, so it
works as-is with Teams, PagerDuty, or a custom listener. Nothing is sent on a
quiet run — the default threshold is `HIGH`. The same behaviour is available
in the HTTP API via the optional `webhookUrl` field on `/v1/analyze-billing`.
A failed webhook is logged and never fails the analysis.

## Use

### CLI

```bash
cloudsealed-jit billing-export.csv
cloudsealed-jit billing-export.csv --json > findings.json
cloudsealed-jit billing-export.csv --html report.html
cloudsealed-jit billing-export.csv --type cost-forecast
cloudsealed-jit billing-export.csv --budget 50000     # when will the trend cross it?
```

`--html` writes a self-contained report (inline CSS, no CDN) alongside
whatever other output is requested — open it straight from disk, or attach it
to an email.

## Forecast

Anomaly detection is reactive — it tells you *after* a spike. `--type
cost-forecast` (or `--budget`) adds a **proactive** projection: it extrapolates
the observed level trend (rolling median slope) forward, carrying the same
weekday seasonality the baseline uses, and — given a budget — predicts the day
the trend crosses it:

```
Projected 30-day spend USD 10,901.01 (trend rising, USD +4.89/day).
At this trend the USD 6,000.00 budget is crossed on day 18 of the horizon.
```

It's a mechanical extrapolation, not a probabilistic prediction — the
`projectedSpend`, `dailyTrend`, and `budgetBreachDay` fields state exactly what
was computed, so the number is auditable rather than a black-box guess. The
forecast is added to the JSON/HTTP response only when requested, so the default
response shape is unchanged.

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
| **FOCUS 1.0** | `ChargePeriodStart` | `BilledCost` |
| Generic | heuristic | heuristic |

[FOCUS](https://focus.finops.org/) is the FinOps Open Cost and Usage
Specification — the vendor-neutral billing format AWS, GCP, Azure and OCI now
export natively. One FOCUS export runs through this analyser unchanged
regardless of which cloud produced it, so a multi-cloud estate is analysed the
same way end to end.

Line items are aggregated to calendar days. Days with no line items are
inserted as zero-spend days rather than skipped. Rows that cannot be parsed are
counted and reported in the summary rather than dropped silently.

## How this compares to other cloud cost anomaly detection tools

cloudsealed-jit does one thing — find cost spikes in a billing export — and
does not try to be a full FinOps platform. If you need a dashboard, live
cloud API connectors, Kubernetes cost allocation, or RI/Savings Plan
management, a commercial platform is the right tool; this is a lighter,
composable piece for the detection step specifically.

| | cloudsealed-jit | Vantage / CloudZero / Finout | AWS Cost Anomaly Detection |
|---|---|---|---|
| Method | Rolling-median + MAD (open, documented, benchmarked) | Proprietary ML | Proprietary ML |
| Multi-cloud | AWS/GCP/Azure/generic CSV | Yes (paid) | AWS only |
| Deployment | Library, CLI, self-hosted API, GitHub Action, MCP tool | SaaS | AWS-managed |
| Cost | Free, open source (MIT) | Paid, usage-based | Free (AWS-native) |
| Dashboard | None (by design — pair with your own) | Yes | Yes |
| Slack/webhook alerts | Yes | Yes | Yes (SNS) |

## FAQ

**How do I detect cost anomalies in an AWS billing export with Python?**
Install `cloudsealed-jit`, then `cloudsealed-jit your-cur-export.csv`. See
[Install](#install) and [Use](#use) above.

**Why not just use mean + standard deviation for anomaly detection?**
Because a handful of large spikes inflates the standard deviation enough to
hide themselves and everything smaller — see
["The problem"](#the-problem) and the measured comparison in
["Does it actually work better?"](#does-it-actually-work-better).

**Can an AI agent call this directly instead of me running the CLI?**
Yes — see [cloudsealed-mcp](https://github.com/cloudsealed/cloudsealed-mcp),
an MCP server that exposes this as a tool for Claude Code, Claude Desktop,
Cursor, and other MCP clients.

**Does this replace AWS Cost Anomaly Detection / GCP's built-in tools?**
Not necessarily — it's cloud-agnostic and works on data you've already
exported, so it's useful alongside native tools when you need one method
across multiple clouds, or want the detection logic to run in CI as a
GitHub Action.

## 🛠️ Extending CloudSealed (Build Your Own Firewall)

**This engine is built to be hackable.** Don't like our Streaming MAD math? Want to write a custom Numba JIT algorithm? **Fork this repository!**

1. **Custom AI Math**: Open `cloudsealed_jit/pipeline_profiler.py` and write your own anomaly logic. The GitHub Action will instantly use it.
2. **New Cloud Parsers**: Want to add support for DigitalOcean or Oracle Cloud billing? Fork the repo and add a new parser.

**We love Community Forks and Pull Requests!** Check our open `good first issue` tickets to start contributing immediately.

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

---

If this saved you from a false-positive cost alert, a star helps other teams find it. Bug reports and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
