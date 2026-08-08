# Architecture — cloudsealed-jit

`cloudsealed-jit` detects structural waste in cloud billing exports. It is a
library, a CLI, and an HTTP service, sharing one analysis core. This document
describes how the pieces fit together and why each boundary is where it is.

```
                    ┌────────────────────────────────────────────┐
                    │                cloudsealed_jit               │
                    │                                              │
  billing export ──▶│  parsing.py ──▶ BillingSeries               │
  (AWS/GCP/Azure/   │      │            (daily totals + per-svc)   │
   generic CSV)     │      │                                       │
                    │      ▼                                       │
                    │  analysis.py ──▶ AnalysisResult              │
                    │      │  ├── kernels.py  (rolling median,     │
                    │      │  │                MAD, modified z)     │
                    │      │  └── baseline → anomalies → waste      │
                    │      │              → recommendations         │
                    │      ▼                                       │
                    │  ┌─────────┬──────────┬─────────────────┐    │
                    │  │ cli.py  │ api.py   │ (import as lib)  │    │
                    │  └─────────┴──────────┴─────────────────┘    │
                    └────────────────────────────────────────────┘
```

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `parsing.py` | Normalise heterogeneous provider exports into one gap-free daily series (`BillingSeries`); count rejected rows instead of dropping them silently. | stdlib only |
| `kernels.py` | Numerical primitives: centred rolling median, MAD, modified z-score. numba-compiled when available, identical pure-NumPy fallback otherwise. | numpy, (numba) |
| `analysis.py` | Baseline model, anomaly detection, waste metric, and derived recommendations. The method lives here. | numpy, parsing, kernels |
| `cli.py` | `cloudsealed-jit export.csv [--json]` — run analysis without a server. | analysis, parsing |
| `api.py` | FastAPI: `GET /health`, `POST /v1/analyze-billing`, `X-Api-Key` auth, payload ceiling. | analysis, parsing, fastapi |

`api.py` at the repository root is a thin compatibility shim re-exporting
`cloudsealed_jit.api` for deployments that still run `uvicorn api:app`.

## Design boundaries and why

**Parsing is separate from analysis.** Providers disagree on schema, date
format, decimal convention, and granularity (one CSV row is a line item, not a
day). Everything provider-specific is confined to `parsing.py`; `analysis.py`
only ever sees a clean daily series. Adding a new export format touches one
module and no statistics.

**Kernels are separate from the method.** `kernels.py` knows nothing about
billing — it is rolling median, MAD, and modified z-score over arbitrary
arrays. That keeps the numerics independently testable and lets the optional
numba acceleration be a drop-in with an identical NumPy fallback, so the
package installs and runs without a compiler toolchain.

**One core, three surfaces.** The CLI, the HTTP API, and direct library import
all call the same `analyze()`. There is no second code path that could drift:
what the service returns is exactly what the CLI prints and what a library
caller computes.

**JIT is an accelerator, never a correctness dependency.** numba compiles the
hot kernels to native code via LLVM for large per-service or hourly series. It
changes speed, not results — the pure-NumPy path is kept and tested so numbers
are byte-reproducible with or without it.

## Data contract

`BillingSeries` (`parsing.py`) → `AnalysisResult` (`analysis.py`). The HTTP
response shape is the field-for-field contract consumed by the CloudSealed
platform (`framework4d-jit-client.ts`) and is treated as stable: field names
and types do not change without versioning the endpoint. See `METHODOLOGY.md`
for how each output number is derived.

## Reproducibility

`benchmarks/masking_benchmark.py` regenerates the numbers in `METHODOLOGY.md`
from a fixed seed, quantifying the advantage of the robust method over the
textbook mean+standard-deviation approach. It has no dependency on private
data and runs in seconds.
