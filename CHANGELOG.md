# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] — 2026-09-30

### Added
- **CloudSealed Compiler v2** — complete rewrite of `compiler.py`:
  - `FStringTransformer`: AST rewriter that converts f-strings (including format specs) to `str()` + concatenation that Numba's nopython mode accepts
  - `ListCompTransformer`: AST rewriter that converts list comprehensions to explicit `for` loops, handling `if` filters
  - `map_python_type_to_numba()`: translates PEP-484 type hints (`Optional[T]`, `List[T]`, `Tuple[T,...]`, `Dict[K,V]`, NumPy scalars, `np.ndarray`) into Numba IR types
  - `CloudSealedCompileError`: human-readable error wrapping Numba compilation failures, with pattern-based hints and a `NUMBA_DISABLE_JIT=1` tip
  - `compilation_stats()` / `reset_compilation_stats()`: global registry of per-function compile time, cache status, and Numba signature
- **`@jitdataclass` v2** — `dataclass_compiler.py` rewritten:
  - Supports default field values (`x: float = 0.0`)
  - User-defined methods on the class are compiled via Numba automatically
  - Nested `@jitdataclass` fields resolve to the correct Numba `StructRef` type
  - Registry (`_jitclass_registry`) for inter-class type resolution
- **Test suite** — `tests/test_compiler.py`: 38 tests covering every AST transform, type mapping case, `@jit` decorator behaviour, diagnostics, `@jitdataclass`, and compilation stats
- **Benchmark suite** — `benchmarks/compiler_benchmark.py`: measures compile time, runtime speedup, kwargs overhead, and AST transform latency

### Changed
- `numba` is fully optional; all compiler and dataclass paths fall back gracefully to pure Python/dataclass mode without errors
- `__init__.py` exports `jit`, `jitdataclass`, `compilation_stats`, `reset_compilation_stats`, `CloudSealedCompileError`

---

## [0.3.x] — 2026

### Added
- Streaming MAD pipeline profiler (`pipeline_profiler.py`) with Numba-accelerated kernels
- GitHub Action (`streaming-action.yml`) for Marketplace distribution: posts viral PR comments with Markdown billing graphs
- Viral PR commenting with Markdown report generation
- Trend-aware spend forecast with budget-breach day prediction

### Changed
- `numba` moved to optional dependency (`[jit]` extra)

---

## [0.2.1] — 2025

### Added
- FOCUS 1.0 billing format support (vendor-neutral multi-cloud standard)
- Slack / generic webhook alerting with threshold filtering
- Self-contained HTML report (`--html`) with inline CSS
- Comparison table vs Vantage / CloudZero / AWS Cost Anomaly Detection
- FAQ section in README

### Fixed
- Module path and dependency installation issues in streaming action

---

## [0.2.0] — 2025

### Added
- Docker image and CI pipeline
- GCP-calibrated realism benchmark
- SOTA comparison and Monte Carlo robustness benchmarks
- `benchmarks/masking_benchmark.py`: measures F1 score vs textbook mean+stddev approach
- `METHODOLOGY.md`: full derivation of rolling-median + MAD scoring

---

## [0.1.0] — 2025

### Added
- Initial `cloudsealed_jit` package: cloud billing waste analysis engine
- `parse_billing_csv()`: heuristic CSV parser supporting AWS CUR, GCP, Azure exports
- `analyze()`: rolling-median baseline + MAD z-score anomaly detection
- Day-of-week seasonality correction (weekday factors from median ratios)
- Waste percentage metric and recommendation engine
- `@jit` / `@jitdataclass` decorators (v1): kwargs unroller and basic Numba wrapper
- FastAPI HTTP service (`api.py`), CLI (`cli.py`), Slack notify (`notify.py`)
