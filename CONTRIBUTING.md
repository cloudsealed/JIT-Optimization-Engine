# Contributing

Contributions are welcome — bug reports, feature requests, documentation improvements, and code changes.

## Setup

```bash
git clone https://github.com/cloudsealed/JIT-Optimization-Engine
cd JIT-Optimization-Engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[jit,api,dev]"
```

## Running tests

```bash
pytest                            # all tests
pytest tests/test_compiler.py -v  # compiler tests only
pytest -x                         # stop on first failure
```

Numba is optional — tests that require it are skipped automatically if it is not installed. To test the JIT paths:

```bash
pip install "numba>=0.57"
pytest tests/test_compiler.py -v
```

## Running benchmarks

```bash
python benchmarks/compiler_benchmark.py
python benchmarks/masking_benchmark.py
```

## Code style

- `ruff` for linting: `ruff check .`
- `black` for formatting: `black .`
- Type hints on all public functions

## Pull requests

1. Fork, create a branch (`git checkout -b fix/my-fix`)
2. Add tests for any changed behaviour
3. Run `pytest` — all tests must pass
4. Open a PR against `main`

For large changes, open an issue first to discuss the approach.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

## Requesting features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
