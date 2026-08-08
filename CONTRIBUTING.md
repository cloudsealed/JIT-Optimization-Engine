# Contributing to JIT-Optimization-Engine

Thank you for your interest in contributing! This project is maintained as a community resource for learning and using JIT compilation and FinOps analytics.

## Before You Contribute

Please read [COMMUNITY.md](COMMUNITY.md) to understand the project's purpose and boundaries.

## Contribution Guidelines

### Types of Contributions We Accept ✅

- **Bug fixes** in existing code
- **Performance improvements** (with benchmarks on synthetic data)
- **Documentation updates** (clarifications, tutorials, examples)
- **New metrics/algorithms** that are generalizable (Z-Score variants, new FinOps heuristics)
- **Example notebooks** demonstrating FinOps analysis
- **Tests** using synthetic telemetry data
- **Numba optimizations** for compute-heavy functions

### Types We Don't Accept ❌

- Real customer billing data or production logs
- Anonymized data without explicit consent from data owner
- Credentials, API keys, or connection strings
- CloudSealed-specific implementations or integrations
- Changes to remove/modify MIT license

## Process

1. **Check existing issues/PRs** to avoid duplicates
2. **Open an issue first** for significant changes (discuss before implementing)
3. **Fork and branch**: `git checkout -b feature/your-feature-name`
4. **Keep data synthetic**: All examples use generated data (`numpy.random`, `faker`)
5. **Update docs**: Explain what your change does and why
6. **Add tests**: Use synthetic data; all tests must be reproducible
7. **Run locally**: Ensure tests pass with `pytest`
8. **Benchmark**: If optimizing, provide before/after metrics (synthetic data only)
9. **Create PR**: Link the issue, describe changes clearly

## Data Policy

**You MUST NOT commit**:
- Real billing data (CSV, JSON, XLSX from real companies)
- Production logs or telemetry
- Customer/infrastructure metrics (even anonymized)
- Cloud credentials (AWS keys, GCP tokens, Azure secrets)
- Internal infrastructure details (account IDs, resource names)

**You CAN use**:
- Synthetic data generated with `numpy.random`, `random.seed()`, or `faker`
- Public datasets (cite source, respect license)
- Realistic but fake scenarios (e.g., "Company A spent $2500/day" as example)

### Quick Check Before Committing

```bash
# Ensure no secrets are committed
git-secrets --scan

# Verify no real company names or account IDs
grep -r "AWS\|GCP\|Azure" . --include="*.csv" --include="*.json"

# Verify all test data is synthetic
grep -r "2026-06\|2026-07" tests/  # Check for real recent dates in sensitive contexts
```

## Code Style

- Follow PEP 8 (use `black` if available)
- Meaningful variable names
- Docstrings for all functions
- Type hints where possible
- Keep functions focused and testable

## Numba Optimization

When using Numba JIT:
- Add `@numba.jit(nopython=True)` for CPU-intensive functions
- Document Numba limitations (no list comprehensions, certain types)
- Benchmark with synthetic data to prove speedup
- Include fallback pure-Python version for testing

```python
@numba.jit(nopython=True)
def compute_z_score(values):
    """Calculate Z-score. Numba-compiled for performance."""
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / std
```

## Documentation

When adding a feature:
1. Update README.md if it's user-facing
2. Add docstrings with purpose, inputs, outputs
3. Include example usage in docstrings or `examples/`
4. Explain FinOps concepts in comments
5. Link to external resources (AWS FinOps, NIST, etc.)

## Testing

- Write tests for any new algorithms
- Use synthetic data only (reproducible seeds)
- Verify tests pass: `pytest tests/`
- Test edge cases (empty data, single value, extreme values)
- Benchmark performance: `pytest --benchmark-only`

## Performance & Benchmarking

If you optimize code:
1. Run benchmarks before/after on synthetic data
2. Document hardware (CPU, RAM, OS)
3. Include benchmark results in PR description
4. Ensure improvements work with real-world data shapes

Example:
```
Before: 1.2s for 1M records
After:  0.3s for 1M records (75% improvement)
Method: Numba JIT compilation + parallelism
Hardware: 8-core i7, 16GB RAM
```

## Licensing

By contributing, you agree that your work will be distributed under the MIT license. You confirm you have the right to contribute and that you're not violating any third-party licenses (especially for external datasets).

## Example Datasets (OK to Use)

- [Kaggle FinOps Datasets](https://www.kaggle.com/search?q=cloud+cost)
- [AWS Sample Data](https://docs.aws.amazon.com/cur/latest/userguide/sample-data.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- Any dataset with public license (CC0, CC-BY, etc.)

**Always cite the source in your commit message and documentation.**

## Questions?

- Check [COMMUNITY.md](COMMUNITY.md) FAQ
- Open an issue with `[question]` tag
- Review existing PRs for examples of what's accepted

Thank you for contributing to the community! 🙏
