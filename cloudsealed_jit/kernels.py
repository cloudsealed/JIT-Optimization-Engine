"""
Numerical kernels for baseline estimation.

Cloud spend series contain the very spikes we are trying to detect. Estimating
a baseline with a mean and flagging outliers with a standard-deviation z-score
fails on this data: a handful of large spikes inflate the standard deviation
enough to hide themselves and everything near them. This is the classic
masking effect.

We therefore use robust statistics throughout:

    baseline  = centred rolling median
    scale     = median absolute deviation (MAD)
    z         = 0.6745 * (x - median) / MAD          [Iglewicz & Hoaglin, 1993]

The 0.6745 constant makes the MAD a consistent estimator of the standard
deviation for normally distributed data, so the resulting score keeps the
familiar "number of deviations" interpretation while staying resistant to
roughly half the sample being contaminated.

A centred rolling median over a window is O(n * w log w) in the naive form.
For the daily series produced by :mod:`cloudsealed_jit.parsing` that is
irrelevant, but the same kernels run over per-service series and over hourly
exports spanning years, where it is not. Numba compiles them to native code
via LLVM; the pure NumPy path is kept as a fallback so the package installs
and runs without a compiler toolchain.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rolling_median", "mad", "modified_zscores", "JIT_ENABLED"]

_MAD_TO_SIGMA = 0.6745


try:  # pragma: no cover - depends on the runtime environment
    from numba import njit

    JIT_ENABLED = True
except ImportError:  # pragma: no cover
    JIT_ENABLED = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in used when numba is unavailable."""
        def wrap(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True)
def _rolling_median_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median with shrinking windows at the edges."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    half = window // 2
    buffer = np.empty(window, dtype=np.float64)

    for i in range(n):
        lo = i - half
        hi = i + half + 1
        if lo < 0:
            lo = 0
        if hi > n:
            hi = n

        size = hi - lo
        for k in range(size):
            buffer[k] = values[lo + k]

        # Insertion sort: window is small and the data is near-sorted.
        for a in range(1, size):
            key = buffer[a]
            b = a - 1
            while b >= 0 and buffer[b] > key:
                buffer[b + 1] = buffer[b]
                b -= 1
            buffer[b + 1] = key

        if size % 2 == 1:
            out[i] = buffer[size // 2]
        else:
            out[i] = 0.5 * (buffer[size // 2 - 1] + buffer[size // 2])

    return out


def rolling_median(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Centred rolling median.

    Args:
        values: 1-D series.
        window: window length in samples. Forced odd and clamped to the series
            length so short inputs degrade to a global median rather than fail.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return series.copy()

    window = max(1, min(int(window), series.size))
    if window % 2 == 0:
        window -= 1
    if window < 1:
        window = 1
    return _rolling_median_kernel(series, window)


def mad(values: np.ndarray, centre: np.ndarray | float | None = None) -> float:
    """Median absolute deviation about ``centre`` (default: the median)."""
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return 0.0
    if centre is None:
        centre = float(np.median(series))
    return float(np.median(np.abs(series - centre)))


def modified_zscores(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Robust z-scores of ``values`` against ``baseline``.

    Falls back to a standard-deviation scale when the MAD is zero, which
    happens on series that are constant apart from a few spikes. If both
    scales are zero the series is flat and every score is zero.
    """
    series = np.asarray(values, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    residuals = series - base

    scale = float(np.median(np.abs(residuals)))
    if scale > 0.0:
        return _MAD_TO_SIGMA * residuals / scale

    fallback = float(np.std(residuals))
    if fallback > 0.0:
        return residuals / fallback

    return np.zeros_like(series)
