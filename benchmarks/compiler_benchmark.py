"""
CloudSealed Compiler Benchmark
==============================

Measures compilation overhead, runtime speedup, and kwargs dispatch
latency for the CloudSealed @jit decorator vs raw Numba and pure Python.

Run:
    python benchmarks/compiler_benchmark.py
"""

import time
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_us(func, *args, n=1000):
    """Average execution time in microseconds over n calls."""
    # Warm up
    func(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        func(*args)
    return (time.perf_counter() - t0) / n * 1e6


def _compilation_time_us(decorator_fn):
    """Time a single decoration (compilation) in microseconds."""
    src = '''
def _bench_target(a: float, b: float) -> float:
    return a * b + a - b
'''
    ns = {"float": float}
    exec(src, ns)
    func = ns["_bench_target"]

    t0 = time.perf_counter()
    decorator_fn(func)
    return (time.perf_counter() - t0) * 1e6


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_compilation_time():
    """Compare @jit decoration time vs raw @njit."""
    try:
        from numba import njit
        from cloudsealed_jit.compiler import jit, reset_compilation_stats
    except ImportError:
        print("SKIP: numba not installed")
        return

    reset_compilation_stats()

    raw_time = _compilation_time_us(lambda f: njit(cache=False)(f))
    cs_time = _compilation_time_us(lambda f: jit(cache=False)(f))

    overhead = cs_time - raw_time
    print(f"  Raw Numba @njit:       {raw_time:>10.1f} µs")
    print(f"  CloudSealed @jit:      {cs_time:>10.1f} µs")
    print(f"  AST transform overhead:{overhead:>10.1f} µs ({overhead/raw_time*100:.1f}% of Numba)")


def bench_runtime_speedup():
    """Compare @jit compiled vs pure Python on a numeric kernel."""
    try:
        from cloudsealed_jit.compiler import jit
    except ImportError:
        print("SKIP: numba not installed")
        return

    def pure_python_dot(a, b):
        total = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total

    @jit(cache=False)
    def jit_dot(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for i in range(a.shape[0]):
            total += a[i] * b[i]
        return total

    sizes = [100, 1_000, 10_000]
    for n in sizes:
        a = np.random.rand(n)
        b = np.random.rand(n)
        py_us = _time_us(pure_python_dot, a, b, n=100 if n > 1000 else 1000)
        jit_us = _time_us(jit_dot, a, b, n=100 if n > 1000 else 1000)
        speedup = py_us / jit_us if jit_us > 0 else float("inf")
        print(f"  n={n:>6}: Python {py_us:>10.1f} µs | @jit {jit_us:>8.1f} µs | {speedup:>6.1f}x speedup")


def bench_kwargs_overhead():
    """Measure the cost of kwargs dispatch vs positional-only."""
    try:
        from cloudsealed_jit.compiler import jit
    except ImportError:
        print("SKIP: numba not installed")
        return

    @jit(cache=False)
    def add(a: float, b: float) -> float:
        return a + b

    # Warm up
    add(1.0, 2.0)
    add(a=1.0, b=2.0)

    pos_us = _time_us(add, 1.0, 2.0, n=100_000)
    kw_us = _time_us(lambda: add(a=1.0, b=2.0), n=100_000)

    print(f"  Positional args: {pos_us:>8.3f} µs/call")
    print(f"  Keyword args:    {kw_us:>8.3f} µs/call")
    print(f"  Kwargs overhead: {kw_us - pos_us:>8.3f} µs/call")


def bench_fstring_transform():
    """Measure AST rewrite time for f-string heavy functions."""
    from cloudsealed_jit.compiler import rewrite_function_ast

    def lots_of_fstrings(a, b, c):
        x = f"value a: {a}"
        y = f"value b: {b}"
        z = f"combined: {a} + {b} = {c}"
        return x + y + z

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        rewrite_function_ast(lots_of_fstrings)
        times.append((time.perf_counter() - t0) * 1e6)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"  F-string AST rewrite: avg {avg:.1f} µs | p50 {p50:.1f} µs | p99 {p99:.1f} µs")


def bench_listcomp_transform():
    """Measure AST rewrite time for list comprehension transforms."""
    from cloudsealed_jit.compiler import rewrite_function_ast

    def with_listcomp(items):
        result = [x * 2 for x in items]
        filtered = [x for x in result if x > 5]
        return filtered

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        rewrite_function_ast(with_listcomp)
        times.append((time.perf_counter() - t0) * 1e6)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    print(f"  ListComp AST rewrite: avg {avg:.1f} µs | p50 {p50:.1f} µs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CloudSealed Compiler Benchmark Suite")
    print("=" * 60)

    print("\n1. Compilation Time (CloudSealed @jit vs raw Numba @njit)")
    bench_compilation_time()

    print("\n2. Runtime Speedup (@jit vs pure Python)")
    bench_runtime_speedup()

    print("\n3. Kwargs Dispatch Overhead")
    bench_kwargs_overhead()

    print("\n4. F-String AST Transform Latency")
    bench_fstring_transform()

    print("\n5. List Comprehension AST Transform Latency")
    bench_listcomp_transform()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
