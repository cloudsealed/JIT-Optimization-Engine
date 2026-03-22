import numpy as np
import os
import sys

# Ensure the src directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engine.jit_kernels import execute_hpmce_kernel

def test_hpmce_kernel_logic():
    """
    Unit test to validate the stochastic drift mathematical logic 
    inside the LLVM-compiled JIT kernel.
    """
    # 1. Setup Mock Data
    size = 100
    values = np.linspace(100, 200, size).astype(np.float64)  # Trending up
    times_ms = np.arange(size).astype(np.float64) * 1000     # 1 second intervals
    telemetry = np.ones(size).astype(np.float64)             # Positive signals
    weights = np.ones(size).astype(np.float64)
    sig_indices = np.array([0, 10, 20]).astype(np.int64)
    
    # 2. Define Parameters
    target = 10.0
    stop = 5.0
    window_ms = 50000.0 # 50 seconds window

    # 3. Execute Kernel
    g, l, o, g_w, l_w = execute_hpmce_kernel(
        values, times_ms, telemetry, weights, sig_indices, target, stop, window_ms
    )

    # 4. Assertions (Validation)
    assert g > 0, "Engine failed to detect positive target drift."
    assert l == 0, "Engine incorrectly triggered stop loss on an upward trend."
    assert (g_w + l_w) > 0, "Weight aggregation failed."
    print("Mission-Critical Kernel Logic: VALIDATED")

if __name__ == "__main__":
    test_hpmce_kernel_logic()