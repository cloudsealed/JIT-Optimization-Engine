from numba import njit
import numpy as np

@njit
def execute_hpmce_kernel(values, times_ms, telemetry, weights, sig_indices, target, stop, window_ms):
    """
    HIGH-PERFORMANCE JIT KERNEL (LLVM-compiled):
    Executes low-latency statistical simulations at machine-code level.
    Designed to bypass the Python GIL for true parallel processing 
    in mission-critical fiscal environments.
    """
    g, l, o, g_w, l_w = 0.0, 0.0, 0.0, 0.0, 0.0
    last_entry_t = -999999999.0
    cooldown = 30000.0 # 30s Safety cooldown interval
    
    for k in range(len(sig_indices)):
        idx = sig_indices[k]
        if times_ms[idx] - last_entry_t < cooldown: continue
        
        last_entry_t = times_ms[idx]
        entry_val = values[idx]
        side = 1.0 if telemetry[idx] > 0 else -1.0
        
        end_t = times_ms[idx] + window_ms
        for j in range(idx + 1, len(values)):
            if times_ms[j] > end_t: break
            
            drift = (values[j] - entry_val) * side
            if drift >= target:
                g += 1.0; g_w += weights[idx]; break
            elif drift <= -stop:
                l += 1.0; l_w += weights[idx]; break
        else: o += 1.0
        
    return g, l, o, g_w, l_w