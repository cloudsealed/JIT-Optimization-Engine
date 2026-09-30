import numpy as np
try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

def _make_streaming_evaluator():
    def _eval(current_val, window_samples, z_threshold):
        # Mediana da janela deslizante
        sorted_win = np.sort(window_samples)
        n = len(sorted_win)
        mid = n // 2
        if n % 2 == 1:
            med = sorted_win[mid]
        else:
            med = (sorted_win[mid - 1] + sorted_win[mid]) / 2.0

        # Desvio absoluto mediano (MAD)
        devs = np.abs(window_samples - med)
        sorted_devs = np.sort(devs)
        if n % 2 == 1:
            mad = sorted_devs[mid]
        else:
            mad = (sorted_devs[mid - 1] + sorted_devs[mid]) / 2.0

        if mad < 1e-6:
            return 0.0, False

        # Modified Z-Score (Iglewicz & Hoaglin)
        z_score = 0.6745 * (current_val - med) / mad
        is_spike = z_score >= z_threshold
        return z_score, is_spike

    if HAVE_NUMBA:
        return numba.njit(fastmath=True)(_eval)
    return _eval

evaluate_job_anomaly = _make_streaming_evaluator()

class PipelineCostProfiler:
    """
    Monitora execuções de pipelines de CI em tempo real.
    Garante alertas de anomalia de custo antes do fechamento do ciclo.
    """
    def __init__(self, window_size: int = 30, z_threshold: float = 3.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history = []

    def observe_and_score(self, cost_usd: float) -> dict:
        if len(self.history) < 7:
            self.history.append(cost_usd)
            return {"z_score": 0.0, "is_anomaly": False, "status": "warmup"}

        window = np.array(self.history[-self.window_size:], dtype=np.float64)
        z_score, is_anomaly = evaluate_job_anomaly(cost_usd, window, self.z_threshold)
        self.history.append(cost_usd)

        return {
            "cost": cost_usd,
            "z_score": float(z_score),
            "is_anomaly": bool(is_anomaly),
            "action": "HALT_PIPELINE" if is_anomaly else "CONTINUE"
        }
