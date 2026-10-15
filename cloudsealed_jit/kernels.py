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


@njit(cache=True)
def _linear_regression_kernel(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Cálculo ultrarrápido de regressão linear para burn rate via JIT."""
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0
    x_mean = 0.0
    y_mean = 0.0
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        x_diff = x[i] - x_mean
        numerator += x_diff * (y[i] - y_mean)
        denominator += x_diff * x_diff

    slope = numerator / denominator if denominator != 0.0 else 0.0
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict_burn_rate(
    spend_history: list[float] | np.ndarray,
    days_ahead: int = 15,
) -> dict[str, float]:
    """Prediz o gasto acumulado futuro de nuvem usando o kernel robusto compilado por JIT."""
    y = np.asarray(spend_history, dtype=np.float64)
    if y.size == 0:
        return {"current_total": 0.0, "projected_total": 0.0, "confidence": 0.0, "daily_trend": 0.0}

    n = y.size
    x = np.arange(1, n + 1, dtype=np.float64)

    # 1. Obter tendência linear acelerada
    slope, intercept = _linear_regression_kernel(x, y)

    # 2. Resíduos e estabilidade (MAD)
    baseline = slope * x + intercept
    z = modified_zscores(y, baseline)
    anomalies = np.sum(np.abs(z) >= 3.5)
    confidence = max(0.60, min(0.99, 1.0 - (float(anomalies) / float(n))))

    current_total = float(np.sum(y))
    # Projeção futura
    future_x = np.arange(n + 1, n + days_ahead + 1, dtype=np.float64)
    future_y = np.maximum(0.0, slope * future_x + intercept)
    projected_total = current_total + float(np.sum(future_y))

    return {
        "current_total": round(current_total, 2),
        "projected_total": round(projected_total, 2),
        "daily_trend": round(float(slope), 4),
        "confidence": round(confidence, 2),
    }


@njit(cache=True)
def _monte_carlo_simulation_kernel(
    n: int,
    n_sims: int,
    horizon: int,
    slope: float,
    intercept: float,
    sigma: float,
) -> np.ndarray:
    results = np.empty(n_sims, dtype=np.float64)
    for s in range(n_sims):
        sim_total = 0.0
        for d in range(1, horizon + 1):
            t = n + d
            drift = slope * t + intercept
            # Box-Muller transform para amostragem gaussiana no JIT
            u1 = np.random.random()
            while u1 <= 1e-12:
                u1 = np.random.random()
            u2 = np.random.random()
            z_norm = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            val = drift + z_norm * sigma
            if val < 0.0:
                val = 0.0
            sim_total += val
        results[s] = sim_total
    return results


def monte_carlo_finops_var(
    spend_history: list[float] | np.ndarray,
    n_simulations: int = 10000,
    horizon_days: int = 30,
    confidence_level: float = 0.99,
) -> dict[str, float]:
    """Calcula Value-at-Risk (VaR) e Expected Shortfall (CVaR) de FinOps via Monte Carlo JIT."""
    y = np.asarray(spend_history, dtype=np.float64)
    if y.size == 0:
        return {
            "expected_spend": 0.0,
            "var_spend": 0.0,
            "cvar_spend": 0.0,
            "worst_case_spend": 0.0,
            "tail_risk_ratio": 1.0,
            "confidence_level": confidence_level,
        }

    n = y.size
    x = np.arange(1, n + 1, dtype=np.float64)
    slope, intercept = _linear_regression_kernel(x, y)

    # Dispersão via MAD para resistir a outliers
    baseline = slope * x + intercept
    mad_val = mad(y - baseline)
    sigma = max(1.0, mad_val / _MAD_TO_SIGMA)

    # 10.000 iterações estocásticas compiladas via LLVM em milissegundos
    simulated_totals = _monte_carlo_simulation_kernel(
        n, n_simulations, horizon_days, slope, intercept, sigma
    )
    simulated_totals.sort()

    idx_var = int(min(n_simulations - 1, max(0, np.floor(confidence_level * n_simulations))))
    expected_spend = float(np.mean(simulated_totals))
    var_spend = float(simulated_totals[idx_var])
    tail_spends = simulated_totals[idx_var:]
    cvar_spend = float(np.mean(tail_spends)) if tail_spends.size > 0 else var_spend
    worst_case = float(simulated_totals[-1])

    return {
        "expected_spend": round(expected_spend, 2),
        "var_spend": round(var_spend, 2),
        "cvar_spend": round(cvar_spend, 2),
        "worst_case_spend": round(worst_case, 2),
        "tail_risk_ratio": round(cvar_spend / expected_spend, 3) if expected_spend > 0 else 1.0,
        "confidence_level": confidence_level,
    }


def calculate_unit_economics_elasticity(
    cost_series: list[float] | np.ndarray,
    volume_series: list[float] | np.ndarray,
) -> dict[str, float | str]:
    """Avalia se a arquitetura de nuvem escala de forma Sublinear (saudável) ou Superlinear (O(N^2) colapso)."""
    c = np.asarray(cost_series, dtype=np.float64)
    v = np.asarray(volume_series, dtype=np.float64)

    if c.size < 2 or v.size < 2 or c.size != v.size:
        return {
            "elasticity": 1.0,
            "scaling_regime": "LINEAR",
            "unit_cost_trend": 0.0,
            "verdict": "Dados insuficientes para cálculo de elasticidade.",
        }

    c_mean = float(np.mean(c))
    v_mean = float(np.mean(v))

    delta_c = (c[-1] - c[0]) / (c_mean if c_mean > 0 else 1.0)
    delta_v = (v[-1] - v[0]) / (v_mean if v_mean > 0 else 1.0)

    elasticity = float(delta_c / delta_v) if abs(delta_v) > 1e-6 else 1.0

    if elasticity < 0.85:
        regime = "SUBLINEAR"
        verdict = "Escalabilidade excelente: economia de escala comprovada (custo cresce menos que o volume)."
    elif elasticity <= 1.25:
        regime = "LINEAR"
        verdict = "Escalabilidade saudável: custo cresce proporcionalmente ao volume de transações."
    else:
        regime = "SUPERLINEAR"
        verdict = "Alerta crítico de arquitetura: custo cresce superlinearmente (O(N^2)). Gargalos de banco/acoplamento."

    unit_cost_initial = c[0] / v[0] if v[0] > 0 else 0.0
    unit_cost_final = c[-1] / v[-1] if v[-1] > 0 else 0.0
    unit_cost_trend = round(float((unit_cost_final - unit_cost_initial) / (unit_cost_initial if unit_cost_initial > 0 else 1.0)) * 100, 2)

    return {
        "elasticity": round(elasticity, 3),
        "scaling_regime": regime,
        "unit_cost_trend_percent": unit_cost_trend,
        "verdict": verdict,
    }


# Matriz de intensidade de carbono em kg CO2e / kWh por região (Fontes EPA / GHG Protocol)
_GRID_CARBON_INTENSITY = {
    "us-east-1": 0.380,      # N. Virginia (mista gás/carvão)
    "us-west-2": 0.085,      # Oregon (hidrelétrica abundante)
    "sa-east-1": 0.075,      # São Paulo (matriz majoritariamente renovável/hidro)
    "eu-west-1": 0.220,      # Irlanda (vento/gás)
    "eu-central-1": 0.310,   # Frankfurt
    "ap-southeast-1": 0.410, # Singapura (dependência gás fóssil)
    "default": 0.320,
}


@njit(cache=True)
def _estimate_server_kwh_kernel(
    vcpu_hours: np.ndarray,
    memory_gb_hours: np.ndarray,
    pue: float = 1.2, # Power Usage Effectiveness médio de Hyperscalers
) -> float:
    """Calcula consumo de energia (kWh) de servidores em microssegundos via JIT.
    Estimativa: ~10W por vCPU ativo médio + ~0.38W por GB de RAM, ajustado por PUE.
    """
    total_kwh = 0.0
    n = vcpu_hours.shape[0]
    for i in range(n):
        # 0.010 kW por vCPU-hora + 0.00038 kW por GB-hora
        server_kw = (vcpu_hours[i] * 0.010) + (memory_gb_hours[i] * 0.00038)
        total_kwh += server_kw * pue
    return total_kwh


def calculate_green_finops_footprint(
    vcpu_hours: list[float] | np.ndarray,
    memory_gb_hours: list[float] | np.ndarray,
    region: str = "us-east-1",
    carbon_credit_cost_per_ton_usd: float = 45.0, # Preço médio mercado voluntário de carbono
) -> dict[str, float | str]:
    """Calcula a Pegada de Carbono ESG (CO2e) e o Passivo Financeiro de Taxação Verde.
    Permite relatórios de sustentabilidade exigidos para IPO, SEC e auditorias ESG europeias.
    """
    v_arr = np.asarray(vcpu_hours, dtype=np.float64)
    m_arr = np.asarray(memory_gb_hours, dtype=np.float64)

    if v_arr.size == 0 or m_arr.size == 0 or v_arr.size != m_arr.size:
        return {
            "total_kwh": 0.0,
            "metric_tons_co2e": 0.0,
            "carbon_offset_cost_usd": 0.0,
            "esg_grade": "A+",
            "region": region,
            "recommendation": "Sem dados suficientes para auditoria de carbono.",
        }

    kwh = _estimate_server_kwh_kernel(v_arr, m_arr, pue=1.18)
    intensity = _GRID_CARBON_INTENSITY.get(region.lower(), _GRID_CARBON_INTENSITY["default"])

    # kg CO2e = kWh * kg CO2e/kWh
    kg_co2e = kwh * intensity
    metric_tons_co2e = kg_co2e / 1000.0

    offset_cost = metric_tons_co2e * carbon_credit_cost_per_ton_usd

    # Classificação ESG Rating de Infraestrutura
    if metric_tons_co2e < 1.0:
        esg_grade = "A+"
        rec = "Pegada de carbono exemplar. Mantenha instâncias em regiões de alta energia renovável."
    elif metric_tons_co2e < 10.0:
        esg_grade = "A"
        rec = "Emissões dentro dos parâmetros de sustentabilidade aceitos pela Taxonomia Verde."
    elif metric_tons_co2e < 50.0:
        esg_grade = "B"
        rec = f"Consumo moderado de carbono. Considere migrar workloads em batch para 'sa-east-1' ou 'us-west-2'."
    else:
        esg_grade = "C"
        rec = f"Alto impacto ambiental. Risco de passivo regulatório ESG de ${round(offset_cost, 2)} USD/ano."

    return {
        "total_kwh": round(float(kwh), 2),
        "metric_tons_co2e": round(float(metric_tons_co2e), 4),
        "kg_co2e": round(float(kg_co2e), 2),
        "carbon_offset_cost_usd": round(float(offset_cost), 2),
        "esg_grade": esg_grade,
        "region": region,
        "recommendation": rec,
    }


@njit(cache=True)
def _calculate_egress_waste_kernel(
    egress_gb_series: np.ndarray,
    nat_gateway_gb_series: np.ndarray,
    rate_per_gb_egress: float = 0.09,      # Média AWS Internet Egress ($0.09/GB)
    rate_per_gb_nat: float = 0.045,        # AWS NAT Gateway Data Processing ($0.045/GB)
    cloudflare_zero_egress_rate: float = 0.015, # Cloudflare R2 / Workers bandwidth cost
) -> tuple[float, float, float, float]:
    """Cálculo ultrarrápido em LLVM de arbitragem de saída de dados e NAT Gateway."""
    n = egress_gb_series.shape[0]
    total_egress_gb = 0.0
    total_nat_gb = 0.0
    for i in range(n):
        total_egress_gb += egress_gb_series[i]
        total_nat_gb += nat_gateway_gb_series[i]

    current_egress_cost = total_egress_gb * rate_per_gb_egress
    current_nat_cost = total_nat_gb * rate_per_gb_nat
    total_current_network_cost = current_egress_cost + current_nat_cost

    # Otimizado: VPC Gateway Endpoints gratuitos para S3/DynamoDB eliminam NAT Gateway em ~65% dos casos
    optimized_nat_cost = current_nat_cost * 0.35
    # Arbitragem Cloudflare / Zero Egress reduz o custo de saída em ~75%
    optimized_egress_cost = total_egress_gb * cloudflare_zero_egress_rate

    optimized_total = optimized_nat_cost + optimized_egress_cost
    potential_savings = total_current_network_cost - optimized_total

    return total_current_network_cost, optimized_total, potential_savings, total_egress_gb + total_nat_gb


def calculate_cloud_egress_arbitrage(
    egress_gb_history: list[float] | np.ndarray,
    nat_gateway_gb_history: list[float] | np.ndarray,
) -> dict[str, float | str]:
    """Perícia de tráfego de rede e arbitragem de Egress / NAT Gateway.
    Detecta o custo oculto de NAT Gateways e tráfego inter-regional vs Cloudflare.
    """
    e_arr = np.asarray(egress_gb_history, dtype=np.float64)
    n_arr = np.asarray(nat_gateway_gb_history, dtype=np.float64)

    if e_arr.size == 0 or n_arr.size == 0 or e_arr.size != n_arr.size:
        return {
            "current_network_cost_usd": 0.0,
            "optimized_network_cost_usd": 0.0,
            "potential_savings_usd": 0.0,
            "savings_ratio_percent": 0.0,
            "total_traffic_gb": 0.0,
            "arbitrage_verdict": "Dados insuficientes de tráfego de rede para análise.",
        }

    current, optimized, savings, total_gb = _calculate_egress_waste_kernel(e_arr, n_arr)
    ratio = (savings / current * 100.0) if current > 0 else 0.0

    verdict = "Arbitragem de alta economia: implementar VPC Endpoints para S3/Dynamo e tunelamento via Cloudflare reduz a conta em até " + f"{round(ratio, 1)}%."

    return {
        "current_network_cost_usd": round(float(current), 2),
        "optimized_network_cost_usd": round(float(optimized), 2),
        "potential_savings_usd": round(float(savings), 2),
        "savings_ratio_percent": round(float(ratio), 1),
        "total_traffic_gb": round(float(total_gb), 1),
        "arbitrage_verdict": verdict,
    }


