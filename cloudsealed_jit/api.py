"""
HTTP surface for the waste analyser.

The request and response schemas are the contract consumed by the CloudSealed
Framework4D assessment engine (``framework4d-jit-client.ts``). Field names and
types are part of that contract and must not change without a version bump.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from . import __version__
from .analysis import analyze
from .kernels import JIT_ENABLED
from .notify import post_webhook
from .parsing import ParseError, parse_billing_csv

logger = logging.getLogger("cloudsealed_jit.api")

#: Reject payloads above this size before parsing. Billing exports are large,
#: but an unbounded body is a denial-of-service surface.
MAX_CSV_BYTES = int(os.getenv("JIT_MAX_CSV_BYTES", str(64 * 1024 * 1024)))

#: When set, every request must present a matching ``X-Api-Key`` header.
API_KEY = os.getenv("JIT_OPTIMIZATION_API_KEY", "")


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


class AnalyzeBillingRequest(BaseModel):
    companyName: str = Field(min_length=1, max_length=200)
    csvContent: str = Field(description="Raw contents of the billing export.")
    analysisType: Optional[Literal["waste-audit", "cost-forecast", "efficiency"]] = (
        "waste-audit"
    )
    webhookUrl: Optional[str] = Field(
        default=None,
        description="If set, POST the result here (Slack incoming webhook or generic "
        "listener) when an anomaly reaches CRITICAL/HIGH severity.",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Monthly budget; if set, the response includes a forecast projecting "
        "when the current spend trend crosses it.",
    )


class CostAnomaly(BaseModel):
    date: str
    expectedCost: float
    actualCost: float
    deviation: float
    zScore: float
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str


class Metrics(BaseModel):
    averageDailyCost: float
    stdDeviation: float
    sharpeRatio: float
    wastePercentage: float


class Recommendation(BaseModel):
    title: str
    description: str
    potentialSavings: float
    effort: Literal["LOW", "MEDIUM", "HIGH"]


class Forecast(BaseModel):
    horizonDays: int
    projectedSpend: float
    runRateSpend: float
    dailyTrend: float
    budget: Optional[float] = None
    budgetBreachDay: Optional[int] = None
    method: str


class AnalyzeBillingResponse(BaseModel):
    anomalies: list[CostAnomaly]
    metrics: Metrics
    recommendations: list[Recommendation]
    summary: str
    forecast: Optional[Forecast] = None


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

app = FastAPI(
    title="CloudSealed JIT Optimization Engine",
    version=__version__,
    description=(
        "Detects structural waste in cloud billing exports using a robust, "
        "day-of-week aware baseline and modified z-scores."
    ),
)


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Enforce ``X-Api-Key`` when an API key is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key.",
        )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "jitEnabled": JIT_ENABLED,
    }


@app.post(
    "/v1/analyze-billing",
    response_model=AnalyzeBillingResponse,
    # `forecast` is only present when requested (cost-forecast or a budget).
    # Excluding None keeps the default response byte-identical to the frozen
    # Framework4D contract, so this stays a purely additive change.
    response_model_exclude_none=True,
    dependencies=[Depends(require_api_key)],
)
def analyze_billing(payload: AnalyzeBillingRequest) -> AnalyzeBillingResponse:
    """Analyse a cloud billing export and return waste findings.

    Returns 422 when the export cannot be interpreted, so callers can tell a
    malformed file from an internal failure.
    """
    size = len(payload.csvContent.encode("utf-8", errors="ignore"))
    if size > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Billing export is {size:,} bytes; limit is {MAX_CSV_BYTES:,}.",
        )

    started = time.perf_counter()
    try:
        series = parse_billing_csv(payload.csvContent)
    except ParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    try:
        result = analyze(series, payload.analysisType or "waste-audit", budget=payload.budget)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Analysis failed for %s", payload.companyName)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed.",
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "analyzed company=%s days=%d rows=%d anomalies=%d elapsed_ms=%.1f",
        payload.companyName, series.span_days, series.rows_parsed,
        len(result.anomalies), elapsed_ms,
    )

    if payload.webhookUrl:
        try:
            post_webhook(payload.webhookUrl, result, company_name=payload.companyName)
        except Exception:  # pragma: no cover - defensive, notify() already swallows network errors
            logger.exception("Webhook dispatch failed for %s", payload.companyName)

    return AnalyzeBillingResponse(**result.to_dict())


class PredictBurnRateRequest(BaseModel):
    spendHistory: list[float] = Field(description="Historical daily cloud spend values.")
    daysAhead: int = Field(default=15, description="Number of future days to project.")


class PredictBurnRateResponse(BaseModel):
    current_total: float
    projected_total: float
    daily_trend: float
    confidence: float


@app.post(
    "/v1/predict-burn-rate",
    response_model=PredictBurnRateResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_burn_rate_endpoint(payload: PredictBurnRateRequest) -> PredictBurnRateResponse:
    """Project cloud burn rate using LLVM JIT-accelerated robust regression."""
    from .kernels import predict_burn_rate
    res = predict_burn_rate(payload.spendHistory, payload.daysAhead)
    return PredictBurnRateResponse(**res)


class MonteCarloVaRRequest(BaseModel):
    spendHistory: list[float]
    nSimulations: int = 10000
    horizonDays: int = 30
    confidenceLevel: float = 0.99


class MonteCarloVaRResponse(BaseModel):
    expected_spend: float
    var_spend: float
    cvar_spend: float
    worst_case_spend: float
    tail_risk_ratio: float
    confidence_level: float


@app.post(
    "/v1/finops/monte-carlo-var",
    response_model=MonteCarloVaRResponse,
    dependencies=[Depends(require_api_key)],
)
def monte_carlo_var_endpoint(payload: MonteCarloVaRRequest) -> MonteCarloVaRResponse:
    """Simulate 10,000 future paths in milliseconds using LLVM JIT to estimate Tail Risk (VaR & CVaR)."""
    from .kernels import monte_carlo_finops_var
    res = monte_carlo_finops_var(
        payload.spendHistory,
        n_simulations=payload.nSimulations,
        horizon_days=payload.horizonDays,
        confidence_level=payload.confidenceLevel,
    )
    return MonteCarloVaRResponse(**res)


class UnitEconomicsRequest(BaseModel):
    costSeries: list[float]
    volumeSeries: list[float]


class UnitEconomicsResponse(BaseModel):
    elasticity: float
    scaling_regime: str
    unit_cost_trend_percent: float
    verdict: str


@app.post(
    "/v1/finops/unit-economics",
    response_model=UnitEconomicsResponse,
    dependencies=[Depends(require_api_key)],
)
def unit_economics_endpoint(payload: UnitEconomicsRequest) -> UnitEconomicsResponse:
    """Evaluate whether cloud cost scales Sublinearly O(N) or Superlinearly O(N^2)."""
    from .kernels import calculate_unit_economics_elasticity
    res = calculate_unit_economics_elasticity(payload.costSeries, payload.volumeSeries)
    return UnitEconomicsResponse(**res)


class GreenFinOpsRequest(BaseModel):
    vcpuHours: list[float]
    memoryGbHours: list[float]
    region: str = "us-east-1"
    carbonCreditCostPerTonUsd: float = 45.0


class GreenFinOpsResponse(BaseModel):
    total_kwh: float
    metric_tons_co2e: float
    kg_co2e: float
    carbon_offset_cost_usd: float
    esg_grade: str
    region: str
    recommendation: str


@app.post(
    "/v1/finops/green-carbon",
    response_model=GreenFinOpsResponse,
    dependencies=[Depends(require_api_key)],
)
def green_carbon_endpoint(payload: GreenFinOpsRequest) -> GreenFinOpsResponse:
    """Calculate datacenter carbon footprint (CO2e) and ESG compliance taxation."""
    from .kernels import calculate_green_finops_footprint
    res = calculate_green_finops_footprint(
        payload.vcpuHours,
        payload.memoryGbHours,
        payload.region,
        payload.carbonCreditCostPerTonUsd,
    )
    return GreenFinOpsResponse(**res)


class EgressArbitrageRequest(BaseModel):
    egressGbHistory: list[float]
    natGatewayGbHistory: list[float]


class EgressArbitrageResponse(BaseModel):
    current_network_cost_usd: float
    optimized_network_cost_usd: float
    potential_savings_usd: float
    savings_ratio_percent: float
    total_traffic_gb: float
    arbitrage_verdict: str


@app.post(
    "/v1/finops/egress-arbitrage",
    response_model=EgressArbitrageResponse,
    dependencies=[Depends(require_api_key)],
)
def egress_arbitrage_endpoint(payload: EgressArbitrageRequest) -> EgressArbitrageResponse:
    """Evaluate network traffic waste, NAT Gateway overspend, and Cloudflare routing arbitrage."""
    from .kernels import calculate_cloud_egress_arbitrage
    res = calculate_cloud_egress_arbitrage(payload.egressGbHistory, payload.natGatewayGbHistory)
    return EgressArbitrageResponse(**res)




def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("JIT_HOST", "0.0.0.0"),
        port=int(os.getenv("JIT_PORT", "8091")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
