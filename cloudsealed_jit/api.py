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


class AnalyzeBillingResponse(BaseModel):
    anomalies: list[CostAnomaly]
    metrics: Metrics
    recommendations: list[Recommendation]
    summary: str


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
        result = analyze(series, payload.analysisType or "waste-audit")
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


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("JIT_HOST", "0.0.0.0"),
        port=int(os.getenv("JIT_PORT", "8091")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
