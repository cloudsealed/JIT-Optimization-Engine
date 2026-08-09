"""
cloudsealed-jit — cloud billing waste analysis.

Detects structural cost waste in cloud billing exports by modelling a robust
day-of-week aware baseline and measuring the excess spend above it.

Public API:
    parse_billing_csv(csv_text) -> BillingSeries
    analyze(series, analysis_type="waste-audit") -> AnalysisResult
"""

from .parsing import BillingSeries, ParseError, parse_billing_csv
from .analysis import AnalysisResult, analyze

__version__ = "0.2.1"

__all__ = [
    "BillingSeries",
    "ParseError",
    "parse_billing_csv",
    "AnalysisResult",
    "analyze",
    "__version__",
]
