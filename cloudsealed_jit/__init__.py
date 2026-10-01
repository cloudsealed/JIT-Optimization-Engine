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
from .compiler import jit, compilation_stats, reset_compilation_stats, CloudSealedCompileError
from .dataclass_compiler import jitdataclass

__version__ = "0.4.2"

__all__ = [
    "BillingSeries",
    "ParseError",
    "parse_billing_csv",
    "AnalysisResult",
    "analyze",
    "jit",
    "jitdataclass",
    "compilation_stats",
    "reset_compilation_stats",
    "CloudSealedCompileError",
    "__version__",
]
