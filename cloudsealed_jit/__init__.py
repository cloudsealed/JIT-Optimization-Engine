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
from .kernels import (
    predict_burn_rate,
    monte_carlo_finops_var,
    calculate_unit_economics_elasticity,
    calculate_green_finops_footprint,
    calculate_cloud_egress_arbitrage,
)

__version__ = "0.5.2"

__all__ = [
    "BillingSeries",
    "ParseError",
    "parse_billing_csv",
    "AnalysisResult",
    "analyze",
    "predict_burn_rate",
    "monte_carlo_finops_var",
    "calculate_unit_economics_elasticity",
    "calculate_green_finops_footprint",
    "calculate_cloud_egress_arbitrage",
    "jit",
    "jitdataclass",
    "compilation_stats",
    "reset_compilation_stats",
    "CloudSealedCompileError",
    "__version__",
]

