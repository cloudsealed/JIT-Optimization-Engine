"""
Billing export parsing.

Cloud providers export cost data with different schemas and different
granularities. A single day of spend is spread across many line items -- one
per resource, SKU or usage type. Any analysis that treats a CSV row as an
observation is measuring line-item size, not daily spend.

This module normalises the common export formats into a single daily series:

    date (UTC calendar day) -> total cost, and cost per service

Supported layouts, detected from the header row:

    AWS Cost and Usage Report   lineItem/UsageStartDate, lineItem/UnblendedCost
    GCP billing export          usage_start_time, cost, service.description
    Azure cost export           Date / UsageDateTime, Cost / CostInBillingCurrency
    FOCUS 1.0                    ChargePeriodStart, BilledCost, ServiceName
    Generic                     heuristic match on date / cost / service columns

FOCUS (the FinOps Open Cost and Usage Specification) is the vendor-neutral
billing format that AWS, GCP, Azure and OCI now export natively. Supporting it
means one analysis runs across every cloud's export unchanged.

Rows that cannot be parsed are counted and reported rather than silently
dropped, so callers can tell a partially-understood file from a clean one.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Sequence

__all__ = ["BillingSeries", "ParseError", "parse_billing_csv"]


class ParseError(ValueError):
    """Raised when a billing export cannot be interpreted at all."""


# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------

# Explicit, provider-specific column names take priority over the heuristics.
# Order matters: the first match wins.
_DATE_COLUMNS: Sequence[str] = (
    "lineitem/usagestartdate",          # AWS CUR
    "lineitem/usagestarttime",
    "bill/billingperiodstartdate",
    "chargeperiodstart",                # FOCUS 1.0 (FinOps Open Cost and Usage Spec)
    "billingperiodstart",               # FOCUS 1.0 (fallback: monthly period)
    "usage_start_time",                 # GCP
    "usage_date",
    "usagedatetime",                    # Azure
    "date",
    "billingdate",
    "usagestart",
)

_COST_COLUMNS: Sequence[str] = (
    "lineitem/unblendedcost",           # AWS CUR
    "lineitem/blendedcost",
    "lineitem/netunblendedcost",
    "billedcost",                       # FOCUS 1.0 (invoiced amount)
    "cost",                             # GCP / generic
    "costinbillingcurrency",            # Azure
    "pretaxcost",
    "amortizedcost",
    "effectivecost",                    # FOCUS 1.0 (amortized amount)
    "amount",
)

_SERVICE_COLUMNS: Sequence[str] = (
    "product/productname",              # AWS CUR
    "lineitem/productcode",
    "service.description",              # GCP
    "service_description",
    "service",
    "servicename",                      # Azure / FOCUS 1.0
    "servicecategory",                  # FOCUS 1.0 (fallback: coarse category)
    "metercategory",
    "product",
    "resourcegroup",
)

# Substrings used when no explicit column name matches. Deliberately narrow:
# matching "cost" alone would also match "costcenter".
_DATE_HINTS = ("date", "day", "time", "period")
_COST_HINTS = ("cost", "amount", "charge", "spend")
_SERVICE_HINTS = ("service", "product", "category", "resource")

_COST_EXCLUDE = ("center", "code", "currency", "type", "category")


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def _pick(headers: Sequence[str], explicit: Sequence[str], hints: Sequence[str],
          exclude: Sequence[str] = ()) -> str | None:
    lookup = {_normalise(h): h for h in headers}

    for candidate in explicit:
        key = _normalise(candidate)
        if key in lookup:
            return lookup[key]

    for key, original in lookup.items():
        if any(bad in key for bad in exclude):
            continue
        if any(hint in key for hint in hints):
            return original
    return None


# --------------------------------------------------------------------------
# Value coercion
# --------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S %Z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y",
)


def _parse_date(raw: str) -> date | None:
    text = (raw or "").strip()
    if not text:
        return None

    # ISO-8601 with timezone offsets and fractional seconds.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Trailing timezone name that strptime cannot handle, e.g. "... UTC".
    head = text.split(" ")[0]
    if head != text:
        return _parse_date(head)
    return None


def _parse_cost(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    text = text.replace("$", "").replace("R$", "").replace("US$", "").strip()

    # "1.234,56" (pt-BR) vs "1,234.56" (en-US): the last separator is decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # Ambiguous. Treat as decimal only when it looks like one.
        head, _, tail = text.rpartition(",")
        text = f"{head}.{tail}" if len(tail) in (1, 2) else text.replace(",", "")

    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value else None  # reject NaN


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class BillingSeries:
    """A billing export normalised to daily totals.

    Attributes:
        days: calendar days present in the file, ascending and gap-free.
        costs: total cost for each day in ``days``.
        by_service: service name -> per-day cost, aligned with ``days``.
        currency: currency code if the export declared one.
        rows_parsed: line items successfully interpreted.
        rows_skipped: line items rejected (unparseable date or cost).
    """

    days: list[date]
    costs: list[float]
    by_service: dict[str, list[float]] = field(default_factory=dict)
    currency: str | None = None
    rows_parsed: int = 0
    rows_skipped: int = 0

    def __post_init__(self) -> None:
        if len(self.days) != len(self.costs):
            raise ValueError("days and costs must have the same length")

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs))

    @property
    def span_days(self) -> int:
        return len(self.days)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.days)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _fill_gaps(daily: dict[date, float]) -> tuple[list[date], list[float]]:
    """Return a contiguous daily series, inserting 0.0 for missing days.

    Gaps matter: a day with no line items is a day with no spend, and leaving
    it out would distort both the weekly seasonality model and the averages.
    """
    if not daily:
        return [], []
    start, end = min(daily), max(daily)
    days: list[date] = []
    costs: list[float] = []
    current = start
    one_day = (datetime.fromordinal(2) - datetime.fromordinal(1))
    while current <= end:
        days.append(current)
        costs.append(round(daily.get(current, 0.0), 6))
        current = (datetime.combine(current, datetime.min.time()) + one_day).date()
    return days, costs


def parse_billing_csv(csv_text: str, *, max_services: int = 25) -> BillingSeries:
    """Parse a cloud billing export into a daily :class:`BillingSeries`.

    Args:
        csv_text: raw contents of the CSV export.
        max_services: keep only the N costliest services in ``by_service``.

    Raises:
        ParseError: the file is empty, has no header, or has no usable
            date/cost column pair.
    """
    if not csv_text or not csv_text.strip():
        raise ParseError("Billing export is empty.")

    reader = csv.DictReader(io.StringIO(csv_text))
    headers = reader.fieldnames
    if not headers:
        raise ParseError("Billing export has no header row.")

    date_col = _pick(headers, _DATE_COLUMNS, _DATE_HINTS)
    cost_col = _pick(headers, _COST_COLUMNS, _COST_HINTS, exclude=_COST_EXCLUDE)
    service_col = _pick(headers, _SERVICE_COLUMNS, _SERVICE_HINTS)
    currency_col = _pick(headers, ("currency", "billingcurrency",
                                   "lineitem/currencycode"), ("currency",))

    if date_col is None or cost_col is None:
        raise ParseError(
            "Could not identify a date and a cost column. "
            f"Columns found: {', '.join(headers[:20])}"
        )

    daily: dict[date, float] = defaultdict(float)
    per_service: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    currency: str | None = None
    parsed = skipped = 0

    for row in reader:
        day = _parse_date(row.get(date_col, ""))
        cost = _parse_cost(row.get(cost_col, ""))
        if day is None or cost is None:
            skipped += 1
            continue

        daily[day] += cost
        if service_col:
            name = (row.get(service_col) or "").strip() or "unattributed"
            per_service[name][day] += cost
        if currency is None and currency_col:
            currency = (row.get(currency_col) or "").strip() or None
        parsed += 1

    if parsed == 0:
        raise ParseError(
            f"No line items could be parsed from {skipped} rows. "
            f"Checked column '{date_col}' for dates and '{cost_col}' for costs."
        )

    days, costs = _fill_gaps(daily)

    # Keep the costliest services; the long tail is noise for waste analysis.
    by_service: dict[str, list[float]] = {}
    ranked = sorted(per_service.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    index = {day: i for i, day in enumerate(days)}
    for name, series in ranked[:max_services]:
        aligned = [0.0] * len(days)
        for day, value in series.items():
            aligned[index[day]] = round(value, 6)
        by_service[name] = aligned

    return BillingSeries(
        days=days,
        costs=costs,
        by_service=by_service,
        currency=currency,
        rows_parsed=parsed,
        rows_skipped=skipped,
    )
