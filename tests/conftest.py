"""Synthetic billing exports with known properties.

Every generator produces data whose correct answer is known in advance, so the
tests assert on behaviour rather than on whatever the code happens to output.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pytest


def build_csv(rows: list[tuple[str, str, float]]) -> str:
    """Render (date, service, cost) tuples as a generic billing export."""
    buffer = io.StringIO()
    buffer.write("date,service,cost,currency\n")
    for day, service, cost in rows:
        buffer.write(f"{day},{service},{cost:.4f},USD\n")
    return buffer.getvalue()


@pytest.fixture
def stable_billing() -> str:
    """60 days of flat spend with mild noise and no anomalies."""
    start = date(2026, 1, 1)
    rows = []
    for i in range(60):
        day = start + timedelta(days=i)
        # Deterministic wobble, well inside normal variation.
        wobble = 1.0 if i % 3 == 0 else -1.0
        rows.append((day.isoformat(), "compute", 100.0 + wobble))
    return build_csv(rows)


@pytest.fixture
def spiked_billing() -> str:
    """60 flat days with a single 5x spike on day 30."""
    start = date(2026, 1, 1)
    rows = []
    for i in range(60):
        day = start + timedelta(days=i)
        cost = 100.0 + (1.0 if i % 3 == 0 else -1.0)
        if i == 30:
            cost = 500.0
        rows.append((day.isoformat(), "compute", cost))
    return build_csv(rows)


@pytest.fixture
def weekend_idle_billing() -> str:
    """56 days where 'staging' runs flat all week and 'prod' follows traffic."""
    start = date(2026, 1, 5)  # a Monday
    rows = []
    for i in range(56):
        day = start + timedelta(days=i)
        weekend = day.weekday() >= 5
        rows.append((day.isoformat(), "staging", 50.0))
        rows.append((day.isoformat(), "prod", 20.0 if weekend else 100.0))
    return build_csv(rows)
