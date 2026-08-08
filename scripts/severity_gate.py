"""Fail the workflow step if any anomaly meets or exceeds MIN_SEVERITY."""

from __future__ import annotations

import json
import os
import sys

ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def main() -> int:
    report_path = os.environ["REPORT_JSON"]
    min_severity = os.environ["MIN_SEVERITY"].strip().upper()

    if min_severity not in ORDER:
        print(f"::error::fail-on-severity must be one of {list(ORDER)}, got '{min_severity}'.")
        return 2

    with open(report_path, encoding="utf-8") as f:
        anomalies = json.load(f)["anomalies"]

    threshold = ORDER[min_severity]
    blocking = [a for a in anomalies if ORDER.get(a["severity"], 0) >= threshold]

    if blocking:
        print(f"::error::{len(blocking)} anomaly(ies) at or above {min_severity}:")
        for a in blocking:
            print(f"::error::  {a['date']} — {a['severity']} — {a['description']}")
        return 1

    print(f"No anomalies at or above {min_severity}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
