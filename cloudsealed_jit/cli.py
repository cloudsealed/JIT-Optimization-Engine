"""Command-line entry point: analyse a billing export without running a server."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import analyze
from .notify import post_webhook
from .parsing import ParseError, parse_billing_csv
from .report import render_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cloudsealed-jit",
        description="Detect structural waste in a cloud billing export.",
    )
    parser.add_argument("csv", help="path to the billing export, or - for stdin")
    parser.add_argument(
        "--type",
        default="waste-audit",
        choices=("waste-audit", "cost-forecast", "efficiency"),
        help="analysis mode (default: waste-audit)",
    )
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="monthly budget; if set, project when the current trend crosses it "
        "(implies a forecast even outside --type cost-forecast).",
    )
    parser.add_argument(
        "--html",
        default="",
        metavar="PATH",
        help="also write a self-contained HTML report to this path",
    )
    parser.add_argument(
        "--webhook-url",
        default="",
        help="POST the result to this URL (Slack incoming webhook or generic listener) "
        "when an anomaly reaches --webhook-min-severity.",
    )
    parser.add_argument(
        "--webhook-min-severity",
        default="HIGH",
        choices=("LOW", "MEDIUM", "HIGH", "CRITICAL"),
        help="minimum anomaly severity that triggers the webhook (default: HIGH)",
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.csv == "-" else open(args.csv, encoding="utf-8").read()

    try:
        result = analyze(parse_billing_csv(text), args.type, budget=args.budget)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.webhook_url:
        post_webhook(args.webhook_url, result, min_severity=args.webhook_min_severity)

    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(render_html(result))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(result.summary)
    if result.anomalies:
        print("\nAnomalies")
        for a in result.anomalies:
            print(
                f"  {a.date}  {a.severity:<8} expected {a.expectedCost:>12,.2f}  "
                f"actual {a.actualCost:>12,.2f}  ({a.deviation:+.1f}%, z={a.zScore})"
            )
    if result.recommendations:
        print("\nRecommendations")
        for r in result.recommendations:
            print(f"  [{r.effort:<6}] {r.title} — {r.potentialSavings:,.2f}/30d")
            print(f"           {r.description}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
