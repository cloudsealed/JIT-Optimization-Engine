"""Push a finished analysis to a chat webhook (Slack, or any generic listener).

Stdlib-only (``urllib``), matching the dependency-free style of
``scripts/post_pr_comment.py``. A webhook is a side effect, not part of the
analysis contract: network failures are logged and swallowed rather than
raised, so a dead webhook can never turn a successful analysis into an error.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from .analysis import AnalysisResult

logger = logging.getLogger("cloudsealed_jit.notify")

WebhookFormat = Literal["slack", "generic"]

_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
_FORBIDDEN_HOSTS = {"localhost", "metadata.google.internal"}


def validate_webhook_url(url: str) -> None:
    """Validate that ``url`` is an external HTTPS endpoint and not an SSRF target."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Webhook URL must use https and include a valid hostname.")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if hostname in _FORBIDDEN_HOSTS or hostname.endswith(".internal"):
        raise ValueError(f"Webhook hostname '{hostname}' is not allowed.")

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve webhook hostname '{hostname}': {exc}") from exc

    if not addr_infos:
        raise ValueError(f"No IP addresses resolved for '{hostname}'.")

    for info in addr_infos:
        raw_ip = info[4][0]
        # Strip IPv6 scope ID if present
        ip_str = raw_ip.split("%", 1)[0]
        addr = ipaddress.ip_address(ip_str)
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise ValueError(f"Webhook resolved to non-public IP address: {addr}")


def detect_format(url: str) -> WebhookFormat:
    return "slack" if "hooks.slack.com" in url else "generic"


def render_slack_payload(result: AnalysisResult, company_name: str = "") -> dict:
    metrics = result.metrics
    header = f"💰 cloudsealed-jit — {company_name}" if company_name else "💰 cloudsealed-jit"
    lines = [
        f"*{header}*",
        result.summary,
        f"Waste: *{metrics.wastePercentage:.1f}%*  |  "
        f"Avg daily cost: *{metrics.averageDailyCost:,.2f}*  |  "
        f"Stability: *{metrics.sharpeRatio:.2f}*",
    ]
    for a in result.anomalies[:10]:
        lines.append(f"• `{a.date}` *{a.severity}* — {a.description}")
    for r in result.recommendations[:5]:
        lines.append(f"→ *{r.title}* — {r.potentialSavings:,.2f}/30d, effort {r.effort}")
    return {"text": "\n".join(lines)}


def render_generic_payload(result: AnalysisResult) -> dict:
    return result.to_dict()


def post_webhook(
    url: str,
    result: AnalysisResult,
    *,
    format: WebhookFormat | None = None,
    company_name: str = "",
    min_severity: str = "HIGH",
) -> bool:
    """POST an analysis to a webhook. Returns whether it was sent.

    Nothing is sent if no anomaly reaches ``min_severity``, so a quiet run
    does not spam the channel.
    """
    threshold = _SEVERITY_ORDER.get(min_severity.upper(), _SEVERITY_ORDER["HIGH"])
    if not any(_SEVERITY_ORDER.get(a.severity, 0) >= threshold for a in result.anomalies):
        return False

    try:
        validate_webhook_url(url)
    except ValueError as exc:
        logger.warning("Blocked unsafe webhook URL %s: %s", url, exc)
        return False

    resolved_format = format or detect_format(url)
    payload = (
        render_slack_payload(result, company_name)
        if resolved_format == "slack"
        else render_generic_payload(result)
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Failed to post webhook to %s: %s", url, exc)
        return False
