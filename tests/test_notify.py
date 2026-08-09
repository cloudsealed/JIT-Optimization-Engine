import json
from unittest.mock import MagicMock, patch

from cloudsealed_jit.analysis import Anomaly, AnalysisResult, Metrics, Recommendation
from cloudsealed_jit.notify import (
    detect_format,
    post_webhook,
    render_generic_payload,
    render_slack_payload,
)


def _result(severity: str = "CRITICAL") -> AnalysisResult:
    return AnalysisResult(
        anomalies=[
            Anomaly(
                date="2026-01-31",
                expectedCost=100.0,
                actualCost=500.0,
                deviation=400.0,
                zScore=7.8,
                severity=severity,
                description="Spend above baseline.",
            )
        ],
        metrics=Metrics(averageDailyCost=110.0, stdDeviation=50.0, sharpeRatio=2.2, wastePercentage=6.3),
        recommendations=[
            Recommendation(title="Idle weekend spend", description="...", potentialSavings=200.5, effort="MEDIUM")
        ],
        summary="1 anomaly found.",
    )


def test_detect_format():
    assert detect_format("https://hooks.slack.com/services/x") == "slack"
    assert detect_format("https://example.com/webhook") == "generic"


def test_render_slack_payload_includes_key_fields():
    payload = render_slack_payload(_result(), company_name="Acme")
    assert "text" in payload
    assert "Acme" in payload["text"]
    assert "6.3%" in payload["text"]
    assert "Idle weekend spend" in payload["text"]


def test_render_generic_payload_is_full_result_dict():
    payload = render_generic_payload(_result())
    assert payload["metrics"]["wastePercentage"] == 6.3
    assert len(payload["anomalies"]) == 1


def test_post_webhook_sends_when_severity_meets_threshold():
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.read.return_value = b""
    with patch("urllib.request.urlopen", return_value=mock_response) as urlopen:
        sent = post_webhook("https://hooks.slack.com/services/x", _result("CRITICAL"))
    assert sent is True
    urlopen.assert_called_once()
    request = urlopen.call_args[0][0]
    body = json.loads(request.data)
    assert "text" in body


def test_post_webhook_skips_when_below_threshold():
    with patch("urllib.request.urlopen") as urlopen:
        sent = post_webhook("https://example.com/webhook", _result("LOW"), min_severity="HIGH")
    assert sent is False
    urlopen.assert_not_called()


def test_post_webhook_swallows_network_errors():
    with patch("urllib.request.urlopen", side_effect=OSError("boom")):
        sent = post_webhook("https://example.com/webhook", _result("CRITICAL"))
    assert sent is False
