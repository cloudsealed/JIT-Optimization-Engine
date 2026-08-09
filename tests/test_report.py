from cloudsealed_jit.analysis import Anomaly, AnalysisResult, Metrics, Recommendation
from cloudsealed_jit.report import render_html


def _result() -> AnalysisResult:
    return AnalysisResult(
        anomalies=[
            Anomaly(
                date="2026-01-31",
                expectedCost=100.0,
                actualCost=500.0,
                deviation=400.0,
                zScore=7.8,
                severity="CRITICAL",
                description="Spend above baseline.",
            )
        ],
        metrics=Metrics(averageDailyCost=110.0, stdDeviation=50.0, sharpeRatio=2.2, wastePercentage=6.3),
        recommendations=[
            Recommendation(title="Idle weekend spend", description="...", potentialSavings=200.5, effort="MEDIUM")
        ],
        summary="1 anomaly found.",
    )


def test_render_html_is_well_formed():
    output = render_html(_result(), company_name="Acme")
    assert output.strip().startswith("<!DOCTYPE html>")
    assert output.count("<html") == output.count("</html>")
    assert output.count("<table>") == output.count("</table>")


def test_render_html_includes_key_values():
    output = render_html(_result(), company_name="Acme")
    assert "Acme" in output
    assert "6.3%" in output
    assert "CRITICAL" in output
    assert "Idle weekend spend" in output


def test_render_html_handles_no_findings():
    empty = AnalysisResult(summary="No anomalies.")
    output = render_html(empty)
    assert "None found." in output
    assert "None." in output


def test_render_html_escapes_untrusted_text():
    result = _result()
    result.anomalies[0].description = "<script>alert(1)</script>"
    output = render_html(result)
    assert "<script>alert(1)</script>" not in output
    assert "&lt;script&gt;" in output
