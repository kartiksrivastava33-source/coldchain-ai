
from src.evaluation import evaluate_report, has_required_sections


SAMPLE_REPORT = """
1. Investigation Summary
This investigation analyzes historical shipment data and live weather.

2. Shipment Telemetry
Record ID: 1
Temperature: 0.5744°C

3. Risk Assessment
Risk Classification: Moderate Risk
Delay Probability: 0.8853
Route Risk Index: 1.1821

4. Environmental Conditions
Historical temperature and current weather are distinguished.

5. Applicable SOP Rules, Including Each Trigger's Status
The applicable rules were evaluated.

6. Recommended Actions
Flag the shipment for operational review.

7. Evidence Quality and Missing Information
Cargo type is unknown. Port identity is not available.
"""


def test_actual_report_heading_format_is_recognized():
    assert has_required_sections(SAMPLE_REPORT)


def test_complete_report_and_simulated_action_pass():
    action = {"execution_mode": "SIMULATED_ONLY"}

    result = evaluate_report(SAMPLE_REPORT, action)

    assert result["passed"] == 5
    assert result["total"] == 5
    assert result["score"] == 1.0
    assert result["status"] == "PASS"


def test_missing_section_causes_review():
    incomplete_report = SAMPLE_REPORT.replace(
        "3. Risk Assessment",
        "3. Risk Analysis",
    )

    result = evaluate_report(
        incomplete_report,
        {"execution_mode": "SIMULATED_ONLY"},
    )

    assert result["status"] == "REVIEW"
    assert result["checks"]["all_required_sections"] is False


def test_missing_action_safety_causes_review():
    result = evaluate_report(SAMPLE_REPORT, {})

    assert result["checks"]["action_safety_disclosed"] is False
    assert result["status"] == "REVIEW"


def test_historical_and_live_distinction_is_required():
    report = SAMPLE_REPORT.replace(
        "live weather", "environmental observations"
    ).replace(
        "current weather", "environmental observations"
    )

    result = evaluate_report(
        report,
        {"execution_mode": "SIMULATED_ONLY"},
    )

    assert result["checks"]["historical_vs_live_distinction"] is False
    assert result["status"] == "REVIEW"
