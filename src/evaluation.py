
"""Deterministic, offline evaluation checks for ColdChain AI reports."""

import re

REQUIRED_SECTIONS = (
    "Investigation Summary",
    "Shipment Telemetry",
    "Risk Assessment",
    "Environmental Conditions",
    "Applicable SOP Rules",
    "Recommended Actions",
    "Evidence Quality and Missing Information",
)


def has_required_sections(report: str) -> bool:
    """Recognize numbered, Markdown, and plain-text report headings."""
    headings = []

    for line in report.splitlines():
        heading = line.strip()

        # Remove Markdown heading markers.
        heading = re.sub(r"^#{1,6}\s*", "", heading)

        # Remove numbering such as '1. ', '2) ', or '3 - '.
        heading = re.sub(r"^\d+\s*[.)-]\s*", "", heading)

        headings.append(heading.strip().lower())

    for required in REQUIRED_SECTIONS:
        required = required.lower()

        if not any(
            heading == required or heading.startswith(required + ",")
            or heading.startswith(required + ":")
            or heading.startswith(required + " -")
            for heading in headings
        ):
            return False

    return True


def evaluate_report(report: str, proposed_action: dict | None = None) -> dict:
    """Score report completeness and action safety without an LLM."""

    text = report or ""
    lower = text.lower()

    # Check whether the proposed action is explicitly simulation-only.
    action_mode = ""
    if isinstance(proposed_action, dict):
        action_mode = str(
            proposed_action.get("execution_mode", "")
        ).upper()

    action_safety = (
        action_mode == "SIMULATED_ONLY"
        or "simulated_only" in lower
        or "no real operational actions" in lower
        or "not executed" in lower
    )

    checks = {
        "all_required_sections": has_required_sections(text),

        "historical_vs_live_distinction": (
            "historical" in lower
            and ("live" in lower or "current weather" in lower)
        ),

        "missing_information_disclosed": any(
            phrase in lower
            for phrase in (
                "unknown",
                "not available",
                "missing information",
                "cannot be confirmed",
            )
        ),

        "risk_dimensions_separated": (
            "risk classification" in lower
            and "delay probability" in lower
            and "route risk" in lower
        ),

        "action_safety_disclosed": action_safety,
    }

    passed = sum(checks.values())
    total = len(checks)

    return {
        "passed": passed,
        "total": total,
        "score": round(passed / total, 2),
        "checks": checks,
        "status": "PASS" if passed == total else "REVIEW",
    }
