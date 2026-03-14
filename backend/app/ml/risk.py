from __future__ import annotations


def get_risk_assessment(probability: float) -> tuple[str, str]:
    """Map fraud probability to a user-facing recommendation and severity label."""

    if probability < 0.30:
        return "Low risk", "low"
    if probability < 0.70:
        return "Suspicious", "medium"
    return "High probability of fraud", "high"
