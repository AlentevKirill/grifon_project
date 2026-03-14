from __future__ import annotations

from app.ml.risk import get_risk_assessment


def test_risk_assessment_low_threshold() -> None:
    recommendation, level = get_risk_assessment(0.10)
    assert recommendation == "Low risk"
    assert level == "low"


def test_risk_assessment_medium_threshold() -> None:
    recommendation, level = get_risk_assessment(0.55)
    assert recommendation == "Suspicious"
    assert level == "medium"


def test_risk_assessment_high_threshold() -> None:
    recommendation, level = get_risk_assessment(0.92)
    assert recommendation == "High probability of fraud"
    assert level == "high"
