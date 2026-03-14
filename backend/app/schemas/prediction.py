from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Single transaction prediction details."""

    row_index: int = Field(ge=0)
    fraud_probability_percent: float = Field(ge=0.0, le=100.0)
    recommendation: str
    risk_level: str


class PredictionResponse(BaseModel):
    """Batch prediction response for uploaded CSV content."""

    total_rows: int = Field(ge=0)
    predictions: list[PredictionItem]
