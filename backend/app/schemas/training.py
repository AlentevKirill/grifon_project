from __future__ import annotations

from pydantic import BaseModel, Field


class MetricsSchema(BaseModel):
    """Evaluation metrics for binary classification quality."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)


class TrainResponse(BaseModel):
    """Payload returned when model training completes."""

    message: str
    target_column: str
    class_distribution: dict[str, int]
    validation_metrics: MetricsSchema
    test_metrics: MetricsSchema


class ModelStatusResponse(BaseModel):
    """Current model availability and metadata summary."""

    is_trained: bool
    feature_count: int = Field(default=0, ge=0)
    target_column: str | None = None
    class_distribution: dict[str, int] = Field(default_factory=dict)
    validation_metrics: MetricsSchema | None = None
    test_metrics: MetricsSchema | None = None
