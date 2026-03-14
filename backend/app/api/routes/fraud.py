from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PredictionResponse
from app.schemas.training import ModelStatusResponse, TrainResponse
from app.services.model_service import fraud_model_service

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])


@router.get("/status", response_model=ModelStatusResponse)
def get_model_status() -> ModelStatusResponse:
    """Return current model availability and metrics metadata."""

    status_payload = fraud_model_service.get_status()
    return ModelStatusResponse(**status_payload)


@router.post("/train", response_model=TrainResponse)
def train_model() -> TrainResponse:
    """Train the fraud model using the project dataset and persist artifacts."""

    result_payload = fraud_model_service.train_model()
    return TrainResponse(**result_payload)


@router.post("/predict", response_model=PredictionResponse)
async def predict_transactions(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict fraud probability for each transaction in an uploaded CSV file."""

    prediction_payload = await fraud_model_service.predict_csv(file)
    return PredictionResponse(**prediction_payload)
