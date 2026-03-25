from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.ml.model import FraudDetectorNN
from app.ml.risk import get_risk_assessment
from app.ml.trainer import ARTIFACT_VERSION, TrainingResult, train_and_persist_model


class FraudModelService:
    """Service responsible for model training lifecycle and prediction workflow."""

    XLSX_MIME_TYPES = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
    }

    DEFAULT_TEMPLATE_FILENAME = "Template Data.xlsx"

    def __init__(self) -> None:
        self._model: FraudDetectorNN | None = None
        self._preprocessor: Any | None = None
        self._metadata: dict[str, Any] | None = None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _required_artifact_paths() -> dict[str, Path]:
        """Return required frozen artifact files for runtime inference."""

        return {
            "fraud_model.pt": settings.model_path,
            "preprocessor.joblib": settings.preprocessor_path,
            "model_metadata.json": settings.metadata_path,
        }

    def _missing_required_artifacts(self) -> dict[str, Path]:
        """Return missing runtime artifacts keyed by expected filename."""

        return {
            artifact_name: artifact_path
            for artifact_name, artifact_path in self._required_artifact_paths().items()
            if not artifact_path.exists()
        }

    def _is_artifacts_ready(self) -> bool:
        return len(self._missing_required_artifacts()) == 0

    def _load_metadata(self) -> dict[str, Any]:
        if not settings.metadata_path.exists():
            return {}

        try:
            return json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model metadata is corrupted and cannot be parsed.",
            ) from exc

    def _validate_artifact_integrity(self, metadata: dict[str, Any], checkpoint: dict[str, Any], preprocessor: Any) -> None:
        metadata_version = int(metadata.get("artifact_version", 0))
        checkpoint_version = int(checkpoint.get("artifact_version", 0))
        if metadata_version != ARTIFACT_VERSION or checkpoint_version != ARTIFACT_VERSION:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model artifact version is incompatible with the running application.",
            )

        expected_hashes: dict[str, str] = metadata.get("artifact_hashes", {})
        expected_model_hash = expected_hashes.get("model_sha256")
        expected_preprocessor_hash = expected_hashes.get("preprocessor_sha256")

        if not expected_model_hash or not expected_preprocessor_hash:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model metadata is missing required artifact integrity hashes.",
            )

        actual_model_hash = self._sha256_file(settings.model_path)
        actual_preprocessor_hash = self._sha256_file(settings.preprocessor_path)
        if actual_model_hash != expected_model_hash or actual_preprocessor_hash != expected_preprocessor_hash:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Artifact integrity validation failed. Retrain and republish model artifacts.",
            )

        model_input_dim = int(checkpoint.get("input_dim", 0))
        metadata_input_dim = int(metadata.get("input_dim", 0))
        if model_input_dim <= 0 or metadata_input_dim <= 0 or model_input_dim != metadata_input_dim:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model artifact metadata is inconsistent: input dimension mismatch.",
            )

        preprocessor_output_dim = int(getattr(preprocessor, "output_dim_", 0))
        if preprocessor_output_dim <= 0 or preprocessor_output_dim != model_input_dim:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Preprocessor output dimension does not match model input dimension.",
            )

    def get_status(self) -> dict[str, Any]:
        metadata = self._load_metadata()
        if not self._is_artifacts_ready():
            return {
                "is_trained": False,
                "feature_count": 0,
                "target_column": None,
                "class_distribution": {},
                "validation_metrics": None,
                "test_metrics": None,
            }

        return {
            "is_trained": True,
            "feature_count": len(metadata.get("feature_columns", [])),
            "target_column": metadata.get("target_column"),
            "class_distribution": metadata.get("class_distribution", {}),
            "validation_metrics": metadata.get("validation_metrics"),
            "test_metrics": metadata.get("test_metrics"),
        }

    def train_model(self) -> dict[str, Any]:
        if not settings.dataset_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "Training dataset not found. Expected file at project root: "
                    f"{settings.dataset_path.name}"
                ),
            )

        try:
            training_result: TrainingResult = train_and_persist_model(
                dataset_path=settings.dataset_path,
                preprocessor_path=settings.preprocessor_path,
                model_path=settings.model_path,
                metadata_path=settings.metadata_path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        self._model = None
        self._preprocessor = None
        self._metadata = None

        result_payload = asdict(training_result)
        return {
            "message": "Model training completed and frozen artifacts were saved successfully.",
            **result_payload,
        }

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._preprocessor is not None and self._metadata is not None:
            return

        missing_artifacts = self._missing_required_artifacts()
        if missing_artifacts:
            missing_details = ", ".join(
                f"{artifact_name} ({artifact_path})"
                for artifact_name, artifact_path in missing_artifacts.items()
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "Required frozen model artifacts are missing: "
                    f"{missing_details}. Run offline training to produce and save "
                    "backend/artifacts files before inference startup."
                ),
            )

        metadata = self._load_metadata()

        try:
            checkpoint = torch.load(settings.model_path, map_location="cpu", weights_only=False)
            preprocessor = joblib.load(settings.preprocessor_path)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to load model artifacts.",
            ) from exc

        self._validate_artifact_integrity(metadata=metadata, checkpoint=checkpoint, preprocessor=preprocessor)

        input_dim = int(checkpoint.get("input_dim", 0))
        model = FraudDetectorNN(input_dim=input_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._model = model
        self._preprocessor = preprocessor
        self._metadata = metadata

    def get_prediction_template_path(self) -> Path:
        """Return the physical XLSX template path used by the template download endpoint."""

        backend_template_path = settings.template_path
        root_template_path = settings.project_root / self.DEFAULT_TEMPLATE_FILENAME

        if backend_template_path.exists():
            return backend_template_path

        if root_template_path.exists():
            try:
                backend_template_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root_template_path, backend_template_path)
            except OSError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Template file is present at project root but could not be staged for download.",
                ) from exc
            return backend_template_path

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Prediction template file is missing. Expected at either "
                f"{backend_template_path} or {root_template_path}."
            ),
        )

    @classmethod
    def _validate_xlsx_upload_metadata(cls, file: UploadFile, size_bytes: int) -> None:
        filename = (file.filename or "").strip()
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded XLSX file must include a valid filename.",
            )

        if not filename.lower().endswith(".xlsx"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be in .xlsx format.",
            )

        if file.content_type and file.content_type not in cls.XLSX_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file content type is not supported for XLSX uploads.",
            )

        if size_bytes == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded XLSX file is empty.")

        max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        if size_bytes > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds the {settings.max_upload_size_mb}MB size limit.",
            )

    @classmethod
    async def _read_upload_dataframe(cls, file: UploadFile) -> pd.DataFrame:
        content = await file.read()
        cls._validate_xlsx_upload_metadata(file=file, size_bytes=len(content))

        try:
            dataframe = pd.read_excel(BytesIO(content), engine="openpyxl")
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="XLSX parsing is unavailable because openpyxl is not installed.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid XLSX workbook.",
            ) from exc

        if dataframe.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded XLSX has no rows.")

        return dataframe

    def _validate_prediction_schema(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        assert self._metadata is not None

        expected_columns: list[str] = self._metadata.get("feature_columns", [])
        missing_columns = [column for column in expected_columns if column not in dataframe.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded XLSX is missing required columns: {', '.join(missing_columns)}",
            )

        return dataframe[expected_columns].copy()

    async def predict_xlsx(self, file: UploadFile) -> dict[str, Any]:
        self._ensure_model_loaded()

        dataframe = await self._read_upload_dataframe(file)
        model_input = self._validate_prediction_schema(dataframe)

        assert self._model is not None
        assert self._preprocessor is not None

        transformed = self._preprocessor.transform(model_input)
        tensor_x = torch.tensor(np.asarray(transformed, dtype=np.float32), dtype=torch.float32)

        with torch.no_grad():
            logits = self._model(tensor_x).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        predictions: list[dict[str, Any]] = []
        for index, probability in enumerate(probabilities.tolist()):
            recommendation, risk_level = get_risk_assessment(probability)
            predictions.append(
                {
                    "row_index": index,
                    "fraud_probability_percent": round(probability * 100, 2),
                    "recommendation": recommendation,
                    "risk_level": risk_level,
                }
            )

        return {"total_rows": len(predictions), "predictions": predictions}


fraud_model_service = FraudModelService()
