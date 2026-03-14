from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.dataset import DatasetBundle, load_training_dataset
from app.ml.model import FraudDetectorNN
from app.ml.preprocessing import TabularFeaturePreprocessor

RANDOM_STATE = 42
MAX_SVD_COMPONENTS = 256
ARTIFACT_VERSION = 1


@dataclass(slots=True)
class TrainingResult:
    """Serializable training result for API responses and metadata persistence."""

    target_column: str
    feature_columns: list[str]
    class_distribution: dict[str, int]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


@dataclass(slots=True)
class PersistedArtifacts:
    """Artifact paths written by a training run."""

    preprocessor_path: Path
    model_path: Path
    metadata_path: Path


def _set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds for reproducible local training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute ROC-AUC safely when one class may be absent due to very small splits."""

    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return 0.5


def _calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, probabilities),
    }


def _evaluate_model(model: FraudDetectorNN, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(features, dtype=torch.float32)
        logits = model(tensor_x).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return _calculate_metrics(labels, probabilities)


def _train_model(
    model: FraudDetectorNN,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    pos_weight: float,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> FraudDetectorNN:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32).unsqueeze(1),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_f1 = -1.0
    patience = 6
    no_improvement_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate_model(model, val_x, val_y)
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _resample_training_data(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training set using SMOTE, with robust fallback for tiny minority classes."""

    class_counts = np.bincount(labels)
    minority_count = int(class_counts[1]) if len(class_counts) > 1 else 0

    if minority_count >= 2:
        k_neighbors = min(5, minority_count - 1)
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    else:
        sampler = RandomOverSampler(random_state=RANDOM_STATE)

    resampled_x, resampled_y = sampler.fit_resample(features, labels)
    return np.asarray(resampled_x, dtype=np.float32), np.asarray(resampled_y, dtype=np.int64)


def _prepare_datasets(bundle: DatasetBundle) -> tuple[Any, Any, Any, Any, Any, Any, TabularFeaturePreprocessor]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        bundle.features,
        bundle.target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=bundle.target,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    preprocessor = TabularFeaturePreprocessor(max_svd_components=MAX_SVD_COMPONENTS)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    train_x_balanced, train_y_balanced = _resample_training_data(
        np.asarray(x_train_processed, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
    )

    return (
        train_x_balanced,
        train_y_balanced,
        np.asarray(x_val_processed, dtype=np.float32),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(x_test_processed, dtype=np.float32),
        np.asarray(y_test, dtype=np.int64),
        preprocessor,
    )


def _sha256_file(path: Path) -> str:
    """Compute deterministic SHA-256 hash for artifact integrity validation."""

    hasher = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _persist_artifacts(
    *,
    preprocessor: TabularFeaturePreprocessor,
    model: FraudDetectorNN,
    train_input_dim: int,
    result: TrainingResult,
    preprocessor_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> PersistedArtifacts:
    """Persist preprocessor/model/metadata atomically to target paths."""

    feature_metadata = getattr(preprocessor, "feature_metadata_", None)

    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor_tmp_path = preprocessor_path.with_suffix(f"{preprocessor_path.suffix}.tmp")
    model_tmp_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    metadata_tmp_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")

    joblib.dump(preprocessor, preprocessor_tmp_path)
    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "state_dict": model.state_dict(),
            "input_dim": int(train_input_dim),
        },
        model_tmp_path,
    )

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_stage": "frozen",
        "target_column": result.target_column,
        "feature_columns": result.feature_columns,
        "class_distribution": result.class_distribution,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "input_dim": int(train_input_dim),
        "feature_space": {
            "original_feature_count": int(feature_metadata.original_feature_count)
            if feature_metadata is not None
            else len(result.feature_columns),
            "transformed_feature_count": int(feature_metadata.transformed_feature_count)
            if feature_metadata is not None
            else int(train_input_dim),
            "output_feature_count": int(feature_metadata.output_feature_count)
            if feature_metadata is not None
            else int(train_input_dim),
            "uses_dimensionality_reduction": bool(feature_metadata.uses_dimensionality_reduction)
            if feature_metadata is not None
            else False,
        },
        "artifact_hashes": {
            "model_sha256": _sha256_file(model_tmp_path),
            "preprocessor_sha256": _sha256_file(preprocessor_tmp_path),
        },
    }

    metadata_tmp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preprocessor_tmp_path.replace(preprocessor_path)
    model_tmp_path.replace(model_path)
    metadata_tmp_path.replace(metadata_path)

    return PersistedArtifacts(
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def train_and_persist_model(
    dataset_path: Path,
    preprocessor_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> TrainingResult:
    """Train, evaluate, and persist a fraud detection model and preprocessing artifacts."""

    _set_random_seeds()

    bundle = load_training_dataset(str(dataset_path))

    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        preprocessor,
    ) = _prepare_datasets(bundle)

    positives = int((bundle.target == 1).sum())
    negatives = int((bundle.target == 0).sum())
    if positives == 0:
        raise ValueError("Training dataset does not contain fraudulent transactions (positive class).")

    pos_weight = max(1.0, negatives / positives)

    model = FraudDetectorNN(input_dim=train_x.shape[1])
    model = _train_model(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        pos_weight=pos_weight,
    )

    validation_metrics = _evaluate_model(model, val_x, val_y)
    test_metrics = _evaluate_model(model, test_x, test_y)

    result = TrainingResult(
        target_column=bundle.target_column,
        feature_columns=bundle.features.columns.tolist(),
        class_distribution={"non_fraud": negatives, "fraud": positives},
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    _persist_artifacts(
        preprocessor=preprocessor,
        model=model,
        train_input_dim=int(train_x.shape[1]),
        result=result,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return result
