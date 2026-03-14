from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import settings
from app.ml.trainer import train_and_persist_model


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for offline model training."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the fraud detection model offline and persist frozen artifacts "
            "(preprocessor.joblib, fraud_model.pt, model_metadata.json)."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=settings.dataset_path,
        help="Path to the training CSV dataset.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=settings.preprocessor_path,
        help="Output path for preprocessor.joblib.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.model_path,
        help="Output path for fraud_model.pt.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=settings.metadata_path,
        help="Output path for model_metadata.json.",
    )
    return parser


def main() -> int:
    """Train and persist model artifacts for inference-only runtime."""

    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    preprocessor_path = args.preprocessor_path.resolve()
    model_path = args.model_path.resolve()
    metadata_path = args.metadata_path.resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Training dataset is missing. Expected CSV at "
            f"'{dataset_path}'. Place Bank_Transaction_Fraud_Detection.csv at the project root "
            "or pass --dataset-path."
        )

    print("== Offline Fraud Model Training ==")
    print(f"Dataset: {dataset_path}")
    print(f"Preprocessor output: {preprocessor_path}")
    print(f"Model output: {model_path}")
    print(f"Metadata output: {metadata_path}")

    result = train_and_persist_model(
        dataset_path=dataset_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    print("Training completed successfully.")
    print("Validation metrics:")
    print(json.dumps(result.validation_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(result.test_metrics, indent=2))
    print("Frozen artifacts were saved and are ready for runtime inference.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
