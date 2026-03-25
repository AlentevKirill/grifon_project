from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables when present."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Bank Fraud Detection API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    backend_dir: Path = Path(__file__).resolve().parents[2]
    project_root: Path = backend_dir.parent
    artifacts_dir: Path = backend_dir / "artifacts"
    dataset_path: Path = project_root / "Bank_Transaction_Fraud_Detection.csv"
    template_path: Path = backend_dir / "app" / "templates" / "Template Data.xlsx"

    preprocessor_path: Path = artifacts_dir / "preprocessor.joblib"
    model_path: Path = artifacts_dir / "fraud_model.pt"
    metadata_path: Path = artifacts_dir / "model_metadata.json"

    max_upload_size_mb: int = Field(default=20, ge=1, le=200)
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )


settings = Settings()
