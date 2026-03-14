# Bank Transaction Fraud Detection Web Application

This project is a full-stack application for detecting fraudulent banking transactions.

- **Backend:** FastAPI + PyTorch
- **Frontend:** React (Vite)
- **Training dataset:** `Bank_Transaction_Fraud_Detection.csv` in the project root

---

## Features

- Neural-network-based fraud detection model
- Handles class imbalance with:
  - **Stratified train/validation/test split**
  - **SMOTE synthetic minority oversampling** (with safe fallback for tiny minority samples)
  - **Weighted loss (`BCEWithLogitsLoss` with `pos_weight`)**
- Evaluation metrics include:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Frontend supports CSV upload and displays, per transaction:
  - Fraud probability (%)
  - Recommendation (`Low risk`, `Suspicious`, `High probability of fraud`)
  - Visual risk indicator (badge + progress bar)

---

## Project Structure

```text
.
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  ├─ core/
│  │  ├─ ml/
│  │  ├─ schemas/
│  │  └─ services/
│  ├─ artifacts/              # Saved preprocessor/model/metadata after training
│  ├─ tests/
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  ├─ package.json
│  └─ vite.config.js
├─ run_project.ps1            # Generic launcher for backend + frontend
├─ run_with_trained_model.ps1 # Launcher that requires frozen trained artifacts
├─ train_offline.ps1          # Offline trainer that generates frozen artifacts
└─ Bank_Transaction_Fraud_Detection.csv
```

---

## One-Command Offline Training (Windows PowerShell)

Use this script to train the model offline and generate frozen artifacts required by inference-only startup.

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_offline.ps1
```

What this script does:

1. Validates Python availability
2. Optionally installs backend dependencies
3. Verifies dataset file exists
4. Runs offline training entrypoint (`python -m app.ml.train_offline`)
5. Persists artifacts atomically:
   - `backend/artifacts/preprocessor.joblib`
   - `backend/artifacts/fraud_model.pt`
   - `backend/artifacts/model_metadata.json`

Optional flags:

```powershell
# Skip backend dependency installation
powershell -ExecutionPolicy Bypass -File .\train_offline.ps1 -SkipInstall

# Train with a custom dataset path
powershell -ExecutionPolicy Bypass -File .\train_offline.ps1 -DatasetPath ".\data\Bank_Transaction_Fraud_Detection.csv"

# Write artifacts to custom locations
powershell -ExecutionPolicy Bypass -File .\train_offline.ps1 -PreprocessorPath ".\backend\artifacts\preprocessor.joblib" -ModelPath ".\backend\artifacts\fraud_model.pt" -MetadataPath ".\backend\artifacts\model_metadata.json"
```

After training succeeds, start inference-only runtime:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1
```

---

## One-Command Startup (Trained Model Required, Windows PowerShell)

Use this launcher when you want to run the web app in inference mode with frozen model artifacts only.

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1
```

What this launcher does:

1. Verifies required artifacts exist:
   - `backend/artifacts/preprocessor.joblib`
   - `backend/artifacts/fraud_model.pt`
   - `backend/artifacts/model_metadata.json`
2. Validates artifact metadata and required SHA-256 fields
3. Validates artifact SHA-256 hashes against metadata
4. Optionally installs dependencies
5. Starts backend and frontend in separate PowerShell windows

Optional flags:

```powershell
# Skip dependency installation
powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1 -SkipInstall

# Custom ports
powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1 -BackendPort 8001 -FrontendPort 5174

# Do not auto-open browser
powershell -ExecutionPolicy Bypass -File .\run_with_trained_model.ps1 -NoBrowser
```

Default URLs:

- **Application UI:** `http://127.0.0.1:5173`
- **Backend API:** `http://127.0.0.1:8000`
- **Backend health check:** `http://127.0.0.1:8000/health`

---

## One-Command Startup (Generic, Windows PowerShell)

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_project.ps1
```

This generic launcher starts backend/frontend and can be used during development when you may still train a model from the UI.

---

## Dataset Requirements

The training dataset must be placed at the project root and named exactly:

- `Bank_Transaction_Fraud_Detection.csv`

### Target column

The backend auto-detects target column names among common variants:

- `is_fraud`, `fraud`, `label`, `target`, `class`, `isfraud`, `is_fraudulent`, `fraudulent`

Target values must be binary (0/1), or mappable values such as:

- Positive: `1`, `true`, `yes`, `fraud`
- Negative: `0`, `false`, `no`, `legit`, `legitimate`

### Prediction CSV format

Uploaded CSVs for inference must contain the **same feature columns** used at training time (target column must be excluded).

---

## Backend Setup and Run

From project root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Backend base URL:

- `http://127.0.0.1:8000`

Health check:

- `GET /health`

---

## Frontend Setup and Run

From `frontend/`:

```powershell
npm install
npm run dev
```

Frontend URL (default):

- `http://127.0.0.1:5173`

Vite proxy forwards `/api/*` requests to backend `http://127.0.0.1:8000`.

---

## API Endpoints

All API endpoints are under prefix: `/api/v1/fraud`

### 1) Check model status

- `GET /api/v1/fraud/status`

Returns whether model artifacts exist and model metrics metadata.

### 2) Train model

- `POST /api/v1/fraud/train`

Runs training on `Bank_Transaction_Fraud_Detection.csv`, saves artifacts to `backend/artifacts/`, and returns validation/test metrics.

### 3) Predict from CSV

- `POST /api/v1/fraud/predict`
- Form-data field: `file` (CSV)

Returns per-row fraud probabilities, recommendations, and risk levels.

---

## Model Training Workflow

1. Load dataset and detect target column.
2. Normalize target to binary 0/1.
3. Stratified split:
   - 70% train
   - 15% validation
   - 15% test
4. Build preprocessing pipeline:
   - Numeric: median imputation + standardization
   - Categorical: mode imputation + one-hot encoding
5. Apply resampling on training data only:
   - SMOTE (or fallback oversampling for extremely rare fraud samples)
6. Train PyTorch neural network with weighted BCE loss.
7. Early stopping using validation F1 score.
8. Evaluate using precision/recall/F1/ROC-AUC.
9. Persist artifacts:
   - `preprocessor.joblib`
   - `fraud_model.pt`
   - `model_metadata.json`

---

## Test and Build Verification

From project root:

```powershell
python -m pytest backend/tests -q
```

From `frontend/`:

```powershell
npm run build
```

Current verification status:

- Backend tests: **13 passed**
- Frontend production build: **successful**

---

## Security Notes

- File upload size is limited (`max_upload_size_mb` in backend settings).
- CSV format and required feature columns are strictly validated before inference.
- CORS is not wildcard by default; allowed origins are configurable via settings.
- In production:
  - set strict `allowed_origins`
  - run behind HTTPS + reverse proxy
  - add authentication/authorization for train/predict endpoints
  - enable request logging and rate limiting

---

## Quick Usage Flows

### Inference-only runtime with trained artifacts

1. Run `train_offline.ps1` from project root.
2. Run `run_with_trained_model.ps1` from project root.
3. Open `http://127.0.0.1:5173`.
4. Upload CSV with transaction features.
5. Review fraud probabilities and recommendations.

### Development flow

1. Run `run_project.ps1` from project root.
2. Open `http://127.0.0.1:5173`.
3. Click **Train model** (when dataset or training code changes).
4. Upload CSV with transaction features.
5. Review fraud probabilities and recommendations.
