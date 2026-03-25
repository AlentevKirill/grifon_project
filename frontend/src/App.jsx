import { useMemo, useState } from "react";

import { predictXlsx } from "./api/client";
import PredictionTable from "./components/PredictionTable";
import TemplateGuideSection from "./components/TemplateGuideSection";
import UploadSection from "./components/UploadSection";

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 KB";
  }

  const kb = bytes / 1024;
  if (kb < 1024) {
    return `${kb.toFixed(1)} KB`;
  }

  return `${(kb / 1024).toFixed(2)} MB`;
}

function createPredictionSummary(predictions) {
  const summary = {
    lowRisk: 0,
    suspicious: 0,
    highRisk: 0,
    averageProbability: 0,
  };

  if (!predictions.length) {
    return summary;
  }

  let probabilitySum = 0;

  for (const prediction of predictions) {
    const normalizedRiskLevel = String(prediction.risk_level || "").toLowerCase();
    const probability = Number(prediction.fraud_probability_percent) || 0;

    probabilitySum += probability;

    if (normalizedRiskLevel === "low") {
      summary.lowRisk += 1;
    } else if (normalizedRiskLevel === "medium") {
      summary.suspicious += 1;
    } else if (normalizedRiskLevel === "high") {
      summary.highRisk += 1;
    }
  }

  summary.averageProbability = probabilitySum / predictions.length;
  return summary;
}

function App() {
  const [predictions, setPredictions] = useState([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [uploadedFileInfo, setUploadedFileInfo] = useState(null);

  const hasPredictions = predictions.length > 0;

  const predictionSummary = useMemo(() => createPredictionSummary(predictions), [predictions]);

  const handlePredict = async (file) => {
    setIsPredicting(true);
    setError("");
    setMessage("");

    try {
      const payload = await predictXlsx(file);
      const nextPredictions = payload.predictions || [];
      const totalRows = Number(payload.total_rows) || nextPredictions.length;

      setPredictions(nextPredictions);
      setUploadedFileInfo({
        fileName: file.name,
        fileSize: formatFileSize(file.size),
        totalRows,
      });
      setMessage(`Processed ${totalRows} transaction(s) from ${file.name}.`);
    } catch (requestError) {
      setPredictions([]);
      setUploadedFileInfo(null);
      setError(requestError.message);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <main className="page">
      <section className="container">
        <header className="hero card">
          <p className="eyebrow">Fraud Detection Dashboard</p>
          <h1>Analyze Uploaded Transactions</h1>
          <p className="subtitle">
            Upload your XLSX file and review fraud probability, risk recommendation, and row-level results.
          </p>
        </header>

        <TemplateGuideSection />

        <UploadSection disabled={isPredicting} isPredicting={isPredicting} onPredict={handlePredict} />

        {message && <p className="message success">{message}</p>}
        {error && <p className="message error">{error}</p>}

        {uploadedFileInfo && (
          <section className="card summary-card">
            <h2>Uploaded file summary</h2>
            <div className="summary-grid">
              <div className="summary-item">
                <span>File name</span>
                <strong>{uploadedFileInfo.fileName}</strong>
              </div>
              <div className="summary-item">
                <span>File size</span>
                <strong>{uploadedFileInfo.fileSize}</strong>
              </div>
              <div className="summary-item">
                <span>Transactions processed</span>
                <strong>{uploadedFileInfo.totalRows}</strong>
              </div>
              <div className="summary-item">
                <span>Average fraud probability</span>
                <strong>{predictionSummary.averageProbability.toFixed(2)}%</strong>
              </div>
              <div className="summary-item">
                <span>Low risk</span>
                <strong>{predictionSummary.lowRisk}</strong>
              </div>
              <div className="summary-item">
                <span>Suspicious</span>
                <strong>{predictionSummary.suspicious}</strong>
              </div>
              <div className="summary-item">
                <span>High probability of fraud</span>
                <strong>{predictionSummary.highRisk}</strong>
              </div>
            </div>
          </section>
        )}

        {hasPredictions && <PredictionTable predictions={predictions} />}
      </section>
    </main>
  );
}

export default App;
