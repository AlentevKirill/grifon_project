function MetricItem({ label, value }) {
  const normalizedValue = typeof value === "number" ? (value * 100).toFixed(2) : "N/A";
  return (
    <div className="metric-item">
      <span>{label}</span>
      <strong>{normalizedValue === "N/A" ? normalizedValue : `${normalizedValue}%`}</strong>
    </div>
  );
}

function ModelStatusCard({ status }) {
  if (!status) {
    return (
      <section className="card">
        <h2>Model status</h2>
        <p>Loading...</p>
      </section>
    );
  }

  return (
    <section className="card">
      <h2>Model status</h2>
      <p>
        <strong>Frozen model available:</strong> {status.is_trained ? "Yes" : "No"}
      </p>
      <p>
        <strong>Features:</strong> {status.feature_count}
      </p>
      {status.target_column && (
        <p>
          <strong>Target column:</strong> {status.target_column}
        </p>
      )}

      {status.class_distribution?.non_fraud !== undefined && (
        <div className="metrics-grid">
          <div className="metric-item">
            <span>Non-fraud samples</span>
            <strong>{status.class_distribution.non_fraud}</strong>
          </div>
          <div className="metric-item">
            <span>Fraud samples</span>
            <strong>{status.class_distribution.fraud}</strong>
          </div>
        </div>
      )}

      {status.validation_metrics && (
        <>
          <h3>Validation metrics</h3>
          <div className="metrics-grid">
            <MetricItem label="Precision" value={status.validation_metrics.precision} />
            <MetricItem label="Recall" value={status.validation_metrics.recall} />
            <MetricItem label="F1-score" value={status.validation_metrics.f1_score} />
            <MetricItem label="ROC-AUC" value={status.validation_metrics.roc_auc} />
          </div>
        </>
      )}

      {status.test_metrics && (
        <>
          <h3>Test metrics</h3>
          <div className="metrics-grid">
            <MetricItem label="Precision" value={status.test_metrics.precision} />
            <MetricItem label="Recall" value={status.test_metrics.recall} />
            <MetricItem label="F1-score" value={status.test_metrics.f1_score} />
            <MetricItem label="ROC-AUC" value={status.test_metrics.roc_auc} />
          </div>
        </>
      )}
    </section>
  );
}

export default ModelStatusCard;
