function RiskBadge({ riskLevel }) {
  const normalized = (riskLevel || "").toLowerCase();
  const className = `risk-badge ${normalized}`;
  const label = normalized ? normalized.toUpperCase() : "UNKNOWN";

  return <span className={className}>{label}</span>;
}

function ProbabilityBar({ value }) {
  const safeValue = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;

  return (
    <div className="progress-wrapper" aria-label={`Fraud probability ${safeValue.toFixed(2)} percent`}>
      <div className="progress-bar" style={{ width: `${safeValue}%` }} />
      <span className="progress-value">{safeValue.toFixed(2)}%</span>
    </div>
  );
}

function PredictionTable({ predictions }) {
  return (
    <section className="card">
      <h2>Prediction results</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Fraud probability</th>
              <th>Recommendation</th>
              <th>Risk level</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((item) => (
              <tr key={`${item.row_index}-${item.recommendation}`}>
                <td>{item.row_index}</td>
                <td>
                  <ProbabilityBar value={item.fraud_probability_percent} />
                </td>
                <td>{item.recommendation}</td>
                <td>
                  <RiskBadge riskLevel={item.risk_level} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default PredictionTable;
