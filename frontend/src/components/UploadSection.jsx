import { useState } from "react";

function UploadSection({ disabled, isPredicting, onPredict }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile || disabled) {
      return;
    }

    await onPredict(selectedFile);
  };

  return (
    <section className="card upload-card">
      <h2>Upload Excel (.xlsx) for Analysis</h2>
      <p className="hint">
        Upload an XLSX file matching the training feature schema (without the fraud label) to get per-transaction risk predictions.
      </p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <label className="file-input" htmlFor="transactions-file">
          <input
            id="transactions-file"
            type="file"
            accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            onChange={handleFileChange}
            disabled={disabled}
            aria-label="Upload XLSX file"
          />
          <span>{selectedFile ? selectedFile.name : "Choose XLSX file"}</span>
        </label>

        <button type="submit" className="primary" disabled={disabled || !selectedFile}>
          {isPredicting ? "Analyzing transactions..." : "Run prediction"}
        </button>
      </form>
    </section>
  );
}

export default UploadSection;
