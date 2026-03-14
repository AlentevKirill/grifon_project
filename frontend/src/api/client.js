const API_BASE = "/api/v1/fraud";

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    const message = payload?.detail || payload?.message || "Request failed.";
    throw new Error(message);
  }

  return payload;
}

export async function getModelStatus() {
  const response = await fetch(`${API_BASE}/status`, {
    method: "GET",
  });
  return parseResponse(response);
}

export async function trainModel() {
  const response = await fetch(`${API_BASE}/train`, {
    method: "POST",
  });
  return parseResponse(response);
}

export async function predictCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}
