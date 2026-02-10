
import { PredictionResult } from '../types.ts';

const API_BASE = 'http://localhost:8000';

/**
 * Fetch all historical crash points for map display.
 * Calls the FastAPI backend GET /api/crashes endpoint.
 */
export const initEngine = async (): Promise<any[]> => {
  const response = await fetch(`${API_BASE}/api/crashes`);
  if (!response.ok) {
    throw new Error(`Failed to load crash data: ${response.statusText}`);
  }
  const data = await response.json();
  // Map the backend response to the format the frontend expects
  return data.map((point: any) => ({
    DEC_LATITUDE: point.lat.toString(),
    DEC_LONGITUDE: point.lng.toString(),
    MAX_SEVERITY_LEVEL: point.severity.toString(),
  }));
};

/**
 * Predict crash risk at a given lat/lng.
 * Calls the FastAPI backend POST /api/predict endpoint.
 */
export const predictRisk = async (lat: number, lng: number): Promise<PredictionResult> => {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat, lng, top_n: 3 }),
  });
  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }
  return await response.json();
};
