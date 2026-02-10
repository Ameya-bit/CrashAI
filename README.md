
# CrashAI | Crash Risk Prediction Platform

A full-stack crash risk analysis tool for Pennsylvania. Uses an XGBoost model trained on 110,000+ PennDOT crash records to predict crash severity at any location and recommend infrastructure improvements.

## Architecture

```
React Frontend (Vite, port 3000)
    ↓  fetch()
FastAPI Backend (Uvicorn, port 8000)
    ↓  loads on startup
Trained XGBoost Model + Processed Crash Data (models/exports/)
```

## Quick Start

### Prerequisites
- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`

### 1. Set up Python environment
```bash
cd CrashAI
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Train the model (one-time)

> **Skip this step if `models/exports/` already contains `model.json`, `processed_crashes.csv`, and `metadata.json`.**

Place the 8 PennDOT CSV files in `models/`:
- `CRASH_2024.csv`, `VEHICLE_2024.csv`, `COMMVEH_2024.csv`, `CYCLE_2024.csv`
- `FLAGS_2024.csv`, `PERSON_2024.csv`, `ROADWAY_2024.csv`, `TRAILVEH_2024.csv`

Then run:
```bash
python models/train.py
```
This trains the XGBoost classifier and exports 3 files to `models/exports/`:
| File | Description |
|---|---|
| `model.json` | Trained XGBoost model |
| `processed_crashes.csv` | Merged crash data with coordinates and encoded features |
| `metadata.json` | Severity mappings, feature importances, encoder info |

### 3. Start the backend
```bash
.\venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```
The backend loads the model and crash data on startup. You should see:
```
✅ CrashAI backend ready!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4. Start the frontend
In a **separate terminal**:
```bash
npm install   # first time only
npm run dev
```
Open **http://localhost:3000** in your browser.

## Features

### Interactive Map
- **Click any location** on the map to analyze crash risk at that point
- **1,000 historical crash points** displayed on the map (sampled from 110k, biased toward high-severity)
- **Street/Satellite toggle** for different map views
- **Coordinate search** — enter lat/lng to jump to any location

### Crash Risk Prediction
- **Risk Score (1–5)** — weighted severity prediction normalized from the XGBoost model output
- **7 Severity Probabilities** — individual probability bars for each severity level:
  - Not Injured, Minor Injury, Moderate Injury, Major Injury, Killed, Injury (Unknown), Unknown
- Predictions are based on the **nearest historical crash** to the clicked location

### Priority Interventions
- **SHAP-based analysis** identifies the top 3 infrastructure factors contributing to crash severity
- **Value-aware recommendations** — suggestions vary based on actual conditions (e.g., "dark road" → lighting upgrade, "icy road" → winter maintenance)
- **Current condition badges** show detected values using PennDOT codes (e.g., "Dark (no street lights)", "Snow Covered")

### Fixable Features Analyzed
| Feature | Description |
|---|---|
| `ILLUMINATION` | Lighting conditions |
| `ROAD_CONDITION` | Road surface state |
| `TCD_TYPE` | Traffic control device type |
| `TCD_FUNC_CD` | Traffic control device functionality |
| `ROADWAY_CLEARED` | Post-incident clearance status |
| `WORK_ZONE_IND` | Work zone presence |
| `WORK_ZONE_LOC` | Work zone location |
| `WORK_ZONE_TYPE` | Work zone configuration |
| `LANE_CLOSED` | Lane closure status |
| `TFC_DETOUR_IND` | Traffic detour status |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check (model loaded, crash count) |
| `GET` | `/api/crashes?limit=1000` | Historical crash points for map display |
| `POST` | `/api/predict` | Predict crash risk at `{lat, lng}` |

### Example predict request
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.3077, "lng": -75.3576}'
```

## Project Structure
```
CrashAI/
├── backend/
│   └── main.py              # FastAPI app with prediction endpoints
├── models/
│   ├── train.py              # Training pipeline script
│   ├── crashai.py            # Original Colab notebook (reference)
│   ├── exports/
│   │   ├── model.json        # Trained XGBoost model
│   │   ├── processed_crashes.csv
│   │   └── metadata.json
│   └── *.csv                 # Raw PennDOT data (not in git)
├── services/
│   └── mockData.ts           # API client (calls FastAPI backend)
├── App.tsx                   # React map + sidebar UI
├── types.ts                  # TypeScript interfaces
├── index.html                # HTML entry point
├── index.tsx                 # React entry point
├── package.json              # Node dependencies
└── vite.config.ts            # Vite configuration
```
