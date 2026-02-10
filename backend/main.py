"""
CrashAI FastAPI Backend
-----------------------
Serves crash risk predictions from the pre-trained XGBoost model.
Loads exported artifacts from models/exports/ on startup.

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import pairwise_distances_argmin_min

# ---------------------------
# Paths
# ---------------------------
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "exports")

# ---------------------------
# Global state (loaded at startup)
# ---------------------------
state = {}


# ---------------------------
# Startup: load all artifacts
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading CrashAI model and data...")

    # Load metadata
    with open(os.path.join(EXPORTS_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(EXPORTS_DIR, "model.json"))

    # Load processed crash data
    df = pd.read_csv(os.path.join(EXPORTS_DIR, "processed_crashes.csv"))

    # Extract the encoded feature columns for prediction
    features = metadata["fixable_features"]
    encoded_cols = [f"{col}_encoded" for col in features]
    X_map = df[encoded_cols].copy()
    X_map.columns = features  # Rename back so model recognizes them

    # Build severity mappings
    severity_levels = metadata["severity_levels"]
    inverse_severity_map = {int(k): v for k, v in metadata["inverse_severity_map"].items()}

    # Store everything in global state
    state["model"] = model
    state["df_map"] = df
    state["X_map"] = X_map
    state["features"] = features
    state["severity_levels"] = severity_levels
    state["inverse_severity_map"] = inverse_severity_map
    state["metadata"] = metadata
    state["explainer"] = shap.TreeExplainer(model)

    print(f"  Loaded {len(df)} crash records")
    print(f"  Model features: {features}")
    print(f"  Severity levels: {severity_levels}")
    print("✅ CrashAI backend ready!")

    yield  # App runs here

    state.clear()


# ---------------------------
# App
# ---------------------------
app = FastAPI(title="CrashAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Request/Response models
# ---------------------------
class PredictRequest(BaseModel):
    lat: float
    lng: float
    top_n: int = 3


class SeverityProb(BaseModel):
    level: int
    label: str
    probability: float


class TopFix(BaseModel):
    feature: str
    impact: float
    recommendation: str
    currentValue: str


class PredictResponse(BaseModel):
    riskScore: float
    severityProbs: list[SeverityProb]
    topFixes: list[TopFix]


class CrashPoint(BaseModel):
    lat: float
    lng: float
    severity: int


# ---------------------------
# Human-readable feature names and value-aware recommendations
# ---------------------------
FEATURE_NAMES = {
    "ILLUMINATION": "Lighting Infrastructure",
    "TCD_TYPE": "Traffic Control Devices",
    "ROAD_CONDITION": "Road Surface Integrity",
    "LANE_CLOSED": "Work Zone Protection",
    "TCD_FUNC_CD": "Signal Hardware",
    "WORK_ZONE_IND": "Work Zone Indicator",
    "WORK_ZONE_LOC": "Work Zone Location",
    "WORK_ZONE_TYPE": "Work Zone Type",
    "ROADWAY_CLEARED": "Roadway Clearance",
    "TFC_DETOUR_IND": "Traffic Detour Status",
}

# PennDOT code descriptions for feature values
FEATURE_VALUE_LABELS = {
    "ILLUMINATION": {
        0: "Dark (no street lights)",
        1: "Dark (street lights on)",
        2: "Dark (street lights off)",
        3: "Dusk",
        4: "Dawn",
        5: "Daylight",
    },
    "ROAD_CONDITION": {
        0: "Dry",
        1: "Wet",
        2: "Sand/Mud/Gravel",
        3: "Snow Covered",
        4: "Icy",
        5: "Water/Puddles",
        6: "Oil/Other",
        7: "Slush",
        9: "Other/Unknown",
    },
    "TCD_TYPE": {
        0: "No Control",
        1: "Stop Sign",
        2: "Signal",
        3: "Flashing Signal",
        4: "Yield Sign",
        5: "School Zone Sign",
        7: "RR Crossing",
    },
    "TCD_FUNC_CD": {
        0: "No TCD Present",
        1: "TCD Functioning",
        2: "TCD Not Functioning",
        3: "TCD Obscured",
        4: "TCD Missing",
    },
    "ROADWAY_CLEARED": {
        0: "Not Cleared",
        1: "Partially Cleared",
        2: "Fully Cleared",
    },
    "WORK_ZONE_IND": {
        "N": "No Work Zone",
        "Y": "Active Work Zone",
    },
    "TFC_DETOUR_IND": {
        "N": "No Detour",
        "Y": "Detour Active",
        "U": "Unknown",
    },
}

# Value-aware recommendations: (feature, condition) -> recommendation
RECOMMENDATIONS = {
    "ILLUMINATION": {
        "dark": "Install or upgrade street lighting — this area has poor nighttime illumination. Deploy high-output LED fixtures with increased pole density.",
        "dusk_dawn": "Enhance transitional lighting for dusk/dawn conditions. Add retroreflective lane markers and solar-powered warning flashers.",
        "default": "Review lighting adequacy at this location and consider upgrading fixtures.",
    },
    "ROAD_CONDITION": {
        "wet": "Address wet-road hazards — apply High Friction Surface Treatment (HFST) and improve drainage to reduce hydroplaning risk.",
        "snow_ice": "Prioritize winter maintenance — increase plowing/salting frequency and deploy anti-icing pre-treatment on this road segment.",
        "debris": "Schedule surface cleaning and address drainage issues causing sand, mud, or gravel accumulation on roadway.",
        "default": "Schedule road surface inspection and resurfacing to correct surface deficiencies.",
    },
    "TCD_TYPE": {
        "no_control": "This uncontrolled intersection needs traffic control — evaluate for stop sign, signal, or roundabout installation.",
        "signal": "Optimize signal timing and phase sequencing. Consider adding protected left-turn phases and pedestrian intervals.",
        "stop": "Evaluate stop sign visibility and compliance. Consider upgrading to LED-illuminated signs or adding rumble strips on approach.",
        "default": "Review traffic control devices and implement high-visibility retroreflective signage.",
    },
    "TCD_FUNC_CD": {
        "not_func": "Critical: Traffic control device is non-functional. Prioritize immediate repair and deploy temporary controls.",
        "obscured": "Traffic control signage is obscured — clear vegetation, relocate signs, or add overhead duplicates for visibility.",
        "missing": "Missing traffic control device — install replacement immediately and evaluate if upgraded controls are warranted.",
        "default": "Modernize traffic control hardware and detection sensors for real-time optimization.",
    },
    "ROADWAY_CLEARED": {
        "not_cleared": "Roadway not cleared after incidents — improve emergency response times and deploy rapid clearance protocols.",
        "partial": "Only partial clearance achieved — review incident management procedures and consider dedicated tow/clearance contracts.",
        "default": "Maintain current clearance operations and review response time metrics.",
    },
    "WORK_ZONE_IND": {
        "active": "Active work zone present — enhance channelization, extend advance warning signage, and ensure compliant taper lengths.",
        "default": "No work zone present — focus on other infrastructure improvements.",
    },
    "WORK_ZONE_LOC": {
        "default": "Review work zone placement to minimize conflict points with through traffic.",
    },
    "WORK_ZONE_TYPE": {
        "default": "Evaluate work zone configuration and transition to designs with improved positive barriers.",
    },
    "LANE_CLOSED": {
        "closed": "Lane closure detected — enhance longitudinal buffers, improve merging taper geometry, and add dynamic message signs.",
        "default": "No lane closures detected. Focus on other improvement areas.",
    },
    "TFC_DETOUR_IND": {
        "active": "Active traffic detour in effect — verify detour signage adequacy, add GPS-compatible routing updates, and monitor detour safety.",
        "unknown": "Detour status unknown — establish monitoring and improve incident communication protocols.",
        "default": "No detour currently active. Deploy automated detour routing systems for future incidents.",
    },
}

SEVERITY_LABELS = {
    0: "Not Injured",
    1: "Minor Injury",
    2: "Moderate Injury",
    3: "Major Injury",
    4: "Killed",
    8: "Injury (Severity Unknown)",
    9: "Unknown",
}


def get_feature_value_label(feature: str, value) -> str:
    """Get human-readable label for a feature's encoded value."""
    labels = FEATURE_VALUE_LABELS.get(feature, {})
    if value is None:
        return "Unknown"
    # Try direct lookup with original value
    str_val = str(value).strip()
    label = labels.get(str_val, None)
    if label:
        return label
    # Try integer lookup
    try:
        int_val = int(float(value))
        label = labels.get(int_val, None)
        if label:
            return label
        return f"Code {int_val}"
    except (ValueError, TypeError):
        return str_val if str_val else "Unknown"


def get_recommendation(feature: str, raw_value) -> str:
    """Get a value-aware recommendation based on feature and its current value."""
    recs = RECOMMENDATIONS.get(feature, {})
    str_val = str(raw_value).strip() if raw_value is not None else ""
    try:
        val = int(float(raw_value))
    except (ValueError, TypeError):
        val = -1

    if feature == "ILLUMINATION":
        if val in (0, 1, 2):
            return recs.get("dark", recs["default"])
        elif val in (3, 4):
            return recs.get("dusk_dawn", recs["default"])
        return recs["default"]
    elif feature == "ROAD_CONDITION":
        if val in (1, 5):
            return recs.get("wet", recs["default"])
        elif val in (3, 4, 7):
            return recs.get("snow_ice", recs["default"])
        elif val in (2, 6):
            return recs.get("debris", recs["default"])
        return recs["default"]
    elif feature == "TCD_TYPE":
        if val == 0:
            return recs.get("no_control", recs["default"])
        elif val == 2:
            return recs.get("signal", recs["default"])
        elif val == 1:
            return recs.get("stop", recs["default"])
        return recs["default"]
    elif feature == "TCD_FUNC_CD":
        if val == 2:
            return recs.get("not_func", recs["default"])
        elif val == 3:
            return recs.get("obscured", recs["default"])
        elif val == 4:
            return recs.get("missing", recs["default"])
        return recs["default"]
    elif feature == "ROADWAY_CLEARED":
        if val == 0:
            return recs.get("not_cleared", recs["default"])
        elif val == 1:
            return recs.get("partial", recs["default"])
        return recs["default"]
    elif feature == "WORK_ZONE_IND":
        if str_val.upper() == "Y":
            return recs.get("active", recs["default"])
        return recs["default"]
    elif feature == "LANE_CLOSED":
        if val > 0:
            return recs.get("closed", recs["default"])
        return recs["default"]
    elif feature == "TFC_DETOUR_IND":
        if str_val.upper() == "Y":
            return recs.get("active", recs["default"])
        elif str_val.upper() == "U":
            return recs.get("unknown", recs["default"])
        return recs["default"]

    return recs.get("default", f"Review {feature} conditions at this location.")


# ---------------------------
# Prediction logic (from crashai.py, SHAP-based version)
# ---------------------------
def predict_at_location(lat: float, lng: float, top_n: int = 3) -> PredictResponse:
    """
    Predict severity probabilities and top crash causes at a given location.
    Uses the same logic as predict_crash_risk_at_location from crashai.py.
    """
    model = state["model"]
    df_map = state["df_map"]
    X_map = state["X_map"]
    severity_levels = state["severity_levels"]
    inverse_severity_map = state["inverse_severity_map"]
    explainer = state["explainer"]
    metadata = state["metadata"]

    # 1. Find the closest crash
    coords = df_map[["DEC_LATITUDE", "DEC_LONGITUDE"]].values
    closest_idx, _ = pairwise_distances_argmin_min(
        np.array([[lat, lng]]), coords
    )
    closest_idx = closest_idx[0]

    # 2. Extract features for this crash
    row_features = X_map.iloc[closest_idx : closest_idx + 1]

    # 3. Predict probabilities
    probs = model.predict_proba(row_features)[0]
    severity_probs_dict = {
        inverse_severity_map[i]: float(probs[i])
        for i in range(len(severity_levels))
    }

    # 4. Compute SHAP values
    shap_values = explainer.shap_values(row_features)

    # 5. Pick the relevant class SHAP (highest predicted probability)
    if isinstance(shap_values, list):  # multi-class
        top_class = probs.argmax()
        row_shap = np.array(shap_values[top_class][0], dtype=float).flatten()
    else:
        row_shap = np.array(shap_values[0], dtype=float).flatten()

    # 6. Build feature importance DataFrame
    features_list = []
    shap_vals = []
    values = []

    for i, col in enumerate(X_map.columns):
        features_list.append(col)
        shap_vals.append(float(row_shap[i]))
        val = row_features.iloc[0, i]
        if isinstance(val, (np.ndarray, list)):
            val = val.item()
        values.append(val)

    row_shap_df = pd.DataFrame({
        "feature": features_list,
        "shap_value": shap_vals,
        "feature_value": values,
    })

    # 7. Select top_n by absolute SHAP
    top_row_features = row_shap_df.reindex(
        row_shap_df["shap_value"].abs().sort_values(ascending=False).index
    ).head(top_n)

    # 8. Compute a risk score (weighted average of severity * probability)
    risk_score = sum(
        sev * prob for sev, prob in severity_probs_dict.items()
    )
    # Normalize to a 1-5 scale (severity levels go from 0-9)
    risk_score = max(1.0, min(5.0, 1.0 + (risk_score / 9.0) * 4.0))

    # 9. Build severity probability list for all 7 levels
    severity_prob_list = []
    for i in range(len(severity_levels)):
        sev_level = inverse_severity_map[i]
        severity_prob_list.append(SeverityProb(
            level=int(sev_level),
            label=SEVERITY_LABELS.get(int(sev_level), f"Level {sev_level}"),
            probability=round(float(probs[i]), 4),
        ))

    # 10. Get raw feature values from the original (non-encoded) data for recommendations
    closest_raw_row = df_map.iloc[closest_idx]

    # 11. Build topFixes from SHAP results with value-aware recommendations
    top_fixes = []
    for _, r in top_row_features.iterrows():
        feat = r["feature"]
        raw_val = closest_raw_row.get(feat, r["feature_value"])
        top_fixes.append(TopFix(
            feature=FEATURE_NAMES.get(feat, feat),
            impact=round(abs(float(r["shap_value"])) * 10, 2),
            recommendation=get_recommendation(feat, raw_val),
            currentValue=get_feature_value_label(feat, raw_val),
        ))

    return PredictResponse(
        riskScore=round(risk_score, 1),
        severityProbs=severity_prob_list,
        topFixes=top_fixes,
    )


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/api/crashes", response_model=list[CrashPoint])
async def get_crashes(limit: int = 1000):
    """Return crash locations for map display (sampled to avoid browser overload)."""
    df = state["df_map"]

    # Sample if dataset is larger than the limit
    if len(df) > limit:
        # Bias sampling toward higher severity crashes so important points show up
        weights = df["MAX_SEVERITY_LEVEL"].fillna(0).astype(float) + 1.0
        sample_df = df.sample(n=limit, weights=weights, random_state=42)
    else:
        sample_df = df

    points = []
    for _, row in sample_df.iterrows():
        lat = row["DEC_LATITUDE"]
        lng = row["DEC_LONGITUDE"]
        sev = row.get("MAX_SEVERITY_LEVEL", 0)
        if pd.notna(lat) and pd.notna(lng):
            points.append(CrashPoint(
                lat=float(lat),
                lng=float(lng),
                severity=int(sev),
            ))
    return points


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict crash risk at a given lat/lng."""
    try:
        return predict_at_location(req.lat, req.lng, req.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "crashes_loaded": len(state.get("df_map", [])),
        "model_loaded": "model" in state,
    }
