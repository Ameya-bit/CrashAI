# -*- coding: utf-8 -*-
"""
CrashAI Training Pipeline
--------------------------
Loads PennDOT crash CSVs, merges them, trains an XGBoost severity classifier,
and exports the trained model + processed data for use by the prediction API.

Usage:
    python train.py

Outputs (in ./exports/):
    - model.json              : Trained XGBoost model
    - processed_crashes.csv   : Merged crash data with lat/lon and encoded features
    - metadata.json           : Severity mappings, fixable features list, feature importances
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing the CSV files
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

CSV_FILES = {
    "crash":    "CRASH_2024.csv",
    "vehicle":  "VEHICLE_2024.csv",
    "commveh":  "COMMVEH_2024.csv",
    "cycle":    "CYCLE_2024.csv",
    "flags":    "FLAGS_2024.csv",
    "person":   "PERSON_2024.csv",
    "roadway":  "ROADWAY_2024.csv",
    "trailveh": "TRAILVEH_2024.csv",
}

FIXABLE_FEATURES = [
    'LANE_CLOSED', 'WORK_ZONE_IND', 'WORK_ZONE_LOC', 'WORK_ZONE_TYPE',
    'ILLUMINATION', 'ROADWAY_CLEARED', 'ROAD_CONDITION',
    'TCD_FUNC_CD', 'TCD_TYPE', 'TFC_DETOUR_IND'
]

TARGET = 'MAX_SEVERITY_LEVEL'
LAT_COL = 'DEC_LATITUDE'
LON_COL = 'DEC_LONGITUDE'


# ---------------------------
# Helper Functions
# ---------------------------
def aggregate_dataset(df, source_name, numeric_cols=[], categorical_cols=[]):
    """Aggregate per CRN safely, ignoring missing columns."""
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = 'mean'
    for col in categorical_cols:
        agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'

    if len(agg_dict) == 0:
        agg_df = df[['CRN']].drop_duplicates()
        agg_df[f'SOURCE_{source_name}'] = 1
        return agg_df

    agg_df = df.groupby('CRN').agg(agg_dict).reset_index()
    agg_df[f'SOURCE_{source_name}'] = 1
    return agg_df


# ---------------------------
# Step 1: Load all datasets
# ---------------------------
def load_data():
    print("Loading CSV files...")
    dataframes = {}
    for key, filename in CSV_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found, skipping.")
            continue
        dataframes[key] = pd.read_csv(filepath)
        print(f"  Loaded {filename}: {len(dataframes[key])} rows")
    return dataframes


# ---------------------------
# Step 2: Aggregate and merge
# ---------------------------
def merge_data(dataframes):
    print("\nAggregating datasets...")

    crash_df = dataframes["crash"]

    aggregations = {
        "vehicle": {
            "numeric_cols": ['TRAVEL_SPD', 'DVR_PRES_IND', 'PEOPLE_IN_UNIT', 'COMM_VEH_IND'],
            "categorical_cols": ['VEH_TYPE', 'BODY_TYPE'],
        },
        "commveh": {
            "numeric_cols": ['COMM_FEATURE_1', 'COMM_FEATURE_2'],
            "categorical_cols": ['VEH_TYPE'],
        },
        "cycle": {
            "numeric_cols": ['TRAVEL_SPD'],
            "categorical_cols": ['VEH_TYPE'],
        },
        "trailveh": {
            "numeric_cols": ['TRAVEL_SPD'],
            "categorical_cols": ['VEH_TYPE'],
        },
        "person": {
            "numeric_cols": ['AGE'],
            "categorical_cols": ['GENDER'],
        },
        "flags": {
            "numeric_cols": [],
            "categorical_cols": ['FLAG_TYPE'],
        },
        "roadway": {
            "numeric_cols": ['SPEED_LIMIT'],
            "categorical_cols": ['RDWY_ALIGNMENT'],
        },
    }

    agg_dfs = [crash_df]
    for key, config in aggregations.items():
        if key in dataframes:
            agg_df = aggregate_dataset(
                dataframes[key],
                source_name=key,
                numeric_cols=config["numeric_cols"],
                categorical_cols=config["categorical_cols"],
            )
            agg_dfs.append(agg_df)
            print(f"  Aggregated {key}: {len(agg_df)} rows")

    print("\nMerging all datasets on CRN...")
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='CRN', how='outer',
                                      suffixes=('', f'_dup_{id(right)}')),
        agg_dfs
    )

    # Drop any accidental duplicate columns
    dup_cols = [col for col in merged_df.columns if '_dup_' in col]
    if dup_cols:
        merged_df.drop(columns=dup_cols, inplace=True)

    # Fill numeric NaNs with 0
    for col in merged_df.select_dtypes(include='number').columns:
        merged_df[col] = merged_df[col].fillna(0)

    print(f"  Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    return merged_df


# ---------------------------
# Step 3: Train XGBoost model
# ---------------------------
def train_model(merged_df):
    print("\nPreparing features for training...")

    # Keep only fixable features that exist in the data
    features = [col for col in FIXABLE_FEATURES if col in merged_df.columns]
    print(f"  Using features: {features}")

    X = merged_df[features].copy()

    # Fill numeric NaNs with 0
    numeric_cols = X.select_dtypes(include='number').columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # Encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include='object').columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = {
            "classes": le.classes_.tolist()
        }

    # Prepare target
    y_raw = pd.to_numeric(merged_df[TARGET], errors='coerce').fillna(0).astype(int)
    severity_levels = sorted(y_raw.unique().tolist())
    severity_map = {v: i for i, v in enumerate(severity_levels)}
    inverse_severity_map = {i: v for v, i in severity_map.items()}
    y = y_raw.map(severity_map)

    print(f"  Severity levels: {severity_levels}")
    print(f"  Training samples: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(severity_levels),
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_test_orig = y_test.map(inverse_severity_map)
    y_pred_orig = pd.Series(y_pred).map(inverse_severity_map)

    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred_orig))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_orig, y_pred_orig))

    # Feature importances
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_.tolist()
    }).sort_values(by='importance', ascending=False)

    print("\nTop fixable features contributing to severity:")
    print(importance_df.to_string(index=False))

    return model, features, severity_levels, inverse_severity_map, label_encoders, importance_df


# ---------------------------
# Step 4: Prepare map data
# ---------------------------
def prepare_map_data(merged_df, features, label_encoders):
    """Prepare the geo-referenced crash data with encoded features for prediction."""
    print("\nPreparing map data for spatial predictions...")

    df_map = merged_df.dropna(subset=[LAT_COL, LON_COL]).copy()
    X_map = df_map[features].copy()

    # Encode categorical features the same way as training
    for col in X_map.select_dtypes(include='object').columns:
        X_map[col] = X_map[col].astype(str)
        le = LabelEncoder()
        le.fit(merged_df[col].astype(str))
        X_map[col] = le.transform(X_map[col].astype(str))

    print(f"  Map data: {len(df_map)} crash locations with coordinates")
    return df_map, X_map


# ---------------------------
# Step 5: Export everything
# ---------------------------
def export_artifacts(model, df_map, X_map, features, severity_levels,
                     inverse_severity_map, label_encoders, importance_df):
    """Export trained model, processed data, and metadata."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 1. Save the XGBoost model
    model_path = os.path.join(EXPORT_DIR, "model.json")
    model.save_model(model_path)
    print(f"\n  Saved model to {model_path}")

    # 2. Save processed crash data (lat/lon + raw features for spatial lookup)
    # Include the encoded features as separate columns
    export_df = df_map[[LAT_COL, LON_COL, 'CRN', TARGET] + features].copy()
    for col in X_map.columns:
        export_df[f'{col}_encoded'] = X_map[col].values

    crashes_path = os.path.join(EXPORT_DIR, "processed_crashes.csv")
    export_df.to_csv(crashes_path, index=False)
    print(f"  Saved processed crashes to {crashes_path}")

    # 3. Save metadata
    metadata = {
        "fixable_features": features,
        "severity_levels": [int(s) for s in severity_levels],
        "inverse_severity_map": {str(k): int(v) for k, v in inverse_severity_map.items()},
        "label_encoders": label_encoders,
        "feature_importances": importance_df.to_dict(orient='records'),
        "lat_col": LAT_COL,
        "lon_col": LON_COL,
        "target": TARGET,
    }

    metadata_path = os.path.join(EXPORT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    print(f"\n✅ All artifacts exported to {EXPORT_DIR}/")


# ---------------------------
# Main
# ---------------------------
def main():
    print("=" * 60)
    print("CrashAI Training Pipeline")
    print("=" * 60)

    # Load
    dataframes = load_data()
    if "crash" not in dataframes:
        print("ERROR: CRASH_2024.csv is required. Exiting.")
        return

    # Merge
    merged_df = merge_data(dataframes)

    # Train
    model, features, severity_levels, inverse_severity_map, label_encoders, importance_df = train_model(merged_df)

    # Prepare map data
    df_map, X_map = prepare_map_data(merged_df, features, label_encoders)

    # Export
    export_artifacts(model, df_map, X_map, features, severity_levels,
                     inverse_severity_map, label_encoders, importance_df)

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
