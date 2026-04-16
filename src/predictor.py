import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Paths to saved model artifacts
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"

# Ordinal encoding for product type — must match training exactly
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

# Numerical columns that need scaling — must match renamed columns from preprocessor exactly
NUMERICAL_COLUMNS = [
    "air_temp_K",
    "process_temp_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
    "temp_difference",
    "power",
]


def load_artifacts():
    """
    Loads the trained model, fitted scaler, and feature list from disk.
    Raises clear errors if any artifact is missing.
    """
    for path in [MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required artifact not found: '{path}'. "
                "Please run the training pipeline first."
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def preprocess_input(input_data: dict, scaler, feature_names: list) -> pd.DataFrame:
    """
    Applies the exact same preprocessing steps used during training
    to a single row of raw input data.
    """
    df = pd.DataFrame([{
        "Type": TYPE_MAPPING[input_data["type"]],
        "air_temp_K": input_data["air_temperature"],
        "process_temp_K": input_data["process_temperature"],
        "rotational_speed_rpm": input_data["rotational_speed"],
        "torque_Nm": input_data["torque"],
        "tool_wear_min": input_data["tool_wear"],
    }])

    # Engineer the same features created during training
    df["temp_difference"] = df["process_temp_K"] - df["air_temp_K"]
    df["power"] = df["torque_Nm"] * df["rotational_speed_rpm"]

    # Scale numerical columns using the saved scaler
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

    # Reorder columns to exactly match training feature order
    df = df[feature_names]

    return df


def predict_local(input_data: dict) -> dict:
    """
    Runs inference using the local joblib model.
    This is the fallback path when SageMaker is not available.
    """
    model, scaler, feature_names = load_artifacts()
    X = preprocess_input(input_data, scaler, feature_names)
    prediction = int(model.predict(X)[0])
    failure_prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction": prediction,
        "failure_probability": round(failure_prob, 4),
        "result": "FAILURE" if prediction == 1 else "NO FAILURE",
        "confidence": round(failure_prob if prediction == 1 else 1 - failure_prob, 4),
        "inference_backend": "local",
    }


def predict(input_data: dict) -> dict:
    """
    Main prediction function with graceful SageMaker fallback.

    Priority:
    1. SageMaker endpoint — if SAGEMAKER_ENDPOINT_NAME is set and endpoint is live
    2. Local joblib model — fallback when SageMaker is not available

    This pattern means the same codebase works in both environments:
    - Local development: uses joblib automatically
    - Production with SageMaker: uses endpoint automatically
    - Production without SageMaker: falls back to joblib gracefully
    """
    try:
        from src.sagemaker_predictor import is_sagemaker_available, predict_sagemaker
        if is_sagemaker_available():
            print("Using SageMaker endpoint for inference.")
            return predict_sagemaker(input_data)
    except Exception as e:
        print(f"SageMaker unavailable, falling back to local model. Reason: {e}")

    return predict_local(input_data)