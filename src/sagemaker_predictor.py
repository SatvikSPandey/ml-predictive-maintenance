"""
sagemaker_predictor.py

Handles inference against a deployed SageMaker endpoint.
Used by predictor.py as the primary inference path when AWS
credentials and a valid endpoint name are available.

Falls back gracefully to local joblib model if:
  - AWS credentials are not configured
  - SAGEMAKER_ENDPOINT_NAME environment variable is not set
  - The endpoint is not available
"""

import boto3
import json
import numpy as np
import pandas as pd
import os
from botocore.exceptions import NoCredentialsError, ClientError

# Set this environment variable to your deployed endpoint name
# e.g. export SAGEMAKER_ENDPOINT_NAME=xgboost-2024-01-01-00-00-00-000
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Ordinal encoding — must match training exactly
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

# Numerical columns in exact training order
NUMERICAL_COLUMNS = [
    "air_temp_K",
    "process_temp_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
    "temp_difference",
    "power",
]


def is_sagemaker_available() -> bool:
    """
    Checks whether SageMaker inference is available by verifying:
    1. SAGEMAKER_ENDPOINT_NAME environment variable is set
    2. AWS credentials are configured
    3. The endpoint exists and is in service

    Returns True only if all three conditions are met.
    This is the same graceful fallback pattern used in Project 9 for Bedrock.
    """
    if not ENDPOINT_NAME:
        return False

    try:
        client = boto3.client("sagemaker", region_name=REGION)
        response = client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return response["EndpointStatus"] == "InService"
    except (NoCredentialsError, ClientError):
        return False


def preprocess_for_sagemaker(input_data: dict) -> str:
    """
    Preprocesses raw input into a CSV string for SageMaker inference.
    SageMaker's built-in XGBoost container expects a CSV row with
    features in the exact same order used during training.

    Note: We apply scaling using the saved local scaler — the same
    scaler fitted during training — to ensure consistency.
    """
    import joblib
    import json

    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_list.json", "r") as f:
        feature_names = json.load(f)

    df = pd.DataFrame([{
        "Type": TYPE_MAPPING[input_data["type"]],
        "air_temp_K": input_data["air_temperature"],
        "process_temp_K": input_data["process_temperature"],
        "rotational_speed_rpm": input_data["rotational_speed"],
        "torque_Nm": input_data["torque"],
        "tool_wear_min": input_data["tool_wear"],
    }])

    df["temp_difference"] = df["process_temp_K"] - df["air_temp_K"]
    df["power"] = df["torque_Nm"] * df["rotational_speed_rpm"]
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])
    df = df[feature_names]

    # Convert to CSV string — SageMaker built-in XGBoost expects this format
    return ",".join([str(v) for v in df.values[0]])


def predict_sagemaker(input_data: dict) -> dict:
    """
    Sends a prediction request to the deployed SageMaker endpoint.
    Returns the same response format as the local predictor so the
    FastAPI layer doesn't need to know which backend is being used.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    csv_input = preprocess_for_sagemaker(input_data)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=csv_input,
    )

    # SageMaker XGBoost returns raw probability as a float string
    failure_prob = float(response["Body"].read().decode("utf-8"))
    prediction = 1 if failure_prob >= 0.5 else 0

    return {
        "prediction": prediction,
        "failure_probability": round(failure_prob, 4),
        "result": "FAILURE" if prediction == 1 else "NO FAILURE",
        "confidence": round(failure_prob if prediction == 1 else 1 - failure_prob, 4),
        "inference_backend": "sagemaker",
    }