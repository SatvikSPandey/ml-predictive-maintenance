"""
sagemaker_trainer.py

Documents how to train and deploy the XGBoost model on AWS SageMaker.
This is a reference implementation — to actually run this, you need:
  1. AWS credentials configured (aws configure)
  2. An S3 bucket to store data and model artifacts
  3. A SageMaker execution role with appropriate permissions

To run a real training job, execute this script directly:
  python src/sagemaker_trainer.py

NOTE: This will incur AWS costs. Delete the endpoint after testing.
"""

import boto3
import sagemaker
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────
REGION = "us-east-1"
BUCKET_NAME = "ml-predictive-maintenance-satvik"  # Change to your bucket name
PREFIX = "predictive-maintenance"
INSTANCE_TYPE = "ml.m5.large"  # Cheapest general purpose SageMaker instance
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")


def get_sagemaker_session():
    """
    Creates and returns a SageMaker session.
    SageMaker session manages interactions with the SageMaker API
    and the S3 bucket where data and artifacts are stored.
    """
    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    return sm_session


def prepare_and_upload_data(session):
    """
    Prepares the training data in the format SageMaker's built-in XGBoost
    container expects: CSV with target column FIRST, no header row.

    Uploads train and validation splits to S3.
    Returns S3 URIs for train and validation data.
    """
    from src.data_loader import load_raw_data
    from src.preprocessor import split_and_preprocess

    print("Preparing data for SageMaker...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_preprocess(df)

    # SageMaker built-in XGBoost requires: target column first, no header
    train_data = pd.concat([y_train.reset_index(drop=True),
                            X_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([y_test.reset_index(drop=True),
                          X_test.reset_index(drop=True)], axis=1)

    # Save locally first
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_data.to_csv("data/processed/train_sagemaker.csv", index=False, header=False)
    val_data.to_csv("data/processed/val_sagemaker.csv", index=False, header=False)

    # Upload to S3
    train_s3_uri = session.upload_data(
        path="data/processed/train_sagemaker.csv",
        bucket=BUCKET_NAME,
        key_prefix=f"{PREFIX}/data/train"
    )
    val_s3_uri = session.upload_data(
        path="data/processed/val_sagemaker.csv",
        bucket=BUCKET_NAME,
        key_prefix=f"{PREFIX}/data/val"
    )

    print(f"Train data uploaded to: {train_s3_uri}")
    print(f"Validation data uploaded to: {val_s3_uri}")
    return train_s3_uri, val_s3_uri


def train_on_sagemaker(session, train_s3_uri, val_s3_uri):
    """
    Launches a SageMaker training job using the built-in XGBoost container.

    The built-in XGBoost container is maintained by AWS — no Docker setup needed.
    Hyperparameters match what we found best during local GridSearchCV tuning.

    SageMaker spins up an ml.m5.large instance, trains, saves the model to S3,
    then shuts down the instance automatically.
    """
    print("Starting SageMaker training job...")

    # Get the URI of AWS's built-in XGBoost container for this region
    xgboost_container = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1"
    )

    # Define the training job
    estimator = sagemaker.estimator.Estimator(
        image_uri=xgboost_container,
        role=SAGEMAKER_ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET_NAME}/{PREFIX}/models",
        sagemaker_session=session,
    )

    # Hyperparameters matching our best GridSearchCV result
    estimator.set_hyperparameters(
        objective="binary:logistic",
        num_round=200,
        max_depth=5,
        eta=0.1,           # learning_rate
        subsample=0.8,
        scale_pos_weight=28,  # handles class imbalance
        eval_metric="logloss",
    )

    # Define data channels
    from sagemaker.inputs import TrainingInput
    train_input = TrainingInput(train_s3_uri, content_type="text/csv")
    val_input = TrainingInput(val_s3_uri, content_type="text/csv")

    # Launch the training job — this blocks until training is complete
    estimator.fit({"train": train_input, "validation": val_input})

    print(f"Training complete. Model artifact saved to S3.")
    return estimator


def deploy_endpoint(estimator):
    """
    Deploys the trained model as a SageMaker real-time endpoint.
    The endpoint stays running and accepts inference requests.

    IMPORTANT: Delete the endpoint after testing to avoid ongoing charges.
    Use: predictor.delete_endpoint()
    """
    print("Deploying SageMaker endpoint...")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        serializer=sagemaker.serializers.CSVSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")
    print("IMPORTANT: Delete this endpoint after testing to avoid charges!")
    print(f"To delete: predictor.delete_endpoint()")
    return predictor


def delete_endpoint(endpoint_name: str):
    """
    Deletes a SageMaker endpoint to stop incurring charges.
    Always call this after testing.
    """
    client = boto3.client("sagemaker", region_name=REGION)
    client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' deleted successfully.")


if __name__ == "__main__":
    session = get_sagemaker_session()
    train_uri, val_uri = prepare_and_upload_data(session)
    estimator = train_on_sagemaker(session, train_uri, val_uri)
    predictor = deploy_endpoint(estimator)

    # Test the endpoint
    test_input = "1,0.5,-0.3,0.2,0.8,-0.1,0.4,1200.0"
    result = predictor.predict(test_input)
    print(f"Test prediction: {result}")

    # ALWAYS delete after testing
    delete_endpoint(predictor.endpoint_name)