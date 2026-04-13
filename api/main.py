from fastapi import FastAPI, HTTPException
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.predictor import predict, load_artifacts
import json

# Initialize the FastAPI app with metadata
# This metadata appears automatically in the API documentation
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered REST API for predicting CNC machine failures using the AI4I 2020 dataset.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Confirms the API is running and the model artifacts are loadable.
    Monitoring systems call this regularly to verify the service is alive.
    """
    try:
        load_artifacts()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            message="Model artifacts loaded successfully."
        )
    except FileNotFoundError as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            message=str(e)
        )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """
    Returns metadata about the deployed model.
    Useful for debugging and understanding what the model expects as input.
    """
    try:
        _, _, feature_names = load_artifacts()
        return ModelInfoResponse(
            features=feature_names,
            model_type="XGBoost Classifier",
            description=(
                "Binary classifier trained on the AI4I 2020 Predictive Maintenance dataset. "
                "Predicts machine failure from CNC sensor readings."
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_failure(request: PredictionRequest):
    """
    Main prediction endpoint.
    Accepts CNC machine sensor readings and returns a failure prediction.
    Input is validated automatically by Pydantic before reaching this function.
    """
    try:
        input_data = {
            "type": request.type,
            "air_temperature": request.air_temperature,
            "process_temperature": request.process_temperature,
            "rotational_speed": request.rotational_speed,
            "torque": request.torque,
            "tool_wear": request.tool_wear,
        }
        result = predict(input_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))