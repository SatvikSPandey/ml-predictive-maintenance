from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Defines the exact input the /predict endpoint expects.
    Field() lets us set realistic value ranges and descriptions
    that appear automatically in the API documentation.
    """
    type: str = Field(
        ...,
        description="Product quality type: L (Low), M (Medium), or H (High)",
        pattern="^[LMH]$"
    )
    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin",
        ge=295.0,
        le=305.0
    )
    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin",
        ge=305.0,
        le=315.0
    )
    rotational_speed: float = Field(
        ...,
        description="Rotational speed in RPM",
        ge=1000.0,
        le=3000.0
    )
    torque: float = Field(
        ...,
        description="Torque in Nm",
        ge=3.0,
        le=80.0
    )
    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes",
        ge=0.0,
        le=260.0
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "M",
                    "air_temperature": 298.1,
                    "process_temperature": 308.6,
                    "rotational_speed": 1551.0,
                    "torque": 42.8,
                    "tool_wear": 0.0
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Defines the exact output the /predict endpoint returns.
    """
    prediction: int = Field(..., description="0 = No Failure, 1 = Failure")
    failure_probability: float = Field(..., description="Probability of failure (0.0 to 1.0)")
    result: str = Field(..., description="Human readable result: NO FAILURE or FAILURE")
    confidence: float = Field(..., description="Confidence in the predicted class")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str


class ModelInfoResponse(BaseModel):
    features: list
    model_type: str
    description: str