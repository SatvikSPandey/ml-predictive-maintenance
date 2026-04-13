import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path

# Mapping for ordinal encoding of product quality type
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

# Columns to drop — they carry no predictive value
COLUMNS_TO_DROP = ["UDI", "Product ID"]

# The target column we are predicting
TARGET_COLUMN = "Machine failure"

# Rename columns to remove characters XGBoost does not allow (brackets)
COLUMN_RENAME_MAP = {
    "Air temperature [K]": "air_temp_K",
    "Process temperature [K]": "process_temp_K",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "tool_wear_min",
}

# Numerical columns that will be scaled (using renamed names)
NUMERICAL_COLUMNS = [
    "air_temp_K",
    "process_temp_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
    "temp_difference",
    "power",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing ones based on physical domain knowledge.
    temp_difference: captures heat dissipation stress on the machine.
    power: captures mechanical power load (torque x speed).
    """
    df = df.copy()
    df["temp_difference"] = df["process_temp_K"] - df["air_temp_K"]
    df["power"] = df["torque_Nm"] * df["rotational_speed_rpm"]
    return df


def preprocess(df: pd.DataFrame, scaler=None, fit_scaler: bool = True):
    """
    Full preprocessing pipeline:
    1. Drops useless columns
    2. Encodes the Type column
    3. Renames columns to be XGBoost-compatible
    4. Engineers new features
    5. Separates features (X) from target (y)
    6. Scales numerical features

    If fit_scaler=True, fits a new StandardScaler on this data (use for training).
    If fit_scaler=False, uses the provided scaler to transform (use for inference).

    Returns: X (DataFrame), y (Series), scaler
    """
    df = df.copy()

    # Step 1: Drop useless columns
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Step 2: Encode Type column (L=0, M=1, H=2)
    df["Type"] = df["Type"].map(TYPE_MAPPING)

    # Step 3: Rename columns to remove XGBoost-incompatible characters
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Step 4: Engineer new features
    df = engineer_features(df)

    # Step 5: Separate features and target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Also drop the failure sub-type columns — they are consequences of failure,
    # not causes. Using them would be data leakage (the model would be cheating).
    failure_subtypes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    X = X.drop(columns=failure_subtypes)

    # Step 6: Scale numerical columns
    if fit_scaler:
        scaler = StandardScaler()
        X[NUMERICAL_COLUMNS] = scaler.fit_transform(X[NUMERICAL_COLUMNS])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit_scaler=False")
        X[NUMERICAL_COLUMNS] = scaler.transform(X[NUMERICAL_COLUMNS])

    return X, y, scaler


def split_and_preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into train and test sets, then preprocesses each.
    The scaler is fitted on training data only, then applied to test data.
    This prevents data leakage.

    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Split before preprocessing to prevent leakage
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )

    # Fit scaler on training data only
    X_train, y_train, scaler = preprocess(train_df, fit_scaler=True)

    # Apply fitted scaler to test data — never refit on test data
    X_test, y_test, _ = preprocess(test_df, scaler=scaler, fit_scaler=False)

    feature_names = list(X_train.columns)

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size:  {len(X_test)} rows")
    print(f"Features:   {feature_names}")
    print(f"Failure rate in train: {y_train.mean():.2%}")
    print(f"Failure rate in test:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def save_scaler_and_features(scaler, feature_names: list):
    """
    Saves the fitted scaler and feature list to the models/ directory.
    These are needed at inference time to process new inputs the same way.
    """
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/feature_list.json", "w") as f:
        json.dump(feature_names, f)
    print("Scaler saved to models/scaler.pkl")
    print("Feature list saved to models/feature_list.json")