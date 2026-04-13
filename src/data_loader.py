import pandas as pd
from pathlib import Path

# These are the exact columns we expect in the raw dataset.
# If any are missing, we catch it immediately and stop.
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_raw_data(filepath: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    """
    Loads the raw AI4I 2020 Predictive Maintenance dataset from a CSV file.
    Validates that the file exists and all expected columns are present.
    Returns a pandas DataFrame.
    """

    path = Path(filepath)

    # Check the file actually exists before trying to read it
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please download ai4i2020.csv and place it in data/raw/"
        )

    df = pd.read_csv(path)

    # Check all expected columns are present
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following expected columns are missing from the dataset: {missing_cols}"
        )

    print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    return df