"""
=========================================================
DATA LOADER MODULE
This module is responsible for loading, validating, preprocessing, and saving the employee attrition dataset. It ensures that the dataset is clean and ready for analysis or modeling.
"""

from pathlib import Path
import pandas as pd

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Employee-Attrition.xlsx"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "employee_attrition_clean.csv"
)

# =========================================================
# Constants
# =========================================================

COLUMNS_TO_DROP = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber",
]

TARGET_COLUMN = "Attrition"

TARGET_MAPPING = {
    "Yes": 1,
    "No": 0,
}


# =========================================================
# Data Loader
# =========================================================

def load_employee_data():
    """
    Load, validate, preprocess and save the employee dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    # -----------------------------------------------------
    # Validate Dataset
    # -----------------------------------------------------

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{RAW_DATA_PATH}"
        )

    # -----------------------------------------------------
    # Read Dataset
    # -----------------------------------------------------

    try:
        df = pd.read_excel(RAW_DATA_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read dataset.\n{exc}"
        )

    # -----------------------------------------------------
    # Basic Validation
    # -----------------------------------------------------

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    expected_values = {"Yes", "No"}
    actual_values = set(df[TARGET_COLUMN].dropna().unique())

    if not actual_values.issubset(expected_values):
        raise ValueError(
            f"Unexpected values in target column: {actual_values}"
        )

    # -----------------------------------------------------
    # Drop Constant / Identifier Columns
    # -----------------------------------------------------

    df.drop(
        columns=COLUMNS_TO_DROP,
        errors="ignore",
        inplace=True,
    )

    # -----------------------------------------------------
    # Encode Target
    # -----------------------------------------------------

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .map(TARGET_MAPPING)
        .astype("int64")
    )

    # -----------------------------------------------------
    # Create Processed Folder
    # -----------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save Cleaned Dataset
    # -----------------------------------------------------

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("EMPLOYEE ATTRITION DATASET SUMMARY")
    print("=" * 65)

    print(f"\nDataset Shape : {df.shape}")

    print("\nColumns")
    print("-" * 65)
    print(df.columns.tolist())

    print("\nData Types")
    print("-" * 65)
    print(df.dtypes)

    print("\nMissing Values")
    print("-" * 65)
    print(df.isnull().sum())

    print("\nDuplicate Records")
    print("-" * 65)
    print(df.duplicated().sum())

    print("\nTarget Distribution")
    print("-" * 65)
    print(df[TARGET_COLUMN].value_counts())

    print("\nTarget Percentage")
    print("-" * 65)
    print(
        (
            df[TARGET_COLUMN]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nProcessed Dataset Saved To")
    print("-" * 65)
    print(PROCESSED_DATA_PATH)

    print("\nDataset loaded successfully.")
    print("=" * 65)

    return df