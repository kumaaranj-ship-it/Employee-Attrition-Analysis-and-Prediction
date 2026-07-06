"""
=========================================================
Description : Feature Engineering & Data Preprocessing
=========================================================
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_attrition_clean.csv"
)

TARGET_COLUMN = "Attrition"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# =========================================================
# Load Dataset
# =========================================================

def load_processed_data():
    """
    Load cleaned dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    return df


# =========================================================
# Feature Engineering
# =========================================================

def create_features(df):
    """
    Create HR analytical features.
    """

    df = df.copy()

    df["IncomePerYearExperience"] = (
        df["MonthlyIncome"]
        /
        (df["TotalWorkingYears"] + 1)
    )

    df["YearsSincePromotionRatio"] = (
        df["YearsSinceLastPromotion"]
        /
        (df["YearsAtCompany"] + 1)
    )

    df["TenureRatio"] = (
        df["YearsAtCompany"]
        /
        (df["Age"] + 1)
    )

    df["CompaniesPerYear"] = (
        df["NumCompaniesWorked"]
        /
        (df["TotalWorkingYears"] + 1)
    )

    return df


# =========================================================
# Data Preprocessing
# =========================================================

def preprocess_data(df):

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    categorical_columns = (
        X.select_dtypes(include="object")
        .columns
        .tolist()
    )

    numerical_columns = (
        X.select_dtypes(exclude="object")
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(

        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]

    )

    categorical_pipeline = Pipeline(

        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )

        ]

    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )
        # =====================================================
    # Feature Names
    # =====================================================

    encoded_columns = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_columns)
    )

    feature_names = (
        numerical_columns +
        encoded_columns.tolist()
    )

    # =====================================================
    # Convert to DataFrame
    # =====================================================

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )
    
    # =====================================================
    # Column Validation
    # =====================================================

    assert list(X_train_processed.columns) == feature_names, (
        "Training feature names do not match."
    )

    assert list(X_test_processed.columns) == feature_names, (
        "Testing feature names do not match."
    )

    # =====================================================
    # Validation
    # =====================================================

    if len(feature_names) != X_train_processed.shape[1]:
        raise ValueError(
            "Feature count mismatch after preprocessing."
        )

    if X_train_processed.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature names detected."
        )

    # =====================================================
    # Console Summary
    # =====================================================

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(f"\nOriginal Dataset Shape      : {df.shape}")

    print(f"Training Samples            : {len(X_train)}")
    print(f"Testing Samples             : {len(X_test)}")

    print(f"\nNumerical Features          : {len(numerical_columns)}")
    print(f"Categorical Features        : {len(categorical_columns)}")
    print(f"Encoded Categorical Columns : {len(encoded_columns)}")
    print(f"Final Feature Count         : {len(feature_names)}")

    print("\nTarget Distribution (Train)")
    print(y_train.value_counts())

    print("\nTarget Distribution (Test)")
    print(y_test.value_counts())

    print("\nProcessed Training Shape")
    print(X_train_processed.shape)

    print("\nProcessed Testing Shape")
    print(X_test_processed.shape)

    print("\nFeature Name Validation")
    print("PASSED")

    print("=" * 70)

    return (

        X_train_processed,

        X_test_processed,

        y_train,

        y_test,

        preprocessor,

        feature_names,

    )


# =========================================================
# Public API
# =========================================================

def get_processed_data():
    """
    Returns fully processed data for model training.

    Returns
    -------
    X_train : pandas.DataFrame
    X_test : pandas.DataFrame
    y_train : pandas.Series
    y_test : pandas.Series
    preprocessor : ColumnTransformer
    feature_names : list
    """

    df = load_processed_data()

    df = create_features(df)

    return preprocess_data(df)


# =========================================================
# Main
# =========================================================

def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    ) = get_processed_data()

    print("\nFirst Five Engineered Features\n")
    print(X_train.head())


if __name__ == "__main__":
    main()