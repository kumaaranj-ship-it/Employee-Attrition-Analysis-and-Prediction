"""
=========================================================
Description : Model Evaluation & Best Model Selection
=========================================================
"""

from pathlib import Path
from tracemalloc import start
import joblib
import pandas as pd
import json
import os
import time


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from prediction import train_all_models

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

BEST_MODEL_FILE = (
    MODEL_DIR / "best_model.joblib"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)
COMPARISON_FILE = (
    MODEL_DIR /
    "model_comparison.csv"
)

METRICS_FILE = (
    MODEL_DIR /
    "model_metrics.json"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# Evaluate Individual Model
# =========================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):
    """
    Evaluate a single trained model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

    }

    return metrics


# =========================================================
# Evaluate All Models
# =========================================================

def evaluate_all_models():

    (
    trained_models,

    training_times,

    X_train,

    X_test,

    y_train,

    y_test,

    preprocessor,

    feature_names,

    ) = train_all_models()

    results = []

    print("\n" + "=" * 75)
    print("MODEL EVALUATION")
    print("=" * 75)

    for model_name, model in trained_models.items():

        print(f"Evaluating {model_name}...")

        metrics = evaluate_model(

            model,

            model_name,

            X_test,

            y_test

        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by=[

            "F1 Score",

            "ROC-AUC",

            "Accuracy"

        ],

        ascending=False

    ).reset_index(drop=True)

    print("\n")
    print("=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)

    print(
    results_df
    .round(4)
    .to_string(index=False)
)
    save_model_comparison(results_df)

    return (

    results_df,

    trained_models,

    training_times,

    X_train,

    X_test,

    y_train,

    y_test,

    preprocessor,

    feature_names,

    )
# =========================================================
# Save Model Comparison
# =========================================================

def save_model_comparison(results_df):
    """
    Save model comparison results.
    """

    results_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    print("\nModel comparison saved.")

    print(COMPARISON_FILE)
    
# =========================================================
# Save Model Metrics
# =========================================================

def save_model_metrics(
    best_model_name,
    best_metrics,
    training_times,
    prediction_time,
    X_train,
    X_test,
):
    """
    Save model evaluation summary for the Streamlit dashboard.
    """

    import json
    import os

    model_path = MODEL_DIR / "best_model.joblib"

    metrics = {

        "best_model": best_model_name,

        "accuracy": round(best_metrics["Accuracy"], 4),

        "precision": round(best_metrics["Precision"], 4),

        "recall": round(best_metrics["Recall"], 4),

        "f1_score": round(best_metrics["F1 Score"], 4),

        "roc_auc": round(best_metrics["ROC-AUC"], 4),

        "training_times": {
            model: round(time_taken, 4)
            for model, time_taken in training_times.items()
        },

        "prediction_time": round(
            prediction_time,
            6
        ),

        "train_samples": len(X_train),

        "test_samples": len(X_test),

        "feature_count": X_train.shape[1],

        "model_size_mb": round(
            os.path.getsize(model_path) / (1024 * 1024),
            2
        )

    }

    with open(
        METRICS_FILE,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nModel metrics saved successfully.")


# =========================================================
# Select Best Model
# =========================================================

def select_best_model(

    results_df,

    trained_models

):
    """
    Select the best model based on

    1. F1 Score
    2. ROC-AUC
    3. Accuracy
    """

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    print("\n" + "=" * 75)
    print("BEST MODEL SELECTED")
    print("=" * 75)

    print(f"\nModel : {best_model_name}")
    best = results_df.iloc[0]

    print()

    print(f"Model      : {best['Model']}")

    print(f"Accuracy   : {best['Accuracy']:.4f}")

    print(f"Precision  : {best['Precision']:.4f}")

    print(f"Recall     : {best['Recall']:.4f}")

    print(f"F1 Score   : {best['F1 Score']:.4f}")

    print(f"ROC-AUC    : {best['ROC-AUC']:.4f}")
    print("=" * 75)

    return (
        best_model_name,
        best_model,
        best
    )


import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Save Best Model
# =========================================================

def save_best_model(
    best_model,
    preprocessor
):
    """
    Save the best model and preprocessing pipeline.
    """

    model_path = MODEL_DIR / "best_model.joblib"

    preprocessor_path = (
        MODEL_DIR /
        "preprocessor.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print("\nSaved Successfully")

    print(model_path)

    print(preprocessor_path)


# =========================================================
# Confusion Matrix
# =========================================================

def plot_confusion_matrix(
    best_model,
    X_test,
    y_test
):

    predictions = best_model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",
        
        linewidths=1,
        square=True,

        cbar=False

    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR /
        "confusion_matrix.png",

        dpi=300

    )
    plt.show()
    plt.close()

    print("Confusion Matrix Saved")


# =========================================================
# ROC Curve
# =========================================================

def plot_roc_curves(
    trained_models,
    X_test,
    y_test
):

    plt.figure(figsize=(8,6))

    for model_name, model in trained_models.items():

        probabilities = model.predict_proba(
            X_test
        )[:,1]

        fpr, tpr, _ = roc_curve(

            y_test,

            probabilities

        )

        auc = roc_auc_score(

            y_test,

            probabilities

        )

        plt.plot(

            fpr,

            tpr,

            label=f"{model_name} ({auc:.3f})"

        )

    plt.plot(

        [0,1],

        [0,1],

        linestyle="--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve Comparison")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR /

        "roc_curve.png",

        dpi=300

    )
    plt.show()

    plt.close()

    print("ROC Curve Saved")


# =========================================================
# Feature Importance
# =========================================================

def plot_feature_importance(
    best_model,
    feature_names
):
    """
    Plot feature importance for both tree-based models
    and Logistic Regression.
    """

    # ---------------------------------------------
    # Tree-based Models
    # ---------------------------------------------

    if hasattr(best_model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": best_model.feature_importances_

        })

    # ---------------------------------------------
    # Logistic Regression
    # ---------------------------------------------

    elif hasattr(best_model, "coef_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": abs(best_model.coef_[0])

        })

    # ---------------------------------------------
    # Unsupported Models
    # ---------------------------------------------

    else:

        print("\nFeature importance unavailable.")

        return

    importance = (

        importance

        .sort_values(

            by="Importance",

            ascending=False

        )

        .head(20)

    )

    print("\n")
    print("="*75)
    print("TOP 20 IMPORTANT FEATURES")
    print("="*75)

    print(
        importance.to_string(index=False)
    )

    plt.figure(figsize=(10,8))

    sns.barplot(

        data=importance,

        x="Importance",

        y="Feature"

    )

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR /

        "feature_importance.png",

        dpi=300

    )

    plt.show()

    plt.close()

    print("\nFeature Importance Saved")


# =========================================================
# Main
# =========================================================

def main():

    (
    results_df,

    trained_models,

    training_times,

    X_train,

    X_test,

    y_train,

    y_test,

    preprocessor,

    feature_names,

    ) = evaluate_all_models()

    (

        best_model_name,

        best_model,
        
        best_metrics

    ) = select_best_model(

        results_df,

        trained_models,

    )
    
    start = time.perf_counter()
    best_model.predict(X_test)
    prediction_time = time.perf_counter() - start

    save_best_model(
        best_model,
        preprocessor,
    )

    save_model_metrics(
        best_model_name,
        best_metrics,
        training_times,
        prediction_time,
        X_train,
        X_test,
    )

    plot_confusion_matrix(

        best_model,

        X_test,

        y_test,

    )

    plot_roc_curves(

        trained_models,

        X_test,

        y_test,

    )

    plot_feature_importance(

        best_model,

        feature_names,

    )

    print("\n" + "="*75)

    print("MODEL EVALUATION COMPLETED")

    print("="*75)

    print(f"\nBest Model : {best_model_name}")

    print("\nSaved Job Libs and Plots to Outputs Directory")
    print("----------------------------")

    print("best_model.joblib")

    print("preprocessor.joblib")

    print("confusion_matrix.png")

    print("roc_curve.png")

    print("feature_importance.png")
    
    print("model_metrics.json")

    print("="*75)


if __name__ == "__main__":

    main()