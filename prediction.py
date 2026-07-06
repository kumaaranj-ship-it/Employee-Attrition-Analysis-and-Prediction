"""
=========================================================
Employee Attrition Analysis and Prediction
Module      : prediction.py
Author      : Kumaaran J
Description : Machine Learning Model Training
=========================================================
"""

import time

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from featureengineering import get_processed_data

# =========================================================
# Global Random State
# =========================================================

RANDOM_STATE = 42

# =========================================================
# Individual Model Training Functions
# =========================================================


def train_logistic(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def train_extra_trees(X_train, y_train):

    model = ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def train_adaboost(X_train, y_train):

    model = AdaBoostClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def train_gradient_boosting(X_train, y_train):

    model = GradientBoostingClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# Train All Models
# =========================================================


def train_all_models():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    ) = get_processed_data()

    trainers = {

        "Logistic Regression": train_logistic,

        "Decision Tree": train_decision_tree,

        "Random Forest": train_random_forest,

        "Extra Trees": train_extra_trees,

        "AdaBoost": train_adaboost,

        "Gradient Boosting": train_gradient_boosting,

    }

    trained_models = {}
    training_times = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    for model_name, trainer in trainers.items():
        print(f"\nTraining {model_name}...")

        start = time.perf_counter()
        model = trainer(X_train, y_train)
        end = time.perf_counter()
        elapsed_time = end - start

        print(f"Completed in {elapsed_time:.3f} seconds")

        trained_models[model_name] = model
        training_times[model_name] = elapsed_time

        print("\n" + "=" * 70)
        print(f"Successfully trained {len(trained_models)} models.")
        print("=" * 70)

    print("\nTraining Time Summary")
    print("-" * 70)

    for model_name, train_time in training_times.items():
        print(f"{model_name:<25} : {train_time:.3f} sec")

    return (
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
# Main
# =========================================================


def main():

    train_all_models()


if __name__ == "__main__":

    main()