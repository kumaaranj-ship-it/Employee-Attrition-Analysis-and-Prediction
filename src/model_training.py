import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =========================================
# LOAD FEATURED DATASET
# =========================================

file_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\featured_employee_attrition.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========\n")


# =========================================
# ENCODE CATEGORICAL COLUMNS
# =========================================

label_encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

print("\n========== CATEGORICAL ENCODING COMPLETED ==========\n")


# =========================================
# FEATURES & TARGET
# =========================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT COMPLETED ==========\n")

print(f"Training Rows: {X_train.shape[0]}")
print(f"Testing Rows: {X_test.shape[0]}")


# =========================================
# LOGISTIC REGRESSION
# =========================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_predictions)

print(f"\nLogistic Regression Accuracy: {log_accuracy:.4f}")


# =========================================
# RANDOM FOREST
# =========================================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


# =========================================
# XGBOOST
# =========================================

xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")


# =========================================
# BEST MODEL SELECTION
# =========================================

models = {
    "Logistic Regression": log_accuracy,
    "Random Forest": rf_accuracy,
    "XGBoost": xgb_accuracy
}

best_model_name = max(models, key=models.get)

print(f"\nBest Model: {best_model_name}")


# =========================================
# SAVE BEST MODEL
# =========================================

if best_model_name == "Logistic Regression":
    best_model = log_model

elif best_model_name == "Random Forest":
    best_model = rf_model

else:
    best_model = xgb_model


model_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\models\best_attrition_model.pkl"

joblib.dump(best_model, model_path)

print("\n========== MODEL SAVED SUCCESSFULLY ==========\n")

print(f"Saved Model Path:\n{model_path}")