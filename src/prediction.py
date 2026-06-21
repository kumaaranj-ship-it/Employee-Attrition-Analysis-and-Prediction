import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder


# =========================================
# LOAD FEATURE ENGINEERED DATASET
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


# =========================================
# LOAD TRAINED MODEL
# =========================================

model_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\models\best_attrition_model.pkl"

model = joblib.load(model_path)

print("\n========== MODEL LOADED ==========\n")


# =========================================
# PREPARE INPUT FEATURES
# =========================================

X = df.drop("Attrition", axis=1)


# =========================================
# MAKE PREDICTIONS
# =========================================

predictions = model.predict(X)

probabilities = model.predict_proba(X)[:, 1]


# =========================================
# ADD PREDICTIONS TO DATAFRAME
# =========================================

df["PredictedAttrition"] = predictions

df["AttritionProbability"] = probabilities


# =========================================
# CREATE RISK LEVELS
# =========================================

def assign_risk_level(probability):

    if probability >= 0.75:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


df["RiskLevel"] = df["AttritionProbability"].apply(assign_risk_level)


# =========================================
# SHOW HIGH RISK EMPLOYEES
# =========================================

high_risk_employees = df[
    df["RiskLevel"] == "High Risk"
]

print("\n========== HIGH RISK EMPLOYEES ==========\n")

print(
    high_risk_employees[
        [
            "Age",
            "Department",
            "JobRole",
            "MonthlyIncome",
            "OverTime",
            "JobSatisfaction",
            "AttritionProbability",
            "RiskLevel"
        ]
    ].head(10)
)


# =========================================
# SAVE PREDICTIONS
# =========================================

output_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\employee_attrition_predictions.csv"

df.to_csv(output_path, index=False)

print("\n========== PREDICTIONS SAVED ==========\n")

print(f"Prediction file saved at:\n{output_path}")