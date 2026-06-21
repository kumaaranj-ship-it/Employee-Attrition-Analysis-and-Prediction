import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# =========================================
# LOAD FEATURED DATASET
# =========================================

file_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\featured_employee_attrition.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========\n")


# =========================================
# ENCODE CATEGORICAL VARIABLES
# =========================================

categorical_columns = df.select_dtypes(include="object").columns

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder


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


# =========================================
# LOAD TRAINED MODEL
# =========================================

model_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\models\best_attrition_model.pkl"

model = joblib.load(model_path)

print("\n========== MODEL LOADED ==========\n")


# =========================================
# MAKE PREDICTIONS
# =========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =========================================
# EVALUATION METRICS
# =========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)


print(f"\nAccuracy Score      : {accuracy:.4f}")

print(f"Precision Score     : {precision:.4f}")

print(f"Recall Score        : {recall:.4f}")

print(f"F1 Score            : {f1:.4f}")

print(f"ROC-AUC Score       : {roc_auc:.4f}")


# =========================================
# CLASSIFICATION REPORT
# =========================================

report = classification_report(y_test, y_pred)

print("\n========== CLASSIFICATION REPORT ==========\n")

print(report)


# =========================================
# SAVE CLASSIFICATION REPORT
# =========================================

report_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\metrics\classification_report.txt"

with open(report_path, "w") as file:

    file.write(report)


# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

cm_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\charts\confusion_matrix.png"

plt.savefig(cm_path)

plt.show()


# =========================================
# ROC CURVE
# =========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

roc_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\charts\roc_curve.png"

plt.savefig(roc_path)

plt.show()


print("\n========== MODEL EVALUATION COMPLETED ==========\n")