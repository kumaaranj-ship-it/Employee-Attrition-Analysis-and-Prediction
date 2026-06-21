import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# LOAD CLEANED DATASET
# =========================================

file_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\processed_employee_attrition.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED SUCCESSFULLY ==========\n")


# =========================================
# CREATE CHART STYLE
# =========================================

sns.set_style("whitegrid")

output_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\charts"


# =========================================
# 1. ATTRITION DISTRIBUTION
# =========================================

plt.figure(figsize=(6, 4))

sns.countplot(x="Attrition", data=df)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Employee Count")

plt.savefig(f"{output_path}/attrition_distribution.png")

plt.show()


# =========================================
# 2. DEPARTMENT VS ATTRITION
# =========================================

plt.figure(figsize=(8, 5))

sns.countplot(x="Department", hue="Attrition", data=df)

plt.title("Department vs Attrition")
plt.xticks(rotation=15)

plt.savefig(f"{output_path}/department_vs_attrition.png")

plt.show()


# =========================================
# 3. OVERTIME VS ATTRITION
# =========================================

plt.figure(figsize=(6, 4))

sns.countplot(x="OverTime", hue="Attrition", data=df)

plt.title("OverTime vs Attrition")

plt.savefig(f"{output_path}/overtime_vs_attrition.png")

plt.show()


# =========================================
# 4. JOB SATISFACTION VS ATTRITION
# =========================================

plt.figure(figsize=(7, 5))

sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)

plt.title("Job Satisfaction vs Attrition")

plt.savefig(f"{output_path}/job_satisfaction_vs_attrition.png")

plt.show()


# =========================================
# 5. MONTHLY INCOME DISTRIBUTION
# =========================================

plt.figure(figsize=(8, 5))

sns.histplot(df["MonthlyIncome"], bins=30, kde=True)

plt.title("Monthly Income Distribution")

plt.savefig(f"{output_path}/monthly_income_distribution.png")

plt.show()


# =========================================
# 6. AGE DISTRIBUTION
# =========================================

plt.figure(figsize=(8, 5))

sns.histplot(df["Age"], bins=20, kde=True)

plt.title("Employee Age Distribution")

plt.savefig(f"{output_path}/age_distribution.png")

plt.show()


# =========================================
# 7. CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(14, 10))

numeric_df = df.select_dtypes(include=["int64", "float64"])

correlation_matrix = numeric_df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig(f"{output_path}/correlation_heatmap.png")

plt.show()


# =========================================
# ATTRITION RATE
# =========================================

attrition_rate = df["Attrition"].mean() * 100

print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

print("\n========== EDA COMPLETED SUCCESSFULLY ==========\n")