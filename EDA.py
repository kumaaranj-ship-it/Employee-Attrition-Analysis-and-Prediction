"""
=========================================================
Exploratory Data Analysis (EDA)
=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# Plot Settings
# =========================================================

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

TARGET_COLUMN = "Attrition"


# =========================================================
# Utility Functions
# =========================================================

def save_plot(filename: str):
    """
    Save current matplotlib figure.
    """
    filepath = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Saved : {filepath.name}")


# =========================================================
# Load Dataset
# =========================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column missing.")

    return df


# =========================================================
# Dataset Summary
# =========================================================

def dataset_summary(df):

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nShape : {df.shape}")

    print("\nMissing Values")
    print("-" * 70)
    print(df.isnull().sum())

    print("\nDuplicate Records")
    print("-" * 70)
    print(df.duplicated().sum())

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include="object").columns

    print(f"\nNumeric Features     : {len(numeric)}")
    print(f"Categorical Features : {len(categorical)}")

    attrition_rate = (
        df[TARGET_COLUMN]
        .mean()
        * 100
    )

    print(f"\nOverall Attrition Rate : {attrition_rate:.2f}%")

    print("=" * 70)


# =========================================================
# 1 Attrition Distribution
# =========================================================

def plot_attrition_distribution(df):

    plt.figure(figsize=(7,5))

    ax = sns.countplot(
        x=TARGET_COLUMN,
        data=df,
        palette="Set2"
    )

    total = len(df)

    for p in ax.patches:

        percentage = 100 * p.get_height() / total

        ax.annotate(
            f"{percentage:.1f}%",
            (
                p.get_x() + p.get_width()/2,
                p.get_height()
            ),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.title("Employee Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Employees")

    save_plot("01_attrition_distribution.png")


# =========================================================
# 2 Correlation Heatmap
# =========================================================

def plot_heatmap(df):

    plt.figure(figsize=(14,10))

    corr = df.select_dtypes(include="number").corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False,
        linewidths=.5
    )

    plt.title("Correlation Heatmap")

    save_plot("02_correlation_heatmap.png")


# =========================================================
# 3 Monthly Income
# =========================================================

def plot_income(df):

    plt.figure(figsize=(8,6))

    sns.boxplot(
        x=TARGET_COLUMN,
        y="MonthlyIncome",
        data=df
    )

    plt.title("Monthly Income vs Attrition")

    save_plot("03_monthly_income_vs_attrition.png")


# =========================================================
# 4 Age
# =========================================================

def plot_age(df):

    plt.figure(figsize=(8,6))

    sns.boxplot(
        x=TARGET_COLUMN,
        y="Age",
        data=df
    )

    plt.title("Age vs Attrition")

    save_plot("04_age_vs_attrition.png")


# =========================================================
# 5 Overtime
# =========================================================

def plot_overtime(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        x="OverTime",
        hue=TARGET_COLUMN,
        data=df
    )

    plt.title("OverTime vs Attrition")

    save_plot("05_overtime_vs_attrition.png")


# =========================================================
# 6 Job Satisfaction
# =========================================================

def plot_job_satisfaction(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        x="JobSatisfaction",
        hue=TARGET_COLUMN,
        data=df
    )

    plt.title("Job Satisfaction vs Attrition")

    save_plot("06_job_satisfaction_vs_attrition.png")


# =========================================================
# 7 Department
# =========================================================

def plot_department(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        y="Department",
        hue=TARGET_COLUMN,
        data=df
    )

    plt.title("Department vs Attrition")

    save_plot("07_department_vs_attrition.png")


# =========================================================
# 8 Business Travel
# =========================================================

def plot_business_travel(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        y="BusinessTravel",
        hue=TARGET_COLUMN,
        data=df
    )

    plt.title("Business Travel vs Attrition")

    save_plot("08_business_travel_vs_attrition.png")


# =========================================================
# 9 Work Life Balance
# =========================================================

def plot_worklife(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        x="WorkLifeBalance",
        hue=TARGET_COLUMN,
        data=df
    )

    plt.title("Work Life Balance vs Attrition")

    save_plot("09_worklife_balance_vs_attrition.png")


# =========================================================
# 10 Numeric Histograms
# =========================================================

def plot_histograms(df):

    numeric = df.select_dtypes(include="number")

    numeric.hist(
        figsize=(18,16),
        bins=20
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontsize=18
    )

    save_plot("10_numeric_feature_histograms.png")


# =========================================================
# Main
# =========================================================

def main():

    print("\nLoading processed dataset...")

    df = load_dataset()

    dataset_summary(df)

    print("\nGenerating visualizations...\n")

    plot_attrition_distribution(df)

    plot_heatmap(df)

    plot_income(df)

    plot_age(df)

    plot_overtime(df)

    plot_job_satisfaction(df)

    plot_department(df)

    plot_business_travel(df)

    plot_worklife(df)

    plot_histograms(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nPlots saved to:\n{OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()