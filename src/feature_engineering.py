import pandas as pd

file_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\processed_employee_attrition.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========\n")


# =========================================
# AGE GROUP FEATURE
# =========================================

def create_age_group(age):

    if age < 30:
        return "Young Adult"

    elif age < 40:
        return "Adult"

    else:
        return "Senior"


df["AgeGroup"] = df["Age"].apply(create_age_group)

print("\n========== AGE GROUP CREATED ==========\n")
print(df["AgeGroup"].value_counts())


# =========================================
# TENURE GROUP FEATURE
# =========================================

def create_tenure_group(years):

    if years < 3:
        return "New Employee"

    elif years < 7:
        return "Mid Tenure"

    else:
        return "Long Tenure"


df["TenureGroup"] = df["YearsAtCompany"].apply(create_tenure_group)

print("\n========== TENURE GROUP CREATED ==========\n")
print(df["TenureGroup"].value_counts())


# =========================================
# INCOME BAND FEATURE
# =========================================

def create_income_band(income):

    if income < 3000:
        return "Low Income"

    elif income < 8000:
        return "Medium Income"

    else:
        return "High Income"


df["IncomeBand"] = df["MonthlyIncome"].apply(create_income_band)

print("\n========== INCOME BAND CREATED ==========\n")
print(df["IncomeBand"].value_counts())


# =========================================
# TOTAL SATISFACTION SCORE
# =========================================

df["TotalSatisfaction"] = (
    df["EnvironmentSatisfaction"] +
    df["JobSatisfaction"] +
    df["RelationshipSatisfaction"] +
    df["WorkLifeBalance"]
)

print("\n========== TOTAL SATISFACTION FEATURE CREATED ==========\n")


# =========================================
# SAVE FEATURE ENGINEERED DATASET
# =========================================

output_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\featured_employee_attrition.csv"

df.to_csv(output_path, index=False)

print("\n========== FEATURE ENGINEERING COMPLETED ==========\n")
print(f"Feature engineered dataset saved at:\n{output_path}")