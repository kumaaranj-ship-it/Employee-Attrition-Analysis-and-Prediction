import pandas as pd

def load_data(file_path):
    
    df = pd.read_excel(file_path)
    return df

file_path = "C:/Users/Kumaaran J/OneDrive/Desktop/Python Programming/Employee Attrition/data/raw/Employee-Attrition.xlsx"

print("loading data...")
df = load_data(file_path)

print("data loaded successfully!")

print("first 5 rows of the dataset:", df.head())
print("statistical summary:", df.describe())
print("Columns in the dataset:", df.columns.tolist)
print("nullvalues in the dataset:", df.isnull().sum())
print(df.duplicated().sum())

df.columns = df.columns.str.strip().str.replace(" ", "_")

print("\n========== CLEANED COLUMN NAMES ==========\n")
print(df.columns)

columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=columns_to_drop, inplace=True)

print("\n========== COLUMNS AFTER DROPPING ==========\n")
print(df.columns)

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print("\n========== ATTRITION VALUE COUNTS ==========\n")
print(df["Attrition"].value_counts())

output_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\data\processed_employee_attrition.csv"

df.to_csv(output_path, index=False)

print("\n========== DATA PREPROCESSING COMPLETED ==========")
print(f"\nProcessed file saved at:\n{output_path}")