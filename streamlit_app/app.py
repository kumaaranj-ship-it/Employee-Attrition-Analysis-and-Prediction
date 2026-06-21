import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)


# =========================================
# LOAD PREDICTION DATA
# =========================================

file_path = r"C:\Users\Kumaaran J\OneDrive\Desktop\Python Programming\Employee Attrition\outputs\employee_attrition_predictions.csv"

df = pd.read_csv(file_path)


# =========================================
# DASHBOARD TITLE
# =========================================

st.title("Employee Attrition Analysis and Prediction Dashboard")

st.markdown("---")


# =========================================
# KPI METRICS
# =========================================

total_employees = len(df)

attrition_count = df["Attrition"].sum()

attrition_rate = round((attrition_count / total_employees) * 100, 2)

high_risk_count = len(
    df[df["RiskLevel"] == "High Risk"]
)


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Employees", total_employees)

col2.metric("Employees Left", int(attrition_count))

col3.metric("Attrition Rate", f"{attrition_rate}%")

col4.metric("High Risk Employees", high_risk_count)


st.markdown("---")


# =========================================
# ATTRITION DISTRIBUTION
# =========================================

st.subheader("Attrition Distribution")

attrition_chart = px.histogram(
    df,
    x="Attrition",
    color="Attrition",
    title="Employee Attrition Distribution"
)

st.plotly_chart(attrition_chart, use_container_width=True)


# =========================================
# DEPARTMENT VS ATTRITION
# =========================================

st.subheader("Department Wise Attrition")

department_chart = px.histogram(
    df,
    x="Department",
    color="Attrition",
    barmode="group",
    title="Department vs Attrition"
)

st.plotly_chart(department_chart, use_container_width=True)


# =========================================
# OVERTIME VS ATTRITION
# =========================================

st.subheader("OverTime Impact on Attrition")

overtime_chart = px.histogram(
    df,
    x="OverTime",
    color="Attrition",
    barmode="group",
    title="OverTime vs Attrition"
)

st.plotly_chart(overtime_chart, use_container_width=True)


# =========================================
# JOB SATISFACTION ANALYSIS
# =========================================

st.subheader("Job Satisfaction Analysis")

satisfaction_chart = px.histogram(
    df,
    x="JobSatisfaction",
    color="Attrition",
    barmode="group",
    title="Job Satisfaction vs Attrition"
)

st.plotly_chart(satisfaction_chart, use_container_width=True)


# =========================================
# RISK LEVEL DISTRIBUTION
# =========================================

st.subheader("Employee Risk Level Distribution")

risk_chart = px.pie(
    df,
    names="RiskLevel",
    title="Risk Level Distribution"
)

st.plotly_chart(risk_chart, use_container_width=True)


# =========================================
# HIGH RISK EMPLOYEES
# =========================================

st.subheader("High Risk Employees")

high_risk_df = df[
    df["RiskLevel"] == "High Risk"
]

st.dataframe(
    high_risk_df[
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
    ]
)


# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    "Employee Attrition Analysis and Prediction Project using Machine Learning and Streamlit"
)