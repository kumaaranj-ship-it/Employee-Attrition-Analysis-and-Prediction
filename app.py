# ==========================================================
# Employee Attrition Analysis & Prediction Dashboard
# Streamlit Application
# ==========================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_attrition_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "best_model.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "preprocessor.joblib"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

# ==========================================================
# Model Evaluation Files
# ==========================================================

MODEL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "model_metrics.json"
)

MODEL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "model_comparison.csv"
)

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        st.error(f"Dataset not found:\n{DATA_PATH}")
        st.stop()

    return pd.read_csv(DATA_PATH)

# ==========================================================
# Load Model Artifacts
# ==========================================================

@st.cache_resource
def load_model_artifacts():

    if not MODEL_PATH.exists():
        st.error("Model file not found.")
        st.stop()

    if not PREPROCESSOR_PATH.exists():
        st.error("Preprocessor file not found.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

# ==========================================================
# Load Model Metrics
# ==========================================================

@st.cache_data
def load_model_metrics():

    import json

    if not MODEL_METRICS_PATH.exists():

        st.error("model_metrics.json not found.")

        st.stop()

    with open(MODEL_METRICS_PATH, "r") as file:

        return json.load(file)


# ==========================================================
# Load Model Comparison
# ==========================================================

@st.cache_data
def load_model_comparison():

    if not MODEL_COMPARISON_PATH.exists():

        st.error("model_comparison.csv not found.")

        st.stop()

    return pd.read_csv(MODEL_COMPARISON_PATH)

# ==========================================================
# Initialize Resources
# ==========================================================

df = load_dataset()

model, preprocessor = load_model_artifacts()

model_metrics = load_model_metrics()

model_comparison = load_model_comparison()
# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📊 HR Analytics")

st.sidebar.caption("Employee Attrition Prediction")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Exploratory Data Analysis",
        "🤖 Attrition Prediction",
        "📈 Model Evaluation",
    ],
)

# ==========================================================
# EDA PAGE
# ==========================================================

def display_plot(title, filename, insight):
    """
    Display a saved plot with title and business insight.
    """

    image_path = PLOTS_DIR / filename

    if image_path.exists():

        st.markdown(f"#### {title}")

        st.image(
            image_path,
            use_container_width=True
        )

        st.info(insight)

    else:

        st.warning(f"{filename} not found.")


def show_eda_page():

    st.title("📊 Exploratory Data Analysis")

    st.markdown("---")

    # =====================================================
    # Dataset Overview
    # =====================================================

    st.subheader("Dataset Overview")

    total_employees = len(df)

    total_features = df.shape[1]

    attrition_rate = (
        df["Attrition"].mean() * 100
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "👥 Total Employees",
            total_employees
        )

    with c2:
        st.metric(
            "📋 Total Features",
            total_features
        )

    with c3:
        st.metric(
            "📉 Attrition Rate",
            f"{attrition_rate:.2f}%"
        )

    st.markdown("---")

    # =====================================================
    # Dataset Preview
    # =====================================================

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("EDA Visualizations")

    # =====================================================
    # Distribution Analysis
    # =====================================================

    with st.expander(
        "📈 Distribution Analysis",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            display_plot(

                "Attrition Distribution",

                "01_attrition_distribution.png",

                "Shows the proportion of employees who stayed versus those who left."

            )

        with col2:

            display_plot(

                "Correlation Heatmap",

                "02_correlation_heatmap.png",

                "Highlights relationships among numerical features."

            )

    # =====================================================
    # Compensation & Demographics
    # =====================================================

    with st.expander(
        "💰 Compensation & Demographics"
    ):

        col1, col2 = st.columns(2)

        with col1:

            display_plot(

                "Monthly Income vs Attrition",

                "03_monthly_income_vs_attrition.png",

                "Compare salary distribution between retained and exited employees."

            )

        with col2:

            display_plot(

                "Age vs Attrition",

                "04_age_vs_attrition.png",

                "Shows whether employee age influences attrition."

            )

    # =====================================================
    # Employee Behaviour
    # =====================================================

    with st.expander(
        "👨‍💼 Employee Behaviour"
    ):

        col1, col2 = st.columns(2)

        with col1:

            display_plot(

                "Overtime vs Attrition",

                "05_overtime_vs_attrition.png",

                "Employees working overtime are generally more likely to leave."

            )

        with col2:

            display_plot(

                "Job Satisfaction vs Attrition",

                "06_job_satisfaction_vs_attrition.png",

                "Lower job satisfaction usually results in higher attrition."

            )

    # =====================================================
    # Organization Analysis
    # =====================================================

    with st.expander(
        "🏢 Organization Analysis"
    ):

        col1, col2 = st.columns(2)

        with col1:

            display_plot(

                "Department vs Attrition",

                "07_department_vs_attrition.png",

                "Compare employee turnover across departments."

            )

        with col2:

            display_plot(

                "Business Travel vs Attrition",

                "08_business_travel_vs_attrition.png",

                "Frequent travel may influence employee retention."

            )

    # =====================================================
    # Work-Life Analysis
    # =====================================================

    with st.expander(
        "⚖ Work-Life Analysis"
    ):

        col1, col2 = st.columns(2)

        with col1:

            display_plot(

                "Work-Life Balance",

                "09_worklife_balance_vs_attrition.png",

                "Healthy work-life balance contributes to employee retention."

            )

        with col2:

            display_plot(

                "Numeric Feature Distribution",

                "10_numeric_feature_histograms.png",

                "Distribution of all numerical variables in the dataset."

            )
            
# ==========================================================
# Prediction Helper
# ==========================================================

def predict_employee(user_inputs):
    """
    Create a complete employee profile using default values,
    update it with user inputs, perform prediction and return
    prediction details.
    """

    # Use the first employee as a template
    employee = df.iloc[0].copy()

    # Update with user inputs
    for key, value in user_inputs.items():
        employee[key] = value

    # Convert to DataFrame
    input_df = pd.DataFrame([employee])

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    input_df["IncomePerYearExperience"] = (
        input_df["MonthlyIncome"] /
        (input_df["TotalWorkingYears"] + 1)
    )

    input_df["YearsSincePromotionRatio"] = (
        input_df["YearsSinceLastPromotion"] /
        (input_df["YearsAtCompany"] + 1)
    )

    input_df["TenureRatio"] = (
        input_df["YearsAtCompany"] /
        (input_df["Age"] + 1)
    )

    input_df["CompaniesPerYear"] = (
        input_df["NumCompaniesWorked"] /
        (input_df["TotalWorkingYears"] + 1)
    )

    # Prediction
    X = preprocessor.transform(input_df)

    # Restore feature names expected by the trained model
    if hasattr(model, "feature_names_in_"):

        X = pd.DataFrame(
        X,
        columns=model.feature_names_in_
    )

    X = X[model.feature_names_in_]

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    return prediction, probability
# ==========================================================
# ATTRITION PREDICTION PAGE
# ==========================================================

def show_prediction_page():

    st.title("🤖 Employee Attrition Prediction")

    st.markdown("---")

    st.subheader("Employee Information")
    
    department = st.selectbox(
    "Department",
    sorted(df["Department"].unique())
    )
    
    available_roles = (
    df.loc[
        df["Department"] == department,
        "JobRole"
    ]
    .drop_duplicates()
    .sort_values()
    .tolist()
    )

    job_role = st.selectbox(
    "Job Role",
    available_roles
    )

    # =====================================================
    # INPUT FORM
    # =====================================================

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # ============================================
        # LEFT COLUMN
        # ============================================

        with col1:

            # =====================================================
            # Department & Job Role Selection
            # =====================================================

              
            business_travel = st.selectbox(
                "Business Travel",
                sorted(df["BusinessTravel"].unique())
            )

            overtime = st.selectbox(
                "OverTime",
                sorted(df["OverTime"].unique())
            )

            distance = st.number_input(
                "Distance From Home",
                min_value=1,
                max_value=100,
                value=10
            )

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                max_value=50000,
                value=6500
            )

        # ============================================
        # RIGHT COLUMN
        # ============================================

        with col2:

            total_years = st.number_input(
                "Total Working Years",
                min_value=0,
                max_value=50,
                value=10
            )

            years_company = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=40,
                value=5
            )

            years_role = st.number_input(
                "Years In Current Role",
                min_value=0,
                max_value=25,
                value=3
            )

            years_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                max_value=20,
                value=2
            )

        st.markdown("---")

        st.subheader("Employee Satisfaction")

        c1, c2, c3 = st.columns(3)

        with c1:

            job_satisfaction = st.selectbox(
                "Job Satisfaction",
                [1, 2, 3, 4],
                index=2
            )

        with c2:

            environment = st.selectbox(
                "Environment Satisfaction",
                [1, 2, 3, 4],
                index=2
            )

        with c3:

            worklife = st.selectbox(
                "Work Life Balance",
                [1, 2, 3, 4],
                index=2
            )

        submitted = st.form_submit_button(
            "🔍 Predict Attrition",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        user_inputs = {

            "Department": department,
            
            "JobRole": job_role,

            "BusinessTravel": business_travel,

            "OverTime": overtime,

            "DistanceFromHome": distance,

            "MonthlyIncome": monthly_income,

            "TotalWorkingYears": total_years,

            "YearsAtCompany": years_company,

            "YearsInCurrentRole": years_role,

            "YearsSinceLastPromotion": years_promotion,

            "JobSatisfaction": job_satisfaction,

            "EnvironmentSatisfaction": environment,

            "WorkLifeBalance": worklife

        }

        prediction, probability = predict_employee(
            user_inputs
        )

        st.markdown("---")

        # =====================================================
        # Prediction Summary
        # =====================================================

        st.subheader("Prediction Summary")

        prediction_text = (
            "Employee is likely to Leave"
            if prediction == 1
            else "Employee is likely to Stay"
        )

        probability_percent = probability * 100

        if probability_percent >= 70:
            risk_level = "🔴 High Risk"

        elif probability_percent >= 40:
            risk_level = "🟠 Medium Risk"

        else:
            risk_level = "🟢 Low Risk"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                prediction_text
            )

        with col2:
            st.metric(
                "Attrition Probability",
                f"{probability_percent:.2f}%"
            )

        with col3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.markdown("---")
        
        # =====================================================
        # Similar Employee Profiles
        # =====================================================

        st.subheader("Similar Employee Profiles")

        filtered_df = df.copy()

        filtered_df = filtered_df[
            (filtered_df["Department"] == department)
        ]

        filtered_df = filtered_df[
            (filtered_df["JobRole"] == job_role)
        ]

        filtered_df = filtered_df[
            (filtered_df["BusinessTravel"] == business_travel)
        ]

        filtered_df = filtered_df[
            (filtered_df["OverTime"] == overtime)
        ]

        st.write(
            f"Matching Employees : {len(filtered_df)}"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # HR Recommendations
        # =====================================================

        st.subheader("HR Recommendations")

        recommendations = []

        if overtime == "Yes":
            recommendations.append(
                "• Review overtime allocation and employee workload."
            )

        if job_satisfaction <= 2:
            recommendations.append(
                "• Conduct a Job Satisfaction discussion."
            )

        if environment <= 2:
            recommendations.append(
                "• Improve workplace environment and engagement."
            )

        if worklife <= 2:
            recommendations.append(
                "• Encourage a healthier work-life balance."
            )

        if years_promotion >= 5:
            recommendations.append(
                "• Review career growth and promotion opportunities."
            )

        if monthly_income < 5000:
            recommendations.append(
                "• Evaluate employee compensation."
            )

        if len(recommendations) == 0:
            st.success(
                "No significant HR concerns identified based on the provided information."
            )
        else:
            for rec in recommendations:
                st.write(rec)

        st.markdown("---")
        
# ==========================================================
# MODEL EVALUATION PAGE
# ==========================================================

def show_model_evaluation_page():

    st.title("📈 Model Evaluation")

    st.markdown("---")

    # =====================================================
    # Best Model
    # =====================================================

    st.subheader("🏆 Best Model")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Algorithm",
            model_metrics["best_model"]
        )

        st.metric(
            "Accuracy",
            f'{model_metrics["accuracy"]:.4f}'
        )

    with c2:

        st.metric(
            "Precision",
            f'{model_metrics["precision"]:.4f}'
        )

        st.metric(
            "Recall",
            f'{model_metrics["recall"]:.4f}'
        )

    with c3:

        st.metric(
            "F1 Score",
            f'{model_metrics["f1_score"]:.4f}'
        )

        st.metric(
            "ROC-AUC",
            f'{model_metrics["roc_auc"]:.4f}'
        )

    st.markdown("---")
    
    # =====================================================
    # Model Comparison
    # =====================================================

    st.subheader("📊 Model Comparison")

    st.dataframe(

        model_comparison,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")
    
    # =====================================================
    # Computational Performance
    # =====================================================

    st.subheader("⚡ Computational Performance")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Training Samples",
            model_metrics["train_samples"]
        )

        st.metric(
            "Testing Samples",
            model_metrics["test_samples"]
        )

    with c2:

        st.metric(
            "Feature Count",
            model_metrics["feature_count"]
        )

        model_size = MODEL_PATH.stat().st_size

        if model_size < 1024 * 1024:
            model_size_display = f"{model_size / 1024:.1f} KB"
        else:
            model_size_display = f"{model_size / (1024 * 1024):.2f} MB"

        st.metric(
            "Model Size",
            model_size_display
        )

    with c3:

        st.metric(
            "Prediction Time (sec)",
            model_metrics["prediction_time"]
        )

    st.subheader("Training Time")

    training_df = pd.DataFrame(

        model_metrics["training_times"].items(),

        columns=[

            "Model",

            "Training Time (sec)"

        ]

    )

    st.dataframe(

        training_df,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")
    
    # =====================================================
    # Evaluation Plots
    # =====================================================

    st.subheader("📈 Evaluation Visualizations")

    col1, col2 = st.columns(2)

    with col1:

        st.image(

            PLOTS_DIR /

            "confusion_matrix.png",

            caption="Confusion Matrix",

            use_container_width=True

        )

    with col2:

        st.image(

            PLOTS_DIR /

            "roc_curve.png",

            caption="ROC Curve",

            use_container_width=True

        )

    st.image(

        PLOTS_DIR /

        "feature_importance.png",

        caption="Feature Importance",

        use_container_width=True

    )

    st.markdown("---")

    # =====================================================
    # Executive Summary
    # =====================================================

    st.subheader("📋 Executive Summary")

    st.success(

        f"""
The selected **{model_metrics['best_model']}** model achieved the
highest overall performance during evaluation.

It obtained an Accuracy of **{model_metrics['accuracy']:.2%}**,
a Precision of **{model_metrics['precision']:.2%}**,
a Recall of **{model_metrics['recall']:.2%}**, and
an F1 Score of **{model_metrics['f1_score']:.2%}**.

Based on these metrics, this model was selected as the production model
for predicting employee attrition.
"""
    )

# ==========================================================
# Page Routing
# ==========================================================

if page == "📊 Exploratory Data Analysis":

    show_eda_page()

elif page == "🤖 Attrition Prediction":

    show_prediction_page()

elif page == "📈 Model Evaluation":

    show_model_evaluation_page()