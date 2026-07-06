# Employee Attrition Analysis and Prediction

## Overview

Employee attrition is one of the most critical challenges faced by Human Resource departments. Losing experienced employees increases recruitment costs, impacts productivity, and affects business continuity.

This project is an end-to-end Enterprise HR Analytics platform that predicts employee attrition using Machine Learning while providing actionable retention insights through an interactive Streamlit dashboard.

The project follows a modular architecture where every stage of the Machine Learning pipeline is implemented independently for better maintainability, testing, and scalability.

---

# Business Objectives

The solution aims to help HR departments by:

- Predicting employee attrition risk.
- Identifying employees likely to resign.
- Understanding key drivers behind employee turnover.
- Supporting proactive retention strategies.
- Reducing hiring and onboarding costs.

---

# Key Business KPIs

- Overall Attrition Rate
- High-Risk Employee Count
- Department-wise Attrition
- Monthly Income vs Attrition
- Overtime Impact
- Job Satisfaction Impact
- Estimated Cost Savings Through Early Intervention

---

# Project Directory

```
Employee Attrition/
│
├── data/
│   └── raw/
│       └── Employee-Attrition.xlsx
│
├── requirements.txt
├── README.md
├── dataloader.py
├── EDA.py
├── featureengineering.py
├── prediction.py
├── model_evaluation.py
└── streamlitapp.py
```

---

# Project Workflow

```
Raw Dataset
      │
      ▼
dataloader.py
      │
      ▼
EDA.py
      │
      ▼
featureengineering.py
      │
      ▼
prediction.py
      │
      ▼
model_evaluation.py
      │
      ▼
Champion Model
      │
      ▼
streamlitapp.py
```

---

# Dataset Location

The dataset should be placed inside:

```
data/raw/
```

Example:

```
data/raw/Employee-Attrition.xlsx
```

---

# Dataset Schema

## Target Variable

| Column | Values |
|---------|--------|
| Attrition | Yes / No |

Target Encoding:

```
Yes → 1
No  → 0
```

---

## Numerical Features

- Age
- DailyRate
- DistanceFromHome
- HourlyRate
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- PercentSalaryHike
- TotalWorkingYears
- TrainingTimesLastYear
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

---

## Ordinal Features

- Education
- EnvironmentSatisfaction
- JobInvolvement
- JobLevel
- JobSatisfaction
- PerformanceRating
- RelationshipSatisfaction
- WorkLifeBalance

---

## Nominal Features

- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- OverTime

---

## Columns Removed

- EmployeeCount
- Over18
- StandardHours
- EmployeeNumber

---

# Machine Learning Pipeline

The project follows this sequence:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Feature Scaling
6. Categorical Encoding
7. Model Training
8. Model Evaluation
9. Champion Model Selection
10. Streamlit Deployment

---

# Models Planned

The project benchmarks multiple classification algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The best-performing model is selected based on:

- F1 Score
- ROC-AUC
- Precision
- Recall

---

# Evaluation Metrics

Each trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

The Champion Model will be selected primarily using:

- Highest F1 Score
- Highest ROC-AUC
- Target Threshold > 85%

---

# Feature Engineering

The preprocessing pipeline includes:

- Missing Value Validation
- Custom Feature Generation
- One-Hot Encoding
- Feature Scaling
- Pipeline Serialization using Joblib

---

# Project Outputs

The project generates:

- Cleaned Dataset
- Statistical Summary
- Correlation Heatmap
- Attrition Distribution Charts
- Feature Importance
- Trained Models
- Champion Model
- Serialized Preprocessing Pipeline
- Interactive Streamlit Dashboard

---

# Streamlit Dashboard Features

The dashboard allows HR users to:

- Enter employee information
- Predict attrition probability
- Display risk category
- View confidence score
- Receive HR retention recommendations

Risk Levels:

- Low Risk
- Medium Risk
- High Risk

---

# Installation

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Project Execution Order

Run the modules in the following sequence:

```bash
python dataloader.py
python EDA.py
python featureengineering.py
python prediction.py
python model_evaluation.py
streamlit run streamlitapp.py
```

---

# Future Enhancements

Potential future improvements include:

- SHAP Explainability
- Hyperparameter Optimization
- Automated Model Retraining
- MLflow Experiment Tracking
- Docker Containerization
- CI/CD Pipeline Integration
- Cloud Deployment (Azure/AWS/GCP)
- Real-time HR Data Integration

---

# Author

Employee Attrition Analysis and Prediction

Enterprise HR Analytics Platform

Built using Python, Scikit-learn, Streamlit, and Machine Learning.