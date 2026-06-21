# Employee Attrition Analysis and Prediction

## Project Overview

Employee Attrition Analysis and Prediction is a Machine Learning and HR Analytics project designed to identify employees who are at risk of leaving the organization. The project analyzes employee-related factors such as job satisfaction, overtime, salary, work-life balance, and department information to uncover key drivers of attrition.

The project also includes predictive modeling and an interactive Streamlit dashboard for HR decision-making and workforce management.

---

# Business Problem

Employee attrition can lead to:

- Increased recruitment costs
- Reduced productivity
- Team disruptions
- Loss of experienced employees

This project helps HR teams:

- Identify at-risk employees
- Understand attrition patterns
- Improve employee retention strategies
- Make data-driven workforce decisions

---

# Project Objectives

- Analyze employee attrition patterns
- Perform Exploratory Data Analysis (EDA)
- Build Machine Learning models to predict attrition
- Evaluate model performance using classification metrics
- Identify high-risk employees
- Create an interactive Streamlit dashboard

---

# Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Plotly
- Streamlit
- Joblib

---

# Project Structure

```bash
Employee Attrition/
│
├── data/
│   ├── processed_employee_attrition.csv
│   └── featured_employee_attrition.csv
│
├── models/
│   └── best_attrition_model.pkl
│
├── outputs/
│   ├── charts/
│   ├── metrics/
│   └── employee_attrition_predictions.csv
│
├── src/
│   ├── dataloader.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction.py
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Project Workflow

## 1. Data Loading and Preprocessing
- Loaded Excel dataset
- Cleaned missing and duplicate data
- Standardized column names
- Removed unnecessary columns
- Encoded target variable

## 2. Exploratory Data Analysis (EDA)
Performed analysis on:
- Attrition distribution
- Department-wise attrition
- Overtime impact
- Job satisfaction trends
- Salary distribution
- Correlation heatmaps

## 3. Feature Engineering
Created additional features such as:
- Age Groups
- Tenure Groups
- Income Bands
- Total Satisfaction Score

## 4. Machine Learning Models
Implemented:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## 5. Model Evaluation
Evaluated models using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## 6. Prediction System
Generated:
- Attrition probability
- Employee risk levels
- High-risk employee identification

## 7. Streamlit Dashboard
Built an interactive dashboard for:
- HR analytics
- Visualization
- Employee risk analysis
- Business insights

---

# Key Insights

- Employees working overtime showed higher attrition rates
- Lower job satisfaction increased employee turnover
- Certain departments experienced higher attrition
- Employees with lower work-life balance were more likely to leave

---

# Machine Learning Workflow

```text
Raw Dataset
    ↓
Data Preprocessing
    ↓
EDA
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Prediction System
    ↓
Streamlit Dashboard
```

---

# How to Run the Project

## Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Step 2: Run Data Preprocessing

```bash
python src/dataloader.py
```

---

## Step 3: Run EDA

```bash
python src/eda.py
```

---

## Step 4: Run Feature Engineering

```bash
python src/feature_engineering.py
```

---

## Step 5: Train Models

```bash
python src/model_training.py
```

---

## Step 6: Evaluate Model

```bash
python src/model_evaluation.py
```

---

## Step 7: Generate Predictions

```bash
python src/prediction.py
```

---

## Step 8: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

# Dashboard Features

- KPI Metrics
- Attrition Analysis
- Department Analysis
- Overtime Analysis
- Risk Distribution
- High-Risk Employee Table
- Interactive Visualizations

---

# Future Improvements

- Hyperparameter tuning
- SHAP feature importance analysis
- Real-time employee prediction form
- Cloud deployment
- Database integration
- Advanced HR analytics

---

# Conclusion

This project demonstrates the complete end-to-end Machine Learning workflow for HR Analytics, including preprocessing, visualization, predictive modeling, evaluation, and deployment using Streamlit.

The solution helps organizations proactively identify employee attrition risks and improve workforce retention strategies.

---

# Author

Kumaaran J