📊 Customer Churn Analysis & Prediction : End-to-End Machine Learning Project

This project implements a complete Customer Churn Prediction System using Machine Learning.
It processes telecom customer data, performs detailed Exploratory Data Analysis (EDA), builds preprocessing pipelines, trains ML models, and deploys a Streamlit prediction app for real-time churn prediction.

The system predicts whether a customer is likely to churn (leave the company) based on demographics, service usage patterns, billing information, and contract details.

🔥 Project Pipeline Overview:
Dataset → Cleaning → Encoding → Scaling → Feature Engineering
                              ↓
                         ML Training
                    Logistic Regression
                    Random Forest (Final)
                              ↓
                   Model Saving (Joblib)
                              ↓
                Streamlit Web App Deployment

🧭 Environment Setup:
1) Create & activate environment
python -m venv venv
venv\Scripts\activate        # Windows

2) Install dependencies
pip install -r requirements.txt

📂 Folder Structure
Telco_Customer_Churn_Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── src/
│   ├── preprocess.py         → Data cleaning & preprocessing pipeline
│   ├── train.py              → Model training script
│   └── predict_app.py        → Streamlit web app for predictions
│
├── models/
│   ├── model.joblib          → Final Random Forest model + preprocessor
│   └── preprocessor.joblib   → Saved preprocessing pipeline
│
├── requirements.txt
└── README.md

🧹 How Data is Cleaned:

The preprocessing script performs:

Dropping customerID

Converting TotalCharges → numeric

Safe string cleaning using:

df[col] = df[col].astype("string").str.strip()


Handling missing values:

Numeric → median

Categorical → most frequent

Encoding categorical variables using OneHotEncoder

Scaling numeric features using StandardScaler

Building a unified preprocessing pipeline (ColumnTransformer)

📊 Exploratory Data Analysis (EDA):

The EDA reveals:

Churn rate ≈ 26% (imbalanced dataset)

High risk factors:

Month-to-month contract

High monthly charges

Fiber Optic internet

Low tenure

No online security / tech support

Longer tenure → lower churn

Two-year contracts have lowest churn

🤖 Model Training:

Run the training script:

python src/train.py


This performs:

1.Preprocessing

2.Train-test split

Training two models:

1.Logistic Regression (baseline)

2.Random Forest (final model)

* Evaluation using:

Classification Report

ROC-AUC

Confusion Matrix

* Saving:

model.joblib

preprocessor.joblib

🏆 Model Performance
✔ Logistic Regression

Good baseline

ROC-AUC ≈ 0.83

✔ Random Forest (Final Model)

Best performance

ROC-AUC ≈ 0.90

Highest recall for churn class

Selected as deployment model

💾 Artifacts Generated

After training:

models/
 ├── model.joblib
 └── preprocessor.joblib

🌐 Streamlit Web App

Run the app:

streamlit run src/predict_app.py


* The UI allows entering:

Gender

Tenure

Contract type

Monthly charges

Internet service

Payment method

Security features

And more…

The app predicts:

Churn Probability (0–1)

Final Prediction (Churn / Not Churn)

Accessible at:

http://localhost:8501

🧠 System Behavior (Internal Logic):

1.User inputs customer details

2.Preprocessor transforms input using saved pipeline

3.Random Forest model predicts churn

4.Probability & label returned

5.Streamlit displays final result

📌 Example Query:

Input:

Tenure: 2 months
Contract: Month-to-month
Internet: Fiber Optic
MonthlyCharges: 85


Output:

Churn Probability: 0.78
Prediction: Churn

📦 Technologies Used:
Machine Learning

Scikit-Learn

Logistic Regression

Random Forest

Joblib

Data Processing

Pandas

NumPy

ColumnTransformer

OneHotEncoder

StandardScaler

Visualization

Matplotlib

Seaborn

Deployment

Streamlit

📈 Future Enhancements:

1.Add SHAP explainability

2.Use SMOTE to handle imbalance

3.Hyperparameter tuning (RandomizedSearchCV / GridSearchCV)

4.Use XGBoost or LightGBM

5.Deploy on Streamlit Cloud

6.Add database connection for real-time customer data


