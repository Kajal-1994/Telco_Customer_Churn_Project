from re import M
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from train import Model_Path

Model_Path = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\model.joblib"

st.set_page_config(page_title = "Customer Churn Prediction", layout = "centered")
st.title("Customer Churn Prediction App")   

if not Path(Model_Path).exists():
    st.error(f"Model file not found at :\n{Model_Path}\n\nPlease train the model first.")
else:
    model, preprocessor = joblib.load(Model_Path)
    st.write("Enter customer details:")

#Inputs matching Telco dataset columns (except customerID, Churn)

    gender = st.selectbox("Gender",["Female","Male"])
    SeniorCitizen = st.selectbox("Senior Citizen (0= No, 1 = Yes)",[0,1])
    Partner = st.selectbox("Partner",["Yes","No"])
    Dependents = st.selectbox("Dependents",["Yes","No"])
    tenure = st.number_input("Tenure (in months)",min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service",["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=2000.0)

    if st.button("Predict Churn"):

        # Build input row with same columns as X during training
        input_dict = {
           "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,  
        }
        
        input_df = pd.DataFrame([input_dict])

        # Apply same preprocessing
        
        Xt = preprocessor.transform(input_df)

        # Predict
        prob = model.predict_proba(Xt)[:, 1][0]
        pred = model.predict(Xt)[0]

        st.subheader("Result")
        st.write(f"**Churn Probability:** {prob:.3f}")
        st.write("**Prediction:**", "Churn" if pred == 1 else "Not Churn")