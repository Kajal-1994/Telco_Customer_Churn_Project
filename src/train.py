import os
# from typing import final
import joblib
import numpy as np
from numpy._core import numeric
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
from preprocess import load_data,basic_cleaning,get_feature_target,build_preprocessor,save_preprocessor
import preprocess
from pathlib import Path

Data_Path = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\data\\Telco-Customer-Churn.csv"
Model_Path = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\model.joblib"
PREPROC_PATH = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\preprocessor.joblib"

# Evaluate model performance

def evaluate_model(model,X_test,y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model,"predict_proba") else None

    print("====Classification Report====" )
    print(classification_report(y_test, preds,target_names=['No Churn','Churn']))

    if probs is not None:
        print("ROC AUC:", roc_auc_score(y_test, probs))
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)


def main():
    
    #1.Load and clean data
    
    print("Loading dataset:",Data_Path)
    df = load_data(Data_Path)
    df = basic_cleaning(df)

    #2. Get features and target
    
    X,y = get_feature_target(df,target_col='Churn')
    if y is None:
        raise ValueError("Target variable 'Churn' not found in the dataset.")
    
    # Build and fit preprocessor

    preprocessor,numeric_cols,cat_cols = build_preprocessor(X)
    fitted_preproc = preprocessor.fit(X)
    
    # Transform full dataset to numeric matrix for modeling
    
    Xt = fitted_preproc.transform(X)

    # ensure models folder exists and save preprocessor
    
    Path(PREPROC_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_preprocessor(fitted_preproc,(PREPROC_PATH))
    print("Saved preprocessor to",PREPROC_PATH)
    
    # train test split (use startify to preserve class ratio)

    X_train,X_test,y_train,y_test = train_test_split(Xt,y,test_size=0.2,random_state=42,stratify=y)
    
    
    # Logistic Regression Model

    print("Training Logistics Regression Model...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train,y_train)
    print("Evaluating Logistics Regression Model...")
    evaluate_model(lr,X_test,y_test)
    
    
    # Random Forest Classifier

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    rf.fit(X_train,y_train) 
    print("Evaluating Random Forest....")
    evaluate_model(rf,X_test,y_test)
    
    # Save final pipeline (model + preprocessor)
    
    final_model = rf
    Path("Model_Path").mkdir(parents=True, exist_ok=True)

    # save a tuple (model, preprocessor) so inference can load both
    joblib.dump((final_model,fitted_preproc),str(Model_Path))
    print("Saved pipeline (model + preprocessor) to:",Model_Path)


if __name__ == "__main__":
    main()

