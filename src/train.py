import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
from preprocess import load_data,basic_cleaning,get_feature_target,build_preprocessor,save_preprocessor
from pathlib import Path

Data_Path = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\data\\Telco-Customer-Churn.csv" # Where CSV dataset is located.
Model_Path = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\models\\model.joblib"                  # Where you want to save the final model file.
PREPROC_PATH = "C:\\Users\\STPIM\\Desktop\\Telco_Customer_Churn_Prediction\\models\\preprocessor.joblib"         # Where you want to save the fitted preprocessor.


# Evaluate model performance
# A function to evaluate a trained model on test data.
def evaluate_model(model,X_test,y_test):
    
    # Uses the model to predict class labels(0 or 1)for the test data.
    preds = model.predict(X_test)
    
    # If the model supports predict_proba,get teh probability of class 1(churn).
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model,"predict_proba") else None

    
    print("====Classification Report====" )
    
    # Shows precision,recall,F1-score and support for both classes.
    print(classification_report(y_test, preds,target_names=['No Churn','Churn']))

    if probs is not None:
        print("ROC AUC:", roc_auc_score(y_test, probs))
    
    
    # compute the confusion matrix(TP,TN,FP,FN)
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)


# Main function that runs the whole pipeline
def main():
    
    # 1.Load and clean data
    
    print("Loading dataset:",Data_Path)
    df = load_data(Data_Path)  # Loads the CSV into dataframe.
    df = basic_cleaning(df)    # Drop ID,convert totalcharges,strip stings,etc.

    
    #2. Get features and target
    # X = all columns except Churn
    # y = Churn converted from Yes/No to 1/0
    
    X,y = get_feature_target(df,target_col='Churn')
    if y is None:
        raise ValueError("Target variable 'Churn' not found in the dataset.")
    
    
    # Build and fit preprocessor
    # Create a ColumnsTranformer with pipelines for each.
    preprocessor,numeric_cols,cat_cols = build_preprocessor(X)
    
    # Fits the preprocessor on all features X
    fitted_preproc = preprocessor.fit(X)
    
    
    # Transform full dataset to numeric matrix for modeling
    # Converts feature DataFrame into a numeric NumPy array suitable for ML models.
    Xt = fitted_preproc.transform(X)

    # ensure models folder exists and save preprocessor
    
    Path(PREPROC_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_preprocessor(fitted_preproc,(PREPROC_PATH))
    print("Saved preprocessor to",PREPROC_PATH)
    
    
    # train test split (use startify= y ensures the churn vs non-churn ratio).
    # Splits data into 80% train,20% test.
    X_train,X_test,y_train,y_test = train_test_split(Xt,y,test_size=0.2,random_state=42,stratify=y)
    
    
    # Logistic Regression Model

    print("Training Logistics Regression Model...")
    
    # Create a Logistics regression model allowing up to 1000 iterations to converge.
    lr = LogisticRegression(max_iter=1000)
    
    # Trains the model on training data
    lr.fit(X_train,y_train)
   
    print("Evaluating Logistics Regression Model...")
    
    # Calls the function to print classification reports,ROC_AUC and confusion matrix.
    evaluate_model(lr,X_test,y_test)
    
    
    # Random Forest Classifier

    print("\nTraining Random Forest...")
    
    # n_estimators=200 → 200 decision trees.
    # random_state=42 → reproducible results.
    # n_jobs=-1 → use all CPU cores for faster training.
    rf = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    
    # Trains Random Forest on training data.
    rf.fit(X_train,y_train) 
    
    print("Evaluating Random Forest....")
    evaluate_model(rf,X_test,y_test)
    
    # Save final pipeline (model + preprocessor)
    
    final_model = rf
    Path(Model_Path).parent.mkdir(parents=True, exist_ok=True)

    # save a tuple (model, preprocessor) so inference can load both
    joblib.dump((final_model,fitted_preproc),(Model_Path))
    print("Saved pipeline (model + preprocessor) to:",Model_Path)


if __name__ == "__main__":
    main()

