import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the Telco churn dataset."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - drop customerID if present
    - convert TotalCharges to numeric (coerce errors)
    - convert object-like columns to pandas 'string' dtype then strip whitespace
    """
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric and leave NaNs if any
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Find columns that are object or string-like and safely strip whitespace
    candidate_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in candidate_cols:
        # Convert to pandas string dtype to preserve NA and support .str accessor
        df[c] = df[c].astype("string").str.strip()

        # If column becomes empty strings and you prefer NA, replace '' with <NA>
       #df[c].replace("", pd.NA, inplace=True)
        df[c] = df[c].replace("", pd.NA)
    return df

def get_feature_target(df: pd.DataFrame, target_col: str = 'Churn'):
    """Return X (features) and y (target as 0/1)"""
    if target_col in df.columns:
        y = df[target_col].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
        y = None
    return X, y

def build_preprocessor(X: pd.DataFrame, numeric_override=None):
    """Auto-detect numeric & categorical features and return a ColumnTransformer"""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    if numeric_override is not None:
        numeric_features = numeric_override

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, cat_features)
    ], remainder='drop')

    return preprocessor, numeric_features, cat_features

def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(preprocessor, path)