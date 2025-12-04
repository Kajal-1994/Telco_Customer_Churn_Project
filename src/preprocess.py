import os
import joblib           # Used to save and load python objects        
import pandas as pd     # Pandas is used for handling tables(Dataframes)
import numpy as np      # Numpy is used for numerical operations

# OneHotEncoder-> Convert categorical text columns into number
# StandardScaler-> Scales numeric columns(like MonthlyCharges) 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Apply different tranformations to different columns in one object
from sklearn.compose import ColumnTransformer

# Used to chain multiple steps together
from sklearn.pipeline import Pipeline

# Fill missing values in columns(e.g.,with median or most frequent value)
from sklearn.impute import SimpleImputer


# Uses pandas to read the CSV file into a dataframe.
# Returns loaded dataframe

csv_path = r"C:\Users\STPIM\Desktop\Telco_Customer_Churn_Prediction\data\Telco-Customer-Churn.csv"

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the Telco churn dataset."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)         
    return df                          


# Takes dataframe as input and returns a cleaned Dataframe

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:  
    """
    Basic cleaning:
    - drop customerID if present
    - convert TotalCharges to numeric (coerce errors)
    - convert object-like columns to pandas 'string' dtype then strip whitespace
    """
    df = df.copy()                     # Creates a copy of the dataframe so the original is not changed directly.

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric and leave NaNs if any
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Find columns that are object or string-like and safely strip whitespace
    candidate_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in candidate_cols:
        
        # Converts that column to Pandas string dtype..str.strip() removes spaces at the begining and endind of the text
        df[c] = df[c].astype("string").str.strip()

       # If column becomes empty strings and you prefer NA, replace '' with <NA>
       # df[c].replace("", pd.NA, inplace=True)
        df[c] = df[c].replace("", pd.NA)
    return df


# A function to split the dataset into feature (X) and target(y).

def get_feature_target(df: pd.DataFrame, target_col: str = 'Churn'):
    """Return X (features) and y (target as 0/1)"""
    if target_col in df.columns:
        y = df[target_col].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
        y = None
    return X, y                # Returns both feature matrix and target vector.


# A function to create the preprocessing pipeline for the features.

def build_preprocessor(X: pd.DataFrame, numeric_override=None):
    """Auto-detect numeric & categorical features and return a ColumnTransformer"""
    
    # Select a columns that are numeric(int/float) and convert it into a list
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Select a columns that are categorical/boolean and converts into a list.
    cat_features = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    # If it is given then replace the detected numeric columns with numeric features.
    # This allows manual control if auto-detection isn't correct
    if numeric_override is not None:
        numeric_features = numeric_override

    
    # Builds a pipeline for numeric columns.
    # SimpleImputer- Replaces missing numeric values with the median of that column.
    # StandardScaler- Scales numeric features to have mean 0 and standard deviation 1
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    
    # Builds a pipeline for categorical columns
    # SimpleImputer- Fills the missing values using the most common value of that column.
    # OneHotEncoder- Converts categories into one-hot vector.
    # handle_unknown='ignore' → if a new category appears at prediction time, ignore instead of error.
    # sparse_output=False → returns a dense NumPy array instead of a sparse matrix.
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    
    # Creates a ColumnTransformer that:
    # Applies num_pipe to numeric_features
    # Applies cat_pipe to cat_features
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, cat_features)
    ], remainder='drop')              # remainder='drop' → any other columns (not specified) are dropped.

    
    # Returns full preprocessor object,List of numeric and categorical columns.
    return preprocessor, numeric_features, cat_features


# Function to save the fitted preprocessor.
def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)  # Ensures the folder exists before saving.
    joblib.dump(preprocessor, path)                           # Saves the preprocessor object to the given path