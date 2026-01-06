
print("preprocess.py is being executed")
import pandas as pd
from typing import Dict, Any

def preprocess_data(
    data: pd.DataFrame,
    drop_first: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess the input DataFrame and return a dict containing
    X (features), y (target), and feature_columns.

    This function does not modify the original `data` argument.
    """

    df = data.copy()

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Optional: print unique values for debugging/inspection
    if verbose:
        for column in df.columns:
            unique_values = df[column].unique()
            print(f"Unique values in '{column}':")
            print(unique_values)
            print()

    # Encode education levels (safe mapping)
    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "Doctorate": 4}
    if 'education_of_employee' in df.columns:
        df['education_of_employee'] = df['education_of_employee'].map(edu_map)
        df['education_of_employee'] = pd.to_numeric(
            df['education_of_employee'], errors='coerce'
        ).fillna(0).astype(int)

    # Convert Y/N columns to 1/0 safely
    yn_cols = ['has_job_experience', 'requires_job_training', 'full_time_position']
    for col in yn_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Y': 1, 'N': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Map case_status to binary (required)
    if 'case_status' not in df.columns:
        raise KeyError("'case_status' column is required in the input DataFrame")
    df['case_status'] = df['case_status'].replace({'Certified': 1, 'Denied': 0})
    df['case_status'] = pd.to_numeric(df['case_status'], errors='coerce')
    if df['case_status'].isnull().any():
        raise ValueError("'case_status' contains values that cannot be mapped to 0/1")
    df['case_status'] = df['case_status'].astype(int)

    # Specify independent and dependent variables
    # Drop case_status (target) and case_id (identifier, not a feature)
    cols_to_drop = ['case_status']
    if 'case_id' in df.columns:
        cols_to_drop.append('case_id')
    X = df.drop(cols_to_drop, axis=1)
    y = df['case_status']

    # One-hot encode categorical columns if they exist
    cat_cols = [c for c in ['continent', 'region_of_employment', 'unit_of_wage'] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dtype=int, drop_first=drop_first)

    if verbose:
        X.info()

    feature_columns = X.columns.tolist()

    return {
        "X": X,
        "y": y,
        "feature_columns": feature_columns,
    }