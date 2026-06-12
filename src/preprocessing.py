import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def load_data(filepath):
    """
    Loads raw CSV data from the given filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df, is_training=True):
    """
    Cleans the dataset by:
    - Handling missing values (imputing reviews_per_month with 0).
    - Dropping unused columns (id, host_id, name, host_name, last_review).
    - Dropping the high-cardinality neighbourhood column.
    - Outlier filtering (price <= 355 and minimum_nights <= 30) - applied if is_training=True.
    """
    # Create a copy to prevent SettingWithCopyWarning
    df_clean = df.copy()
    
    # 1. Drop unused text and ID columns
    unused_cols = ["id", "host_id", "name", "host_name", "last_review"]
    cols_to_drop = [col for col in unused_cols if col in df_clean.columns]
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        
    # 2. Impute missing reviews_per_month with 0
    if "reviews_per_month" in df_clean.columns:
        df_clean["reviews_per_month"] = df_clean["reviews_per_month"].fillna(0.0)
        
    # 3. Outlier treatment (quantiles computed as in original notebook: price <= 355 and minimum_nights <= 30)
    if is_training:
        if "price" in df_clean.columns:
            # 95th percentile is 355.0 in the original dataset
            df_clean = df_clean[df_clean["price"] <= 355.0]
        if "minimum_nights" in df_clean.columns:
            # 95th percentile is 30.0 in the original dataset
            df_clean = df_clean[df_clean["minimum_nights"] <= 30.0]
            
    # 4. Drop high-cardinality neighbourhood column
    if "neighbourhood" in df_clean.columns:
        df_clean.drop(columns=["neighbourhood"], inplace=True)
        
    return df_clean

def fit_preprocessor(df, categorical_cols=["neighbourhood_group", "room_type"]):
    """
    Fits and returns a OneHotEncoder on the categorical columns.
    We drop the first category of each variable (same as pd.get_dummies(..., drop_first=True))
    to avoid multi-collinearity in Linear Regression.
    """
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[categorical_cols])
    return encoder

def transform_data(df, encoder, categorical_cols=["neighbourhood_group", "room_type"]):
    """
    Transforms the data using the fitted encoder.
    Returns the final feature dataframe (excluding targets like 'price') and target vector (if present).
    """
    df_encoded = df.copy()
    
    # Get encoded features
    encoded_arr = encoder.transform(df_encoded[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    # Create DataFrame from encoded features
    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_feature_names, index=df_encoded.index)
    
    # Concatenate numerical features and encoded categorical features
    df_numerical = df_encoded.drop(columns=categorical_cols)
    df_final = pd.concat([df_numerical, encoded_df], axis=1)
    
    return df_final
