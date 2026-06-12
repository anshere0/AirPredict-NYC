import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering to the Airbnb dataset:
    - availability_ratio: Ratio of listing availability in a year.
    - reviews_density: Number of reviews normalized by minimum nights.
    - high_availability: Binary flag for properties available > 200 days/year.
    """
    df_feat = df.copy()
    
    # 1. availability_ratio = availability_365 / 365.0
    if "availability_365" in df_feat.columns:
        df_feat["availability_ratio"] = df_feat["availability_365"] / 365.0
        
    # 2. reviews_density = number_of_reviews / minimum_nights (clipping at 1 to prevent division by zero)
    if "number_of_reviews" in df_feat.columns and "minimum_nights" in df_feat.columns:
        # Prevent division by zero
        min_nights_safe = df_feat["minimum_nights"].clip(lower=1)
        df_feat["reviews_density"] = df_feat["number_of_reviews"] / min_nights_safe
        
    # 3. high_availability = 1 if availability_365 > 200 else 0
    if "availability_365" in df_feat.columns:
        df_feat["high_availability"] = (df_feat["availability_365"] > 200).astype(int)
        
    return df_feat
