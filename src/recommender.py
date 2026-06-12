import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AirbnbRecommender:
    def __init__(self, model_path=None, scaler_path=None, reference_path=None):
        """
        Initializes the recommender by loading saved artifacts if paths are provided.
        """
        self.model = None
        self.scaler = None
        self.reference_df = None
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if reference_path and os.path.exists(reference_path):
            self.reference_df = pd.read_csv(reference_path)
            
    def get_features(self, df):
        """
        Selects and returns the features used for KNN similarity computation.
        We use location (latitude, longitude), room type, and price tier.
        """
        feature_cols = [
            "latitude", "longitude", "room_type_Private room", 
            "room_type_Shared room", "price"
        ]
        # Ensure all columns exist, fill missing room type columns if needed
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[feature_cols]

    def train_recommender(self, df_processed, df_reference, models_dir="models", data_dir="data"):
        """
        Trains the KNN model and StandardScaler on clean processed listings.
        Saves artifacts (scaler, knn model, reference metadata) for prediction time.
        """
        # Get recommendation features
        features = self.get_features(df_processed)
        
        # Fit scaler
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit NearestNeighbors
        self.model = NearestNeighbors(n_neighbors=6, metric="euclidean")
        self.model.fit(scaled_features)
        
        # Save reference dataframe (keep name, neighbourhood, etc. for display)
        self.reference_df = df_reference.copy()
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save artifacts
        joblib.dump(self.model, os.path.join(models_dir, "recommender.joblib"))
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.joblib"))
        self.reference_df.to_csv(os.path.join(data_dir, "reference_listings.csv"), index=False)
        print("Recommender system trained and artifacts saved successfully.")
        
    def recommend(self, query_features, n_recommendations=5, exclude_id=None):
        """
        Returns recommendations for a given processed listing.
        query_features: Dict or single-row DataFrame representing the query listing.
        n_recommendations: Number of listings to recommend.
        exclude_id: Optional listing index (integer row index in original reference df) to exclude.
        """
        if self.model is None or self.scaler is None or self.reference_df is None:
            raise ValueError("Recommender artifacts not loaded or trained.")
            
        # Convert dictionary to DataFrame if needed
        if isinstance(query_features, dict):
            query_df = pd.DataFrame([query_features])
        else:
            query_df = query_features.copy()
            
        # Select and scale search features
        features = self.get_features(query_df)
        scaled_features = self.scaler.transform(features)
        
        # Query nearest neighbors
        # Ask for n_recommendations + 1 in case the query listing itself is returned
        distances, indices = self.model.kneighbors(scaled_features, n_neighbors=n_recommendations + 1)
        
        indices = indices[0]
        distances = distances[0]
        
        recommendations = []
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            # If the neighbor is the listing itself (distance is virtually 0 or same index), skip it
            if exclude_id is not None and idx == exclude_id:
                continue
            
            rec_listing = self.reference_df.iloc[idx].to_dict()
            rec_listing["similarity_distance"] = float(dist)
            rec_listing["similarity_percentage"] = max(0.0, round(100 * (1 - dist / (1 + dist)), 1))
            recommendations.append(rec_listing)
            
            if len(recommendations) == n_recommendations:
                break
                
        return pd.DataFrame(recommendations)
