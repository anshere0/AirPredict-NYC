import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import joblib

# Import custom modules
from preprocessing import clean_data, transform_data
from features import engineer_features

class AirbnbPredictor:
    def __init__(self, models_dir="models"):
        """
        Loads pre-trained models and preprocessing encoders from the models directory.
        """
        self.models_dir = models_dir
        self.encoder_path = os.path.join(models_dir, "preprocessor.joblib")
        
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Preprocessor not found at {self.encoder_path}. Run training first.")
            
        # Load preprocessor
        self.encoder = joblib.load(self.encoder_path)
        
        # Load models
        self.models = {}
        model_files = {
            "Linear Regression": "linear_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "XGBoost": "xgboost.joblib"
        }
        
        for name, filename in model_files.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                except Exception as e:
                    print(f"Skipping model '{name}' due to error: {e}")
                
        if not self.models:
            raise FileNotFoundError("No models found. Run training first.")

    def predict(self, listing_data, model_name="XGBoost"):
        """
        Predicts the price of a listing given its raw dictionary data.
        listing_data: dict of listing attributes.
        model_name: Name of the model to use ("Linear Regression", "Random Forest", or "XGBoost").
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        
        # 1. Convert single dictionary row into DataFrame
        df_input = pd.DataFrame([listing_data])
        
        # 2. Clean data (is_training=False ensures we do not filter outliers for predictions)
        df_clean = clean_data(df_input, is_training=False)
        
        # 3. Feature engineering
        df_engineered = engineer_features(df_clean)
        
        # 4. Apply categorical encoding
        df_processed = transform_data(df_engineered, self.encoder)
        
        # 5. Extract feature names expected by the model
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
        else:
            # Fallback for older models or generic objects
            expected_features = df_processed.columns
            
        # Reindex dataframe to match the training feature order and fill missing with 0
        df_final = df_processed.reindex(columns=expected_features, fill_value=0)
        
        # 6. Make prediction
        prediction = model.predict(df_final)
        
        # Target price cannot be negative, clip to a minimum of 0
        return max(0.0, float(prediction[0]))
