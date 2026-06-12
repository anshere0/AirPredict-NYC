import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Import our custom modules
from preprocessing import load_data, clean_data, fit_preprocessor, transform_data
from features import engineer_features
from recommender import AirbnbRecommender

def main():
    print("Starting ML Pipeline Training...")
    
    # Define paths
    raw_data_path = "data/AB_NYC_2019.csv"
    models_dir = "models"
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading raw data...")
    df_raw = load_data(raw_data_path)
    print(f"Raw data shape: {df_raw.shape}")
    
    # 2. Clean Data (training mode applies outlier filtering)
    print("Cleaning data and filtering outliers...")
    df_clean = clean_data(df_raw, is_training=True)
    print(f"Cleaned data shape (after outlier removal): {df_clean.shape}")
    
    # Create and save a reference copy of raw data for the recommendation system lookup
    # It must map exactly to the rows that survived cleaning/outlier removal
    df_reference = df_raw.loc[df_clean.index].copy()
    
    # 3. Feature Engineering
    print("Engineering features...")
    df_engineered = engineer_features(df_clean)
    
    # 4. Encoding Categorical Variables
    print("Fitting and applying categorical encoder...")
    categorical_cols = ["neighbourhood_group", "room_type"]
    encoder = fit_preprocessor(df_engineered, categorical_cols)
    df_processed = transform_data(df_engineered, encoder, categorical_cols)
    print(f"Processed feature matrix shape: {df_processed.shape}")
    
    # Save the encoder for inference pipeline
    joblib.dump(encoder, os.path.join(models_dir, "preprocessor.joblib"))
    
    # 5. Split into Features and Target
    X = df_processed.drop(columns=["price"])
    y = df_processed["price"]
    
    # 6. Train-Test Split (80% train, 20% test as per original notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 7. Model Training & Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    
    metrics_summary = {}
    best_model_name = None
    best_r2 = -float("inf")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{name} Metrics -> R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        
        metrics_summary[name] = {
            "R2": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        }
        
        # Track the best model based on R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            
        # Serialize model
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        
    print(f"Best Model: {best_model_name} with R2: {best_r2:.4f}")
    
    # Save metrics JSON to be loaded in Streamlit instantly
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # 8. Train KNN Listing Recommender
    print("Training KNN listing recommender...")
    # Map back original name and description data for recommendation lookup
    recommender = AirbnbRecommender()
    # Fit KNN recommender on processed listings and save reference dataset
    recommender.train_recommender(df_processed, df_reference)
    
    # 9. Generate Feature Importance Plot for the best tree-based model
    print("Generating Feature Importance Plot...")
    best_tree_model = models.get("XGBoost", models.get("Random Forest"))
    if best_tree_model:
        importances = best_tree_model.feature_importances_
    features_list = X.columns
    
    imp_df = pd.DataFrame({"Feature": features_list, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df, hue="Feature", legend=False, palette="viridis")
    plt.title(f"{best_model_name} - Top 10 Feature Importances", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "feature_importance.png"))
    plt.close()
    
    # 10. Fit and save SHAP explainer for the best model
    print("Fitting SHAP Explainer...")
    try:
        import shap
        # Standard Tree Explainer
        explainer = shap.TreeExplainer(best_tree_model)
        joblib.dump(explainer, os.path.join(models_dir, "shap_explainer.joblib"))
        print("SHAP Explainer saved successfully.")
    except Exception as e:
        print(f"SHAP Explainer training skipped/failed: {e}")
        
    print("ML Pipeline Training Completed Successfully!")

if __name__ == "__main__":
    main()
