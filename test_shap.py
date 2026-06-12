import traceback
import joblib
import pandas as pd
try:
    print("Loading preprocessor...")
    preprocessor = joblib.load('models/preprocessor.joblib')
    print("Loading explainer...")
    explainer = joblib.load('models/shap_explainer.joblib')
    print("Loaded explainer successfully.")
except Exception as e:
    traceback.print_exc()
