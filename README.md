# ✈️ AirPredict-NYC

**NYC Airbnb Price Prediction & Recommender Engine with Machine Learning Ensembles and SHAP Explainability**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F00?style=for-the-badge)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=for-the-badge&logo=plotly)](https://plotly.com/)

---

## 📌 Executive Summary

Pricing short-term rental listings in competitive metropolitan markets like New York City is challenging. Under-pricing leaves significant revenue on the table, while over-pricing results in prolonged vacancy.

**AirPredict-NYC** is an end-to-end Machine Learning web application designed to predict optimal Airbnb listing prices across all five NYC boroughs. It trains and compares multiple regression architectures (Linear Regression, Random Forest, and XGBoost), provides full feature explainability using **SHAP (SHapley Additive exPlanations)** waterfall breakdowns, and features a **K-Nearest Neighbors (KNN)** spatial listing recommender plotted on an interactive Plotly map.

---

## 📊 Rigorous Model Evaluation & Results

All models were evaluated on an independent test split of the historical NYC Airbnb dataset utilizing $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
|---|:---:|:---:|:---:|
| **XGBoost Regressor (Champion)** | **0.5661** | **$33.73** | **$48.21** |
| Random Forest Regressor | 0.5644 | $33.79 | $48.31 |
| Linear Regression (Baseline) | 0.4719 | $38.49 | $53.19 |

> *Source: Exact serialized validation evaluation stored in `models/metrics.json`.*

---

## 🔍 Explainable AI with SHAP

Instead of treating the model as a black box, AirPredict-NYC generates interactive **SHAP waterfall charts** for every individual prediction:
- **Baseline Value**: The expected average listing price across NYC.
- **Positive Push Factors**: How premium zip codes (e.g. Manhattan, Brooklyn Heights), entire home/apt listings, and high review density elevate the valuation.
- **Negative Pull Factors**: Minimum night restrictions, shared room types, and outer-borough locations reducing market rate.

---

## 🗺️ Spatial KNN Recommender
In addition to price inference, the application utilizes a **K-Nearest Neighbors (KNN)** distance model in `src/recommender.py` to identify the Top-5 most comparable reference listings based on:
1. Haversine geographical distance (Latitude / Longitude)
2. Room configuration and capacity
3. Historical availability metrics

---

## 📂 Repository Layout

```text
AirPredict-NYC/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Target deployment environment runtime
├── data/
│   ├── AB_NYC_2019.csv         # Raw training dataset
│   └── reference_listings.csv  # Cleaned lookup dataset for KNN recommender
├── models/
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── recommender.joblib
│   ├── scaler.joblib
│   ├── shap_explainer.joblib
│   ├── metrics.json            # Verified evaluation metrics
│   └── feature_importance.png  # Global feature importance visualization
├── notebooks/
│   ├── Minor_Project_3_Airbnb_Price_Prediction_Ansh_Arora.ipynb
│   └── AnshArora_MinorProject3_report.docx
└── src/
    ├── features.py             # Feature engineering & ratio transforms
    ├── predict.py              # Robust inference pipeline & model loader
    ├── preprocessing.py        # Outlier handling & one-hot encoding
    ├── recommender.py          # Spatial KNN similarity logic
    └── train.py                # Automated model training pipeline
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python `3.10` or higher
- `pip` package manager

### 2. Setup & Installation

```bash
# Clone the repository
git clone https://github.com/anshere0/AirPredict-NYC.git
cd AirPredict-NYC

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Re-train the Pipeline (Optional)

```bash
python src/train.py
```

---

## 👤 Author
**Ansh Arora**  
- GitHub: [@anshere0](https://github.com/anshere0)  
- LinkedIn: [Ansh Arora](https://www.linkedin.com/in/ansh-arora0/)
