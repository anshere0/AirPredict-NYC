# 🏠 NYC Airbnb Price Predictor & Recommender

A robust, production-ready Machine Learning application built to predict Airbnb listing prices in New York City and recommend similar listings based on geographical and feature similarity. The project features an interactive Streamlit dashboard, advanced model explainability (SHAP), and a modular architecture.

---

## ✨ Features

- **🔮 Price Prediction Engine**: Utilizes an ensemble of models (Linear Regression, Random Forest, XGBoost) to predict listing prices based on 10+ features including location, room type, minimum nights, and host metrics.
- **📊 Interactive SHAP Explainability**: Features a high-quality, interactive Plotly waterfall chart that breaks down exactly how each individual feature contributes to the final predicted price.
- **🧭 Similar Listing Recommender**: A K-Nearest Neighbors (KNN) based recommendation system that surfaces the Top 5 most similar NYC listings and plots their geographical distribution on an interactive map.
- **📈 Model Comparison Dashboard**: View comparative metrics ($R^2$, MAE, RMSE) and global feature importance plots to understand overall model performance.
- **🛡️ Deployment-Safe Architecture**: Gracefully handles missing heavy dependencies (like XGBoost and SHAP) in constrained environments by falling back to core features without crashing.

---

## 🏗️ Project Architecture

The codebase is refactored into a clean, modular structure, fully separating the training pipeline from the inference engine:

```text
airbnb price prediction/
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Deployment dependencies (with exact versions)
├── runtime.txt             # Python version specification for cloud deployment
├── .gitignore              # Deployment filters
├── data/
│   ├── AB_NYC_2019.csv         # Raw training dataset
│   └── reference_listings.csv  # Cleaned lookup dataset for the KNN recommender
├── models/                 # Serialized weights (.joblib), metrics, and plots
└── src/                    # Core Machine Learning pipeline modules
    ├── features.py         # Feature engineering (ratios, densities)
    ├── predict.py          # Robust inference pipeline and model loader
    ├── preprocessing.py    # Data cleaning and categorical encoding
    ├── recommender.py      # KNN similarity and distance logic
    └── train.py            # Automated training pipeline script
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Training the Models (First Time Setup)
Before running the dashboard, you need to build the model weights and pipelines. You can do this automatically from the terminal:

```bash
python src/train.py
```
*This will process the data, train Linear Regression, Random Forest, and XGBoost models, fit the SHAP explainer, train the KNN recommender, and save everything into the `models/` directory.*

### 3. Launching the App
Start the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project uses entirely relative paths (`os.path`) and robust dependency loading, making it 100% plug-and-play for cloud platforms like **Streamlit Cloud**, **Render**, and **Railway**. 

1. Push your repository to GitHub.
2. Link the repository to your chosen platform.
3. The platform will automatically detect `requirements.txt` and `runtime.txt`, install dependencies, and run `app.py`.

*Note: If a cloud environment fails to compile heavy C++ extensions like XGBoost, the app will safely ignore it and seamlessly fall back to Random Forest.*

---

## 🔬 Technologies Used

- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly (Interactive Waterfall), Matplotlib, Seaborn
