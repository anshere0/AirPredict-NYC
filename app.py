import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="NYC Airbnb Price Predictor & Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .prediction-title {
        color: #38bdf8;
        font-weight: bold;
    }
    .recommendation-card {
        background-color: #0f172a;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to check if trained artifacts are present
def check_models_trained(models_dir="models"):
    required_files = [
        "linear_regression.joblib",
        "random_forest.joblib",
        "preprocessor.joblib",
        "recommender.joblib",
        "scaler.joblib",
        "metrics.json"
    ]
    return all(os.path.exists(os.path.join(models_dir, f)) for f in required_files)

# Dynamic training trigger inside Streamlit
if not check_models_trained():
    st.title("🏠 NYC Airbnb ML Application")
    st.warning("⚠️ Machine Learning models and KNN recommender are not trained yet.")
    st.info("Click the button below to run the training pipeline on the NYC dataset. This will train all models, pre-compute SHAP explainers, and initialize the recommender. (Takes ~30 seconds)")
    
    if st.button("🚀 Train ML Models & Recommender"):
        with st.spinner("Running training pipeline..."):
            try:
                # Add current directory to path just in case
                import sys
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
                
                from src.train import main as train_main
                train_main()
                st.success("✅ Training completed successfully! Reloading application...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.exception(e)
    st.stop()

# Import predictor and recommender classes (only import after training check passes)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.predict import AirbnbPredictor
from src.recommender import AirbnbRecommender
from src.preprocessing import clean_data, transform_data
from src.features import engineer_features

@st.cache_resource
def load_predictor():
    return AirbnbPredictor()

@st.cache_resource
def load_recommender():
    return AirbnbRecommender("models/recommender.joblib", "models/scaler.joblib", "data/reference_listings.csv")

# Initialize models
predictor = load_predictor()
recommender = load_recommender()

# Load pre-computed metrics
with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

# Sidebar input controls
st.sidebar.title("🏠 Listing Input Details")
st.sidebar.markdown("Provide features of the Airbnb listing to predict its price and find similar listings.")

neighbourhood_groups = ["Brooklyn", "Manhattan", "Queens", "Staten Island", "Bronx"]
room_types = ["Entire home/apt", "Private room", "Shared room"]

neighbourhood_group = st.sidebar.selectbox("Neighborhood Group", neighbourhood_groups)
room_type = st.sidebar.selectbox("Room Type", room_types)

latitude = st.sidebar.number_input("Latitude", value=40.7128, format="%.5f")
longitude = st.sidebar.number_input("Longitude", value=-74.0060, format="%.5f")

minimum_nights = st.sidebar.slider("Minimum Nights", min_value=1, max_value=30, value=3)
availability_365 = st.sidebar.slider("Availability (Days/Year)", min_value=0, max_value=365, value=120)
number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=15)
reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, value=1.2, format="%.2f")
calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", min_value=1, value=1)

# Streamlit App Navigation
st.title("🏠 NYC Airbnb Pricing & Recommender Dashboard")
tab1, tab2, tab3 = st.tabs(["🔮 Price Predictor & Explainer", "📊 Model Comparison & Insights", "🧭 Similar Listing Recommender"])

# Tab 1: Predictor
with tab1:
    st.header("Price Prediction & Model Explanation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Predict Listing Price")
        available_models = list(predictor.models.keys())
        if not available_models:
            st.error("No models loaded. Please train models first.")
            st.stop()
        model_name = st.selectbox("Select Model for Prediction", available_models)
        
        # Package user inputs matching feature pipeline
        user_input = {
            "latitude": latitude,
            "longitude": longitude,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": calculated_host_listings_count,
            "availability_365": availability_365,
            "neighbourhood_group": neighbourhood_group,
            "room_type": room_type
        }
        
        if st.button("🔮 Predict Listing Price", use_container_width=True):
            price_pred = predictor.predict(user_input, model_name=model_name)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0px; color: #a1a1aa;">Predicted Listing Price ({model_name})</h3>
                <h1 style="margin: 10px 0px; font-size: 3.5rem; color: #10b981;">${price_pred:.2f} <span style="font-size: 1.5rem; color: #71717a;">/ night</span></h1>
                <p style="margin: 0px; color: #a1a1aa; font-size: 0.9rem;">Price generated using structured regression model trained on 45,000+ NYC Airbnb records.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.subheader("Inference Explainability (SHAP)")
        st.markdown("SHAP (SHapley Additive exPlanations) shows how each input feature contributes to the prediction compared to the average dataset baseline.")
        
        try:
            import shap
            import joblib
            
            # Load preprocessor and SHAP explainer
            preprocessor = joblib.load("models/preprocessor.joblib")
            explainer = joblib.load("models/shap_explainer.joblib")
            
            df_input = pd.DataFrame([user_input])
            df_clean = clean_data(df_input, is_training=False)
            df_engineered = engineer_features(df_clean)
            df_processed = transform_data(df_engineered, preprocessor)
            
            # Align features
            if "XGBoost" in predictor.models and hasattr(predictor.models["XGBoost"], "feature_names_in_"):
                expected_features = predictor.models["XGBoost"].feature_names_in_
            elif "Random Forest" in predictor.models and hasattr(predictor.models["Random Forest"], "feature_names_in_"):
                expected_features = predictor.models["Random Forest"].feature_names_in_
            else:
                expected_features = df_processed.columns
                
            df_final = df_processed.reindex(columns=expected_features, fill_value=0)
            
            # Compute SHAP
            shap_values = explainer(df_final)
            sv = shap_values[0]
            
            import plotly.graph_objects as go
            import numpy as np

            feature_names = sv.feature_names
            shap_vals = sv.values
            feature_vals = sv.data
            base_value = sv.base_values

            abs_vals = np.abs(shap_vals)
            sorted_idx = np.argsort(abs_vals)[::-1]
            
            top_n = 9
            top_indices = sorted_idx[:top_n]
            other_indices = sorted_idx[top_n:]
            
            other_shap = float(np.sum(shap_vals[other_indices])) if len(other_indices) > 0 else 0.0

            measures = ["absolute"]
            y_labels = ["Base Value (Average)"]
            x_vals = [float(base_value)]
            hover_texts = [f"Average predicted price across all listings: <b>${base_value:.2f}</b>"]

            for idx in top_indices:
                feat = feature_names[idx]
                val = feature_vals[idx]
                sv_val = float(shap_vals[idx])
                
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                y_labels.append(f"{feat} = {val_str}")
                measures.append("relative")
                x_vals.append(sv_val)
                
                direction = "Increases" if sv_val > 0 else "Decreases"
                hover_texts.append(f"Feature: <b>{feat}</b><br>Value: {val_str}<br>Effect: {direction} price by <b>${abs(sv_val):.2f}</b>")

            if len(other_indices) > 0:
                y_labels.append(f"{len(other_indices)} Other Features")
                measures.append("relative")
                x_vals.append(other_shap)
                direction = "Increases" if other_shap > 0 else "Decreases"
                hover_texts.append(f"Combined effect of {len(other_indices)} other features<br>Effect: {direction} price by <b>${abs(other_shap):.2f}</b>")

            predicted_value = float(base_value + np.sum(shap_vals))
            y_labels.append("Final Predicted Price")
            measures.append("absolute")
            x_vals.append(predicted_value)
            hover_texts.append(f"Final prediction for this listing: <b>${predicted_value:.2f}</b>")

            text_labels = []
            for v, m in zip(x_vals, measures):
                if m == "relative":
                    text_labels.append(f"+${v:.2f}" if v > 0 else f"-${abs(v):.2f}")
                else:
                    text_labels.append(f"${v:.2f}")

            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="h",
                measure=measures,
                y=y_labels,
                x=x_vals,
                textposition="outside",
                text=text_labels,
                hoverinfo="text",
                hovertext=hover_texts,
                increasing={"marker":{"color":"#ff0051"}}, 
                decreasing={"marker":{"color":"#008bfb"}},
                totals={"marker":{"color":"#64748b"}}
            ))

            fig.update_layout(
                title="Interactive Feature Attributions",
                showlegend=False,
                height=550,
                margin=dict(l=10, r=40, t=40, b=10),
                yaxis=dict(title="", autorange="reversed"),
                xaxis=dict(title="Price Impact (USD)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0")
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute local SHAP explanation:")
            st.exception(e)

# Tab 2: Dashboard comparison
with tab2:
    st.header("Model Evaluation & Insights")
    
    col_metrics, col_plot = st.columns([1, 1.2])
    
    with col_metrics:
        st.subheader("Performance Matrix")
        
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=["R2"], color="#14532d").highlight_min(axis=0, subset=["MAE", "RMSE"], color="#14532d"), use_container_width=True)
        
        st.markdown("""
        **Metric Meanings:**
        - **$R^2$ Score (Coefficient of Determination):** Proportion of price variance explained by features. Closer to 1.0 is better.
        - **MAE (Mean Absolute Error):** Average prediction deviation in dollars. Smaller error is better.
        - **RMSE (Root Mean Squared Error):** Standard deviation of prediction residuals (penalizes larger errors). Smaller error is better.
        """)
        
    with col_plot:
        st.subheader("Global Feature Importance (XGBoost)")
        if os.path.exists("models/feature_importance.png"):
            st.image("models/feature_importance.png", use_container_width=True)
        else:
            st.info("Feature importance plot not found. Run model training.")

# Tab 3: Recommender
with tab3:
    st.header("KNN-Based Similar Listing Recommendations")
    st.markdown("Locates similar listings in New York City based on location similarity, room types, and price levels using K-Nearest Neighbors.")
    
    # Run recommendation for the selected user inputs
    query_row = {
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "price": predictor.predict(user_input, model_name=available_models[0])  # Use best available prediction as search price
    }
    
    # Preprocess dict row for KNN matching
    df_raw_row = pd.DataFrame([query_row])
    df_clean_row = clean_data(df_raw_row, is_training=False)
    df_engineered_row = engineer_features(df_clean_row)
    df_processed_row = transform_data(df_engineered_row, predictor.encoder)
    
    # Run recommend query
    try:
        recs = recommender.recommend(df_processed_row, n_recommendations=5)
        
        col_rec_list, col_rec_map = st.columns([1.2, 1])
        
        with col_rec_list:
            st.subheader("Top 5 Recommended Listings")
            for idx, row in recs.iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="margin: 0px; color: #10b981;">{row['name']}</h4>
                    <p style="margin: 5px 0px 0px 0px; color: #94a3b8; font-size: 0.95rem;">
                        📍 <b>{row['neighbourhood']}</b> ({row['neighbourhood_group']}) &bull; 🏠 <b>{row['room_type']}</b> &bull; 💲 <b>${row['price']:.2f} / night</b>
                    </p>
                    <p style="margin: 2px 0px 0px 0px; color: #64748b; font-size: 0.85rem;">
                        Similarity Match: <b>{row['similarity_percentage']}%</b> (Euclidean Distance: {row['similarity_distance']:.4f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_rec_map:
            st.subheader("Geographical Distribution Map")
            # Build geo df for Streamlit map
            map_data = pd.DataFrame({
                "lat": [latitude] + list(recs["latitude"]),
                "lon": [longitude] + list(recs["longitude"]),
                "name": ["Current Query"] + list(recs["name"])
            })
            st.map(map_data, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error querying recommendation model: {e}")
        st.exception(e)
