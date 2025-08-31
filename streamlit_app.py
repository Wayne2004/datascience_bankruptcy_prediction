import streamlit as st
import numpy as np
import joblib
import pandas as pd

try:
    model = joblib.load("xgb_feature_selected_pipeline.pkl")
except:
    model = None

# =====================
# Page Config
st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# =====================
# Title & Description
st.title("📊 Taiwanese Bankruptcy Prediction System")
st.write(
    "This tool uses machine learning to predict the risk of bankruptcy based on selected financial ratios. "
    "Provide the required inputs in the sidebar and click **Predict** to see results."
)

# =====================
# Sidebar - Input Features
st.sidebar.header("🔢 Enter Financial Ratios")

# Example: Replace these with your top selected features
input_features = {
    "Net Income to Stockholder's Equity": st.sidebar.number_input("Net Income to Stockholder's Equity", value=0.0),
    "Debt Ratio %": st.sidebar.number_input("Debt Ratio %", value=0.0),
    "Persistent EPS (Last 4 Seasons)": st.sidebar.number_input("Persistent EPS (Last 4 Seasons)", value=0.0),
    "Net Profit before Tax / Paid-in Capital": st.sidebar.number_input("Net Profit before Tax / Paid-in Capital", value=0.0),
    "Borrowing Dependency": st.sidebar.number_input("Borrowing Dependency", value=0.0),
    # ... add more from your top 15 features
}

# Convert input into DataFrame (for prediction later)
input_df = pd.DataFrame([input_features])

# =====================
# Main Page - Prediction
if st.sidebar.button("🚀 Predict"):
    if model is not None:
        # Get probability and prediction
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("✅ Prediction Results")
        st.metric("Bankruptcy Risk Probability", f"{prob*100:.2f}%")
        st.metric("Final Classification", "Bankrupt" if pred == 1 else "Non-Bankrupt")
    else:
        st.warning("⚠️ Model not loaded. Please train and load your model first.")
