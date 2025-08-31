import streamlit as st
import numpy as np
import joblib
import pandas as pd

try:
    model = joblib.load("xgb_feature_selected_pipeline.pkl")
except Exception as e:
    model = None
    st.error(f"❌ Error loading with joblib: {repr(e)}")

# =====================
# Page Config
st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# =====================
# Title & Description
st.title("Taiwanese Bankruptcy Prediction System")
st.write(
    "This tool uses machine learning to predict the risk of bankruptcy based on selected financial ratios. "
    "Provide the required inputs in the sidebar and click **Predict** to see results."
)

# =====================
# Sidebar - Input Features
st.sidebar.header("🔢 Enter Financial Ratios")


input_features = {
    " Net Income to Stockholder's Equity": st.sidebar.number_input("Net Income to Stockholder's Equity", value=0.0),
    " Debt ratio %": st.sidebar.number_input("Debt ratio %", value=0.0),
    " Persistent EPS in the Last Four Seasons": st.sidebar.number_input("Persistent EPS in the Last Four Seasons", value=0.0),
    " Net profit before tax/Paid-in capital": st.sidebar.number_input("Net profit before tax/Paid-in capital", value=0.0),
    " Borrowing dependency": st.sidebar.number_input("Borrowing dependency", value=0.0),
    " Per Share Net profit before tax (Yuan ¥)": st.sidebar.number_input("Per Share Net profit before tax (Yuan ¥)", value=0.0),
    " ROA(A) before interest and % after tax": st.sidebar.number_input("ROA(A) before interest and % after tax", value=0.0),
    " Net Value Per Share (A)": st.sidebar.number_input("Net Value Per Share (A)", value=0.0),
    " Net Value Per Share (B)": st.sidebar.number_input("Net Value Per Share (B)", value=0.0),
    " ROA(C) before interest and depreciation before interest": st.sidebar.number_input("ROA(C) before interest and depreciation before interest", value=0.0),
    " Continuous interest rate (after tax)": st.sidebar.number_input("Continuous interest rate (after tax)", value=0.0),
    " ROA(B) before interest and depreciation after tax": st.sidebar.number_input("ROA(B) before interest and depreciation after tax", value=0.0),
    " Net Income to Total Assets": st.sidebar.number_input("Net Income to Total Assets", value=0.0),
    " Degree of Financial Leverage (DFL)": st.sidebar.number_input("Degree of Financial Leverage (DFL)", value=0.0),
    " Retained Earnings to Total Assets": st.sidebar.number_input("Retained Earnings to Total Assets", value=0.0),
}

# Convert input into DataFrame (for prediction later)
input_df = pd.DataFrame([input_features])

# =====================
# PREDICTION HAPPENS HERE --- :D
if st.sidebar.button("📊 Predict"):
    if model is not None:
        # Get probability and prediction
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("✅ Prediction Results")
        st.metric("Bankruptcy Risk Probability", f"{prob*100:.2f}%")
        st.metric("Final Classification", "Bankrupt" if pred == 1 else "Non-Bankrupt")
    else:
        st.warning("⚠️ Model not loaded. Please train and load your model first.")
