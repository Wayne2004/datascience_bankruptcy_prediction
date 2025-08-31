import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
# Session State for history
if "history" not in st.session_state:
    st.session_state.history = []

# =====================
# Tabs
tab1, tab2 = st.tabs(["📊 Prediction", "📜 History"])

with tab1:
    if st.sidebar.button("Predict"):
        if model is not None:
            # Get probability and prediction
            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]

            result = {
                "inputs": input_features,
                "probability": f"{prob*100:.2f}%",
                "classification": "Bankrupt" if pred == 1 else "Non-Bankrupt"
            }

            # Save to session history
            st.session_state.history.append(result)

            # Show results
            st.subheader("✅ Prediction Results" if result["classification"] == "Non-Bankrupt" else "❌ Prediction Results")
            st.metric("Bankruptcy Risk Probability", result["probability"])
            st.metric("Final Classification", result["classification"])

            # =====================
            # Feature Importance Visualization
            st.subheader("🔍 Feature Importance")
            try:
                importances = model.named_steps["classifier"].feature_importances_
                features = input_df.columns
                feat_imp = pd.DataFrame({"Feature": features, "Importance": importances})
                feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)

                plt.figure(figsize=(6, 4))
                plt.barh(feat_imp["Feature"], feat_imp["Importance"])
                plt.gca().invert_yaxis()
                plt.title("Top 10 Important Features")
                st.pyplot(plt)
            except Exception as e:
                st.info("Feature importance not available for this model.")

        else:
            st.warning("⚠️ Model not loaded. Please train and load your model first.")

with tab2:
    st.subheader("📜 Past Predictions")
    if st.session_state.history:
        hist_df = pd.DataFrame([
            {
                **r["inputs"],
                "Risk Probability": r["probability"],
                "Final Classification": r["classification"]
            }
            for r in st.session_state.history
        ])

        st.dataframe(hist_df, use_container_width=True)

        # =====================
        # History Visualizations
        st.subheader("📈 History Trends")
        try:
            hist_df["Risk Probability (%)"] = hist_df["Risk Probability"].str.replace("%", "").astype(float)

            st.line_chart(hist_df["Risk Probability (%)"])
            st.bar_chart(hist_df["Final Classification"].value_counts())
        except Exception as e:
            st.info("Could not generate history charts.")
    else:
        st.info("No predictions made yet.")
