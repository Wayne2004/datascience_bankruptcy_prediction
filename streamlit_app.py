import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import io

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
            st.subheader("✅ Prediction Results" if result["classification"] == "Non-Bankrupt" else "💸 Prediction Results")
            st.metric("Bankruptcy Risk Probability", result["probability"])
            st.metric("Final Classification", result["classification"])

            # =====================
            # Feature Contribution Visualization
            st.subheader("🔍 Feature Contribution for This Prediction")

            classifier = model.named_steps.get("clf", None)

            if classifier is not None:
                # Create a SHAP explainer for the model
                explainer = shap.TreeExplainer(classifier)

                # Compute SHAP values for the single input
                shap_values = explainer.shap_values(input_df)

                # Create dataframe for contributions
                contrib_df = pd.DataFrame({
                    "Feature": input_df.columns,
                    "Contribution": shap_values[0]  # for this single instance
                }).sort_values("Contribution", key=abs, ascending=False)

                # Show top 10 contributions
                top_contrib = contrib_df.head(10)

                fig, ax = plt.subplots(figsize=(6,4))
                colors = ["red" if val > 0 else "blue" for val in top_contrib["Contribution"]]
                ax.barh(top_contrib["Feature"], top_contrib["Contribution"], color=colors)
                ax.set_xlabel("SHAP Value (Impact on Bankruptcy Prediction)")
                ax.set_title("Top Feature Contributions for This Prediction")
                ax.invert_yaxis()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                
                # Display the image with custom width (e.g. scaled down to 400px)
                st.image(buf, width=800)

                st.write("🔵 Negative values push towards 'Non-Bankrupt'  🔴 Positive values push towards 'Bankrupt'.")

                # =====================
                # Business Interpretation
                st.subheader("📌 User Guidance")

                # Risk bucket
                risk_level = "🟢 Low Risk" if prob <= 0.20 else ("🟡 Moderate Risk" if prob <= 0.50 else "🔴 High Risk")
                st.write(f"**Risk Category:** {risk_level}")

                # Top red flags
                red_flags = top_contrib[top_contrib["Contribution"] > 0].head(2)
                if not red_flags.empty:
                    signals = ", and ".join(red_flags["Feature"].tolist())
                    st.write(f"🚨 **Top Warning Signals:** {signals} are pushing the company towards bankruptcy.")

                # Recommendations (from negative contributors)
                positives = top_contrib[top_contrib["Contribution"] < 0].head(2)
                if not positives.empty:
                    recs = ", and ".join(positives["Feature"].tolist())
                    st.write(f"✅ **Improvement Suggestions:** Strengthening {recs} could reduce bankruptcy risk.")

            else:
                st.info("SHAP analysis not available — classifier not found in pipeline.")

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
        try:
            hist_df["Risk Probability (%)"] = hist_df["Risk Probability"].str.replace("%", "").astype(float)

            st.subheader("📈 History Trends")
            st.line_chart(hist_df["Risk Probability (%)"])
            st.subheader("📈 History Cases Count")
            st.bar_chart(hist_df["Final Classification"].value_counts())
        except Exception as e:
            st.info("Could not generate history charts.")
    else:
        st.info("No predictions made yet.")
