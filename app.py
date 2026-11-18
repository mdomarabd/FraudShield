import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==================================================
# App configuration
# ==================================================
st.set_page_config(
    page_title="FraudShield Dashboard",
    layout="wide"
)

st.title("💳 FraudShield — Credit Card Fraud Detection")
st.markdown("### Intelligent, Explainable, Real-Time Fraud Detection")

st.markdown(
    """
    This dashboard demonstrates both:
    - **Offline model performance** (true evaluation on hold-out test set)
    - **Live predictions and evaluation** on uploaded or demo datasets
    """
)

# ==================================================
# Load trained models
# ==================================================
@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgb_fraud_model.pkl")
    logreg_model = joblib.load("models/logreg_fraud_model.pkl")
    return xgb_model, logreg_model

xgb_model, logreg_model = load_models()

# ==================================================
# Load offline evaluation metrics
# ==================================================
@st.cache_data
def load_model_metrics():
    return pd.read_csv("model_metrics.csv")

metrics_df = load_model_metrics()

# ==================================================
# Sidebar — Model selection
# ==================================================
st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["XGBoost (Best Overall)", "Logistic Regression (High Recall)"]
)

if model_choice.startswith("XGBoost"):
    model = xgb_model
    model_name = "XGBoost (Tuned)"
else:
    model = logreg_model
    model_name = "Logistic Regression (Tuned)"

st.sidebar.success(f"Active model: {model_name}")

# ==================================================
# SECTION 1 — REAL OFFLINE METRICS (IMPORTANT)
# ==================================================
st.header("📊 Model Performance (Hold-Out Test Set)")

st.markdown(
    """
    These metrics were computed **offline** on an untouched test dataset  
    and represent the **true generalization performance** of the selected model.
    """
)

model_metrics = metrics_df[metrics_df["Model"] == model_name].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{model_metrics['Accuracy']:.3f}")
col2.metric("Recall (Fraud)", f"{model_metrics['Recall (Fraud)']:.3f}")
col3.metric("Precision (Fraud)", f"{model_metrics['Precision (Fraud)']:.3f}")
col4.metric("F1 Score", f"{model_metrics['F1 (Fraud)']:.3f}")
col5.metric("ROC-AUC", f"{model_metrics['ROC-AUC']:.3f}")

st.divider()

# ==================================================
# Feature engineering (MUST MATCH TRAINING)
# ==================================================
def add_features(df):
    df = df.copy()
    pca_cols = [c for c in df.columns if c.startswith("V")]
    df["pca_magnitude"] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))
    df["log_amount"] = np.log1p(df["Amount"])
    return df

# ==================================================
# SECTION 2 — Input Data
# ==================================================
st.header("📥 Transaction Input")

input_method = st.radio(
    "Select input method:",
    ["Upload CSV", "Use Sample Dataset"]
)

input_df = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV file (same structure as training data)",
        type=["csv"]
    )
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)

else:
    input_df = pd.read_csv("data/processed/streamlit_test_data.csv")
    st.info("Using provided dataset.")
    st.dataframe(input_df)

# ==================================================
# SECTION 3 — Predictions & LIVE Evaluation
# ==================================================
if input_df is not None:

    st.header("🔍 Predictions & Live Evaluation")

    try:
        # Separate features and labels if available
        if "Class" in input_df.columns:
            X_input = input_df.drop(columns=["Class"])
            y_true = input_df["Class"].values
            has_labels = True
        else:
            X_input = input_df.copy()
            has_labels = False

        # Feature engineering
        X_input_fe = add_features(X_input)

        # Predictions
        preds = model.predict(X_input_fe)
        probs = model.predict_proba(X_input_fe)[:, 1]

        # --------------------------------------------------
        # LIVE METRICS (CLEARLY LABELED)
        # --------------------------------------------------
        if has_labels:
            st.subheader("🧪 Live Evaluation on Uploaded / Demo Dataset")

            st.warning(
                """
                The metrics below are computed **only on the currently loaded dataset**.
                They may differ significantly from the true model performance,
                especially for small or synthetic datasets.
                """
            )

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Accuracy", f"{accuracy_score(y_true, preds):.3f}")
            col2.metric("Recall (Fraud)", f"{recall_score(y_true, preds):.3f}")
            col3.metric("Precision (Fraud)", f"{precision_score(y_true, preds):.3f}")
            col4.metric("F1 Score", f"{f1_score(y_true, preds):.3f}")
            col5.metric("ROC-AUC", f"{roc_auc_score(y_true, probs):.3f}")

            # Confusion Matrix (robust)
            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Legitimate", "Actual Fraud"],
                columns=["Predicted Legitimate", "Predicted Fraud"]
            )

            st.dataframe(cm_df)

        else:
            st.info(
                "No ground-truth labels (`Class`) provided. "
                "Live evaluation metrics are not available."
            )

        # --------------------------------------------------
        # Transaction-level predictions
        # --------------------------------------------------
        st.subheader("📄 Transaction-Level Predictions")

        results = input_df.copy()
        results["Prediction"] = preds
        results["Fraud Probability"] = probs
        results["Prediction"] = results["Prediction"].map(
            {0: "Legitimate", 1: "Fraud"}
        )

        st.dataframe(results)

        # --------------------------------------------------
        # Risk interpretation
        # --------------------------------------------------
        st.subheader("⚠️ Risk Interpretation")

        avg_risk = float(probs.mean())
        st.progress(avg_risk)

        if avg_risk < 0.3:
            st.success("🟢 Overall Risk Level: Low")
        elif avg_risk < 0.7:
            st.warning("🟡 Overall Risk Level: Medium")
        else:
            st.error("🔴 Overall Risk Level: High")

    except Exception as e:
        st.error(f"Error during prediction or evaluation: {e}")
