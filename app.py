import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engineer import FeatureEngineer

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Player Engagement Prediction",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
xgb_bundle = joblib.load("model/xgb_model.pkl")
xgb_model = xgb_bundle["model"]
xgb_le = xgb_bundle["label_encoder"]

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest (Ensemble)": joblib.load("model/random_forest.pkl"),
    "XGBoost (Ensemble)": xgb_model
}

# -----------------------------
# Utility: Load sample test data from repo
# -----------------------------
SAMPLE_TEST_PATH = "data/test.csv"

@st.cache_data
def load_sample_test_data():
    return pd.read_csv(SAMPLE_TEST_PATH)

# -----------------------------
# Title & Problem Statement
# -----------------------------
st.title("🎮 Player Engagement Prediction")
st.markdown(
    """
    **Problem Statement:**  
    The objective of this project is to predict player engagement levels using
    multiple machine learning classification models. The application allows
    users to upload test data, select a trained model, and evaluate performance
    using standard classification metrics.
    """
)

st.divider()

# -----------------------------
# Sample Test Data Download
# -----------------------------
st.subheader("📥 Sample Test Dataset")

try:
    sample_df = load_sample_test_data()

    st.download_button(
        label="Download Sample Test CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )

    st.caption("Sample test dataset provided from the project repository.")

except Exception:
    st.warning("Sample test dataset not found in the repository.")

st.divider()

# -----------------------------
# Dataset Upload
# -----------------------------
st.subheader("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload TEST dataset (CSV with 'EngagementLevel' column)",
    type="csv"
)

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)

    if "EngagementLevel" not in test_data.columns:
        st.error("CSV must include 'EngagementLevel' column for evaluation.")
        st.stop()

    X_test = test_data.drop(columns=["EngagementLevel"])
    y_test = test_data["EngagementLevel"]

    # Feature Engineering
    fe = FeatureEngineer()
    X_test = fe.transform(X_test)

    st.divider()

    # -----------------------------
    # Model Selection
    # -----------------------------
    st.subheader("🤖 Model Selection")
    model_choice = st.selectbox(
        "Select a classification model",
        list(models.keys())
    )
    model = models[model_choice]

    # -----------------------------
    # Predictions
    # -----------------------------
    with st.spinner("Evaluating model..."):
        y_pred = model.predict(X_test)

        if model_choice.startswith("XGBoost"):
            y_pred = xgb_le.inverse_transform(y_pred)

    # -----------------------------
    # Metrics
    # -----------------------------
    st.subheader("📊 Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_true_bin = pd.get_dummies(
                xgb_le.transform(y_test) if model_choice.startswith("XGBoost") else y_test
            )
            y_score = model.predict_proba(X_test)
            auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
        except Exception:
            auc = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1 Score", f"{f1:.3f}")
    c5.metric("MCC", f"{mcc:.3f}")
    c6.metric("AUC", f"{auc:.3f}" if auc is not None else "N/A")

    st.divider()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("📌 Confusion Matrix")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # Sample Predictions
    # -----------------------------
    st.subheader("🔍 Sample Predictions")
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.dataframe(results_df.head(10), use_container_width=True)
