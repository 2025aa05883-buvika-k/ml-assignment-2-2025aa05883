import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from feature_engineer import FeatureEngineer

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)

# Load models

# Load XGBoost bundle (model + label encoder)
xgb_bundle = joblib.load("model/xgb_model.pkl")
xgb_model = xgb_bundle["model"]
xgb_le = xgb_bundle["label_encoder"]

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": xgb_model
}

# Streamlit UI

st.title("Player Engagement Prediction – ML Assignment 2")
st.info("Upload the test dataset CSV containing the 'EngagementLevel' column.")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type="csv"
)

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)

    if "EngagementLevel" not in test_data.columns:
        st.error("CSV must include 'EngagementLevel' column for evaluation.")
        st.stop()

    X_test = test_data.drop(columns=["EngagementLevel"])
    y_test = test_data["EngagementLevel"]

    model_choice = st.selectbox(
        "Choose a classification model",
        list(models.keys())
    )
    model = models[model_choice]

    # Predictions

    y_pred = model.predict(X_test)

    # Decode XGBoost predictions
    if model_choice == "XGBoost":
        y_pred = xgb_le.inverse_transform(y_pred)

    # Classification Report

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Evaluation Metrics

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # AUC (OvR, Macro)
    auc = "N/A"
    if hasattr(model, "predict_proba"):
        try:
            if model_choice == "XGBoost":
                y_true_enc = xgb_le.transform(y_test)
                y_true_bin = pd.get_dummies(y_true_enc)
            else:
                y_true_bin = pd.get_dummies(y_test)

            y_score = model.predict_proba(X_test)
            auc = roc_auc_score(
                y_true_bin,
                y_score,
                average="macro",
                multi_class="ovr"
            )
        except Exception:
            auc = "N/A"

    st.subheader("Evaluation Metrics")
    st.table({
        "Accuracy": [acc],
        "Precision (Macro)": [prec],
        "Recall (Macro)": [rec],
        "F1 Score (Macro)": [f1],
        "MCC": [mcc],
        "AUC (OvR, Macro)": [auc]
    })

    # Confusion Matrix

    st.subheader("Confusion Matrix")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots()
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

    # Sample Predictions

    st.subheader("Sample Predictions")
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.dataframe(results_df.head(10))
