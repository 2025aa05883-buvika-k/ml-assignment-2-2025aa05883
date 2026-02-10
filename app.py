import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from feature_engineer import FeatureEngineer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)


# PAGE CONFIG (MUST BE FIRST)

st.set_page_config(
    page_title="Player Engagement Prediction",
    layout="wide"
)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], div, span, p, h1, h2, h3, label, input, select, textarea {
    font-family: 'Poppins', 'Segoe UI Emoji', sans-serif !important;
}

.stApp {
    background: linear-gradient(180deg, #fff5fa 0%, #fde8f1 100%);
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #f3a6c8, #c86b98);
    padding: 2.6rem;
    border-radius: 28px;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    color: white;
    font-size: 3.2rem;
    font-weight: 800;
}
.hero p {
    color: #fff0f7;
    font-size: 1.15rem;
}

/* PROBLEM CARD */
.problem-box {
    background: #fff0f7;
    padding: 1.6rem;
    border-radius: 22px;
    border-left: 6px solid #e75480;
}

/* HEADERS */
h2, h3 {
    color: #7a1f4c;
    font-weight: 700;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #ffe6f0 !important;
    padding: 2rem !important;
    border-radius: 22px !important;
    border: 2px dashed #e75480 !important;
}

/* DROPDOWN */
select {
    background: #ffe6f0 !important;
    border-radius: 14px !important;
    border: 1px solid #e75480 !important;
    padding: 0.5rem !important;
}

/* BUTTONS */
button {
    background: linear-gradient(135deg, #f3a6c8, #c86b98) !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: none !important;
}
button:hover {
    transform: translateY(-2px);
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: #fff0f7;
    padding: 1.4rem;
    border-radius: 20px;
    box-shadow: 0 6px 18px rgba(231,84,128,0.18);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(231,84,128,0.32);
}
[data-testid="stMetricValue"] {
    color: #7a1f4c;
    font-size: 1.7rem;
}

/* TABLE */
thead tr th {
    background-color: #e75480 !important;
    color: white !important;
    font-weight: 600 !important;
}
tbody tr:nth-child(even) {
    background-color: #fff5fa;
}

/* SECTION TITLE */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #7a1f4c;
    margin-top: 1rem;
}
</style>
            
<style>

.sample-table thead tr th {
    background-color: #e75480 !important;
    color: white !important;
    font-weight: 600 !important;
}

.correct-row {
    background-color: #e6f4ea !important;
}

.wrong-row {
    background-color: #fde2ea !important;
}

</style>            
""", unsafe_allow_html=True)


# LOAD MODELS
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


# SAMPLE TEST DATA
SAMPLE_TEST_PATH = "data/test.csv"

@st.cache_data
def load_sample_test_data():
    return pd.read_csv(SAMPLE_TEST_PATH)

# HERO SECTION
st.markdown("""
<div class="hero">
    <h1>🎮 Player Engagement Prediction</h1>
    <p>Multi-model machine learning evaluation with interactive visual analytics</p>
</div>
""", unsafe_allow_html=True)

# PROBLEM STATEMENT
st.markdown("""
<div class="problem-box">
<b>Problem Statement</b><br><br>
The objective of this project is to predict <b>player engagement levels</b>
(<i>High, Medium, Low</i>) using gameplay and demographic features.
The application allows users to upload test data, select a trained
classification model, and evaluate performance using standard multi-class
metrics.
</div>
""", unsafe_allow_html=True)

# SAMPLE DATA DOWNLOAD
st.markdown('<div class="section-title">📥 Sample Test Dataset</div>', unsafe_allow_html=True)

try:
    sample_df = load_sample_test_data()
    st.download_button(
        "Download Sample Test CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
except:
    st.warning("Sample test dataset not found in repository.")

# UPLOAD DATA
st.markdown('<div class="section-title">📂 Upload Test Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file (must include `EngagementLevel`)",
    type="csv"
)

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)

    if "EngagementLevel" not in test_data.columns:
        st.error("CSV must include 'EngagementLevel' column.")
        st.stop()

    X_test = FeatureEngineer().transform(
        test_data.drop(columns=["EngagementLevel"])
    )
    y_test = test_data["EngagementLevel"]

    # MODEL SELECTION
    st.markdown('<div class="section-title">🤖 Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_choice]

    with st.spinner("✨ Evaluating model..."):
        y_pred = model.predict(X_test)
        if model_choice.startswith("XGBoost"):
            y_pred = xgb_le.inverse_transform(y_pred)

    # METRICS
    st.markdown('<div class="section-title">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_true_bin = pd.get_dummies(
                xgb_le.transform(y_test)
                if model_choice.startswith("XGBoost") else y_test
            )
            auc = roc_auc_score(
                y_true_bin,
                model.predict_proba(X_test),
                average="macro",
                multi_class="ovr"
            )
        except:
            auc = None

    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{acc:.3f}")
    cols[1].metric("Precision", f"{prec:.3f}")
    cols[2].metric("Recall", f"{rec:.3f}")
    cols[3].metric("F1 Score", f"{f1:.3f}")
    cols[4].metric("MCC", f"{mcc:.3f}")
    cols[5].metric("AUC (OvR)", f"{auc:.3f}" if auc else "N/A")

    # CONFUSION MATRIX
    st.markdown('<div class="section-title">📌 Confusion Matrix</div>', unsafe_allow_html=True)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Compact figure size to avoid vertical scrolling
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=110)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    st.pyplot(fig, width="content")



    # SAMPLE PREDICTIONS
    st.markdown('<div class="section-title">🔍 Sample Predictions</div>', unsafe_allow_html=True)

    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).head(10)

    def highlight_predictions(row):
        if row["Actual"] == row["Predicted"]:
            return ['background-color: #e6f4ea'] * len(row)   # green
        else:
            return ['background-color: #fde2ea'] * len(row)   # pink

    st.dataframe(
        pred_df.style.apply(highlight_predictions, axis=1)
                .set_table_attributes('class="sample-table"'),
        width="stretch"
    )
    st.caption("🟢 Correct predictions are highlighted in green, while ❌ incorrect predictions are shown in soft pink.")


