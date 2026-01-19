# Machine Learning Assignment 2 [2025aa05883]

End-to-end machine learning classification project implementing multiple models with evaluation metrics and an interactive Streamlit web application.

---

## 🎮 Player Engagement Prediction

## 📌 Problem Statement
The objective of this project is to predict **player engagement levels** (`High`, `Medium`, `Low`) in online gaming environments using demographic and gameplay-related features. By applying multiple machine learning classification models, this project aims to analyze factors influencing player engagement and compare model performance using standard evaluation metrics.

---

## 📊 Dataset Description
- **Source**: *Predict Online Gaming Behavior Dataset* by Rabie El Kharoua (Kaggle, CC BY 4.0 license)  
- **Instances**: ~40,000 rows  
- **Predictive Features**: 12+ (excluding `PlayerID`, which is used only as an identifier)  
- **Target Variable**: `EngagementLevel` (`High`, `Medium`, `Low`)  

### Feature Overview
- Age  
- Gender  
- Location  
- GameGenre  
- PlayTimeHours  
- InGamePurchases  
- GameDifficulty  
- SessionsPerWeek  
- AvgSessionDurationMinutes  
- PlayerLevel  
- AchievementsUnlocked  
- *(Engineered Feature)* TotalPlayTimePerWeek  

The dataset is suitable for **multi-class classification** and contains a mix of numerical and categorical features. Categorical variables were encoded appropriately, and feature engineering was applied consistently across training and inference pipelines.

---

## 🤖 Machine Learning Models Used
The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 📈 Comparison of Evaluation Metrics

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression |          |     |           |        |          |     |
| Decision Tree       |          |     |           |        |          |     |
| kNN                 |          |     |           |        |          |     |
| Naive Bayes         |          |     |           |        |          |     |
| Random Forest       |          |     |           |        |          |     |
| XGBoost             |          |     |           |        |          |     |

**Note:** For multi-class classification, AUC was computed using a **One-vs-Rest (OvR)** strategy with **macro-averaging** to ensure fair evaluation across all classes.

---

## 🔍 Observations on Model Performance

| ML Model Name        | Observation |
|---------------------|-------------|
| Logistic Regression |             |
| Decision Tree       |             |
| kNN                 |             |
| Naive Bayes         |             |
| Random Forest       |             |
| XGBoost             |             |

*(Observations will be added after evaluating all models, focusing on performance differences, class-wise behavior, and robustness.)*

---

## 🚀 Streamlit Application
An interactive **Streamlit web application** was developed and deployed using **Streamlit Community Cloud** with the following features:

- CSV dataset upload (test data only)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix / classification report visualization  

🔗 **Live Streamlit App**: *[Link to be added]*  
🔗 **GitHub Repository**: *[Link to be added]*  

---

## ⚙️ How to Run the Project
1. Clone the GitHub repository  
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
