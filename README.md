# ml-assignment-2-2025aa05883

End-to-end machine learning classification project implementing multiple models with evaluation metrics and an interactive Streamlit web application.

# 🎮 Player Engagement Prediction

## 📌 Problem Statement
The goal of this project is to predict **player engagement levels** (`High`, `Medium`, `Low`) in online gaming environments based on demographic and gameplay metrics. By applying multiple machine learning classification models, we aim to analyze factors influencing player retention and identify the most effective model for engagement prediction.

---

## 📊 Dataset Description
- **Source**: Synthetic dataset created by Rabie El Kharoua (CC BY 4.0 license).  
- **Rows**: ~40,000 instances.  
- **Features**: 12 predictive features (considering `PlayerID` as identifier).  
  - PlayerID
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
  - *(Optional engineered feature: TotalPlayTimePerWeek)*  
- **Target Variable**: `EngagementLevel` (categorical: High, Medium, Low).  

This dataset is suitable for **multi-class classification** and provides a balanced mix of categorical and numerical variables.

---

## 🤖 Models Used
We implemented six classification models on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 📈 Comparison of Evaluation Metrics

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression   |          |     |           |        |    |     |
| Decision Tree         |          |     |           |        |    |     |
| kNN                   |          |     |           |        |    |     |
| Naive Bayes           |          |     |           |        |    |     |
| Random Forest         |          |     |           |        |    |     |
| XGBoost               |          |     |           |        |    |     |

👉 *Note: For multi-class metrics, macro-averaging was used to ensure fair evaluation across all classes.*

---

## 🔍 Observations on Model Performance

| ML Model Name        | Observation about model performance |
|-----------------------|-------------------------------------|
| Logistic Regression   |                                     |
| Decision Tree         |                                     |
| kNN                   |                                     |
| Naive Bayes           |                                     |
| Random Forest         |                                     |
| XGBoost               |                                     |

*(Fill in with your insights, e.g., “Random Forest achieved the highest accuracy and balanced precision/recall, while Naive Bayes struggled with imbalanced classes.”)*

---

## 🚀 Deployment
The project includes a **Streamlit app** deployed on Streamlit Community Cloud.  
Features of the app:
- Dataset upload option (CSV).  
- Model selection dropdown.  
- Display of evaluation metrics.  
- Confusion matrix visualization.  

**Live App Link**: [Insert your Streamlit app URL here]  
**GitHub Repository**: [Insert your repo link here]  
