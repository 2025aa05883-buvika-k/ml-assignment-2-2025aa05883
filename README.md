# Machine Learning Assignment 2 [2025aa05883]

End-to-end machine learning classification project implementing multiple models with evaluation metrics and an interactive Streamlit web application.

---

## 🎮 Player Engagement Prediction

## 📌 Problem Statement
The objective of this project is to predict **player engagement levels** (`High`, `Medium`, `Low`) in online gaming environments using demographic and gameplay-related features. Accurate engagement prediction enables better understanding of player behavior and supports data-driven decisions for improving player retention and game design.

---

## 📊 Dataset Description
- **Dataset Name**: Predict Online Gaming Behavior Dataset  
- **Source**: Kaggle (Rabie El Kharoua, CC BY 4.0 License)  
- **Number of Instances**: 40,034  
- **Number of Features**: 12 (excluding `PlayerID`)  
- **Target Variable**: `EngagementLevel` (`High`, `Medium`, `Low`)  

### Features Used
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

The dataset contains both numerical and categorical features and is suitable for **multi-class classification**.

### Preprocessing and Feature Engineering
- Encoding of categorical variables  
- Scaling of numerical features where required  
- Creation of derived features based on gameplay patterns to support improved model learning  

All preprocessing and feature engineering steps were applied **consistently across all models**.

---

## 🤖 Machine Learning Models Used
The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 📈 Comparison of Evaluation Metrics

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1 Score | MCC |
|---------------------|----------|------|-----------|--------|----------|-----|
| Logistic Regression | 0.8951 | 0.9422 | 0.8897 | 0.8948 | 0.8919 | 0.8351 |
| Decision Tree       | 0.8529 | 0.8842 | 0.8444 | 0.8455 | 0.8449 | 0.7677 |
| kNN                 | 0.8497 | 0.9268 | 0.8514 | 0.8401 | 0.8453 | 0.7610 |
| Naive Bayes         | 0.7890 | 0.9177 | 0.7825 | 0.8105 | 0.7903 | 0.6828 |
| Random Forest       | 0.9288 | 0.9431 | 0.9283 | 0.9211 | 0.9246 | 0.8872 |
| XGBoost             | 0.9303 | 0.9414 | 0.9302 | 0.9223 | 0.9261 | 0.8895 |

**Note:**  
AUC scores are computed using the **One-vs-Rest (OvR)** strategy with **macro-averaging** for multi-class classification.

---

## 🔍 Observations on Model Performance

| ML Model Name        | Observation |
|---------------------|-------------|
| Logistic Regression | Achieved strong performance due to effective feature engineering and scaling, showing good generalization on unseen data. |
| Decision Tree       | Performed reasonably well but showed reduced generalization compared to ensemble models. |
| kNN                 | Delivered stable results but was affected by high dimensionality after feature expansion. |
| Naive Bayes         | Fast and simple model, but lower performance due to independence assumptions not fully holding. |
| Random Forest       | Demonstrated excellent accuracy and robustness by aggregating multiple decision trees. |
| XGBoost             | Achieved the best overall performance by capturing complex non-linear feature interactions. |

---

## 🚀 Streamlit Application
An interactive **Streamlit web application** was developed and deployed using **Streamlit Community Cloud** with the following features:

- Upload CSV test dataset  
- Downloadable sample test dataset  
- Model selection dropdown  
- Display of evaluation metrics (Accuracy, Precision, Recall, F1 Score, MCC, AUC)  
- Confusion matrix visualization  
- Sample prediction preview  

🔗 **Live Streamlit App**:  
https://player-engagement-level-prediction.streamlit.app/

🔗 **GitHub Repository**:  
https://github.com/2025aa05883-buvika-k/ml-assignment-2-2025aa05883

---

## ⚙️ How to Run the Project
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit application:
   ```bash
   streamlit run app.py