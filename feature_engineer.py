# feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Interaction features
        X["WeeklyPlayMinutes"] = X["SessionsPerWeek"] * X["AvgSessionDurationMinutes"]
        X["ProgressionIntensity"] = X["PlayerLevel"] * X["AchievementsUnlocked"]
        # Ratios
        X["Efficiency"] = X["AchievementsUnlocked"] / (X["PlayerLevel"] + 1)
        X["SpendPerHour"] = X["InGamePurchases"] / (X["PlayTimeHours"] + 1)
        return X