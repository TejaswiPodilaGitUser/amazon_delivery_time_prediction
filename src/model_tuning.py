import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import numpy as np
from sklearn.model_selection import KFold

# Define the cross-validation strategy with random_state
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning for Gradient Boosting Regressor
def tune_gradient_boosting(x_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    }
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=cv, scoring='r2', n_jobs=-1)

    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Hyperparameter tuning for Random Forest Regressor
def tune_random_forest(x_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None],
    }
    random_search = RandomizedSearchCV(RandomForestRegressor(), param_dist, n_iter=10, cv=cv, random_state=42, scoring='r2', n_jobs=-1)

    random_search.fit(x_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

# Combined function to tune both models
def train_and_tune_model(x_train, y_train):
    # Tuning Gradient Boosting Model
    print("Tuning Gradient Boosting...")
    best_gb_model, best_gb_params = tune_gradient_boosting(x_train, y_train)
    print("Best Gradient Boosting Model:", best_gb_model)

    # Tuning Random Forest Model
    print("Tuning Random Forest...")
    best_rf_model, best_rf_params = tune_random_forest(x_train, y_train)
    print("Best Random Forest Model:", best_rf_model)

    # Return the best model and parameters
    return best_gb_model if best_gb_params else best_rf_model, best_gb_params if best_gb_params else best_rf_params
