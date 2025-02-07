import os
import time
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from mlflow.models.signature import infer_signature

# Set MLflow experiment
mlflow.set_experiment("Amazon Delivery Time Prediction")

# Load Data
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    return pd.read_csv(file_path)

# Data Preprocessing
def preprocess_data(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype('category').cat.codes  
    
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=["float64", "int64"]).columns] = imputer.fit_transform(df[df.select_dtypes(include=["float64", "int64"]).columns])
    
    return df

# Data Splitting
def split_data(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Model Evaluation
def evaluate_model(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

# Hyperparameter Optimization using Optuna
def hyperparameter_tuning(x_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        model = RandomForestRegressor(**params, random_state=42)
        return cross_val_score(model, x_train, y_train, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return RandomForestRegressor(**study.best_params, random_state=42)

# Model Training & Logging
def train_and_log_model(model_name, model, x_train, x_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=model_name, signature=infer_signature(x_test, y_pred))
        return model, metrics

# Save Best Model
def save_best_model(best_model):
    model_dir = "models/trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved at: {model_path}")

# Main Function
def main():
    file_path = "data/processed/cleaned_data.csv"
    target_column = "Delivery_Time"
    
    df = load_data(file_path)
    df_processed = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(df_processed, target_column)
    
    best_rf_model = hyperparameter_tuning(x_train, y_train)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": best_rf_model,
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
    }
    
    best_model, best_r2 = None, -np.inf
    for model_name, model in models.items():
        trained_model, metrics = train_and_log_model(model_name, model, x_train, x_test, y_train, y_test)
        if metrics["r2"] > best_r2:
            best_r2, best_model = metrics["r2"], trained_model
    
    if best_model:
        save_best_model(best_model)

if __name__ == "__main__":
    main()