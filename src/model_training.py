import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import time
import joblib
from mlflow.models.signature import infer_signature
from sklearn.impute import SimpleImputer

# Set up MLflow experiment
mlflow.set_experiment("Amazon Delivery Time Prediction")

def load_data(file_path):
    """Load cleaned data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Encode categorical variables and convert integer columns to float64."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype('category').cat.codes  

    df[df.select_dtypes(include=["int64", "int32"]).columns] = df.select_dtypes(include=["int64", "int32"]).astype(np.float64)

    # Ensure missing values are imputed
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=["float64"]).columns] = imputer.fit_transform(df[df.select_dtypes(include=["float64"]).columns])

    return df

def split_data(df, target_column):
    """Split the dataset into train and test sets."""
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(x, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics for the model."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def hyperparameter_tuning(x_train, y_train):
    """Perform hyperparameter tuning for Random Forest."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1, scoring='r2', random_state=42)
    rf_random.fit(x_train, y_train)
    print(f"Best Hyperparameters: {rf_random.best_params_}")
    return rf_random.best_estimator_

def cross_validate_model(model, x_train, y_train):
    """Perform cross-validation and return mean score."""
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    print(f"Mean R² Score: {scores.mean():.4f}")
    return scores.mean()

def train_and_log_model(model_name, model, x_train, x_test, y_train, y_test):
    """Train a model, log it to MLflow, and return performance metrics."""
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = evaluate_model(y_test, y_pred)
        
        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(x_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path=model_name, signature=signature, input_example=x_test.iloc[0:1])
        
        print(f"Model: {model_name}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return model, metrics

def save_best_model(best_model):
    """Save the best model with a timestamp and as a standard file."""
    model_dir = "models/trained_models"
    os.makedirs(model_dir, exist_ok=True)

    timestamped_model_path = os.path.join(model_dir, f"best_model_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
    best_model_path = os.path.join(model_dir, "best_model.pkl")

    # Ensure 'best_model.pkl' is a file, not a directory
    if os.path.isdir(best_model_path):
        os.rmdir(best_model_path)

    joblib.dump(best_model, timestamped_model_path)
    joblib.dump(best_model, best_model_path)

    print(f"Best model saved at: {best_model_path}")

def main():
    file_path = "data/processed/cleaned_data.csv"
    target_column = "Delivery_Time"
    
    print("Loading data...")
    df = load_data(file_path)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = split_data(df_processed, target_column)
    
    print("Performing hyperparameter tuning for Random Forest...")
    best_rf_model = hyperparameter_tuning(x_train, y_train)

    print("Cross-validating the best model...")
    cross_validate_model(best_rf_model, x_train, y_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": best_rf_model,
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1),
    }
    
    print("Training and logging models...")
    best_model, best_metrics, best_r2 = None, None, -np.inf
    for model_name, model in models.items():
        trained_model, metrics = train_and_log_model(model_name, model, x_train, x_test, y_train, y_test)
        if metrics["r2"] > best_r2:
            best_r2, best_model, best_metrics = metrics["r2"], trained_model, metrics

    if best_model:
        print("Saving the best model...")
        save_best_model(best_model)

        print("Best model metrics:")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
