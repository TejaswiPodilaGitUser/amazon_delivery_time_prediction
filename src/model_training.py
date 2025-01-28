import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import time
from mlflow.models.signature import infer_signature

# Set up MLflow experiment
mlflow.set_experiment("Amazon Delivery Time Prediction")

# Load the cleaned dataset
def load_data(file_path):
    """Load cleaned data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path)
    return df

# Preprocess the data (ensure categorical variables are encoded properly)
def preprocess_data(df):
    """Preprocess the data, encoding categorical variables and handling integer columns."""
    # Identify categorical columns (non-numeric) and encode them
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes  # Convert categorical to numeric codes
    
    # Convert integer columns to float to handle missing values
    integer_columns = df.select_dtypes(include=["int64", "int32"]).columns.tolist()
    df[integer_columns] = df[integer_columns].astype(np.float64)  # Convert integer to float64
    
    return df

# Split the data into training and testing sets
def split_data(df, target_column):
    """Split the dataset into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values (if necessary)
def handle_missing_values(df):
    """Handle missing values by filling or dropping them."""
    return df.dropna()

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics for the model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# Train and log a model
def train_and_log_model(model_name, model, x_train, x_test, y_train, y_test):
    """Train a model, log it to MLflow, and return performance metrics."""
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(metrics)

        # Infer signature and input example
        input_example = x_test.iloc[0:1]
        signature = infer_signature(x_test, model.predict(x_test))

        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path=model_name, 
            signature=signature, 
            input_example=input_example
        )

        print(f"Model: {model_name}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        return model, metrics

# Main function
def main():
    file_path = "data/processed/cleaned_data.csv"
    target_column = "Delivery_Time"
    
    print("Loading data...")
    df = load_data(file_path)
    print("Handling missing values...")
    df = handle_missing_values(df)
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = split_data(df_processed, target_column)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, min_samples_leaf=2, max_features="sqrt"
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, learning_rate=0.1
        ),
    }
    
    print("Training and logging models...")
    best_model = None
    best_metrics = None
    best_r2 = -np.inf
    for model_name, model in models.items():
        trained_model, metrics = train_and_log_model(model_name, model, x_train, x_test, y_train, y_test)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = trained_model
            best_metrics = metrics
    
    if best_model:
        print("Saving the best model...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        best_model_path = f"models/trained_models/best_model_{timestamp}.pkl"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        mlflow.sklearn.save_model(best_model, best_model_path)
        print(f"Best model saved at: {best_model_path}")
        print("Best model metrics:")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
