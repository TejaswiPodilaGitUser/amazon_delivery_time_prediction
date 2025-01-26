import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os

# Set up MLflow experiment
mlflow.set_experiment("Amazon Delivery Time Prediction")

# Load the cleaned dataset
def load_data(file_path):
    """Load cleaned data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path)
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
    return df.dropna()  # You could also choose to fill missing values if required

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics for the model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# Preprocessing function to handle categorical features
def preprocess_data(df):
    """Preprocess the data, encoding categorical variables."""
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Apply label encoding to each categorical column
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))  # Ensure all values are treated as strings before encoding
    
    # After label encoding, return the transformed dataframe
    return df

# Train and log a model
def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model, log it to MLflow, and return performance metrics."""
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        metrics = evaluate_model(y_test, y_pred)
        
        # Log parameters, metrics, and the model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)
        
        # Print metrics for reference
        print(f"Model: {model_name}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return model, metrics

# Main function
def main():
    # Load data
    file_path = "data/processed/cleaned_data.csv"  # Update path as needed
    target_column = "Delivery_Time"  # Replace with your actual target column
    
    print("Loading data...")
    df = load_data(file_path)
    
    # Handle missing values (if necessary)
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    # Preprocess data (encode categorical features)
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(df_processed, target_column)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    # Train and log models
    print("Training and logging models...")
    best_model = None
    best_metrics = None
    best_r2 = -np.inf
    for model_name, model in models.items():
        trained_model, metrics = train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
        
        # Track the best model based on R² score
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = trained_model
            best_metrics = metrics
    
    # Save the best model
    if best_model:
        print("Saving the best model...")
        best_model_path = "models/trained_models/best_model.pkl"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)  # Ensure the directory exists
        mlflow.sklearn.save_model(best_model, best_model_path)
        print(f"Best model saved at: {best_model_path}")
        print("Best model metrics:")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("No model was trained successfully.")

if __name__ == "__main__":
    main()
