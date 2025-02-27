import os
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from mlflow.models.signature import infer_signature

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("Amazon Delivery Time Prediction")

# Define feature columns and target column
FEATURE_COLUMNS = [
    "Agent_Age", "Agent_Rating", "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude",
    "Weather", "Traffic", "Vehicle", "Area", "Category", "Order_Datetime", "Order_Year", "Order_Month",
    "Order_Day", "Order_Hour", "Order_Minute", "Order_to_Pickup_Duration", "Pickup_Hour", "Pickup_Minute",
    "Distance_km", "Order_Weekday", "Is_Weekend", "Weather_Impact_Score", "Traffic_Impact_Score", "Area_Impact_Score"
]
TARGET_COLUMN = "Delivery_Time"

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully.")
    
    # Convert Order_Datetime to Unix timestamp
    if "Order_Datetime" in df.columns:
        df["Order_Datetime"] = pd.to_datetime(df["Order_Datetime"], errors="coerce").astype('int64') // 10**9
    
    return df

# Encode categorical features
def encode_categorical_columns(df):
    categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
    df[categorical_cols] = df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)
    return df

# Split data into train and test sets
def split_data(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Model evaluation metrics
def evaluate_model(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

# XGBoost Hyperparameter tuning
def hyperparameter_tuning_xgb(x_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True)
        }
        model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42, n_jobs=-1)
        return cross_val_score(model, x_train, y_train, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("✅ XGBoost Hyperparameter tuning completed.")
    return xgb.XGBRegressor(**study.best_params, random_state=42, objective='reg:squarederror')

# Train and log models with MLflow, save models for comparison
import joblib

def train_and_log_model(model_name, model, x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    print(f"\nTraining {model_name}...")
    with mlflow.start_run(run_name=model_name):
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path=model_name, signature=infer_signature(x_test, y_pred))

        # ✅ Save model along with feature names
        model_data = {"model": pipeline, "feature_names": x_train.columns.tolist()}
        model_path = f"models/{model_name.replace(' ', '_')}.pkl"
        joblib.dump(model_data, model_path)
        
        print(f"✅ {model_name} trained and saved at {model_path}.")
        return pipeline, metrics


# Main function
def main():
    try:
        file_path = "data/processed/engineered_data.csv"
        
        df = load_data(file_path)
        df_encoded = encode_categorical_columns(df)
        x_train, x_test, y_train, y_test = split_data(df_encoded)
        
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(alpha=0.1),
            "Ridge Regression": Ridge(alpha=1.0),
            "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        for name, model in models.items():
            _, metrics = train_and_log_model(name, model, x_train, x_test, y_train, y_test)
            model_results[name] = metrics  

        # XGBoost with Hyperparameter Tuning
        best_xgb_model = hyperparameter_tuning_xgb(x_train, y_train)
        _, xgb_metrics = train_and_log_model("XGBoost", best_xgb_model, x_train, x_test, y_train, y_test)
        model_results["XGBoost"] = xgb_metrics  

        # Save model performance comparison
        results_df = pd.DataFrame(model_results).T  
        results_df.to_csv("models/model_performance_comparison.csv", index=True)
        print("\n📊 Model Performance Comparison saved: models/model_performance_comparison.csv\n")
        print(results_df.sort_values(by="r2", ascending=False))  

                # Find the best model based on R2 score
        best_model_name = results_df['r2'].idxmax()
        best_model_r2 = results_df.loc[best_model_name, 'r2']
        print(f"\n🏆 Best Model: {best_model_name} with R² Score: {best_model_r2:.4f}\n")
    
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
