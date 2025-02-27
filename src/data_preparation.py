import pandas as pd
import numpy as np
import logging
import os
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    """Load raw data from CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    if df.empty:
        logging.warning("⚠️ Loaded data is empty. Check the file content.")
        return None

    logging.info(f"✅ Loaded raw data successfully from {file_path}. Shape: {df.shape}")
    return df

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def clean_data(df):
    """Perform data cleaning: Remove duplicates, handle missing values, and ensure schema consistency."""
    
    logging.info("🔍 Initial Data Overview:")
    logging.info(f"\n{df.info()}")

    # Drop 'Order_ID' column as it is not needed for model training
    if "Order_ID" in df.columns:
        df = df.drop(columns=["Order_ID"])
        logging.info("🧹 Dropped 'Order_ID' column.")

    # Handle 'Order_Date' and 'Order_Time' columns (to datetime)
    if "Order_Date" in df.columns and "Order_Time" in df.columns:
        df["Order_Datetime"] = pd.to_datetime(df["Order_Date"] + ' ' + df["Order_Time"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df["Order_Year"] = df["Order_Datetime"].dt.year
        df["Order_Month"] = df["Order_Datetime"].dt.month
        df["Order_Day"] = df["Order_Datetime"].dt.day
        df["Order_Hour"] = df["Order_Datetime"].dt.hour
        df["Order_Minute"] = df["Order_Datetime"].dt.minute
        logging.info("📅 Extracted features from 'Order_Date' and 'Order_Time'.")

    # Handle 'Pickup_Time' column properly
    if "Pickup_Time" in df.columns and "Order_Date" in df.columns:
        df["Pickup_Time"] = pd.to_datetime(df["Order_Date"] + " " + df["Pickup_Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        logging.info(f"❓ Missing values in 'Pickup_Time' after conversion: {df['Pickup_Time'].isna().sum()}")

        # Compute Order-to-Pickup duration
        df["Order_to_Pickup_Duration"] = (df["Pickup_Time"] - df["Order_Datetime"]).dt.total_seconds() / 60
        df["Pickup_Hour"] = df["Pickup_Time"].dt.hour
        df["Pickup_Minute"] = df["Pickup_Time"].dt.minute
        df = df.drop(columns=["Pickup_Time"])
        logging.info("📦 Extracted features from 'Pickup_Time'.")

    # Drop 'Order_Date' and 'Order_Time' (since we extracted needed features)
    if "Order_Date" in df.columns and "Order_Time" in df.columns:
        df = df.drop(columns=["Order_Date", "Order_Time"])
    
    # Compute distance using Haversine formula
    df["Distance_km"] = df.apply(lambda row: haversine(row["Store_Latitude"], row["Store_Longitude"],
                                                         row["Drop_Latitude"], row["Drop_Longitude"]), axis=1)
    logging.info("📏 Computed distance between store and drop location.")

    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    logging.info(f"🧹 Removed {initial_rows - len(df)} duplicate rows.")

    # Convert integer columns to float for consistency
    int_cols = df.select_dtypes(include=['int']).columns
    df[int_cols] = df[int_cols].astype('float64')
    logging.info(f"🔄 Converted integer columns {list(int_cols)} to float to maintain schema consistency.")

    # Strip whitespace from categorical columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
    logging.info("🔠 Stripped whitespace from categorical columns.")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Mean for numeric
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill with most frequent category

    logging.info(f"🛠 Missing values imputed. Remaining missing values count: {df.isnull().sum().sum()}")

    logging.info("✅ Data cleaning completed successfully.")
    return df

if __name__ == "__main__":
    raw_data_path = "data/raw/amazon_delivery.csv"  # Update with actual file path
    cleaned_data_path = "data/processed/cleaned_data.csv"  # Update with desired output file path

    df = load_data(raw_data_path)
    if df is not None:
        df_cleaned = clean_data(df)
        df_cleaned.to_csv(cleaned_data_path, index=False)
        logging.info(f"📂 Cleaned data saved to {cleaned_data_path}.")
