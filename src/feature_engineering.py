import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def estimate_weather_impact(weather_condition):
    impact_mapping = {"stormy": 5, "sandstorms": 5, "fog": 4, "windy": 3, "cloudy": 2, "sunny": 0}
    return impact_mapping.get(str(weather_condition).strip().lower(), 1)

def estimate_traffic_impact(traffic_condition):
    traffic_mapping = {"high": 5, "jam": 4, "medium": 3, "low": 2, "other": 1}
    return traffic_mapping.get(str(traffic_condition).strip().lower(), 1)

def encode_area(area):
    area_mapping = {"metropolitan": 4, "urban": 3, "semi-urban": 2, "others": 1}
    return area_mapping.get(str(area).strip().lower(), 1)

def add_sla_breach_flag(df):
    if {'Expected_Delivery_Time', 'Delivery_Time'}.issubset(df.columns):
        df['Delivery_SLA_Breach'] = (df['Delivery_Time'] > df['Expected_Delivery_Time']).astype(int)
        logging.info("✅ Added SLA breach flag.")
    return df

def add_impact_scores(df):
    if 'Weather' in df.columns:
        df['Weather_Impact_Score'] = df['Weather'].apply(estimate_weather_impact)
    if 'Traffic' in df.columns:
        df['Traffic_Impact_Score'] = df['Traffic'].apply(estimate_traffic_impact)
    if 'Area' in df.columns:
        df['Area_Impact_Score'] = df['Area'].apply(encode_area)
    logging.info("✅ Added impact scores.")
    return df

def add_pickup_time(df):
    if "Pickup_Time" not in df.columns and "Pickup_Hour" in df.columns and "Pickup_Minute" in df.columns:
        df["Pickup_Time"] = df["Pickup_Hour"] * 60 + df["Pickup_Minute"]
    logging.info("✅ Processed Pickup Time.")
    return df

def add_expected_delivery_time(df):
    if 'Category' in df.columns and 'Delivery_Time' in df.columns:
        category_mean_times = df.groupby('Category')['Delivery_Time'].mean()
        df['Expected_Delivery_Time'] = df['Category'].map(category_mean_times)
        logging.info("✅ Added Expected Delivery Time.")
    return df

def extract_datetime_features(df):
    if 'Order_Datetime' in df.columns:
        df = df.copy()  # Ensure we work on a proper copy to avoid SettingWithCopyWarning
        df['Order_Datetime'] = pd.to_datetime(df['Order_Datetime'], errors='coerce')

        # Drop rows where datetime parsing failed
        df.dropna(subset=['Order_Datetime'], inplace=True)

        # Extract datetime features safely
        df.loc[:, 'Order_Year'] = df['Order_Datetime'].dt.year
        df.loc[:, 'Order_Month'] = df['Order_Datetime'].dt.month
        df.loc[:, 'Order_Day'] = df['Order_Datetime'].dt.day
        df.loc[:, 'Order_Hour'] = df['Order_Datetime'].dt.hour
        df.loc[:, 'Order_Minute'] = df['Order_Datetime'].dt.minute
        df.loc[:, 'Order_Weekday'] = df['Order_Datetime'].dt.weekday.astype('float64')  # Convert to float immediately # Monday=0, Sunday=6
        df.loc[:, 'Is_Weekend'] = (df['Order_Weekday'] >= 5).astype(float)  # Ensure float type


        logging.info("✅ Extracted datetime features after handling NaNs.")

    return df

def fill_missing_values(df):
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        logging.warning(f"⚠️ Missing values before filling:\n{missing_values[missing_values > 0]}")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical with mode
            else:
                df[col] = df[col].fillna(df[col].median())  # Fill numeric with median

    missing_values_after = df.isnull().sum()
    if missing_values_after.sum() > 0:
        logging.error(f"❌ NaN values still present after filling:\n{missing_values_after[missing_values_after > 0]}")
        df.dropna(inplace=True)  # Drop any remaining NaN values

    logging.info("✅ Missing values filled completely.")
    return df

def convert_int_to_float(df):
    """ Ensure all integer columns are converted to float64 to prevent MLflow issues. """
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype('float64')
        logging.info(f"✅ Converted integer columns to float64: {list(int_cols)}")
    return df

def feature_engineering(df):
    """ Apply all feature engineering steps in a structured order. """
    
    # Step 1: Fill Missing Values
    df = fill_missing_values(df)

    # Step 2: Apply Feature Engineering
    df = extract_datetime_features(df)
    df = add_pickup_time(df)
    df = add_expected_delivery_time(df)
    df = add_sla_breach_flag(df)
    df = add_impact_scores(df)

    # Step 3: Convert Integers to Float64
    df = convert_int_to_float(df)

    logging.info("✅ Feature engineering completed.")
    return df

if __name__ == "__main__":
    processed_data_path = "data/processed/cleaned_data.csv"
    feature_data_path = "data/processed/engineered_data.csv"

    try:
        data = pd.read_csv(processed_data_path)
        logging.info(f"🔍 Loaded data from {processed_data_path}.")
        
        engineered_data = feature_engineering(data)
        
        engineered_data.to_csv(feature_data_path, index=False)
        logging.info(f"📂 Processed data saved to {feature_data_path}.")
        
    except FileNotFoundError as e:
        logging.error(f"❌ File not found: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
