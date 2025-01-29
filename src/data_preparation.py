import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_distance(row):
    """Calculate geodesic distance between Store and Drop locations."""
    try:
        store_location = (row['Store_Latitude'], row['Store_Longitude'])
        drop_location = (row['Drop_Latitude'], row['Drop_Longitude'])
        return geodesic(store_location, drop_location).km
    except Exception as e:
        logging.warning(f"Error calculating distance for row: {e}")
        return np.nan

def handle_missing_values(df):
    """Impute missing values with the median for numerical columns."""
    imputer = SimpleImputer(strategy='median')  # Use 'mean' for mean imputation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def clean_data(df):
    """Clean the dataset by handling missing values, duplicates, and calculating distances."""
    try:
        logging.info("Starting data cleaning...")
        initial_shape = df.shape

        # Remove duplicates
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

        # Handle missing values (using imputation)
        df = handle_missing_values(df)
        logging.info(f"Missing values imputed. Remaining missing: {df.isnull().sum().sum()}.")

        # Calculate distance (assuming these columns are present in your data)
        if {'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'}.issubset(df.columns):
            df['Distance'] = df.apply(calculate_distance, axis=1)
            logging.info("Distances calculated.")
        else:
            logging.warning("Latitude/Longitude columns missing. Distance not calculated.")

        # Drop rows with NaN in critical columns
        critical_columns = ['Delivery_Time', 'Distance']
        df.dropna(subset=critical_columns, inplace=True)

        logging.info(f"Final data shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise

# Example usage
if __name__ == "__main__":
    raw_data_path = "data/raw/amazon_delivery.csv"
    processed_data_path = "data/processed/cleaned_data.csv"

    try:
        data = pd.read_csv(raw_data_path)
        logging.info("Loaded raw data successfully.")

        cleaned_data = clean_data(data)
        cleaned_data.to_csv(processed_data_path, index=False)
        logging.info(f"Cleaned data saved to {processed_data_path}.")
    except Exception as e:
        logging.error(f"Failed to process the dataset: {e}")
