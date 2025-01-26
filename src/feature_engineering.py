import pandas as pd
from geopy.distance import geodesic

def calculate_distance(row):
    """Calculate distance between store and drop locations."""
    store_coords = (row['Store_Latitude'], row['Store_Longitude'])
    drop_coords = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(store_coords, drop_coords).km

def add_time_features(df):
    """Extract time-based features."""
    df['Order_Hour'] = pd.to_datetime(df['Order_Time']).dt.hour
    df['Order_Day'] = pd.to_datetime(df['Order_Date']).dt.day_name()
    return df

if __name__ == "__main__":
    processed_data_path = "../data/processed/cleaned_data.csv"
    data = pd.read_csv(processed_data_path)

    # Add features
    data['Distance'] = data.apply(calculate_distance, axis=1)
    data = add_time_features(data)

    data.to_csv(processed_data_path, index=False)
    print(f"Feature-engineered data saved to {processed_data_path}.")
