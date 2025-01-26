import pandas as pd
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Current working directory:", os.getcwd())

def load_data(filepath):
    """Load dataset from the specified file path."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Perform basic cleaning: remove duplicates, handle missing values."""
    df = df.drop_duplicates()
    df.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    return df

if __name__ == "__main__":
    raw_data_path = "data/raw/amazon_delivery.csv"
    processed_data_path = "data/processed/cleaned_data.csv"

    data = load_data(raw_data_path)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(processed_data_path, index=False)
    print(f"Cleaned data saved to {processed_data_path}.")
