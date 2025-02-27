import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = "data/processed/engineered_data.csv"  # Replace with the correct file path
data = pd.read_csv(file_path, encoding='utf-8', na_values=['NaN', 'null'])

# Display the first few rows of the data to inspect
print("First few rows of the data:")
print(data.head())

# Check the shape (rows, columns) of the dataset
print("\nData shape (rows, columns):", data.shape)

# Check for any missing values in the dataset
print("\nMissing values in each column:")
print(data.isnull().sum())

# Basic data types information
print("\nData types of each column:")
print(data.dtypes)

# Basic statistics for numerical columns
print("\nBasic statistics of the dataset:")
print(data.describe())

# Plotting setup
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))

# Plot 1 - Bar chart of Delivery Time by Vehicle
vehicle_delivery_time = data.groupby('Vehicle')['Delivery_Time'].mean().sort_values(ascending=False)
axs[0, 0].bar(vehicle_delivery_time.index, vehicle_delivery_time.values, color='skyblue')
axs[0, 0].set_title('Average Delivery Time by Vehicle')
axs[0, 0].set_xlabel('Vehicle')
axs[0, 0].set_ylabel('Average Delivery Time')
axs[0, 0].tick_params(axis='x', rotation=45)

# Plot 2 - Line plot of Delivery Time by Order Hour
order_hour_delivery_time = data.groupby('Order_Hour')['Delivery_Time'].mean()
axs[0, 1].plot(order_hour_delivery_time.index, order_hour_delivery_time.values, marker='o', color='green')
axs[0, 1].set_title('Average Delivery Time by Order Hour')
axs[0, 1].set_xlabel('Order Hour')
axs[0, 1].set_ylabel('Average Delivery Time')

# Plot 3 - Bar chart of Delivery Time by Traffic Impact Score
traffic_delivery_time = data.groupby('Traffic_Impact_Score')['Delivery_Time'].mean()
axs[1, 0].bar(traffic_delivery_time.index, traffic_delivery_time.values, color='orange')
axs[1, 0].set_title('Average Delivery Time by Traffic Impact Score')
axs[1, 0].set_xlabel('Traffic Impact Score')
axs[1, 0].set_ylabel('Average Delivery Time')

# Plot 4 - Area Impact Score vs Delivery Time (Line plot)
area_impact_delivery_time = data.groupby('Area_Impact_Score')['Delivery_Time'].mean()
axs[1, 1].plot(area_impact_delivery_time.index, area_impact_delivery_time.values, marker='s', color='purple')
axs[1, 1].set_title('Average Delivery Time by Area Impact Score')
axs[1, 1].set_xlabel('Area Impact Score')
axs[1, 1].set_ylabel('Average Delivery Time')

# Plot 5 - Bar chart of Delivery Time by Weather
weather_delivery_time = data.groupby('Weather')['Delivery_Time'].mean().sort_values(ascending=False)
axs[2, 0].bar(weather_delivery_time.index, weather_delivery_time.values, color='brown')
axs[2, 0].set_title('Average Delivery Time by Weather')
axs[2, 0].set_xlabel('Weather')
axs[2, 0].set_ylabel('Average Delivery Time')
axs[2, 0].tick_params(axis='x', rotation=45)

# Plot 6 - Line plot of Delivery Time by Order Day
order_day_delivery_time = data.groupby('Order_Day')['Delivery_Time'].mean()
axs[2, 1].plot(order_day_delivery_time.index, order_day_delivery_time.values, marker='x', color='blue')
axs[2, 1].set_title('Average Delivery Time by Order Day')
axs[2, 1].set_xlabel('Order Day')
axs[2, 1].set_ylabel('Average Delivery Time')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
