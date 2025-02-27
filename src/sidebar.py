import streamlit as st
import datetime

# Default values for missing inputs
DEFAULT_VALUES = {
    'Agent_Age': 30, 'Agent_Rating': 4.5, 'Store_Latitude': 22.745049, 'Store_Longitude': 75.892471,
    'Drop_Latitude': 22.765049, 'Drop_Longitude': 75.912471, 'Weather': 'Sunny', 'Traffic': 'Medium',
    'Vehicle': 'motorcycle', 'Area': 'Urban', 'Category': 'General', 'Order_Year': 2022, 'Order_Month': 3,
    'Order_Day': 1, 'Order_Hour': 12, 'Order_Minute': 30, 'Pickup_Hour': 12, 'Pickup_Minute': 45,
    'Pickup_Time': 15, 'Expected_Delivery_Time': 30, 'Delivery_SLA_Breach': 0, 'Weather_Impact_Score': 2,
    'Traffic_Impact_Score': 3, 'Area_Impact_Score': 2, 'Distance_km': 5.0,
    'Is_Weekend': 0, 'Order_Datetime': int(datetime.datetime.now().timestamp()),
    'Order_Weekday': 0, 'Order_to_Pickup_Duration': 15
}

CATEGORY_MAP = {
    'Clothing': 0, 'Electronics': 1, 'Sports': 2, 'Cosmetics': 3, 'Toys': 4, 
    'Shoes': 5, 'Apparel': 6, 'Snacks': 7, 'Outdoor': 8, 'Jewelry': 9, 
    'Kitchen': 10, 'Groceries': 11, 'Books': 12, 'Others': 13, 'Home': 14, 
    'Pet supplies': 15, 'Skin care': 16
}

def get_input_data():
    """Collects user inputs from the sidebar and returns a dictionary."""
    st.sidebar.title("Feature Selection")
    selected_options = st.sidebar.multiselect(
        "Select parameters to enter", 
        ["Order Details", "Agent Details", "Location Details", "Weather & Traffic", "All"], 
        default=["Order Details"]
    )

    input_data = {}

    if "Order Details" in selected_options or "All" in selected_options:
        st.sidebar.subheader("Order Details")
        input_data['Order_Hour'] = st.sidebar.slider("Order Hour", 0, 23, DEFAULT_VALUES['Order_Hour'])
        input_data['Order_Day'] = st.sidebar.slider("Order Day (0=Monday, 6=Sunday)", 0, 6, DEFAULT_VALUES['Order_Day'])
        input_data['Order_Minute'] = st.sidebar.slider("Order Minute", 0, 59, DEFAULT_VALUES['Order_Minute'])
        input_data['Pickup_Time'] = st.sidebar.slider("Pickup Time (min)", 0, 120, DEFAULT_VALUES['Pickup_Time'])
        input_data['Category'] = st.sidebar.selectbox(
            "Order Category", 
            list(CATEGORY_MAP.keys()),  # Dropdown options will match CATEGORY_MAP keys
            index=0  # Default selection
        )


    if "Agent Details" in selected_options or "All" in selected_options:
        st.sidebar.subheader("Agent Details")
        input_data['Agent_Age'] = st.sidebar.slider("Agent Age", 18, 70, DEFAULT_VALUES['Agent_Age'])
        input_data['Agent_Rating'] = st.sidebar.slider("Agent Rating", 1.0, 5.0, DEFAULT_VALUES['Agent_Rating'], step=0.1)

    if "Location Details" in selected_options or "All" in selected_options:
        st.sidebar.subheader("Location Details")
        input_data['Store_Latitude'] = st.sidebar.slider("Store Latitude", 0.0, 90.0, DEFAULT_VALUES['Store_Latitude'])
        input_data['Store_Longitude'] = st.sidebar.slider("Store Longitude", 0.0, 180.0, DEFAULT_VALUES['Store_Longitude'])
        input_data['Drop_Latitude'] = st.sidebar.slider("Drop Latitude", 0.0, 90.0, DEFAULT_VALUES['Drop_Latitude'])
        input_data['Drop_Longitude'] = st.sidebar.slider("Drop Longitude", 0.0, 180.0, DEFAULT_VALUES['Drop_Longitude'])
        input_data['Vehicle'] = st.sidebar.selectbox("Vehicle Type", ["motorcycle", "van", "bicycle"], index=0)
        input_data['Area'] = st.sidebar.selectbox("Delivery Area", ["Urban", "Suburban", "Rural"], index=0)

    if "Weather & Traffic" in selected_options or "All" in selected_options:
        st.sidebar.subheader("Weather & Traffic")
        input_data['Weather'] = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Stormy", "Sandstorms"], index=0)
        input_data['Traffic'] = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"], index=1)

    return input_data, selected_options
