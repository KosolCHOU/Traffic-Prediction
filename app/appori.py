import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Traffic Prediction App",
    page_icon="🚦",
    layout="wide"
)

# Function to create time-based features (same as in your training notebook)
def create_features(df):
    """
    Creates time series features from datetime index
    """
    df = df.copy()
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['dayofmonth'] = df['DateTime'].dt.day
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week
    
    return df

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Try to load the saved model
        with open('traffic_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train and save the model first.")
        return None

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the traffic data
        df = pd.read_csv('traffic.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload traffic.csv file.")
        return None

# Function to make predictions
def predict_traffic(model, input_features):
    # Make prediction
    prediction = model.predict(input_features)
    return prediction[0]

# Main app
def main():
    # Add the header with image and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Add your analytics image here - bigger size
        # Save your image as 'analytics_icon.png' in the same folder as your app
        try:
            st.image("analytics_icon.png", width=400)
        except:
            # Fallback to emoji if image not found
            st.markdown("# 📊")
    
    with col2:
        st.title("Traffic Prediction App")
    
    # Create sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Predict", "Data Exploration"])
    
    # Load model
    model = load_model()
    
    # Load data for visualization and stats
    data = load_data()
    
    if model is None or data is None:
        st.warning("Please ensure model and data files are available.")
        
        # Option to upload model file
        uploaded_model = st.file_uploader("Upload trained model file (pkl)", type=['pkl'])
        if uploaded_model is not None:
            with open('traffic_prediction_model.pkl', 'wb') as f:
                f.write(uploaded_model.getvalue())
            st.success("Model uploaded successfully! Please refresh the page.")
            
        # Option to upload data file
        uploaded_data = st.file_uploader("Upload traffic data (csv)", type=['csv'])
        if uploaded_data is not None:
            data = pd.read_csv(uploaded_data)
            data.to_csv('traffic.csv', index=False)
            st.success("Data uploaded successfully! Please refresh the page.")
        
        return
    
    # Filter data for Junction 1 (as in the training code)
    j_1 = data[data['Junction'] == 1].copy()
    j_1 = create_features(j_1)
    
    # Define the features used in the model
    FEATURES = ['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    
    if page == "Predict":
        st.header("Predict Traffic")
        
        st.subheader("Input Parameters")
        
        # Date and time inputs - restricted to 2017-2019
        prediction_date = st.date_input(
            "Select Date",
            value=datetime.date(2017, 1, 1),
            min_value=datetime.date(2017, 1, 1),
            max_value=datetime.date(2019, 12, 31)
        )
        
        prediction_time = st.time_input(
            "Select Time",
            datetime.time(12, 0)
        )
        
        # Combine date and time
        prediction_datetime = datetime.datetime.combine(prediction_date, prediction_time)
        
        # Create a dataframe with the datetime
        input_df = pd.DataFrame({'DateTime': [prediction_datetime]})
        
        # Create features
        input_df = create_features(input_df)
        
        # Select only the required features
        input_features = input_df[FEATURES]
        
        # Make prediction when the button is clicked
        if st.button("Predict Traffic"):
            with st.spinner('Predicting...'):
                # Get the prediction
                prediction = predict_traffic(model, input_features)
                
                # Display the prediction only (no historical average or difference)
                st.success(f"Predicted Traffic: **{int(prediction)}** vehicles")
                
                # Show traffic pattern chart below the prediction
                st.subheader(f"Traffic Pattern for {prediction_date}")
                
                # Filter data for the selected date
                selected_date_data = j_1[j_1['DateTime'].dt.date == prediction_date]
                
                if not selected_date_data.empty:
                    # Use actual historical data
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x='hour', y='Vehicles', data=selected_date_data, ax=ax, marker='o')
                    ax.set_xticks(range(0, 24))
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Number of Vehicles')
                    ax.set_title(f'Actual Hourly Traffic Pattern on {prediction_date}')
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight the selected hour
                    selected_hour = prediction_time.hour
                    ax.axvline(x=selected_hour, color='red', linestyle='--', alpha=0.7, label=f'Selected Hour ({selected_hour}:00)')
                    ax.legend()
                    
                    st.pyplot(fig)
                else:
                    # Generate predicted values for all 24 hours
                    st.info("No historical data available. Showing predicted traffic pattern for the entire day.")
                    
                    # Create predictions for all 24 hours
                    hourly_predictions = []
                    hours = list(range(24))
                    
                    for hour in hours:
                        # Create datetime for each hour
                        hour_datetime = datetime.datetime.combine(prediction_date, datetime.time(hour, 0))
                        hour_df = pd.DataFrame({'DateTime': [hour_datetime]})
                        hour_df = create_features(hour_df)
                        hour_features = hour_df[FEATURES]
                        
                        # Make prediction for this hour
                        hour_prediction = predict_traffic(model, hour_features)
                        hourly_predictions.append(hour_prediction)
                    
                    # Create the predicted traffic pattern chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(hours, hourly_predictions, marker='o', linewidth=2, markersize=6)
                    ax.set_xticks(range(0, 24))
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Predicted Vehicles')
                    ax.set_title(f'Predicted Hourly Traffic Pattern on {prediction_date}')
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight the selected hour
                    selected_hour = prediction_time.hour
                    ax.axvline(x=selected_hour, color='red', linestyle='--', alpha=0.7, label=f'Selected Hour ({selected_hour}:00)')
                    
                    # Highlight the predicted point for selected hour
                    ax.scatter([selected_hour], [hourly_predictions[selected_hour]], 
                             color='red', s=100, zorder=5, label=f'Your Prediction: {int(hourly_predictions[selected_hour])} vehicles')
                    
                    ax.legend()
                    st.pyplot(fig)
    
    elif page == "Data Exploration":
        st.header("Traffic Data Exploration")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(j_1['Vehicles'].describe())
        
        # Time series plot
        st.subheader("Traffic Over Time")
        
        # Allow user to select date range
        min_date = j_1['DateTime'].min().date()
        max_date = j_1['DateTime'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", min(max_date, start_date + timedelta(days=7)), min_value=start_date, max_value=max_date)
        
        # Filter data based on date range
        filtered_data = j_1[(j_1['DateTime'].dt.date >= start_date) & (j_1['DateTime'].dt.date <= end_date)]
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='DateTime', y='Vehicles', data=filtered_data, ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title(f'Traffic Volume from {start_date} to {end_date}')
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        corr = j_1[FEATURES + ['Vehicles']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        # Display raw data with pagination
        st.subheader("Raw Data Sample")
        page_size = 100
        page_number = st.number_input("Page", min_value=1, max_value=len(j_1)//page_size + 1, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        st.write(j_1.iloc[start_idx:end_idx])

if __name__ == "__main__":
    main()