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
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Navigation logic
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'predict':
        show_predict_page()
    elif st.session_state.current_page == 'explore':
        show_data_exploration_page()

def show_home_page():
    # Custom CSS for beautiful gradient background and styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
    }
    .home-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .home-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    /* Simple Main Predict Button */
    .main-predict-button {
        background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 25px 50px !important;
        font-size: 28px !important;
        font-weight: 600 !important;
        text-align: center !important;
        width: 100% !important;
        height: 100px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: opacity 0.2s ease !important;
    }
    .main-predict-button:hover {
        opacity: 0.9 !important;
    }
    
    /* Simple Professional Secondary Buttons */
    .secondary-button {
        background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        text-align: center !important;
        transition: opacity 0.2s ease !important;
        width: 100% !important;
        height: 60px !important;
    }
    .secondary-button:hover {
        opacity: 0.9 !important;
    }
    
    /* Simple Explore Data Button */
    .explore-button {
        background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        text-align: center !important;
        transition: opacity 0.2s ease !important;
        width: 100% !important;
        height: 60px !important;
    }
    .explore-button:hover {
        opacity: 0.9 !important;
    }
    
    /* CTA Section Enhancement */
    .cta-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 25px;
        padding: 50px 30px;
        margin: 40px 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .cta-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }
    .cta-subtitle {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    
    /* Button Container */
    .button-container {
        margin: 30px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <div class="home-title">🚦 TRAFFIC PREDICTION</div>
        <div class="home-subtitle">
            Predict traffic flow with advanced machine learning algorithms.<br>
            Get accurate forecasts for better traffic management and planning.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("## ✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>Using XGBoost machine learning model trained on historical traffic data to provide reliable forecasts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Visual Analytics</h3>
            <p>Interactive charts and graphs to visualize traffic patterns and understand peak hours.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⏰ Time-based Analysis</h3>
            <p>Analyze traffic patterns by hour, day, week, and seasonal variations for comprehensive insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Data Exploration</h3>
            <p>Explore historical traffic data with advanced filtering and correlation analysis tools.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Call-to-action section
    st.markdown("""
    <div class="cta-section">
        <div class="cta-title">Ready to predict traffic?</div>
        <div class="cta-subtitle">
            Experience the power of AI-driven traffic forecasting<br>
            Get instant, accurate predictions with our advanced system
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the buttons with enhanced styling
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        # MASSIVE Professional Main Predict Button
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        # Apply simple button styles matching your theme
        st.markdown("""
        <style>
        /* Simple Main START PREDICTING Button */
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button:first-child,
        .stButton > button,
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 15px !important;
            padding: 25px 50px !important;
            font-size: 28px !important;
            font-weight: 600 !important;
            text-align: center !important;
            width: 100% !important;
            height: 100px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: opacity 0.2s ease !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stButton"] > button:first-child:hover,
        .stButton > button:hover {
            opacity: 0.9 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # HUGE Main predict button
        if st.button("🚀 START PREDICTING", key="start_btn", help="Click to go to prediction page", 
                    type="primary", use_container_width=True):
            st.session_state.current_page = 'predict'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Secondary buttons with professional styling
        col_a, col_b = st.columns(2, gap="large")
        
        # Simple secondary buttons styling
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            text-align: center !important;
            width: 100% !important;
            height: 60px !important;
            transition: opacity 0.2s ease !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            opacity: 0.9 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col_a:
            if st.button("📊 Explore Data", key="explore_btn", type="secondary", use_container_width=True):
                st.session_state.current_page = 'explore'
                st.rerun()
        
        # Simple Quick Predict button styling
        st.markdown("""
        <style>
        .quick-predict-btn button {
            background: linear-gradient(135deg, #4facfe 0%, #667eea 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            text-align: center !important;
            width: 100% !important;
            height: 60px !important;
            transition: opacity 0.2s ease !important;
        }
        .quick-predict-btn button:hover {
            opacity: 0.9 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown('<div class="quick-predict-btn">', unsafe_allow_html=True)
            if st.button("🔮 Quick Predict", key="predict_btn", use_container_width=True):
                st.session_state.current_page = 'predict'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def show_predict_page():
    # Navigation back to home
    if st.button("← Back to Home", key="back_home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    # Add the header with image and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        try:
            st.image("analytics_icon.png", width=150)
        except:
            st.markdown("# 📊")
    
    with col2:
        st.title("Traffic Prediction App")
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.warning("Please ensure model and data files are available.")
        
        uploaded_model = st.file_uploader("Upload trained model file (pkl)", type=['pkl'])
        if uploaded_model is not None:
            with open('traffic_prediction_model.pkl', 'wb') as f:
                f.write(uploaded_model.getvalue())
            st.success("Model uploaded successfully! Please refresh the page.")
            
        uploaded_data = st.file_uploader("Upload traffic data (csv)", type=['csv'])
        if uploaded_data is not None:
            data = pd.read_csv(uploaded_data)
            data.to_csv('traffic.csv', index=False)
            st.success("Data uploaded successfully! Please refresh the page.")
        
        return
    
    # Filter data for Junction 1
    j_1 = data[data['Junction'] == 1].copy()
    j_1 = create_features(j_1)
    
    # Define the features used in the model
    FEATURES = ['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    
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
                fig, ax = plt.subplots(figsize=(10, 4))  # Made smaller: was (12, 6)
                sns.lineplot(x='hour', y='Vehicles', data=selected_date_data, ax=ax, marker='o', linewidth=2, markersize=4)
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
                fig, ax = plt.subplots(figsize=(10, 4))  # Made smaller: was (12, 6)
                ax.plot(hours, hourly_predictions, marker='o', linewidth=2, markersize=4)
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

def show_data_exploration_page():
    # Navigation back to home
    if st.button("← Back to Home", key="back_home_explore"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    # Add the header with image and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        try:
            st.image("analytics_icon.png", width=150)
        except:
            st.markdown("# 📊")
    
    with col2:
        st.title("Traffic Prediction App")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.warning("Please ensure data file is available.")
        uploaded_data = st.file_uploader("Upload traffic data (csv)", type=['csv'])
        if uploaded_data is not None:
            data = pd.read_csv(uploaded_data)
            data.to_csv('traffic.csv', index=False)
            st.success("Data uploaded successfully! Please refresh the page.")
        return
    
    # Filter data for Junction 1
    j_1 = data[data['Junction'] == 1].copy()
    j_1 = create_features(j_1)
    
    # Define the features used in the model
    FEATURES = ['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    
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
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="explore_start")
    with col2:
        end_date = st.date_input("End Date", min(max_date, start_date + timedelta(days=7)), min_value=start_date, max_value=max_date, key="explore_end")
    
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
    page_number = st.number_input("Page", min_value=1, max_value=len(j_1)//page_size + 1, value=1, key="explore_page")
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    st.write(j_1.iloc[start_idx:end_idx])

if __name__ == "__main__":
    main()