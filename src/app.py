import streamlit as st
import plotly.graph_objects as go
from prophet.plot import plot_plotly
import os
import sys
import traceback
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import data modules
from src.data.load_data import load_raw_data
from src.data.preprocess import get_series_data, train_test_split

# Import forecasting modules with error handling
try:
    from src.models.train_prophet import train_prophet, evaluate_model, load_model_and_metadata
    from src.models.predict import load_model, predict_future, detect_anomalies
    FORECASTING_AVAILABLE = True
except ImportError as e:
    FORECASTING_AVAILABLE = False
    forecasting_error = str(e)

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="TrendPredictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS for Modern UI -------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Base styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container background – subtle gradient */
    .stApp {
        background: linear-gradient(145deg, #f8faff 0%, #eef2f6 100%);
    }

    /* Dark mode support (optional toggle) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(145deg, #1a2634 0%, #0f172a 100%);
            color: #e2e8f0;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f1f5f9;
        }
    }

    /* Header with animated gradient */
    .header-container {
        background: linear-gradient(90deg, #667eea, #764ba2, #6b8cff, #a463f5);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .header-container h1 {
        color: white;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header-container p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Card style for data containers */
    .card {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        color: white;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
        opacity: 0.9;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background: transparent;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        font-weight: 600;
        color: #1e293b;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #10b981, #059669);
        border: none;
    }

    /* Success/warning/info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 8px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown("""
<div class="header-container">
    <h1>📈 TrendPredictor</h1>
    <p>Intelligent time series forecasting with Prophet</p>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data
def load_all_data():
    return load_raw_data()

try:
    df_all = load_all_data()
    unique_ids = sorted(df_all['unique_id'].unique())
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error = str(e)

if not data_loaded:
    st.error(f"Failed to load data: {data_error}")
    st.stop()

# ------------------- Sidebar -------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
    st.markdown("## ⚙️ Controls")

    selected_series = st.selectbox("Select Series (unique_id)", unique_ids)
    forecast_periods = st.slider("Forecast Horizon (hours)", min_value=6, max_value=168, value=72, step=6)
    test_size = st.slider("Test Size (for evaluation)", min_value=0.1, max_value=0.3, value=0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 🔧 Hyperparameters")
    changepoint_prior = st.slider(
        "Changepoint Prior Scale",
        min_value=0.001, max_value=0.5, value=0.05, step=0.005,
        help="Controls trend flexibility"
    )
    seasonality_prior = st.slider(
        "Seasonality Prior Scale",
        min_value=0.1, max_value=20.0, value=10.0, step=0.5,
        help="Controls seasonality strength"
    )
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses Facebook Prophet to forecast time series data. "
        "You can adjust hyperparameters, load existing models, and detect anomalies."
    )

# ------------------- Main Content -------------------
# Get data for selected series
series_df = get_series_data(unique_id=selected_series)
train, test = train_test_split(series_df, test_size=test_size)

# Historical data card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(f"📊 Series {selected_series} - Historical Data")
st.line_chart(series_df.set_index('ds')['y'])
st.markdown("</div>", unsafe_allow_html=True)

# Model path
model_path = os.path.join('models', f'prophet_model_{selected_series}.pkl')

# Check forecasting availability
if not FORECASTING_AVAILABLE:
    st.error(f"Forecasting modules not available: {forecasting_error}")
    st.info("Please ensure prophet is installed and all modules are created.")
    st.stop()

# Model persistence
existing_model, existing_metadata = None, None
if os.path.exists(model_path):
    existing_model, existing_metadata = load_model_and_metadata(selected_series)
    if existing_model is not None:
        st.sidebar.success(f"✅ Existing model (trained on {existing_metadata['train_end'][:10]})")
        use_existing = st.sidebar.checkbox("Use existing model (skip retraining)", value=True)
    else:
        use_existing = False
        st.sidebar.info("Model file found but metadata missing. Will retrain.")
else:
    use_existing = False
    st.sidebar.info("No existing model. Will train a new one.")

# Train button
if st.button("🚀 Train & Forecast"):
    with st.spinner("Processing... This may take a minute."):
        try:
            if use_existing and existing_model is not None:
                model = existing_model
                st.info("Using existing model.")
                if len(test) > 0:
                    eval_df = evaluate_model(model, test)
                    mae = eval_df['abs_error'].mean()
                    rmse = (eval_df['error']**2).mean()**0.5
                else:
                    st.warning("No test data for evaluation.")
            else:
                # Train new model
                model, metadata = train_prophet(
                    train,
                    series_id=selected_series,
                    save_path=model_path,
                    changepoint_prior_scale=changepoint_prior,
                    seasonality_prior_scale=seasonality_prior,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality
                )
                eval_df = evaluate_model(model, test)
                mae = eval_df['abs_error'].mean()
                rmse = (eval_df['error']**2).mean()**0.5
                st.success("Model trained successfully!")

            # Display metrics in cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>MAE</h3>
                    <div class='value'>{mae:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>RMSE</h3>
                    <div class='value'>{rmse:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Train Size</h3>
                    <div class='value'>{len(train)}</div>
                </div>
                """, unsafe_allow_html=True)

            # Plot train/test/prediction
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual Test'))
            fig.add_trace(go.Scatter(x=eval_df['ds'], y=eval_df['yhat'], mode='lines', name='Predicted Test'))

            # Anomaly detection (FIXED: using eval_df directly)
            if len(test) > 0:
                anomalies = detect_anomalies(eval_df)  # Now expects eval_df
                if len(anomalies) > 0:
                    st.warning(f"⚠️ Detected {len(anomalies)} anomalies in test period!")
                    with st.expander("View Anomaly Details"):
                        # anomalies contains all columns from eval_df (including yhat)
                        st.dataframe(anomalies[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']])
                    fig.add_trace(go.Scatter(
                        x=anomalies['ds'],
                        y=anomalies['y'],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='Anomalies'
                    ))
                else:
                    st.success("✅ No anomalies detected.")

            st.plotly_chart(fig, use_container_width=True)

            # Future forecast
            forecast = predict_future(model, periods=forecast_periods)
            fig2 = plot_plotly(model, forecast)
            st.plotly_chart(fig2, use_container_width=True)

            # Download button
            csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"forecast_series_{selected_series}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.code(traceback.format_exc())
else:
    if os.path.exists(model_path):
        st.info("A trained model already exists. Click 'Train & Forecast' to use it or retrain with new settings.")