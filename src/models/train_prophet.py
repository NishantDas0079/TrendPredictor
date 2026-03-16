import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import os
import json
from datetime import datetime

def _convert_to_serializable(obj):
    """Convert numpy types and pandas Timestamps to JSON‑serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj

def train_prophet(train_df, series_id=0, save_path=None, **kwargs):
    """Train Prophet model with optional hyperparameters."""
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        save_path = os.path.join(project_root, 'models', f'prophet_model_{series_id}.pkl')
    
    # Default hyperparameters
    params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False
    }
    params.update(kwargs)
    
    model = Prophet(**params)
    model.fit(train_df)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Prepare metadata with conversion
    metadata = {
        'series_id': _convert_to_serializable(series_id),
        'trained_on': datetime.now().isoformat(),
        'params': _convert_to_serializable(params),
        'train_start': _convert_to_serializable(train_df['ds'].min()),
        'train_end': _convert_to_serializable(train_df['ds'].max()),
        'num_rows': _convert_to_serializable(len(train_df))
    }
    
    metadata_path = save_path.replace('.pkl', '_metadata.json')
    # Write metadata safely: first to a temp file, then replace
    temp_path = metadata_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    os.replace(temp_path, metadata_path)  # atomic on Unix, may not be atomic on Windows but still safer
    
    print(f"Model saved to {save_path}")
    return model, metadata

def evaluate_model(model, test_df):
    """Make predictions on test period and compute errors."""
    future = model.make_future_dataframe(periods=len(test_df), include_history=False)
    forecast = model.predict(future)
    
    comparison = pd.merge(test_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    comparison['error'] = comparison['y'] - comparison['yhat']
    comparison['abs_error'] = abs(comparison['error'])
    
    return comparison

def load_model_and_metadata(series_id=0):
    """Load model and metadata for a given series if they exist and are valid."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, 'models', f'prophet_model_{series_id}.pkl')
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        except (pickle.UnpicklingError, json.JSONDecodeError, EOFError, KeyError) as e:
            print(f"Warning: Corrupted model/metadata for series {series_id}: {e}. Will retrain.")
            # Optionally delete corrupted files to avoid future issues
            # os.remove(model_path)
            # os.remove(metadata_path)
            return None, None
    else:
        return None, None