import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_future(model, periods=30, freq='H'):
    future = model.make_future_dataframe(periods=periods, include_history=True)
    forecast = model.predict(future)
    return forecast

def detect_anomalies(eval_df):
    """
    Detect anomalies where actual values fall outside forecast confidence intervals.
    eval_df: DataFrame from evaluate_model (must contain 'y', 'yhat_lower', 'yhat_upper')
    Returns DataFrame of anomalies (subset of eval_df).
    """
    return eval_df[(eval_df['y'] < eval_df['yhat_lower']) | (eval_df['y'] > eval_df['yhat_upper'])]