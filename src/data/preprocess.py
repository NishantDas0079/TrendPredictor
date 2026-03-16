import pandas as pd
from .load_data import load_raw_data

def get_series_data(unique_id=0):
    """Filter data for a given unique_id and return sorted by ds."""
    df = load_raw_data()
    series_df = df[df['unique_id'] == unique_id].copy()
    series_df = series_df.sort_values('ds')
    # Prophet requires columns 'ds' and 'y'
    return series_df[['ds', 'y']]

def train_test_split(series_df, test_size=0.2):
    """Split time series chronologically."""
    split_idx = int(len(series_df) * (1 - test_size))
    train = series_df.iloc[:split_idx]
    test = series_df.iloc[split_idx:]
    return train, test

if __name__ == "__main__":
    df = get_series_data(unique_id=0)
    train, test = train_test_split(df)
    print(f"Total: {len(df)}, Train: {len(train)}, Test: {len(test)}")