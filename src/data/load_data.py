import pandas as pd
import os

def load_raw_data(filepath=None):
    if filepath is None:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to project root, then into data/raw
        project_root = os.path.dirname(os.path.dirname(script_dir))
        filepath = os.path.join(project_root, 'data', 'raw', 'y_amazon-google-large.csv')
    
    df = pd.read_csv(filepath)
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Date range:", df['ds'].min(), "to", df['ds'].max())
    print("Unique series (unique_id):", df['unique_id'].nunique())
    print(df.head())