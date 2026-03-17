import pandas as pd
df = pd.read_csv("data/raw/y_amazon-google-large.csv", nrows=10000)
df.to_csv("data/raw/sample.csv", index=False)
print("Sample dataset created at data/raw/sample.csv")