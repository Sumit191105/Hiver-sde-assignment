import pandas as pd

FILE_PATH = "data/twcs.csv"

print("Reading dataset...")

df = pd.read_csv(FILE_PATH, nrows=10000)

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Rows loaded:", len(df))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())