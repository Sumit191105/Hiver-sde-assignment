import pandas as pd

FILE_PATH = "data/training_sample.csv"

df = pd.read_csv(FILE_PATH)

print("Total messages:", len(df))

print("\n==============================")
print("SAMPLE TRAINING MESSAGES")
print("==============================\n")

for i, text in enumerate(df["text"].head(50), start=1):
    print(f"{i}. {text}")
    print("-" * 80)