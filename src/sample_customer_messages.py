import pandas as pd

FILE_PATH = "data/amazon_help.csv"

df = pd.read_csv(FILE_PATH)

customers = df[df["inbound"] == True]

print("Total customer tweets:", len(customers))

print("\n==============================")
print("500 CUSTOMER MESSAGES")
print("==============================\n")

sample = customers["text"].sample(
    n=min(500, len(customers)),
    random_state=42
)

for i, text in enumerate(sample, start=1):
    print(f"{i}. {text}")
    print("-" * 80)