import pandas as pd

FILE_PATH = "data/amazon_help.csv"

df = pd.read_csv(FILE_PATH)

customers = df[df["inbound"] == True]

print("Total AmazonHelp customer tweets:", len(customers))

print("\n==============================")
print("SAMPLE CUSTOMER MESSAGES")
print("==============================\n")

for i, text in enumerate(customers["text"].head(100), start=1):
    print(f"{i}. {text}")
    print("-" * 80)