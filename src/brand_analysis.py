import pandas as pd
from collections import Counter

FILE_PATH = "data/twcs.csv"

brand_counts = Counter()

print("Analyzing dataset...")

for chunk in pd.read_csv(FILE_PATH, chunksize=100_000):
    
    # inbound=False means the tweet is from the company/brand
    company_tweets = chunk[chunk["inbound"] == False]

    # Count how many tweets each company account has
    brand_counts.update(company_tweets["author_id"])

print("\n==============================")
print("TOP 20 BRANDS")
print("==============================")

for brand, count in brand_counts.most_common(20):
    print(f"{brand:30} {count}")