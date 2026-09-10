import pandas as pd

FILE_PATH = "data/amazon_help.csv"
OUTPUT_FILE = "data/training_sample.csv"

print("Loading AmazonHelp data...")

df = pd.read_csv(FILE_PATH)

# Only customer messages
customers = df[df["inbound"] == True].copy()

# Take a random sample
sample = customers.sample(
    n=min(5000, len(customers)),
    random_state=42
)

# Keep only useful columns
sample = sample[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
]

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Created training sample.")
print("Rows:", len(sample))
print("Saved to:", OUTPUT_FILE)