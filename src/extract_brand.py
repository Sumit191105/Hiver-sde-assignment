import pandas as pd

FILE_PATH = "data/twcs.csv"
OUTPUT_FILE = "data/amazon_help.csv"

print("Extracting AmazonHelp data...")

first_chunk = True

for chunk in pd.read_csv(FILE_PATH, chunksize=100_000):

    # Keep AmazonHelp tweets and tweets directly related to AmazonHelp
    mask = (
        (chunk["author_id"] == "AmazonHelp")
        |
        (chunk["text"].str.contains("@AmazonHelp", case=False, na=False))
    )

    amazon = chunk[mask]

    if not amazon.empty:
        amazon.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

print(f"\nSaved AmazonHelp data to: {OUTPUT_FILE}")
