import pandas as pd


INPUT_FILE = "data/amazon_help.csv"
OUTPUT_FILE = "data/amazon_conversations.csv"


# Load AmazonHelp data
df = pd.read_csv(INPUT_FILE)


# Create a lookup:
# tweet_id -> tweet text
tweet_lookup = dict(
    zip(
        df["tweet_id"].astype(str),
        df["text"].fillna("").astype(str)
    )
)


pairs = []


for _, row in df.iterrows():

    # We only want customer messages
    if row["inbound"] != True:
        continue

    customer_text = str(row["text"])

    # response_tweet_id tells us which tweet
    # was sent as a response
    response_id = str(row["response_tweet_id"])

    if response_id == "nan" or response_id == "":
        continue

    # A customer tweet can have multiple response IDs
    response_ids = response_id.split(",")

    for response_id in response_ids:

        response_id = response_id.strip()

        if response_id in tweet_lookup:

            reply_text = tweet_lookup[response_id]

            pairs.append({
                "customer_text": customer_text,
                "historical_reply": reply_text
            })


# Convert to DataFrame
pairs_df = pd.DataFrame(pairs)


# Save
pairs_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Conversation pairs created!")
print("Total pairs:", len(pairs_df))
print()
print(pairs_df.head(10).to_string(index=False))