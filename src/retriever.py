import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = "data/amazon_conversations.csv"


# -----------------------------
# 1. Load historical conversations
# -----------------------------

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["customer_text", "historical_reply"]
)

df["customer_text"] = df["customer_text"].astype(str)
df["historical_reply"] = df["historical_reply"].astype(str)


print("Historical conversations:", len(df))


# -----------------------------
# 2. Convert customer messages
#    into TF-IDF vectors
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

customer_vectors = vectorizer.fit_transform(
    df["customer_text"]
)


# -----------------------------
# 3. Retrieval function
# -----------------------------

def retrieve_similar_messages(
    query,
    top_k=3
):

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    ).flatten()

    top_indices = similarities.argsort()[
        -top_k:
    ][::-1]

    results = []

    for index in top_indices:

        results.append({
            "customer_message":
                df.iloc[index]["customer_text"],

            "historical_reply":
                df.iloc[index]["historical_reply"],

            "similarity":
                round(
                    float(similarities[index]),
                    3
                )
        })

    return results