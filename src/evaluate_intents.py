import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from gemini_intent_classifier import classify_intent


AUDIT_FILE = "data/golden_audit.csv"
OUTPUT_FILE = "data/intent_evaluation.csv"


# Load evaluation data
df = pd.read_csv(AUDIT_FILE)

df = df.dropna(
    subset=["text", "human_intent"]
)

df["text"] = df["text"].astype(str)
df["human_intent"] = df["human_intent"].astype(str)

print("Evaluation examples:", len(df))


# Load previous results
if os.path.exists(OUTPUT_FILE):

    previous_df = pd.read_csv(OUTPUT_FILE)

    cache = dict(
        zip(
            previous_df["text"].astype(str),
            previous_df["predicted_intent"].astype(str)
        )
    )

    print(
        "Cached predictions found:",
        len(cache)
    )

else:

    cache = {}

    print("No cached predictions found.")


# Run predictions
for index, row in df.iterrows():

    message = row["text"]

    # Use cached result if available
    if message in cache:

        print(
            f"[{index + 1}/{len(df)}] "
            "Using cached result"
        )

        continue

    print(
        f"\n[{index + 1}/{len(df)}] "
        "Calling Gemini..."
    )

    try:

        predicted_intent = classify_intent(
            message
        )

        cache[message] = predicted_intent

        print(
            "Predicted:",
            predicted_intent
        )

    except Exception as e:

        print("\nGemini API error:")
        print(e)

        print(
            "\nStopping evaluation safely."
        )

        break


# Build results only for completed predictions
results = []

for _, row in df.iterrows():

    message = row["text"]

    if message in cache:

        results.append({
            "tweet_id": row["tweet_id"],
            "text": message,
            "human_intent": row["human_intent"],
            "predicted_intent": cache[message]
        })


results_df = pd.DataFrame(results)


# Save results
results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nCompleted predictions: "
    f"{len(results_df)}/{len(df)}"
)


# Calculate metrics only for completed predictions
if len(results_df) > 0:

    accuracy = accuracy_score(
        results_df["human_intent"],
        results_df["predicted_intent"]
    )

    macro_f1 = f1_score(
        results_df["human_intent"],
        results_df["predicted_intent"],
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        results_df["human_intent"],
        results_df["predicted_intent"],
        average="weighted",
        zero_division=0
    )

    print("\n==============================")
    print("GEMINI INTENT EVALUATION")
    print("==============================")

    print(
        f"\nAccuracy: {accuracy:.3f}"
    )

    print(
        f"Macro F1: {macro_f1:.3f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.3f}"
    )

    print(
        "\nDetailed classification report:\n"
    )

    print(
        classification_report(
            results_df["human_intent"],
            results_df["predicted_intent"],
            zero_division=0
        )
    )

else:

    print(
        "\nNo predictions available yet."
    )


print(
    f"\nResults saved to: {OUTPUT_FILE}"
)