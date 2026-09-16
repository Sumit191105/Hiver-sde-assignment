import os
import pandas as pd

from gemini_intent_classifier import classify_intent


AUDIT_FILE = "data/golden_audit.csv"
OUTPUT_FILE = "data/intent_evaluation.csv"

# Maximum NEW Gemini API calls in one run.
# This keeps us safely below the free-tier per-minute limit.
MAX_NEW_CALLS = 4


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, macro F1 and weighted F1
    without scikit-learn.
    """

    labels = sorted(set(y_true) | set(y_pred))

    total = len(y_true)

    correct = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    f1_scores = []
    supports = []

    for label in labels:

        true_positive = sum(
            true == label and pred == label
            for true, pred in zip(y_true, y_pred)
        )

        false_positive = sum(
            true != label and pred == label
            for true, pred in zip(y_true, y_pred)
        )

        false_negative = sum(
            true == label and pred != label
            for true, pred in zip(y_true, y_pred)
        )

        support = sum(
            true == label
            for true in y_true
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        f1_scores.append(f1)
        supports.append(support)

    macro_f1 = (
        sum(f1_scores) / len(f1_scores)
        if len(f1_scores) > 0
        else 0.0
    )

    total_support = sum(supports)

    weighted_f1 = (
        sum(
            f1 * support
            for f1, support in zip(
                f1_scores,
                supports
            )
        ) / total_support
        if total_support > 0
        else 0.0
    )

    return accuracy, macro_f1, weighted_f1


# ============================================================
# 1. LOAD HUMAN-AUDITED DATA
# ============================================================

df = pd.read_csv(AUDIT_FILE)

df = df.dropna(
    subset=["text", "human_intent"]
)

df["text"] = df["text"].astype(str)
df["human_intent"] = df["human_intent"].astype(str)

print("Evaluation examples:", len(df))


# ============================================================
# 2. LOAD CACHED GEMINI PREDICTIONS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    previous_df = pd.read_csv(OUTPUT_FILE)

    if (
        "text" in previous_df.columns
        and "predicted_intent" in previous_df.columns
    ):

        cache = dict(
            zip(
                previous_df["text"].astype(str),
                previous_df["predicted_intent"].astype(str)
            )
        )

    else:
        cache = {}

    print(
        "Cached predictions found:",
        len(cache)
    )

else:

    cache = {}

    print(
        "No cached predictions found."
    )


# ============================================================
# 3. COUNT NEW CALLS
# ============================================================

new_calls = 0


# ============================================================
# 4. RUN GEMINI PREDICTIONS
# ============================================================

for index, row in df.iterrows():

    # Stop after MAX_NEW_CALLS new API requests.
    if new_calls >= MAX_NEW_CALLS:

        print(
            "\nReached per-run Gemini call limit."
        )

        break

    message = row["text"]

    # --------------------------------------------------------
    # Use cached result if already available
    # --------------------------------------------------------

    if message in cache:

        print(
            f"[{index + 1}/{len(df)}] "
            "Using cached result"
        )

        continue

    # --------------------------------------------------------
    # New Gemini API request
    # --------------------------------------------------------

    print(
        f"\n[{index + 1}/{len(df)}] "
        "Calling Gemini..."
    )

    try:

        predicted_intent = classify_intent(
            message
        )

        cache[message] = predicted_intent

        new_calls += 1

        print(
            "Predicted:",
            predicted_intent
        )

        # ----------------------------------------------------
        # Save immediately after every successful prediction
        # ----------------------------------------------------

        completed_rows = []

        for _, completed_row in df.iterrows():

            completed_message = str(
                completed_row["text"]
            )

            if completed_message in cache:

                completed_rows.append({
                    "tweet_id": completed_row["tweet_id"],
                    "text": completed_message,
                    "human_intent": completed_row["human_intent"],
                    "predicted_intent": cache[
                        completed_message
                    ]
                })

        pd.DataFrame(
            completed_rows
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )

    except Exception as e:

        print(
            "\nGemini API error:"
        )

        print(e)

        print(
            "\nStopping evaluation safely."
        )

        break


# ============================================================
# 5. BUILD COMPLETED RESULTS
# ============================================================

results = []

for _, row in df.iterrows():

    message = str(
        row["text"]
    )

    if message in cache:

        results.append({
            "tweet_id": row["tweet_id"],
            "text": message,
            "human_intent": row["human_intent"],
            "predicted_intent": cache[message]
        })


results_df = pd.DataFrame(
    results
)


# ============================================================
# 6. SAVE FINAL CACHE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nCompleted predictions: "
    f"{len(results_df)}/{len(df)}"
)

print(
    f"New Gemini calls this run: "
    f"{new_calls}"
)


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

if len(results_df) > 0:

    y_true = (
        results_df["human_intent"]
        .tolist()
    )

    y_pred = (
        results_df["predicted_intent"]
        .tolist()
    )

    accuracy, macro_f1, weighted_f1 = (
        calculate_metrics(
            y_true,
            y_pred
        )
    )

    print(
        "\n=============================="
    )

    print(
        "GEMINI INTENT EVALUATION"
    )

    print(
        "=============================="
    )

    print(
        f"\nAccuracy: {accuracy:.3f}"
    )

    print(
        f"Macro F1: {macro_f1:.3f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.3f}"
    )

else:

    print(
        "\nNo predictions available yet."
    )


# ============================================================
# 8. OUTPUT LOCATION
# ============================================================

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)	