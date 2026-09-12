import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


TRAIN_FILE = "data/golden_set_labeled.csv"
AUDIT_FILE = "data/golden_audit.csv"


# -----------------------------
# 1. Load datasets
# -----------------------------

def load_training_data():

    train_df = pd.read_csv(TRAIN_FILE)
    audit_df = pd.read_csv(AUDIT_FILE)

    # Keep audit examples completely unseen
    audit_ids = set(
        audit_df["tweet_id"].astype(str)
    )

    train_df = train_df[
        ~train_df["tweet_id"]
        .astype(str)
        .isin(audit_ids)
    ]

    train_df = train_df.dropna(
        subset=["text", "intent"]
    )

    audit_df = audit_df.dropna(
        subset=["text", "human_intent"]
    )

    train_df["text"] = train_df["text"].astype(str)
    train_df["intent"] = train_df["intent"].astype(str)

    audit_df["text"] = audit_df["text"].astype(str)
    audit_df["human_intent"] = (
        audit_df["human_intent"].astype(str)
    )

    return train_df, audit_df


# -----------------------------
# 2. Build improved model
# -----------------------------

def build_model():

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,

                # Use both words and character patterns.
                # Character n-grams help with short/noisy
                # Twitter-style messages.
                analyzer="char_wb",
                ngram_range=(3, 5),

                min_df=1,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    return model


# -----------------------------
# 3. Train model
# -----------------------------

def train_model():

    train_df, _ = load_training_data()

    model = build_model()

    model.fit(
        train_df["text"],
        train_df["intent"]
    )

    return model


# -----------------------------
# 4. Predict intent
# -----------------------------

def predict_intent(customer_message):

    model = train_model()

    prediction = model.predict(
        [customer_message]
    )

    return prediction[0]


# -----------------------------
# 5. Predict intent + confidence
# -----------------------------

def predict_intent_with_confidence(customer_message):

    model = train_model()

    prediction = model.predict(
        [customer_message]
    )[0]

    probabilities = model.predict_proba(
        [customer_message]
    )[0]

    confidence = max(probabilities)

    return prediction, round(
        float(confidence),
        3
    )


# -----------------------------
# 6. Evaluate model
# -----------------------------

def evaluate_model():

    train_df, audit_df = load_training_data()

    print(
        "Training examples:",
        len(train_df)
    )

    print(
        "Audit examples:",
        len(audit_df)
    )

    model = build_model()

    print("\nTraining improved classifier...")

    model.fit(
        train_df["text"],
        train_df["intent"]
    )

    print("Training complete.")

    predictions = model.predict(
        audit_df["text"]
    )

    accuracy = accuracy_score(
        audit_df["human_intent"],
        predictions
    )

    macro_f1 = f1_score(
        audit_df["human_intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        audit_df["human_intent"],
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n==============================")
    print("CLASSIFIER RESULTS")
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
            audit_df["human_intent"],
            predictions,
            zero_division=0
        )
    )


# -----------------------------
# 7. Run evaluation
# -----------------------------

if __name__ == "__main__":

    evaluate_model()