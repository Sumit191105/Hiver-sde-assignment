import os
import json
import time
import pandas as pd
from google import genai


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = "data/golden_set.csv"
OUTPUT_FILE = "data/golden_set_labeled.csv"

MODEL = "gemini-3.6-flash"
BATCH_SIZE = 20

INTENTS = [
    "DELIVERY_DELAY",
    "DELIVERY_NOT_RECEIVED",
    "ORDER_STATUS",
    "RETURN_REFUND",
    "ACCOUNT_ACCESS",
    "PRIME_MEMBERSHIP",
    "PAYMENT_BILLING",
    "PRODUCT_TECHNICAL",
    "CUSTOMER_SERVICE_ESCALATION",
    "SKIP",
]


# -----------------------------------
# Gemini API
# -----------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# -----------------------------------
# Load data
# -----------------------------------

if os.path.exists(OUTPUT_FILE):
    print("Loading existing labeling progress...")
    df = pd.read_csv(OUTPUT_FILE)
else:
    print("Starting new labeling...")
    df = pd.read_csv(INPUT_FILE)


# Make sure intent column exists
if "intent" not in df.columns:
    df["intent"] = ""

df["intent"] = df["intent"].fillna("").astype(str).str.strip()


# Add confidence column
if "confidence" not in df.columns:
    df["confidence"] = ""

df["confidence"] = df["confidence"].astype(object)


# -----------------------------------
# Classification function
# -----------------------------------

def classify_batch(messages):

    formatted_messages = ""

    for i, message in enumerate(messages):

        formatted_messages += f"""
MESSAGE_ID: {i}
MESSAGE: {message}
"""


    prompt = f"""
You are labeling customer-support messages for an Amazon customer-support dataset.

For every message choose exactly ONE intent.

Available intents:

DELIVERY_DELAY

DELIVERY_NOT_RECEIVED

ORDER_STATUS

RETURN_REFUND

ACCOUNT_ACCESS

PRIME_MEMBERSHIP

PAYMENT_BILLING

PRODUCT_TECHNICAL

CUSTOMER_SERVICE_ESCALATION

SKIP


Definitions:


DELIVERY_DELAY:

Order is late, delayed, or delivery date/time was missed.


DELIVERY_NOT_RECEIVED:

Tracking says delivered, but customer says they did not receive the package.


ORDER_STATUS:

Customer asks where the order is, tracking status, estimated delivery,
or when/how the order will arrive.


RETURN_REFUND:

Returns, refunds, replacements, damaged/wrong items where the main issue
is return or refund.


ACCOUNT_ACCESS:

Login, password, locked account, suspended account,
or account access/security.


PRIME_MEMBERSHIP:

Prime subscription, Prime renewal, Prime cancellation, Prime benefits,
or unexpected Prime membership charge specifically related to Prime.


PAYMENT_BILLING:

Payment, credit/debit card, bank transaction, duplicate charge,
billing, gift card, or money/charge dispute.


PRODUCT_TECHNICAL:

Technical problems with Amazon devices, Fire TV, Kindle, Echo,
apps, websites, or product functionality.


CUSTOMER_SERVICE_ESCALATION:

Customer wants a manager/human, complains that support has not solved
the issue, repeated unresolved complaint, or asks for escalation.


SKIP:

Message is unrelated, just thanks, just a link/tag, too vague,
or does not clearly fit any category.


IMPORTANT:

- Return exactly one result for every message.
- Use the MESSAGE_ID provided.
- Do not skip any message.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not add explanations.


Required format:

[
  {{
    "message_id": 0,
    "intent": "DELIVERY_DELAY",
    "confidence": 0.95
  }}
]


Messages:

{formatted_messages}
"""


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )


    text = response.text.strip()


    # Remove markdown fences if Gemini adds them
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()


    results = json.loads(text)


    # Make sure Gemini returned a list
    if not isinstance(results, list):
        raise ValueError("Gemini response is not a JSON list")


    # Make sure Gemini returned correct number of results
    if len(results) != len(messages):
        raise ValueError(
            f"Expected {len(messages)} results, "
            f"but Gemini returned {len(results)}"
        )


    return results


# -----------------------------------
# Find unlabeled messages
# -----------------------------------

unlabeled_indices = []

for index in df.index:

    intent = str(df.loc[index, "intent"]).strip()

    if intent == "":
        unlabeled_indices.append(index)


print("\n==============================")
print("BATCH GOLDEN SET LABELING")
print("==============================")

print(f"Total messages: {len(df)}")
print(f"Already labeled: {len(df) - len(unlabeled_indices)}")
print(f"Still needing labels: {len(unlabeled_indices)}")
print(f"Batch size: {BATCH_SIZE}")

print("==============================")


# -----------------------------------
# Process only unlabeled messages
# -----------------------------------

for start in range(
    0,
    len(unlabeled_indices),
    BATCH_SIZE
):

    # Actual dataframe indexes for this batch
    batch_indices = unlabeled_indices[
        start:start + BATCH_SIZE
    ]


    # Get only the messages that are still unlabeled
    messages = [
        str(df.loc[index, "text"])
        for index in batch_indices
    ]


    print("\n" + "=" * 80)

    print(
        f"Processing "
        f"{start + 1}-"
        f"{start + len(batch_indices)} "
        f"of {len(unlabeled_indices)}"
    )

    print("=" * 80)


    try:

        # Ask Gemini to classify this batch
        results = classify_batch(messages)


        # -----------------------------------
        # Validate and save results
        # -----------------------------------

        for result in results:

            message_id = int(result["message_id"])

            intent = str(
                result["intent"]
            ).strip()

            confidence = result["confidence"]


            # Check message ID
            if (
                message_id < 0
                or message_id >= len(batch_indices)
            ):
                raise ValueError(
                    f"Invalid message_id: {message_id}"
                )


            # Convert Gemini batch position
            # to actual dataframe index
            actual_index = batch_indices[message_id]


            # If Gemini gives an unknown intent,
            # safely mark it as SKIP
            if intent not in INTENTS:
                intent = "SKIP"


            # Save classification
            df.loc[
                actual_index,
                "intent"
            ] = intent


            df.loc[
                actual_index,
                "confidence"
            ] = confidence


            print(
                f"{actual_index + 1}: "
                f"{intent} "
                f"(confidence={confidence})"
            )


        # -----------------------------------
        # Save after every successful batch
        # -----------------------------------

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )


        print(
            f"\nSaved progress to: "
            f"{OUTPUT_FILE}"
        )


        # Small delay between API requests
        time.sleep(2)


    except Exception as e:

        print("\n==============================")
        print("ERROR")
        print("==============================")

        print(e)


        # Save whatever progress exists
        df.to_csv(
            OUTPUT_FILE,
            index=False
        )


        print("\nProgress saved.")
        print("Stopping safely.")
        print(
            "Run the script again later "
            "to continue from where it stopped."
        )

        break


# -----------------------------------
# Final status
# -----------------------------------

print("\n==============================")
print("LABELING FINISHED / STOPPED")
print("==============================")


print("\nIntent distribution:")

print(
    df["intent"].replace(
        "",
        "UNLABELED"
    ).value_counts()
)


print("\nOutput:")
print(OUTPUT_FILE)