import os
import re

from google import genai


# -----------------------------
# Gemini setup
# -----------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(
    api_key=API_KEY
)


# -----------------------------
# Allowed intents
# -----------------------------

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
    "SKIP"
]


# -----------------------------
# Intent definitions
# -----------------------------

INTENT_DEFINITIONS = """
DELIVERY_DELAY:
The customer says an order is late, delayed, or missed the expected delivery time.

DELIVERY_NOT_RECEIVED:
The tracking says the package was delivered, but the customer says they did not receive it.

ORDER_STATUS:
The customer wants to know where their order is, its tracking status, ETA, or when it will arrive.

RETURN_REFUND:
The main issue is returning an item, getting a refund, replacement, or dealing with a wrong/damaged item.

ACCOUNT_ACCESS:
The customer has a login, password, account lock, account access, or account security problem.

PRIME_MEMBERSHIP:
The issue is specifically about Amazon Prime membership, subscription, renewal, cancellation, benefits, or Prime-specific charges.

PAYMENT_BILLING:
The issue involves payment, credit/debit card, bank payment, duplicate charge, billing, gift card, or disputed money/charge.

PRODUCT_TECHNICAL:
The customer has a technical problem with an Amazon product, device, app, website, or product functionality.

CUSTOMER_SERVICE_ESCALATION:
The customer explicitly wants a human, manager, supervisor, or escalation, or says the issue remains unresolved after previous support.

SKIP:
The message is irrelevant, only a thank-you, contains only a link/tag, is too vague, or does not have a clear support intent.
"""


# -----------------------------
# Classify customer message
# -----------------------------

def classify_intent(customer_message):

    prompt = f"""
You are an intent classification system for Amazon customer support.

Classify the customer's message into EXACTLY ONE of the following intents.

{INTENT_DEFINITIONS}

Customer message:
{customer_message}

Return ONLY the intent name.

Do not add:
- explanations
- punctuation
- quotes
- markdown
- extra text

Valid output must be exactly one of:

{", ".join(INTENTS)}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_result = response.text.strip().upper()

    # Clean accidental formatting
    raw_result = re.sub(
        r"[^A-Z_]",
        "",
        raw_result
    )

    # Make sure Gemini returned a valid intent
    if raw_result in INTENTS:
        return raw_result

    # Fallback
    return "SKIP"


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    test_messages = [
        "My package is two days late",

        "My package says delivered but I never received it",

        "Where is my order?",

        "I want a refund for this item",

        "I cannot log into my account",

        "Why was I charged for Amazon Prime?",

        "I was charged twice for the same order",

        "My Kindle is not turning on",

        "I want to speak to a manager",

        "Thanks Amazon!"
    ]

    print("\n==============================")
    print("GEMINI INTENT CLASSIFIER")
    print("==============================")

    for message in test_messages:

        intent = classify_intent(message)

        print("\nCustomer:")
        print(message)

        print("Intent:")
        print(intent)
        