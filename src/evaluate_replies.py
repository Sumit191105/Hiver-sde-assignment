import os
import pandas as pd

from retriever import retrieve_similar_messages
from response_generator import generate_response
from gemini_intent_classifier import classify_intent


OUTPUT_FILE = "data/reply_evaluation.csv"

# Keep API usage controlled.
MAX_NEW_CALLS = 2


def build_evidence_text(retrieved_examples):
    """
    Convert retrieved historical conversations into
    readable evidence for the judge.
    """

    if not retrieved_examples:
        return "No historical evidence was retrieved."

    evidence = []

    for i, item in enumerate(retrieved_examples, 1):

        customer = item.get("customer_message", "")
        reply = item.get("support_reply", "")
        similarity = item.get("similarity", 0)

        evidence.append(
            f"Example {i} "
            f"(similarity={similarity:.3f})\n"
            f"Customer: {customer}\n"
            f"Historical support reply: {reply}"
        )

    return "\n\n".join(evidence)


def judge_reply(
    customer_message,
    draft_reply,
    retrieved_examples
):
    """
    Ask Gemini to evaluate the generated reply.

    Returns a compact JSON-like dictionary.
    """

    evidence = build_evidence_text(
        retrieved_examples
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

AI-generated reply:
{draft_reply}

Historical support evidence:
{evidence}

Evaluate the AI reply on four dimensions.

Give each score from 1 to 5:

1. groundedness:
Does the reply stay supported by the historical evidence?
It should not invent policies, refunds, dates, guarantees, or actions.

2. helpfulness:
Does it meaningfully address the customer's issue?

3. safety:
Does it avoid unsupported promises and risky claims?

4. actionability:
Does it provide a reasonable next step for the customer?

Return ONLY this format:

groundedness: X
helpfulness: X
safety: X
actionability: X
overall: X
reason: short explanation
"""

    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    scores = {}

    for line in text.splitlines():

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1
        )

        key = key.strip().lower()
        value = value.strip()

        if key in {
            "groundedness",
            "helpfulness",
            "safety",
            "actionability",
            "overall"
        }:

            try:
                scores[key] = int(
                    value.split()[0]
                )
            except ValueError:
                pass

    scores["reason"] = text

    return scores


# ============================================================
# TEST CASES
# ============================================================

test_messages = [
    "My package says delivered but I did not receive it",

    "Where is my order? It was supposed to arrive yesterday",

    "I want to return this item and get my money back",

    "I cannot log into my Amazon account",

    "My Amazon payment was charged twice",

    "My Kindle is connected to WiFi but I cannot download books"
]


# ============================================================
# LOAD CACHE
# ============================================================

if os.path.exists(OUTPUT_FILE):

    previous_df = pd.read_csv(
        OUTPUT_FILE
    )

    if (
        "customer_message" in previous_df.columns
    ):

        cache = {}

        for _, row in previous_df.iterrows():

            cache[
                str(row["customer_message"])
            ] = row.to_dict()

    else:

        cache = {}

    print(
        "Cached reply evaluations:",
        len(cache)
    )

else:

    cache = {}

    print(
        "No cached reply evaluations."
    )


# ============================================================
# RUN EVALUATION
# ============================================================

new_calls = 0

results = []


for index, customer_message in enumerate(
    test_messages,
    1
):

    if customer_message in cache:

        print(
            f"[{index}/{len(test_messages)}] "
            "Using cached result"
        )

        results.append(
            cache[customer_message]
        )

        continue

    if new_calls >= MAX_NEW_CALLS:

        print(
            "\nReached per-run Gemini call limit."
        )

        break

    print(
        f"\n[{index}/{len(test_messages)}]"
    )

    print(
        "Customer:",
        customer_message
    )

    try:

        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        intent = classify_intent(
            customer_message
        )

        print(
            "Intent:",
            intent
        )

        # ----------------------------------------------------
        # Retrieve historical evidence
        # ----------------------------------------------------

        retrieved = retrieve_similar_messages(
            customer_message,
            top_k=3
        )

        print(
            "Retrieved examples:",
            len(retrieved)
        )

        # ----------------------------------------------------
        # Generate reply
        # ----------------------------------------------------

        draft_reply = generate_response(
            customer_message
        )

        print(
            "Draft reply:",
            draft_reply
        )

        # ----------------------------------------------------
        # Judge reply
        # ----------------------------------------------------

        scores = judge_reply(
            customer_message,
            draft_reply,
            retrieved
        )

        row = {
            "customer_message": customer_message,
            "intent": intent,
            "draft_reply": draft_reply,
            "groundedness": scores.get(
                "groundedness",
                ""
            ),
            "helpfulness": scores.get(
                "helpfulness",
                ""
            ),
            "safety": scores.get(
                "safety",
                ""
            ),
            "actionability": scores.get(
                "actionability",
                ""
            ),
            "overall": scores.get(
                "overall",
                ""
            ),
            "judge_reason": scores.get(
                "reason",
                ""
            )
        }

        results.append(row)

        cache[customer_message] = row

        new_calls += 1

        # Save immediately
        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "Judge overall:",
            row["overall"]
        )

    except Exception as e:

        print(
            "\nEvaluation error:"
        )

        print(e)

        print(
            "\nStopping safely."
        )

        break


# ============================================================
# SAVE RESULTS
# ============================================================

if results:

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n=============================="
    )

    print(
        "REPLY QUALITY EVALUATION"
    )

    print(
        "=============================="
    )

    print(
        "\nCompleted evaluations:",
        len(results_df)
    )

    for metric in [
        "groundedness",
        "helpfulness",
        "safety",
        "actionability",
        "overall"
    ]:

        values = pd.to_numeric(
            results_df[metric],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            print(
                f"{metric.capitalize()}: "
                f"{values.mean():.2f}/5"
            )

else:

    print(
        "\nNo reply evaluations completed."
    )


print(
    f"\nResults saved to: {OUTPUT_FILE}"
)