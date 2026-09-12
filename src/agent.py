from gemini_intent_classifier import classify_intent
from retriever import retrieve_similar_messages
from response_generator import generate_response


# -----------------------------
# Decide AUTO-HANDLE or ESCALATE
# -----------------------------

def decide_action(intent, retrieval_results):

    # If customer explicitly wants a human,
    # always escalate.
    if intent == "CUSTOMER_SERVICE_ESCALATION":
        return (
            "ESCALATE",
            "Customer explicitly requested human customer support."
        )

    # SKIP means the message does not contain
    # a clear support request.
    if intent == "SKIP":
        return (
            "ESCALATE",
            "Message does not contain a clear support intent."
        )

    # Check whether we found useful historical evidence.
    if not retrieval_results:
        return (
            "ESCALATE",
            "No historical support examples were found."
        )

    top_similarity = retrieval_results[0]["similarity"]

    # Low similarity means our historical evidence
    # may not be relevant enough.
    if top_similarity < 0.15:
        return (
            "ESCALATE",
            "Historical evidence is not sufficiently similar."
        )

    # Otherwise the system can draft a response.
    return (
        "AUTO_HANDLE",
        "A clear support intent and relevant historical evidence were found."
    )


# -----------------------------
# Complete AI Support Agent
# -----------------------------

def run_agent(customer_message):

    # Step 1:
    # Classify customer intent using Gemini.
    intent = classify_intent(
        customer_message
    )

    # Step 2:
    # Retrieve similar historical Amazon
    # customer-support conversations.
    retrieval_results = retrieve_similar_messages(
        customer_message,
        top_k=3
    )

    # Step 3:
    # Decide whether to auto-handle or escalate.
    action, reason = decide_action(
        intent,
        retrieval_results
    )

    # Step 4:
    # Generate a grounded draft reply.
    draft_reply = generate_response(
        customer_message
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "action": action,
        "reason": reason,
        "draft_reply": draft_reply,
        "retrieved_examples": retrieval_results
    }


# -----------------------------
# Test the complete agent
# -----------------------------

if __name__ == "__main__":

    customer_message = (
        "My package says delivered but I did not receive it"
    )

    result = run_agent(
        customer_message
    )

    print("\n==============================")
    print("AI CUSTOMER SUPPORT AGENT")
    print("==============================")

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nIntent:")
    print(result["intent"])

    print("\nDecision:")
    print(result["action"])

    print("\nReason:")
    print(result["reason"])

    print("\nAI Draft Reply:")
    print(result["draft_reply"])

    print("\nRetrieved Historical Examples:")

    for i, example in enumerate(
        result["retrieved_examples"],
        start=1
    ):

        print(f"\nExample {i}")

        print(
            "Customer:",
            example["customer_message"]
        )

        print(
            "Amazon Reply:",
            example["historical_reply"]
        )

        print(
            "Similarity:",
            example["similarity"]
        )