import os

from google import genai

from retriever import retrieve_similar_messages


# -----------------------------
# 1. Gemini setup
# -----------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)


# -----------------------------
# 2. Generate AI response
# -----------------------------

def generate_response(customer_message):

    # Retrieve similar historical conversations
    results = retrieve_similar_messages(
        customer_message,
        top_k=3
    )

    # Build historical evidence
    evidence = ""

    for i, result in enumerate(results, start=1):

        evidence += f"""
Historical Example {i}

Customer:
{result["customer_message"]}

Amazon Reply:
{result["historical_reply"]}

Similarity Score:
{result["similarity"]}
"""


    # Prompt for Gemini
    prompt = f"""
You are an Amazon customer support assistant.

A customer has sent this message:

CUSTOMER MESSAGE:
{customer_message}

Below are historical Amazon customer-support conversations.
Use these examples as evidence for how Amazon historically handled similar issues.

{evidence}

Instructions:

1. Write a helpful and professional customer-support reply.
2. Ground the reply in the historical examples.
3. Do not invent policies, refunds, delivery dates, or guarantees.
4. Do not mention the historical examples.
5. If account-specific information is required, tell the customer to contact Amazon support.
6. Keep the response concise.
"""


    # Generate response using Gemini
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# -----------------------------
# 3. Test the response generator
# -----------------------------

if __name__ == "__main__":

    customer_message = (
        "My package says delivered but I did not receive it"
    )

    print("\nCustomer:")
    print(customer_message)

    print("\nAI Draft Reply:")

    reply = generate_response(
        customer_message
    )

    print(reply)