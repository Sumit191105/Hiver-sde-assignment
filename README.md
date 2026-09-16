# AmazonHelp AI Support Agent

An AI-powered customer support agent built using the Customer Support on Twitter dataset.

The system takes a customer support message, identifies its intent, retrieves historically similar AmazonHelp conversations, drafts a grounded response, and decides whether the case can be auto-handled or should be escalated to a human.

> **Important:** This project focuses on demonstrating the complete AI support workflow and evaluating its behavior on a small, reproducible subset of the full dataset, rather than training on the entire dataset.

---

## Table of Contents

1. [Problem](#1-problem)
2. [What I Chose](#2-what-i-chose)
3. [Dataset](#3-dataset)
4. [Data Processing](#4-data-processing)
5. [System Architecture](#5-system-architecture)
6. [Supported Intents](#6-supported-intents)
7. [Intent Classification](#7-intent-classification)
8. [Gemini Intent Classifier](#8-gemini-intent-classifier)
9. [Golden Evaluation Set](#9-golden-evaluation-set)
10. [Historical Retrieval](#10-historical-retrieval)
11. [Response Generation](#11-response-generation)
12. [Example](#12-example)
13. [Auto-Handle vs. Escalation](#13-auto-handle-vs-escalation)
14. [Reply Quality Evaluation](#14-reply-quality-evaluation)
15. [What Is Misleading About My Headline Number?](#15-what-is-misleading-about-my-headline-number)
16. [Failure Modes](#16-failure-modes)
17. [What I Would Build Next Week](#17-what-i-would-build-next-week)
18. [What I Would NOT Build](#18-what-i-would-not-build)
19. [Project Structure](#19-project-structure)
20. [Main Components](#20-main-components)
21. [Setup](#21-setup)
22. [Gemini API Key](#22-gemini-api-key)
23. [Running the Project](#23-running-the-project)
24. [Run the Agent](#24-run-the-agent)
25. [Evaluation](#25-evaluation)
26. [Reproducibility](#26-reproducibility)
27. [Limitations](#27-limitations)
28. [Decision Log](#28-decision-log)
29. [Summary](#29-summary)

---

## 1. Problem

Customer support systems receive large numbers of messages covering repetitive issues such as:

- Delivery delays
- Missing deliveries
- Order tracking
- Returns and refunds
- Payment and billing problems
- Account access
- Prime membership
- Product/device problems
- Requests for human support

The goal of this project is to build a small AI support agent that can:

1. Classify the customer's message into a defined support intent.
2. Retrieve historically similar AmazonHelp conversations.
3. Generate a response grounded in those historical examples.
4. Decide whether to automatically handle the request or escalate it to a human.
5. Provide a reason for the routing decision.

---

## 2. What I Chose

### Brand: AmazonHelp

AmazonHelp was selected because it had the largest number of support-authored tweets in the sampled brand analysis.

### Why This Matters

Instead of building a generic chatbot, the system learns from the way the selected brand historically responded to customers. This allows the generated response to be grounded in real historical support interactions.

---

## 3. Dataset

The project uses the **Customer Support on Twitter** dataset:
🔗 https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The original dataset contains approximately 3 million tweets from customer-support conversations involving multiple brands.

**Important fields:**

| Field | Description |
|---|---|
| `tweet_id` | Unique tweet identifier |
| `author_id` | Author of the tweet |
| `inbound` | Whether the tweet is inbound (from a customer) |
| `created_at` | Timestamp |
| `text` | Tweet content |
| `response_tweet_id` | ID of the response tweet |
| `in_response_to_tweet_id` | ID of the tweet being responded to |

For this project:

- `inbound=True` → treated as a **customer message**.
- `inbound=False` → treated as a **support/brand response**.
- Tweet relationships are used to connect customer messages with historical support responses.

---

## 4. Data Processing

The full dataset was not loaded into the model training pipeline. Instead, the workflow was:

```text
Full Customer Support on Twitter dataset
                |
                v
        AmazonHelp filtering
                |
                v
       Customer/support pairs
                |
                v
      Training / retrieval sample
                |
                v
       Evaluation / golden set
```

### Dataset Statistics Used in This Project

| Stage | Size |
|---|---:|
| AmazonHelp support-authored tweets | 169,840 |
| AmazonHelp filtered rows | 280,231 |
| Customer → historical support pairs | 103,274 |
| Training sample | 5,000 |
| Golden evaluation sample | 200 |
| Human-verified audit examples | 89 |

> The original full dataset is intentionally excluded from Git because of its large size.

---

## 5. System Architecture

```text
Customer Message
       |
       v
+----------------------+
| Intent Classification|
|       Gemini         |
+----------+-----------+
           |
           v
        Intent
           |
           v
+----------------------+
| Historical Retrieval |
|        TF-IDF        |
+----------+-----------+
           |
           v
 Similar customer/support
      conversations
           |
           v
+----------------------+
| Response Generation  |
|        Gemini        |
+----------+-----------+
           |
           v
      Draft Reply
           |
           v
+----------------------+
|  Routing Decision    |
| Auto-handle/Escalate |
+----------------------+
           |
           v
    Final Agent Output
```

---

## 6. Supported Intents

The current system uses 10 support intents.

| Intent | Description |
|---|---|
| `DELIVERY_DELAY` | Order is late or delivery is delayed |
| `DELIVERY_NOT_RECEIVED` | Tracking says delivered but customer did not receive it |
| `ORDER_STATUS` | Customer asks where an order is or requests tracking/ETA |
| `RETURN_REFUND` | Return, refund, replacement, damaged, or wrong item issues |
| `ACCOUNT_ACCESS` | Login, password, account lock, or account security problems |
| `PRIME_MEMBERSHIP` | Prime subscription, renewal, cancellation, or Prime benefits |
| `PAYMENT_BILLING` | Payments, cards, billing, duplicate charges, or money-related issues |
| `PRODUCT_TECHNICAL` | Problems with Amazon devices, apps, or product functionality |
| `CUSTOMER_SERVICE_ESCALATION` | Customer explicitly requests human/manager support or escalation |
| `SKIP` | Irrelevant, vague, non-support, or unusable messages |

---

## 7. Intent Classification

Two approaches were implemented.

### 7.1 Simple Baseline

The first baseline uses:

```text
Character TF-IDF + Logistic Regression
```

Character n-grams make the baseline more tolerant of spelling variations and short customer messages.

**Baseline result** — evaluated on **89 human-verified examples**:

| Metric | Result |
|---|---:|
| Accuracy | 29.2% |
| Macro F1 | 15.1% |
| Weighted F1 | 22.7% |

The baseline performs reasonably on common categories such as `SKIP`, but struggles with lower-frequency intents. This provided a useful reference point for evaluating the semantic/LLM approach.

---

## 8. Gemini Intent Classifier

The second classifier uses Gemini with a constrained intent definition.

The model receives:

```text
Customer message + Intent definitions
```

and is instructed to return only one of the supported intent names.

**Example:**

```text
Customer: "My package says delivered but I did not receive it"
Output:   DELIVERY_NOT_RECEIVED
```

### Current Evaluation Snapshot

The Gemini classifier was evaluated on the currently available evaluated subset.

| Metric | Result |
|---|---:|
| Evaluated examples | 13 |
| Accuracy | 46.2% |
| Macro F1 | 29.0% |
| Weighted F1 | 47.6% |

> This is an **early evaluation snapshot**, not a final estimate of production performance. The small evaluation size means these numbers should not be interpreted as statistically stable.

---

## 9. Golden Evaluation Set

A 200-example golden set was created by sampling customer messages. Initial labels were generated using Gemini-based weak labeling across the 10 defined intents.

### Important Labeling Note

The 200 examples should **not** be described as independently hand-labelled. Instead:

- 200 examples were sampled for the golden evaluation set.
- Initial labels were produced using an LLM-assisted labeling process.
- 89 examples were subsequently reviewed and verified by a human.
- The 89 human-verified examples are therefore the strongest manually checked subset currently available.

This distinction matters because using LLM-generated labels as if they were independent human ground truth would inflate confidence in the evaluation.

---

## 10. Historical Retrieval

The system retrieves similar historical AmazonHelp conversations before generating a response, using:

```text
TF-IDF + Word n-grams (1,2) + Cosine similarity
```

The system retrieves the **top 3** historical customer/support examples. Each retrieved example contains:

```text
Historical customer message + Historical AmazonHelp response
```

These retrieved examples are then provided to the response-generation model as evidence.

---

## 11. Response Generation

Gemini generates the draft response using:

1. The current customer message.
2. The detected intent.
3. Retrieved historical AmazonHelp examples.

The response-generation prompt instructs the model to:

- Stay grounded in historical evidence.
- Avoid inventing policies, refunds, guarantees, or delivery dates.
- Avoid claiming actions that were not actually performed.
- Provide practical next steps.
- Escalate to Amazon Support when account/order-specific information is required.

---

## 12. Example

**Customer:**
```text
My package says delivered but I did not receive it
```

**Detected intent:** `DELIVERY_NOT_RECEIVED`

**Retrieved evidence:** three historically similar AmazonHelp conversations (top similarity: `0.511`)

**Generated response:**

```text
I'm sorry to hear that you haven't received your package yet!

Please check the "Find a Missing Package" guide in our Help section on
Amazon for initial steps, such as checking around your delivery location,
checking with neighbors, or verifying your shipping address.

If you've tried these steps and still cannot locate your parcel, please
contact Amazon Customer Support directly through your account so we can
access your order details and assist you further.
```

**Routing:** `AUTO_HANDLE`
**Reason:** A clear support intent and relevant historical evidence were found.

---

## 13. Auto-Handle vs. Escalation

The system uses explicit routing rules.

### Escalate When:

**1. Customer explicitly asks for human support** (`CUSTOMER_SERVICE_ESCALATION`)

> Example: *"I want to speak to a real person."*
> Result: `ESCALATE` — Customer explicitly requested human customer support.

**2. No clear support intent** (`SKIP`)

> Result: `ESCALATE` — Message does not contain a clear support intent.

**3. Historical evidence is too weak**

> If the best retrieved historical example has a similarity below the configured threshold:
> Result: `ESCALATE` — Historical evidence is not sufficiently similar.

### Auto-Handle When:

The system has a **clear support intent** + **relevant historical evidence**, and the case does not trigger an escalation rule.

> The current implementation only drafts a response. It does **not** actually modify orders, issue refunds, access accounts, or perform transactions.

---

## 14. Reply Quality Evaluation

Reply quality was evaluated using an LLM-as-judge rubric across four dimensions:

| Dimension | Question |
|---|---|
| **Groundedness** | Is the response supported by the retrieved historical evidence? |
| **Helpfulness** | Does the response address the customer's actual problem? |
| **Safety** | Does the response avoid unsupported promises, invented actions, or risky instructions? |
| **Actionability** | Does the customer receive clear next steps? |

Each dimension is scored from 1 (poor) to 5 (excellent).

### Current Reply Evaluation Snapshot

Based on **2 completed examples**:

| Metric | Score |
|---|---:|
| Groundedness | 4.50 / 5 |
| Helpfulness | 5.00 / 5 |
| Safety | 5.00 / 5 |
| Actionability | 5.00 / 5 |
| **Overall** | **5.00 / 5** |

> These scores are only an initial smoke-test style evaluation because the sample size is very small. They should **not** be interpreted as evidence that the system consistently achieves 5/5 quality.

---

## 15. What Is Misleading About My Headline Number?

A headline such as **"5.0/5 reply quality"** would be misleading if presented without context.

The current score comes from only **2 evaluated replies**, and the judge itself is an LLM. Therefore:

- The sample is too small for a reliable quality estimate.
- LLM-as-judge scores can contain evaluator bias.
- The current evaluation does not establish production-level performance.
- A larger human-reviewed evaluation set is required before making strong claims.

The purpose of the current result is to demonstrate that the evaluation pipeline works — not to claim the agent has solved customer support.

---

## 16. Failure Modes

### 1. Ambiguous Customer Messages

Some messages don't contain enough context to confidently determine intent.

> *Example:* "Can you help me with this?"
> *Hypothesis:* The message requires conversation context not available from a single tweet.

### 2. Overlap Between Delivery Intents

`DELIVERY_DELAY`, `ORDER_STATUS`, and `DELIVERY_NOT_RECEIVED` can be difficult to distinguish.

> *Example:* "Where is my package? It should have arrived yesterday." — could be `ORDER_STATUS` **or** `DELIVERY_DELAY`.
> *Hypothesis:* Intent definitions need clearer boundaries and potentially hierarchical classification.

### 3. SKIP Dominates the Sampled Dataset

A significant portion of sampled messages are short, irrelevant, vague, or difficult to categorize, which can make accuracy misleading since a classifier can gain credit by performing well on the dominant class.

> *Hypothesis:* Evaluation should report macro-F1 and per-class results rather than accuracy alone.

### 4. Historical Retrieval Can Return Superficially Similar Examples

TF-IDF similarity is based on lexical overlap — two messages can share words while having different support requirements.

> *Hypothesis:* A semantic embedding-based retriever could improve retrieval quality.

### 5. LLM Response Generation Can Overgeneralize

Even with retrieved evidence, an LLM can potentially generate policies, guarantees, or actions not present in the evidence.

> *Mitigation:* The generation prompt explicitly prohibits invented refunds, policies, dates, guarantees, and completed-action claims. The escalation mechanism also reduces risk when evidence is weak.

---

## 17. What I Would Build Next Week

If given another week, I would focus on evaluation quality and retrieval before adding more product features.

1. **Expand the human-labelled evaluation set** — increase the manually verified set from 89 examples toward the required 150–250 range.
2. **Improve intent definitions** — create clearer decision boundaries between `ORDER_STATUS`, `DELIVERY_DELAY`, and `DELIVERY_NOT_RECEIVED`.
3. **Replace TF-IDF retrieval with embeddings** — retrieve historical conversations based on meaning rather than only lexical overlap.
4. **Add confidence calibration** — combine intent confidence + retrieval confidence + intent risk to determine whether the agent should auto-handle a case.
5. **Improve evaluation** — add human reply-quality ratings, LLM-as-judge ratings, agreement analysis between human and LLM judges, per-intent metrics, and retrieval metrics.
6. **Add conversation context** — provide previous messages in the same thread instead of classifying a single tweet, to help with ambiguous and multi-turn conversations.

---

## 18. What I Would NOT Build

The goal is not to build a full Amazon customer-service replacement. I would not attempt to build:

- Real Amazon order management.
- Real refunds.
- Real payment processing.
- Real account authentication.
- A production customer-support dashboard.
- A fully autonomous agent with unrestricted actions.

Those systems require access control, security, auditing, business rules, and production integrations that are outside the scope of this assignment.

The focus here is: **Intent + Evidence + Draft response + Safe routing + Evaluation.**

---

## 19. Project Structure

```text
hiver-sde-assignment/
│
├── data/
│   ├── twcs.csv
│   ├── amazon_help.csv
│   ├── training_sample.csv
│   ├── golden_set.csv
│   ├── golden_set_labeled.csv
│   └── golden_audit.csv
│
├── src/
│   ├── inspect_data.py
│   ├── brand_analysis.py
│   ├── extract_brand.py
│   ├── view_customer_messages.py
│   ├── sample_customer_messages.py
│   ├── create_training_sample.py
│   ├── inspect_training_sample.py
│   ├── create_golden_set.py
│   ├── label_golden_set.py
│   ├── auto_label_golden_set.py
│   ├── gemini_test.py
│   ├── intent_classifier.py
│   ├── extract_conversations.py
│   ├── retriever.py
│   ├── response_generator.py
│   ├── gemini_intent_classifier.py
│   ├── agent.py
│   ├── evaluate_intents.py
│   └── evaluate_replies.py
│
├── .gitignore
└── README.md
```

> Large raw/generated files are excluded from Git where appropriate.

---

## 20. Main Components

| File | Purpose |
|---|---|
| `brand_analysis.py` | Analyzes the dataset and identifies support-heavy brands |
| `extract_brand.py` | Extracts AmazonHelp-related tweets from the full dataset |
| `extract_conversations.py` | Builds customer → historical support-response pairs |
| `retriever.py` | Implements TF-IDF-based historical conversation retrieval |
| `gemini_intent_classifier.py` | Classifies customer messages into the supported intent set using Gemini |
| `response_generator.py` | Generates a grounded customer-support draft using retrieved historical evidence |
| `agent.py` | Combines the complete pipeline: Intent → Retrieval → Response → Routing |
| `intent_classifier.py` | Implements the classical TF-IDF + Logistic Regression baseline |
| `evaluate_intents.py` | Evaluates intent predictions using accuracy, macro F1, weighted F1, per-class performance |
| `evaluate_replies.py` | Evaluates generated replies using an LLM-as-judge rubric |

---

## 21. Setup

### Requirements

- Python 3.10+
- Gemini API key
- Dataset downloaded from Kaggle

### Install Dependencies

```bash
pip install pandas numpy scikit-learn google-genai
```

---

## 22. Gemini API Key

Set the API key as an environment variable.

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

**Linux / macOS:**

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

> Never commit the API key to Git.

---

## 23. Running the Project

From the project root:

```bash
python src/inspect_data.py            # Inspect the raw dataset
python src/brand_analysis.py          # Run brand analysis
python src/extract_brand.py           # Extract AmazonHelp data
python src/extract_conversations.py   # Create historical conversation pairs
python src/create_training_sample.py  # Create a training sample
python src/create_golden_set.py       # Create the golden set
```

---

## 24. Run the Agent

The main agent is implemented in `src/agent.py`. The pipeline is:

```text
Customer message
        ↓
Gemini intent classification
        ↓
TF-IDF historical retrieval
        ↓
Gemini response generation
        ↓
Auto-handle / escalate decision
```

**Example input:**

```text
My package says delivered but I did not receive it
```

**Expected output type:**

```text
Intent: DELIVERY_NOT_RECEIVED
Action: AUTO_HANDLE
Reason: A clear support intent and relevant historical evidence were found.

Draft: A grounded customer-support response based on
       historical AmazonHelp conversations.
```

---

## 25. Evaluation

```bash
python src/evaluate_intents.py   # Run intent evaluation
python src/evaluate_replies.py   # Run reply evaluation
```

Evaluation outputs are cached in the `data/` directory so that completed API calls do not need to be repeated.

---

## 26. Reproducibility

The project is designed to be reproducible on a small subset of the original dataset. The full dataset is approximately 500+ MB and contains millions of tweets, so it is intentionally not committed to the repository.

**To reproduce the main workflow:**

1. Download the dataset.
2. Place `twcs.csv` in `data/`.
3. Install dependencies.
4. Set `GEMINI_API_KEY`.
5. Run the preprocessing scripts.
6. Run the agent.
7. Run the evaluation scripts.

The system uses cached intermediate files where possible to reduce repeated API calls.

---

## 27. Limitations

- **Small human evaluation set** — only 89 examples have currently been human-verified; the assignment's target of 150–250 independently hand-labelled examples has not yet been fully reached.
- **Weakly labelled examples** — the initial 200-example golden set used LLM-assisted labels, which should not be treated as equivalent to independent human annotations.
- **Small reply evaluation** — only two reply examples are currently included in the automated reply-quality evaluation, so the current 5/5 result is a pipeline demonstration rather than a statistically meaningful quality estimate.
- **LLM dependence** — the Gemini classifier and response generator depend on an external API.
- **Retrieval limitations** — TF-IDF retrieval relies on lexical similarity and can miss semantically similar messages with different wording.
- **No real customer actions** — the agent does not modify orders, issue refunds, change accounts, process payments, or contact customers. It only produces a draft response and routing recommendation.

---

## 28. Decision Log

| # | Decision |
|--:|---|
| 1 | Selected AmazonHelp as the target brand |
| 2 | Used historical customer/support pairs as grounding evidence |
| 3 | Limited the system to 10 intents |
| 4 | Built a classical TF-IDF + Logistic Regression baseline |
| 5 | Added Gemini for semantic intent classification |
| 6 | Used TF-IDF retrieval as a simple interpretable retrieval baseline |
| 7 | Retrieved the top 3 historical conversations |
| 8 | Used retrieved historical responses as generation evidence |
| 9 | Added explicit escalation rules |
| 10 | Escalated messages without a clear intent |
| 11 | Escalated when historical evidence was weak |
| 12 | Added an LLM-as-judge reply evaluation |
| 13 | Cached evaluation results to reduce repeated API calls |
| 14 | Kept raw large dataset files out of Git |
| 15 | Avoided claiming weak/LLM labels were independent human labels |

---

## 29. Summary

This project demonstrates a complete AI customer-support workflow:

```text
             CUSTOMER MESSAGE
                    |
                    v
          +--------------------+
          | Intent Classifier  |
          +--------------------+
                    |
                    v
                 INTENT
                    |
                    v
          +--------------------+
          | Historical Search  |
          +--------------------+
                    |
                    v
           SUPPORT EVIDENCE
                    |
                    v
          +--------------------+
          | Response Generator |
          +--------------------+
                    |
                    v
              DRAFT REPLY
                    |
                    v
          +--------------------+
          |  Routing Decision  |
          +--------------------+
             /            \
            /              \
           v                v
     AUTO-HANDLE         ESCALATE
```

The main engineering focus was not simply generating text, but connecting:

**classification → historical evidence → grounded generation → safe routing → evaluation.**

The current results demonstrate that the pipeline works end-to-end, while the evaluation limitations are explicitly documented rather than hidden.