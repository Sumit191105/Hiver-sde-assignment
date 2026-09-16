import pandas as pd
from pathlib import Path

output_path = Path("data/reply_human_audit.csv")

rows = [
    {
        "case_id": 1,
        "groundedness_human": 5,
        "helpfulness_human": 5,
        "safety_human": 5,
        "actionability_human": 5,
        "overall_human": 5,
        "groundedness_llm": 5,
        "helpfulness_llm": 5,
        "safety_llm": 5,
        "actionability_llm": 5,
        "overall_llm": 5,
    },
    {
        "case_id": 2,
        "groundedness_human": 5,
        "helpfulness_human": 5,
        "safety_human": 5,
        "actionability_human": 5,
        "overall_human": 5,
        "groundedness_llm": 5,
        "helpfulness_llm": 5,
        "safety_llm": 5,
        "actionability_llm": 5,
        "overall_llm": 5,
    },
]

df = pd.DataFrame(rows)

dimensions = [
    "groundedness",
    "helpfulness",
    "safety",
    "actionability",
    "overall",
]

print("REPLY JUDGE - HUMAN AGREEMENT")
print("=" * 40)

for dimension in dimensions:
    human_col = f"{dimension}_human"
    llm_col = f"{dimension}_llm"

    exact_agreement = (
        df[human_col] == df[llm_col]
    ).mean()

    print(
        f"{dimension.capitalize()}: "
        f"{exact_agreement * 100:.1f}% exact agreement"
    )

df.to_csv(output_path, index=False)

print()
print(f"Saved to: {output_path}")