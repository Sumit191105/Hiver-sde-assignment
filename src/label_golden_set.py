import pandas as pd

INPUT_FILE = "data/golden_set.csv"
OUTPUT_FILE = "data/golden_set.csv"

df = pd.read_csv(INPUT_FILE)
df["intent"] = df["intent"].fillna("").astype(str)

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
]

print("\n==============================")
print("GOLDEN SET LABELING")
print("==============================")

print("\nIntents:")
for i, intent in enumerate(INTENTS, start=1):
    print(f"{i}. {intent}")

print("\nEnter 1-9 to label a message.")
print("Enter 0 to skip.")
print("Enter 'q' to save and quit.\n")

for index in range(len(df)):

    # Skip already labelled messages
    if pd.notna(df.loc[index, "intent"]) and df.loc[index, "intent"] != "":
        continue

    print("\n" + "=" * 80)
    print(f"Message {index + 1}/{len(df)}")
    print("=" * 80)

    print(df.loc[index, "text"])

    while True:
        choice = input("\nYour choice: ").strip()

        if choice.lower() == "q":
            df.to_csv(OUTPUT_FILE, index=False)
            print("\nProgress saved.")
            exit()

        if choice == "0":
            break

        if choice.isdigit() and 1 <= int(choice) <= 9:
            df.loc[index, "intent"] = INTENTS[int(choice) - 1]
            break

        print("Invalid choice. Enter 1-9, 0, or q.")

    # Save after every message
    df.to_csv(OUTPUT_FILE, index=False)

print("\n==============================")
print("LABELING COMPLETE")
print("==============================")

print("\nIntent distribution:")
print(df["intent"].value_counts())