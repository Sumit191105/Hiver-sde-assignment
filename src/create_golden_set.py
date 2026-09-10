import pandas as pd

INPUT_FILE = "data/training_sample.csv"
OUTPUT_FILE = "data/golden_set.csv"

df = pd.read_csv(INPUT_FILE)

# Select 200 random customer messages
golden = df.sample(
    n=200,
    random_state=42
).copy()

# Add empty label column
golden["intent"] = ""

# Save golden dataset
golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print("==============================")
print("GOLDEN SET CREATED")
print("==============================")
print("Rows:", len(golden))
print("Saved to:", OUTPUT_FILE)