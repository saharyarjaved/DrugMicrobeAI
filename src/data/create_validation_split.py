import os
import pandas as pd
from sklearn.model_selection import train_test_split


# =====================================
# Paths
# =====================================

INPUT_PATH = "data/processed/train_encoded.csv"

TRAIN_PATH = "data/processed/train_split.csv"
VAL_PATH = "data/processed/val_split.csv"


# =====================================
# Load Dataset
# =====================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("Creating Train / Validation Split")
print("=" * 60)

print(f"Total Samples : {len(df)}")


# =====================================
# Split Dataset
# =====================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)


# =====================================
# Create Output Folder
# =====================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# =====================================
# Save Datasets
# =====================================

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

val_df.to_csv(
    VAL_PATH,
    index=False
)


# =====================================
# Summary
# =====================================

print(f"Training Samples   : {len(train_df)}")
print(f"Validation Samples : {len(val_df)}")

print(f"\nTrain Saved : {TRAIN_PATH}")
print(f"Val Saved   : {VAL_PATH}")

print("=" * 60)
print("Validation Split Created Successfully")
print("=" * 60)