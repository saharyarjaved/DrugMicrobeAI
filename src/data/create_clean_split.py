import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/train_encoded.csv"

TRAIN_OUT = "data/processed/clean_train.csv"
TEST_OUT = "data/processed/clean_test.csv"

SEED = 42

# ============================================
# Load complete dataset
# ============================================

df = pd.read_csv(INPUT)

print("=" * 60)
print("Creating Clean Train/Test Split")
print("=" * 60)

print("Original shape:", df.shape)
print("\nOriginal labels:")
print(df["Label"].value_counts().to_dict())

# ============================================
# Remove duplicate Drug-Microbe pairs
# ============================================

df = df.drop_duplicates(
    subset=["Drug_ID", "Microbe_ID", "Label"]
).reset_index(drop=True)

# ============================================
# Stratified split
# ============================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["Label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ============================================
# Safety checks
# ============================================

train_pairs = set(
    zip(
        train_df["Drug_ID"],
        train_df["Microbe_ID"]
    )
)

test_pairs = set(
    zip(
        test_df["Drug_ID"],
        test_df["Microbe_ID"]
    )
)

overlap = train_pairs & test_pairs

# ============================================
# Save
# ============================================

train_df.to_csv(
    TRAIN_OUT,
    index=False
)

test_df.to_csv(
    TEST_OUT,
    index=False
)

# ============================================
# Report
# ============================================

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\nTrain labels:")
print(train_df["Label"].value_counts().to_dict())

print("\nTest labels:")
print(test_df["Label"].value_counts().to_dict())

print("\nTrain/Test pair overlap:", len(overlap))

print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)

print("=" * 60)