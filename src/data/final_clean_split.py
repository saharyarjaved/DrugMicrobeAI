import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/unique_pairs.csv"
TRAIN_OUT = "data/processed/final_train.csv"
TEST_OUT = "data/processed/final_test.csv"

SEED = 42

df = pd.read_csv(INPUT)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["Label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_pairs = set(
    zip(train_df["Drug_ID"], train_df["Microbe_ID"])
)

test_pairs = set(
    zip(test_df["Drug_ID"], test_df["Microbe_ID"])
)

overlap = train_pairs & test_pairs

train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("=" * 60)
print("FINAL CLEAN DATASET SPLIT")
print("=" * 60)

print("Total:", len(df))
print("Train:", len(train_df))
print("Test :", len(test_df))

print("\nTrain labels:")
print(train_df["Label"].value_counts().to_dict())

print("\nTest labels:")
print(test_df["Label"].value_counts().to_dict())

print("\nTrain/Test overlap:", len(overlap))

print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)

print("=" * 60)