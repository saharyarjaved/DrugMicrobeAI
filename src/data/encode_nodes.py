import pandas as pd

# Load train & test
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Combine
all_data = pd.concat([train, test], ignore_index=True)

# Unique nodes
drugs = sorted(all_data["Drug"].unique())
microbes = sorted(all_data["Microbe"].unique())

drug_to_id = {d: i for i, d in enumerate(drugs)}
microbe_to_id = {m: i for i, m in enumerate(microbes)}

# Encode
train["Drug_ID"] = train["Drug"].map(drug_to_id)
train["Microbe_ID"] = train["Microbe"].map(microbe_to_id)

test["Drug_ID"] = test["Drug"].map(drug_to_id)
test["Microbe_ID"] = test["Microbe"].map(microbe_to_id)

# Save
train.to_csv("data/processed/train_encoded.csv", index=False)
test.to_csv("data/processed/test_encoded.csv", index=False)

print("Train:", train.shape)
print("Test :", test.shape)

print("\nFirst 5 rows\n")
print(train.head())