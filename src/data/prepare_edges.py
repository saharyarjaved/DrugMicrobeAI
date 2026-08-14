import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Random seed
random.seed(42)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Positive interactions
positive_edges = list(zip(df["Name"], df["Microbe"]))

print("Positive interactions:", len(positive_edges))

# Unique drugs & microbes
drugs = df["Name"].unique().tolist()
microbes = df["Microbe"].unique().tolist()

positive_set = set(positive_edges)

# Generate negative interactions
negative_edges = []

while len(negative_edges) < len(positive_edges):
    drug = random.choice(drugs)
    microbe = random.choice(microbes)

    if (drug, microbe) not in positive_set:
        negative_edges.append((drug, microbe))

print("Negative interactions:", len(negative_edges))

# Create labeled dataset
data = []

for d, m in positive_edges:
    data.append([d, m, 1])

for d, m in negative_edges:
    data.append([d, m, 0])

dataset = pd.DataFrame(
    data,
    columns=["Drug", "Microbe", "Label"]
)

# Shuffle
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/Test split
train_df, test_df = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    stratify=dataset["Label"]
)

# Save
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

print("\nLabel Distribution")
print(train_df["Label"].value_counts())