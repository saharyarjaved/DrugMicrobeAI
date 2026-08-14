from src.data.link_dataset import (
    load_link_dataset,
    train_test_edges
)

drug, microbe, labels = load_link_dataset(
    "data/processed/train_encoded.csv"
)

print(drug.shape)
print(microbe.shape)
print(labels.shape)

train = train_test_edges(
    drug,
    microbe,
    labels
)

print("\nTrain Size:", len(train[0]))
print("Test Size :", len(train[3]))