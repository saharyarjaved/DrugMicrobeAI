from src.data.create_training_data import create_training_dataset

drug, microbe, labels = create_training_dataset(
    "data/processed/train_encoded.csv",
    num_drugs=1394,
    num_microbes=180
)

print("=" * 50)

print("Total Samples :", len(labels))

print("Positive :", (labels == 1).sum().item())

print("Negative :", (labels == 0).sum().item())

print("=" * 50)