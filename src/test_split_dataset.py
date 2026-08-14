from src.data.split_dataset import split_dataset

split_dataset(
    csv_path="data/processed/train_encoded.csv",
    train_path="data/processed/train_split.csv",
    test_path="data/processed/test_split.csv"
)