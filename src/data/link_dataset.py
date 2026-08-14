import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_link_dataset(csv_path):

    df = pd.read_csv(csv_path)

    drug_ids = torch.tensor(
        df["Drug_ID"].values,
        dtype=torch.long
    )

    microbe_ids = torch.tensor(
        df["Microbe_ID"].values,
        dtype=torch.long
    )

    labels = torch.ones(
        len(df),
        dtype=torch.float
    )

    return drug_ids, microbe_ids, labels


def train_test_edges(
    drug_ids,
    microbe_ids,
    labels,
    test_size=0.2
):

    idx = list(range(len(labels)))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    return (
        drug_ids[train_idx],
        microbe_ids[train_idx],
        labels[train_idx],
        drug_ids[test_idx],
        microbe_ids[test_idx],
        labels[test_idx],
    )