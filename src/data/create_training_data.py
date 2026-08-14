import torch

from src.data.link_dataset import load_link_dataset
from src.data.negative_sampling import generate_negative_edges


def create_training_dataset(csv_path, num_drugs, num_microbes):

    # --------------------
    # Positive Samples
    # --------------------
    drug_ids, microbe_ids, _ = load_link_dataset(csv_path)

    positive_labels = torch.ones(len(drug_ids))

    # --------------------
    # Negative Samples
    # --------------------
    neg_drug, neg_microbe = generate_negative_edges(
        drug_ids,
        microbe_ids,
        num_drugs,
        num_microbes
    )

    negative_labels = torch.zeros(len(neg_drug))

    # --------------------
    # Combine
    # --------------------
    all_drugs = torch.cat(
        [drug_ids, neg_drug]
    )

    all_microbes = torch.cat(
        [microbe_ids, neg_microbe]
    )

    all_labels = torch.cat(
        [positive_labels, negative_labels]
    )

    return (
        all_drugs,
        all_microbes,
        all_labels
    )