import torch
import torch.nn as nn


def train_one_epoch(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels,
    optimizer,
    device,
):

    model.train()
    decoder.train()

    optimizer.zero_grad()

    # -------------------------
    # Node Embeddings
    # -------------------------
    embeddings = model(
        graph.x.float(),
        graph.edge_index
    )

    # -------------------------
    # Drug Embeddings
    # -------------------------
    drug_embedding = embeddings[
        drug_ids
    ]

    # -------------------------
    # Microbe Embeddings
    # -------------------------
    microbe_embedding = embeddings[
        microbe_ids
    ]

    # -------------------------
    # Edge Prediction
    # -------------------------
    prediction = decoder(
        drug_embedding,
        microbe_embedding
    ).squeeze()

    # -------------------------
    # Loss
    # -------------------------
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(
        prediction,
        labels.float()
    )

    loss.backward()

    optimizer.step()

    return loss.item()