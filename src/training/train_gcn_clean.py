import os
import torch
from torch.optim import Adam

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gcn import GCNModel
from src.models.link_predictor import LinkPredictor

from src.config import (
    DEVICE,
    HIDDEN_DIM,
    OUTPUT_DIM,
    LEARNING_RATE,
    EPOCHS
)

# ============================================================
# Device
# ============================================================

print("\n" + "=" * 60)
print("Clean GCN Training")
print("=" * 60)

print(f"Device : {DEVICE}")

# ============================================================
# Build Graph ONLY from clean training data
# ============================================================

graph = build_gcn_graph(
    "data/processed/clean_train.csv"
)

graph = graph.to(DEVICE)

print(f"Graph Nodes : {graph.num_nodes}")

# ============================================================
# Training Dataset
# ============================================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/clean_train.csv",
    num_drugs=graph.num_drugs,
    num_microbes=graph.num_microbes
)

# GCN uses combined node ID space
microbe_ids = microbe_ids + graph.num_drugs

drug_ids = drug_ids.long().to(DEVICE)
microbe_ids = microbe_ids.long().to(DEVICE)
labels = labels.float().to(DEVICE)

print(f"Training Samples : {len(labels)}")

# ============================================================
# Model
# ============================================================

model = GCNModel(
    input_dim=graph.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
).to(DEVICE)

predictor = LinkPredictor(
    OUTPUT_DIM
).to(DEVICE)

# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    list(model.parameters()) +
    list(predictor.parameters()),
    lr=LEARNING_RATE
)

criterion = torch.nn.BCELoss()

# ============================================================
# Save Folder
# ============================================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

# ============================================================
# Training
# ============================================================

best_loss = float("inf")

print("\n" + "=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    model.train()
    predictor.train()

    optimizer.zero_grad()

    embeddings = model(
        graph.x.float(),
        graph.edge_index
    )

    drug_embeddings = embeddings[
        drug_ids
    ]

    microbe_embeddings = embeddings[
        microbe_ids
    ]

    probabilities = predictor(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    loss = criterion(
        probabilities,
        labels
    )

    loss.backward()

    optimizer.step()

    loss_value = loss.item()

    print(
        f"Epoch {epoch:03d} | "
        f"Loss = {loss_value:.6f}"
    )

    if loss_value < best_loss:

        best_loss = loss_value

        torch.save(
            {
                "epoch": epoch,
                "loss": loss_value,
                "gcn_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict()
            },
            "saved_models/best_gcn_clean_model.pth"
        )

        print("Best GCN Model Saved")

# ============================================================
# Completed
# ============================================================

print("\n" + "=" * 60)
print("Clean GCN Training Completed")
print("=" * 60)

print(
    f"Best Loss : {best_loss:.6f}"
)

print(
    "Saved : saved_models/best_gcn_clean_model.pth"
)

print("=" * 60)
