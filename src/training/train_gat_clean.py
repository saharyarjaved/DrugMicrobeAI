import os
import torch
from torch.optim import Adam

from src.config import (
    DEVICE,
    HIDDEN_DIM,
    OUTPUT_DIM,
    HEADS,
    LEARNING_RATE,
    EPOCHS
)

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gat import GATModel
from src.models.edge_decoder import EdgeDecoder

from src.training.train_loop import train_one_epoch


# =====================================
# Device
# =====================================

print(f"\nUsing Device : {DEVICE}\n")


# =====================================
# Graph
# =====================================

graph = build_gcn_graph(
    "data/processed/clean_train.csv"
)

graph = graph.to(DEVICE)


# =====================================
# Training Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/clean_train.csv",
    num_drugs=graph.num_drugs,
    num_microbes=graph.num_microbes
)

# Shift Microbe IDs
microbe_ids = microbe_ids + graph.num_drugs

drug_ids = drug_ids.to(DEVICE)
microbe_ids = microbe_ids.to(DEVICE)
labels = labels.to(DEVICE)


# =====================================
# Model
# =====================================

model = GATModel(
    input_dim=graph.num_nodes,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    heads=HEADS
).to(DEVICE)

decoder = EdgeDecoder(
    OUTPUT_DIM
).to(DEVICE)


# =====================================
# Optimizer
# =====================================

optimizer = Adam(
    list(model.parameters()) +
    list(decoder.parameters()),
    lr=LEARNING_RATE
)


# =====================================
# Save Folder
# =====================================

os.makedirs(
    "saved_models",
    exist_ok=True
)


# =====================================
# Training Loop
# =====================================

best_loss = float("inf")

print("=" * 60)
print("Training GAT Started")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    loss = train_one_epoch(
        model,
        decoder,
        graph,
        drug_ids,
        microbe_ids,
        labels,
        optimizer,
        DEVICE
    )

    print(f"Epoch {epoch:03d} | Loss = {loss:.4f}")

    if loss < best_loss:

        best_loss = loss

        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "gat_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict()
            },
            "saved_models/best_gat_clean_model.pth"
        )


print("\n" + "=" * 60)
print("Training Completed")
print(f"Best Loss : {best_loss:.4f}")
print("Saved : saved_models/best_gat_clean_model.pth")
print("=" * 60)
