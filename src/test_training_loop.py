import os
import torch
from torch.optim import Adam

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gcn import GCNModel
from src.models.edge_decoder import EdgeDecoder

from src.training.train_loop import train_one_epoch
from src.utils.plot_loss import plot_loss

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}\n")

# =====================================
# Graph
# =====================================

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

graph = graph.to(device)

# =====================================
# Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/train_encoded.csv",
    num_drugs=1394,
    num_microbes=180
)

# Shift Microbe IDs
microbe_ids = microbe_ids + graph.num_drugs

drug_ids = drug_ids.to(device)
microbe_ids = microbe_ids.to(device)
labels = labels.to(device)

# =====================================
# Model
# =====================================

model = GCNModel(
    input_dim=graph.num_nodes,
    hidden_dim=128,
    output_dim=64
).to(device)

decoder = EdgeDecoder(64).to(device)

optimizer = Adam(
    list(model.parameters()) +
    list(decoder.parameters()),
    lr=0.001
)

# =====================================
# Create Folders
# =====================================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================================
# Training
# =====================================

best_loss = float("inf")
losses = []

print("=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(1, 101):

    loss = train_one_epoch(
        model,
        decoder,
        graph,
        drug_ids,
        microbe_ids,
        labels,
        optimizer,
        device
    )

    losses.append(loss)

    print(f"Epoch {epoch:03d} | Loss = {loss:.4f}")

    if loss < best_loss:

        best_loss = loss

        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "gcn_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict()
            },
            "saved_models/best_model.pth"
        )

# =====================================
# Plot Loss Curve
# =====================================

plot_loss(
    losses,
    save_path="results/loss_curve.png"
)

# =====================================
# Training Summary
# =====================================

print("\n" + "=" * 60)
print("Training Completed")
print(f"Best Loss : {best_loss:.4f}")
print("Model Saved : saved_models/best_model.pth")
print("Loss Curve : results/loss_curve.png")
print("=" * 60)