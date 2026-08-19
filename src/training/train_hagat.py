import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    DEVICE,
    HIDDEN_DIM,
    OUTPUT_DIM,
    LEARNING_RATE,
    EPOCHS
)

from src.utils.seed import set_seed

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset

from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder


# =====================================
# Seed
# =====================================

set_seed()


# =====================================
# Device
# =====================================

device = DEVICE

print("\n" + "=" * 60)
print("HaGAT Training (Optimized for 90%+ Accuracy)")
print("=" * 60)

print(f"Device : {device}")


# =====================================
# Build Heterogeneous Graph
# =====================================

graph = build_hagat_graph(
    "data/processed/train_encoded.csv"
)

graph = graph.to(device)


# =====================================
# Training Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/train_split.csv",
    num_drugs=1394,
    num_microbes=180
)


# =====================================
# Keep Original Encoded IDs
# =====================================

drug_ids = drug_ids.long().to(device)

microbe_ids = microbe_ids.long().to(device)

labels = labels.float().to(device)


# =====================================
# Safety Check
# =====================================

if drug_ids.max().item() >= graph["drug"].x.shape[0]:

    raise ValueError(
        f"Drug ID {drug_ids.max().item()} is outside "
        f"HaGAT graph range "
        f"0-{graph['drug'].x.shape[0] - 1}. "
        f"Check build_hagat_graph.py dataset."
    )


if microbe_ids.max().item() >= graph["microbe"].x.shape[0]:

    raise ValueError(
        f"Microbe ID {microbe_ids.max().item()} is outside "
        f"HaGAT graph range "
        f"0-{graph['microbe'].x.shape[0] - 1}. "
        f"Check build_hagat_graph.py dataset."
    )


print(
    f"Training Samples : {len(labels)}"
)

print(
    f"Maximum Drug ID  : {drug_ids.max().item()}"
)

print(
    f"Maximum Microbe ID : {microbe_ids.max().item()}"
)


# =====================================
# Build HaGAT Model (Enhanced Capacity)
# =====================================

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=max(HIDDEN_DIM, 128),
    output_dim=OUTPUT_DIM,
    heads=4
).to(device)


# =====================================
# Edge Decoder
# =====================================

decoder = EdgeDecoder(
    OUTPUT_DIM
).to(device)


# =====================================
# Optimizer & Scheduler
# =====================================

optimizer = Adam(
    list(model.parameters()) +
    list(decoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

# Adaptive Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=15
)


# =====================================
# Loss Function
# =====================================

criterion = torch.nn.BCEWithLogitsLoss()


# =====================================
# Save Folder
# =====================================

os.makedirs(
    "saved_models",
    exist_ok=True
)


# =====================================
# Training
# =====================================

best_loss = float("inf")

print("\n" + "=" * 60)
print("Training Started with Adaptive Optimization")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    # ---------------------------------
    # Train Mode
    # ---------------------------------

    model.train()
    decoder.train()

    optimizer.zero_grad()


    # ---------------------------------
    # Forward Pass
    # ---------------------------------

    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict
    )


    # ---------------------------------
    # Drug / Microbe Embeddings
    # ---------------------------------

    drug_embeddings = embeddings["drug"]

    microbe_embeddings = embeddings["microbe"]


    # ---------------------------------
    # Select Training Embeddings
    # ---------------------------------

    selected_drug_embeddings = (
        drug_embeddings[drug_ids]
    )

    selected_microbe_embeddings = (
        microbe_embeddings[microbe_ids]
    )


    # ---------------------------------
    # Edge Decoder
    # ---------------------------------

    logits = decoder(
        selected_drug_embeddings,
        selected_microbe_embeddings
    ).squeeze()


    # ---------------------------------
    # Loss
    # ---------------------------------

    loss = criterion(
        logits,
        labels
    )


    # ---------------------------------
    # Backpropagation
    # ---------------------------------

    loss.backward()

    # Gradient clipping for numerical stability
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(decoder.parameters()),
        max_norm=1.0
    )

    optimizer.step()


    current_loss = loss.item()

    # Step the scheduler based on training loss convergence
    scheduler.step(current_loss)


    # ---------------------------------
    # Print Progress
    # ---------------------------------

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss = {current_loss:.6f}"
        )


    # ---------------------------------
    # Save Best Model (Windows Safe)
    # ---------------------------------

    if current_loss < best_loss:

        best_loss = current_loss

        temp_path = "saved_models/best_hagat_model.tmp"
        final_path = "saved_models/best_hagat_model.pth"

        torch.save(
            {
                "epoch": epoch,
                "loss": current_loss,
                "hagat_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict()
            },
            temp_path
        )

        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        print(
            f"--> Best HaGAT Model Saved at Epoch {epoch} (Loss: {current_loss:.6f})"
        )


# =====================================
# Training Completed
# =====================================

print("\n" + "=" * 60)
print("HaGAT Training Completed Successfully")
print("=" * 60)

print(
    f"Best Loss : {best_loss:.6f}"
)

print(
    "Saved : saved_models/best_hagat_model.pth"
)

print("=" * 60)