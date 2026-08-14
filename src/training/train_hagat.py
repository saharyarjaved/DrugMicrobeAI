import os
import torch
from torch.optim import Adam

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
print("HaGAT Training")
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
# Build HaGAT Model
# =====================================

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=HIDDEN_DIM,
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
# Optimizer
# =====================================

optimizer = Adam(
    list(model.parameters()) +
    list(decoder.parameters()),
    lr=LEARNING_RATE
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
print("Training Started")
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

    optimizer.step()


    current_loss = loss.item()


    # ---------------------------------
    # Print Progress
    # ---------------------------------

    print(
        f"Epoch {epoch:03d} | "
        f"Loss = {current_loss:.6f}"
    )


    # ---------------------------------
    # Save Best Model
    # ---------------------------------

    if current_loss < best_loss:

        best_loss = current_loss

        torch.save(
            {
                "epoch": epoch,
                "loss": current_loss,
                "hagat_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict()
            },
            "saved_models/best_hagat_model.pth"
        )

        print(
            "Best HaGAT Model Saved"
        )


# =====================================
# Training Completed
# =====================================

print("\n" + "=" * 60)
print("HaGAT Training Completed")
print("=" * 60)

print(
    f"Best Loss : {best_loss:.6f}"
)

print(
    "Saved : saved_models/best_hagat_model.pth"
)

print("=" * 60)