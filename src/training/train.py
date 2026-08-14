import os
import csv
import torch
from torch.optim import Adam

from src.utils.seed import set_seed
from src.utils.logger import log

from src.config import (
    DEVICE,
    HIDDEN_DIM,
    OUTPUT_DIM,
    LEARNING_RATE,
    EPOCHS
)

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gcn import GCNModel
from src.models.edge_decoder import EdgeDecoder

from src.training.train_loop import train_one_epoch
from src.training.early_stopping import EarlyStopping


# =====================================
# Seed
# =====================================

set_seed()


# =====================================
# Load Graph
# =====================================

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

graph = graph.to(DEVICE)

log("Graph Loaded")


# =====================================
# Training Dataset
# =====================================

train_drug_ids, train_microbe_ids, train_labels = (
    create_training_dataset(
        "data/processed/train_split.csv",
        num_drugs=graph.num_drugs,
        num_microbes=graph.num_microbes
    )
)

# Shift microbe IDs
train_microbe_ids = (
    train_microbe_ids + graph.num_drugs
)

train_drug_ids = train_drug_ids.to(DEVICE)
train_microbe_ids = train_microbe_ids.to(DEVICE)
train_labels = train_labels.to(DEVICE)

log(f"Training Samples: {len(train_labels)}")


# =====================================
# Validation Dataset
# =====================================

val_drug_ids, val_microbe_ids, val_labels = (
    create_training_dataset(
        "data/processed/val_split.csv",
        num_drugs=graph.num_drugs,
        num_microbes=graph.num_microbes
    )
)

# Shift microbe IDs
val_microbe_ids = (
    val_microbe_ids + graph.num_drugs
)

val_drug_ids = val_drug_ids.to(DEVICE)
val_microbe_ids = val_microbe_ids.to(DEVICE)
val_labels = val_labels.to(DEVICE)

log(f"Validation Samples: {len(val_labels)}")


# =====================================
# Build GCN Model
# =====================================

model = GCNModel(
    input_dim=graph.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
).to(DEVICE)


# =====================================
# Edge Decoder
# =====================================

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
# Learning Rate Scheduler
# =====================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


# =====================================
# Create Folders
# =====================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

os.makedirs(
    "saved_models/checkpoints",
    exist_ok=True
)

os.makedirs(
    "logs",
    exist_ok=True
)


# =====================================
# Training History
# =====================================

history_path = "logs/training_history.csv"

with open(
    history_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Training_Loss",
        "Validation_Loss",
        "Learning_Rate"
    ])


# =====================================
# Early Stopping
# =====================================

early_stopping = EarlyStopping(
    patience=20,
    min_delta=0.0001,
    save_path="saved_models/best_model.pth"
)


# =====================================
# Validation Function
# =====================================

@torch.no_grad()
def calculate_validation_loss():

    model.eval()
    decoder.eval()

    embeddings = model(
        graph.x.float(),
        graph.edge_index
    )

    drug_embeddings = embeddings[
        val_drug_ids
    ]

    microbe_embeddings = embeddings[
        val_microbe_ids
    ]

    logits = decoder(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    criterion = torch.nn.BCEWithLogitsLoss()

    validation_loss = criterion(
        logits,
        val_labels.float()
    )

    return validation_loss.item()


# =====================================
# Training
# =====================================

print("\n" + "=" * 60)
print("GCN Training Started")
print("=" * 60)

print(f"Device        : {DEVICE}")
print(f"Hidden Dim    : {HIDDEN_DIM}")
print(f"Output Dim    : {OUTPUT_DIM}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Max Epochs    : {EPOCHS}")

print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    # ---------------------------------
    # Training
    # ---------------------------------

    train_loss = train_one_epoch(
        model,
        decoder,
        graph,
        train_drug_ids,
        train_microbe_ids,
        train_labels,
        optimizer,
        DEVICE
    )


    # ---------------------------------
    # Validation
    # ---------------------------------

    validation_loss = calculate_validation_loss()


    # ---------------------------------
    # Learning Rate Scheduler
    # ---------------------------------

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ---------------------------------
    # Print Results
    # ---------------------------------

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss = {train_loss:.6f} | "
        f"Val Loss = {validation_loss:.6f} | "
        f"LR = {current_lr:.7f}"
    )


    # ---------------------------------
    # Save Training History
    # ---------------------------------

    with open(
        history_path,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch,
            train_loss,
            validation_loss,
            current_lr
        ])


    # ---------------------------------
    # Early Stopping
    # ---------------------------------

    should_stop = early_stopping.step(
        validation_loss,
        model,
        decoder,
        optimizer,
        epoch
    )


    # ---------------------------------
    # Periodic Checkpoint
    # ---------------------------------

    if epoch % 10 == 0:

        checkpoint_path = (
            f"saved_models/checkpoints/"
            f"epoch_{epoch:03d}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "gcn_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            },
            checkpoint_path
        )

        print(
            f"Checkpoint Saved : {checkpoint_path}"
        )


    # ---------------------------------
    # Stop Training
    # ---------------------------------

    if should_stop:

        print(
            f"\nTraining stopped at epoch {epoch}."
        )

        break


# =====================================
# Training Completed
# =====================================

print("\n" + "=" * 60)
print("GCN Training Completed")
print("=" * 60)

print(
    f"Best Validation Loss : "
    f"{early_stopping.best_loss:.6f}"
)

print(
    "Best Model : "
    "saved_models/best_model.pth"
)

print(
    "History : "
    "logs/training_history.csv"
)

print("=" * 60)