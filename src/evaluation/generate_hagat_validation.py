from pathlib import Path
import os
import torch
import pandas as pd

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_FILE = "data/processed/clean_train.csv"
VAL_FILE = "data/processed/val_split.csv"
MODEL_FILE = "saved_models/best_hagat_clean_model.pth"
OUTPUT_FILE = "experiments/hagat_clean_validation_predictions.pt"

graph = build_hagat_graph(GRAPH_FILE).to(device)

drug_ids, microbe_ids, labels = create_training_dataset(
    VAL_FILE,
    num_drugs=1394,
    num_microbes=180,
)

drug_ids = drug_ids.long().to(device)
microbe_ids = microbe_ids.long().to(device)
labels = labels.float().to(device)

if drug_ids.numel() == 0:
    raise RuntimeError("Validation dataset produced no samples.")

if drug_ids.max().item() >= graph["drug"].x.shape[0]:
    raise ValueError("Validation drug ID exceeds graph range.")

if microbe_ids.max().item() >= graph["microbe"].x.shape[0]:
    raise ValueError("Validation microbe ID exceeds graph range.")

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4,
).to(device)

decoder = EdgeDecoder(64).to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
)

model.load_state_dict(
    checkpoint["hagat_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

model.eval()
decoder.eval()

with torch.no_grad():
    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict,
    )

    logits = decoder(
        embeddings["drug"][drug_ids],
        embeddings["microbe"][microbe_ids],
    ).squeeze()

    probabilities = torch.sigmoid(logits)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

torch.save(
    {
        "y_true": labels.cpu(),
        "y_score": probabilities.cpu(),
    },
    OUTPUT_FILE,
)

print("=" * 60)
print("HaGAT Validation Predictions")
print("=" * 60)
print(f"Samples       : {len(labels)}")
print(f"Positive      : {int(labels.sum().item())}")
print(f"Negative      : {int((labels == 0).sum().item())}")
print(f"Loaded Epoch  : {checkpoint['epoch']}")
print(f"Best Loss     : {checkpoint['loss']:.6f}")
print(f"Saved         : {OUTPUT_FILE}")
print("=" * 60)
