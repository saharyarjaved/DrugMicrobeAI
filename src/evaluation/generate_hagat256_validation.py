from pathlib import Path
import os
import torch

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_FILE = "data/processed/clean_train.csv"
VAL_FILE = "data/processed/val_split.csv"
MODEL_FILE = "saved_models/hyperparameter/hagat_hidden256_confirmed.pth"
OUTPUT_FILE = "experiments/final/hagat_hidden256_validation_predictions.pt"

os.makedirs(
    "experiments/final",
    exist_ok=True,
)

graph = build_hagat_graph(
    GRAPH_FILE
).to(DEVICE)

drug_ids, microbe_ids, labels = create_training_dataset(
    VAL_FILE,
    num_drugs=1394,
    num_microbes=180,
)

drug_ids = drug_ids.long().to(DEVICE)
microbe_ids = microbe_ids.long().to(DEVICE)
labels = labels.float().to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
)

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=256,
    output_dim=64,
    heads=4,
).to(DEVICE)

decoder = EdgeDecoder(64).to(DEVICE)

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

    scores = torch.sigmoid(logits)

torch.save(
    {
        "y_true": labels.cpu(),
        "y_score": scores.cpu(),
    },
    OUTPUT_FILE,
)

print("=" * 70)
print("HaGAT Hidden256 Validation Predictions")
print("=" * 70)
print(f"Samples      : {len(labels)}")
print(f"Positive     : {int(labels.sum().item())}")
print(f"Negative     : {int((labels == 0).sum().item())}")
print(f"Epoch        : {checkpoint['epoch']}")
print(f"Best loss    : {checkpoint['loss']:.6f}")
print(f"Saved        : {OUTPUT_FILE}")
print("=" * 70)
