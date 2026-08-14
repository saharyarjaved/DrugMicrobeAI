import os
import torch
import pandas as pd

from src.data.build_gcn_graph import build_gcn_graph
from src.models.gat import GATModel
from src.models.edge_decoder import EdgeDecoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 60)
print("Clean GAT Evaluation")
print("=" * 60)

print(f"Device : {device}")

# ============================================================
# Load Clean Train Graph
# ============================================================

graph = build_gcn_graph(
    "data/processed/clean_train.csv"
)

graph = graph.to(device)

print(f"Graph Nodes : {graph.num_nodes}")

# ============================================================
# Load Clean Test Data
# IMPORTANT: Use actual labels from clean_test.csv
# ============================================================

test_df = pd.read_csv(
    "data/processed/clean_test.csv"
)

drug_ids = torch.tensor(
    test_df["Drug_ID"].values,
    dtype=torch.long
)

microbe_ids = torch.tensor(
    test_df["Microbe_ID"].values,
    dtype=torch.long
)

labels = torch.tensor(
    test_df["Label"].values,
    dtype=torch.float
)

# GAT uses one combined node space.
# Microbe IDs must be shifted after drug nodes.

microbe_ids = (
    microbe_ids + graph.num_drugs
)

drug_ids = drug_ids.to(device)
microbe_ids = microbe_ids.to(device)
labels = labels.to(device)

print(f"Test Samples : {len(labels)}")
print(
    f"Test Positives : "
    f"{int((labels == 1).sum().item())}"
)
print(
    f"Test Negatives : "
    f"{int((labels == 0).sum().item())}"
)

# ============================================================
# Safety Checks
# ============================================================

if drug_ids.max().item() >= graph.num_drugs:
    raise ValueError(
        f"Drug ID {drug_ids.max().item()} "
        f"outside graph range."
    )

if microbe_ids.max().item() >= graph.num_nodes:
    raise ValueError(
        f"Microbe node ID {microbe_ids.max().item()} "
        f"outside graph range."
    )

# ============================================================
# Model
# ============================================================

model = GATModel(
    input_dim=graph.x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4
).to(device)

# ============================================================
# Decoder
# ============================================================

decoder = EdgeDecoder(
    64
).to(device)

# ============================================================
# Load Clean GAT Checkpoint
# ============================================================

checkpoint = torch.load(
    "saved_models/best_gat_clean_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["gat_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

print(
    f"Loaded Epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Loss : "
    f"{checkpoint['loss']:.6f}"
)

# ============================================================
# Evaluation
# ============================================================

model.eval()
decoder.eval()

with torch.no_grad():

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

    logits = decoder(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).int()

# ============================================================
# NumPy
# ============================================================

y_true = labels.cpu().numpy()
y_pred = predictions.cpu().numpy()
y_score = probabilities.cpu().numpy()

# =====================================
# Save Predictions
# =====================================

os.makedirs("experiments", exist_ok=True)

torch.save(
    {
        "y_true": torch.tensor(y_true),
        "y_pred": torch.tensor(y_pred),
        "y_score": torch.tensor(y_score)
    },
    "experiments/gat_clean_predictions.pt"
)

print("Saved: experiments/gat_clean_predictions.pt")

# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_score
)

# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("Clean GAT Test Results")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1          : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("=" * 60)

# ============================================================
# Save Results
# ============================================================

os.makedirs(
    "experiments",
    exist_ok=True
)

results_file = (
    "experiments/results.csv"
)

new_result = pd.DataFrame(
    [
        {
            "Experiment": "GAT_CLEAN",
            "Model": "GAT_CLEAN",
            "Hidden": 128,
            "Output": 64,
            "Heads": 4,
            "LearningRate": 0.001,
            "Epochs": checkpoint["epoch"],
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    ]
)

if os.path.exists(results_file):

    existing = pd.read_csv(
        results_file
    )

    existing = existing[
        existing["Experiment"]
        .astype(str)
        .str.strip()
        .str.upper()
        != "GAT_CLEAN"
    ]

    results = pd.concat(
        [
            existing,
            new_result
        ],
        ignore_index=True
    )

else:

    results = new_result

results.to_csv(
    results_file,
    index=False
)

print(
    "\nGAT_CLEAN result saved to:"
)

print(
    results_file
)

print("=" * 60)

