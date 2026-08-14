import os
import torch
import pandas as pd

from src.data.build_gcn_graph import build_gcn_graph
from src.models.gcn import GCNModel
from src.models.link_predictor import LinkPredictor

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
print("Clean GCN Evaluation")
print("=" * 60)

print(f"Device : {device}")

# ============================================================
# Build graph ONLY from clean training data
# ============================================================

graph = build_gcn_graph(
    "data/processed/clean_train.csv"
)

graph = graph.to(device)

print(f"Graph Nodes : {graph.num_nodes}")

# ============================================================
# Load actual clean test data
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

# GCN uses combined node space
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
# Safety checks
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

model = GCNModel(
    input_dim=graph.x.shape[1],
    hidden_dim=128,
    output_dim=64
).to(device)

predictor = LinkPredictor(
    64
).to(device)

# ============================================================
# Load best clean model
# ============================================================

checkpoint = torch.load(
    "saved_models/best_gcn_clean_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["gcn_state_dict"]
)

predictor.load_state_dict(
    checkpoint["predictor_state_dict"]
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
predictor.eval()

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

    probabilities = predictor(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    predictions = (
        probabilities >= 0.5
    ).int()

# ============================================================
# Metrics
# ============================================================

y_true = labels.cpu().numpy()
y_pred = predictions.cpu().numpy()
y_score = probabilities.cpu().numpy()

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
print("Clean GCN Test Results")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1          : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("=" * 60)

# ============================================================
# Save results
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
            "Experiment": "GCN_CLEAN",
            "Model": "GCN_CLEAN",
            "Hidden": 128,
            "Output": 64,
            "Heads": "-",
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
        != "GCN_CLEAN"
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
    "\nGCN_CLEAN result saved to:"
)

print(results_file)

print("=" * 60)
# ============================================================
# Save Clean GCN Predictions
# ============================================================

prediction_file = "experiments/gcn_clean_predictions.pt"

torch.save(
    {
        "y_true": torch.tensor(y_true),
        "y_pred": torch.tensor(y_pred),
        "y_score": torch.tensor(y_score)
    },
    prediction_file
)

print(
    f"Predictions saved to: {prediction_file}"
)
