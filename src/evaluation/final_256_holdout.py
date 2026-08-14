from pathlib import Path
import os
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_FILE = "data/processed/clean_train.csv"
TEST_FILE = "data/processed/clean_test.csv"
MODEL_FILE = "saved_models/hyperparameter/hagat_hidden256_confirmed.pth"

OUTPUT_DIR = Path("experiments/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

graph = build_hagat_graph(GRAPH_FILE).to(DEVICE)

drug_ids, microbe_ids, labels = create_training_dataset(
    TEST_FILE,
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

y_true = labels.cpu().numpy()
y_score = scores.cpu().numpy()

# Default threshold for initial evaluation
threshold = 0.5
y_pred = (y_score >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

result = {
    "Model": "HaGAT Hidden256",
    "Hidden": 256,
    "Heads": 4,
    "Output": 64,
    "Epoch": checkpoint["epoch"],
    "Threshold": threshold,
    "Samples": len(y_true),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(
        y_true, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_true, y_pred, zero_division=0
    ),
    "Specificity": specificity,
    "F1": f1_score(
        y_true, y_pred, zero_division=0
    ),
    "Balanced_Accuracy": balanced_accuracy_score(
        y_true, y_pred
    ),
    "MCC": matthews_corrcoef(
        y_true, y_pred
    ),
    "ROC_AUC": roc_auc_score(
        y_true, y_score
    ),
    "PR_AUC": average_precision_score(
        y_true, y_score
    ),
}

output = OUTPUT_DIR / "hagat_hidden256_holdout.csv"

pd.DataFrame([result]).to_csv(
    output,
    index=False,
)

print("=" * 70)
print("FINAL HaGAT 256-HIDDEN HOLDOUT")
print("=" * 70)

for key, value in result.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.4f}")
    else:
        print(f"{key:20s}: {value}")

print(f"\nSaved: {output}")
print("=" * 70)
