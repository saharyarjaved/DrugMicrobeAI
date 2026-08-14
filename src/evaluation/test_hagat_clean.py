import os
import torch
import pandas as pd

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset

from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 60)
print("HaGAT Evaluation")
print("=" * 60)

print(f"Device : {device}")


# =====================================
# Build Graph
# =====================================

graph = build_hagat_graph(
    "data/processed/clean_train.csv"
)

graph = graph.to(device)


# =====================================
# Test Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/clean_test.csv",
    num_drugs=1394,
    num_microbes=180
)

drug_ids = drug_ids.long().to(device)
microbe_ids = microbe_ids.long().to(device)
labels = labels.float().to(device)


# =====================================
# Safety Check
# =====================================

if drug_ids.max().item() >= graph["drug"].x.shape[0]:
    raise ValueError(
        f"Drug ID {drug_ids.max().item()} "
        f"outside graph range."
    )

if microbe_ids.max().item() >= graph["microbe"].x.shape[0]:
    raise ValueError(
        f"Microbe ID {microbe_ids.max().item()} "
        f"outside graph range."
    )


# =====================================
# Model
# =====================================

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4
).to(device)


# =====================================
# Decoder
# =====================================

decoder = EdgeDecoder(
    64
).to(device)


# =====================================
# Load Best Model
# =====================================

checkpoint = torch.load(
    "saved_models/best_hagat_clean_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["hagat_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

print(
    f"Loaded Epoch : {checkpoint['epoch']}"
)

print(
    f"Best Loss : {checkpoint['loss']:.6f}"
)


# =====================================
# Evaluation
# =====================================

model.eval()
decoder.eval()

with torch.no_grad():

    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict
    )

    drug_embeddings = embeddings["drug"]

    microbe_embeddings = embeddings["microbe"]

    selected_drug_embeddings = (
        drug_embeddings[drug_ids]
    )

    selected_microbe_embeddings = (
        microbe_embeddings[microbe_ids]
    )

    logits = decoder(
        selected_drug_embeddings,
        selected_microbe_embeddings
    ).squeeze()

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).int()


# =====================================
# NumPy
# =====================================

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
    "experiments/hagat_clean_predictions.pt"
)

print("Saved: experiments/hagat_clean_predictions.pt")


# =====================================
# Metrics
# =====================================

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


# =====================================
# Print Results
# =====================================

print("\n" + "=" * 60)
print("HaGAT Test Results")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1          : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("=" * 60)


# =====================================
# Save Results
# =====================================

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
            "Experiment": "HAGAT_CLEAN",
            "Model": "HaGAT",
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


# =====================================
# Append / Create Results
# =====================================

if os.path.exists(results_file):

    existing = pd.read_csv(
        results_file
    )

    # Remove previous HaGAT result
    # to avoid duplicate entries

    existing = existing[
        existing["Model"]
        .astype(str)
        .str.strip()
        .str.upper()
        != "HAGAT"
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


# =====================================
# Save CSV
# =====================================

results.to_csv(
    results_file,
    index=False
)


print(
    f"\nHaGAT result saved to:"
)

print(
    results_file
)

print("=" * 60)

