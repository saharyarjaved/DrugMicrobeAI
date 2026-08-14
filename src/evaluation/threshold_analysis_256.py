from pathlib import Path
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

PREDICTIONS_FILE = "experiments/final/hagat_hidden256_validation_predictions.pt"
OUTPUT_FILE = "experiments/final/hagat_hidden256_threshold_analysis.csv"

data = torch.load(
    PREDICTIONS_FILE,
    map_location="cpu",
)

y_true = data["y_true"].numpy()
y_score = data["y_score"].numpy()

rows = []

for threshold in [x / 100 for x in range(30, 71)]:
    y_pred = (y_score >= threshold).astype(int)

    rows.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "F1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "MCC": matthews_corrcoef(
                y_true, y_pred
            ),
        }
    )

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("=" * 70)
print("HaGAT Hidden256 Validation Threshold Analysis")
print("=" * 70)

for label, column in [
    ("Best F1", "F1"),
    ("Best Balanced Accuracy", "Balanced_Accuracy"),
    ("Best MCC", "MCC"),
]:
    row = results.loc[
        results[column].idxmax()
    ]

    print(f"\n{label}:")
    print(row.to_string())

print(f"\nSaved: {OUTPUT_FILE}")
print("=" * 70)
