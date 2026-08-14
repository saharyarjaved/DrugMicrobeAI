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

PREDICTIONS_FILE = "experiments/hagat_clean_validation_predictions.pt"
OUTPUT_FILE = "experiments/hagat_clean_threshold_analysis.csv"

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
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "F1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_true,
                y_pred,
            ),
            "MCC": matthews_corrcoef(
                y_true,
                y_pred,
            ),
        }
    )

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

best_f1 = results.loc[
    results["F1"].idxmax()
]

best_balanced = results.loc[
    results["Balanced_Accuracy"].idxmax()
]

best_mcc = results.loc[
    results["MCC"].idxmax()
]

print("\n" + "=" * 70)
print("HaGAT Validation Threshold Analysis")
print("=" * 70)

print("\nBest F1 threshold:")
print(best_f1.to_string())

print("\nBest Balanced Accuracy threshold:")
print(best_balanced.to_string())

print("\nBest MCC threshold:")
print(best_mcc.to_string())

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)
