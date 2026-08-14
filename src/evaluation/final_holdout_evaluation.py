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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

THRESHOLD = 0.32

PREDICTIONS_FILE = "experiments/hagat_clean_predictions.pt"
OUTPUT_FILE = "experiments/hagat_clean_final_holdout.csv"

data = torch.load(
    PREDICTIONS_FILE,
    map_location="cpu",
)

y_true = data["y_true"].numpy()
y_score = data["y_score"].numpy()
y_pred = (y_score >= THRESHOLD).astype(int)

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
    "Model": "HaGAT Clean",
    "Dataset": "Clean Holdout",
    "Threshold": THRESHOLD,
    "Samples": len(y_true),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
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
    "Specificity": specificity,
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
    "ROC_AUC": roc_auc_score(
        y_true,
        y_score,
    ),
    "PR_AUC": average_precision_score(
        y_true,
        y_score,
    ),
}

pd.DataFrame([result]).to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("HaGAT Final Clean Holdout")
print("=" * 70)

for key, value in result.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.4f}")
    else:
        print(f"{key:20s}: {value}")

print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
