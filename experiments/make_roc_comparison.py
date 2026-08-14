import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

files = {
    "GCN": "experiments/gcn_clean_predictions.pt",
    "GAT": "experiments/gat_clean_predictions.pt",
    "HaGAT": "experiments/hagat_clean_predictions.pt"
}

plt.figure(figsize=(8, 7))

for model, path in files.items():
    p = torch.load(path, map_location="cpu")

    y_true = p["y_true"].numpy()
    y_score = p["y_score"].numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.plot(
        fpr,
        tpr,
        label=f"{model} (AUC={auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Clean ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output = "experiments/clean_roc_comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("ROC-AUC Results:")
for model, path in files.items():
    p = torch.load(path, map_location="cpu")
    y_true = p["y_true"].numpy()
    y_score = p["y_score"].numpy()
    print(f"{model}: {roc_auc_score(y_true, y_score):.4f}")

print(f"\nSaved: {output}")
