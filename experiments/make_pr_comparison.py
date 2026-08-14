import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score
    )

    ap = average_precision_score(
        y_true,
        y_score
    )

    plt.plot(
        recall,
        precision,
        label=f"{model} (AP={ap:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Clean Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output = "experiments/clean_pr_comparison.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Average Precision Results:")

for model, path in files.items():
    p = torch.load(path, map_location="cpu")
    y_true = p["y_true"].numpy()
    y_score = p["y_score"].numpy()

    ap = average_precision_score(
        y_true,
        y_score
    )

    print(f"{model}: {ap:.4f}")

print(f"\nSaved: {output}")
