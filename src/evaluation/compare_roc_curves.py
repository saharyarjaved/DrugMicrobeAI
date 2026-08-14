import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


PREDICTION_FILES = {
    "GCN": "experiments/gcn_clean_predictions.pt",
    "GAT": "experiments/gat_clean_predictions.pt",
    "HaGAT": "experiments/hagat_clean_predictions.pt",
}


def load_predictions(path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    y_true = data["y_true"].numpy()
    y_score = data["y_score"].numpy()

    return y_true, y_score


def main():

    os.makedirs("experiments/plots", exist_ok=True)

    plt.figure(figsize=(8, 7))

    for model_name, path in PREDICTION_FILES.items():

        if not os.path.exists(path):
            print(f"Missing prediction file: {path}")
            continue

        y_true, y_score = load_predictions(path)

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        print(
            f"{model_name:<8} ROC-AUC = {roc_auc:.4f}"
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve Comparison - Drug-Microbe Interaction Prediction"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        "experiments/plots/"
        "roc_curve_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print()
    print("=" * 60)
    print("ROC comparison saved to:")
    print(output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()