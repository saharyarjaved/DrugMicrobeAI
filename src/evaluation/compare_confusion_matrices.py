import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


PREDICTION_FILES = {
    "GCN": "experiments/gcn_clean_predictions.pt",
    "GAT": "experiments/gat_clean_predictions.pt",
    "HaGAT": "experiments/hagat_clean_predictions.pt",
}


def main():

    os.makedirs(
        "experiments/plots",
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    for ax, (model_name, path) in zip(
        axes,
        PREDICTION_FILES.items()
    ):

        if not os.path.exists(path):
            print(f"Missing prediction file: {path}")
            continue

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        y_true = data["y_true"].numpy()
        y_pred = data["y_pred"].numpy()

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        print()
        print("=" * 60)
        print(f"{model_name} Confusion Matrix")
        print("=" * 60)
        print(f"True Negative  : {tn}")
        print(f"False Positive : {fp}")
        print(f"False Negative : {fn}")
        print(f"True Positive  : {tp}")

        image = ax.imshow(
            cm,
            interpolation="nearest",
            cmap="Blues"
        )

        ax.set_title(
            f"{model_name}"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "True Label"
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No Interaction", "Interaction"]
        )

        ax.set_yticklabels(
            ["No Interaction", "Interaction"]
        )

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14
                )

    fig.suptitle(
        "Confusion Matrix Comparison - Drug-Microbe Interaction Prediction",
        fontsize=14
    )

    plt.tight_layout()

    output_path = (
        "experiments/plots/"
        "confusion_matrix_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 60)
    print("Confusion matrix comparison saved to:")
    print(output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()