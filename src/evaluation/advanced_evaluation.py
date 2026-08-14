import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# =====================================
# Create Results Folder
# =====================================

os.makedirs(
    "results",
    exist_ok=True
)


# =====================================
# Load Prediction Data
# =====================================

prediction_file = (
    "experiments/predictions.pt"
)


if not os.path.exists(prediction_file):

    raise FileNotFoundError(
        f"Prediction file not found: "
        f"{prediction_file}"
    )


predictions = torch.load(
    prediction_file,
    map_location="cpu"
)


# =====================================
# Extract Models
# =====================================

models = [
    "GCN",
    "GAT",
    "HaGAT"
]


# =====================================
# Confusion Matrices
# =====================================

for model_name in models:

    if model_name not in predictions:

        print(
            f"Skipping {model_name}: "
            f"prediction data not found."
        )

        continue

    data = predictions[
        model_name
    ]

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Interaction",
            "Interaction"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    output_file = (
        f"results/"
        f"{model_name.lower()}_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved : {output_file}"
    )


# =====================================
# ROC Curve
# =====================================

plt.figure(
    figsize=(9, 7)
)


for model_name in models:

    if model_name not in predictions:

        continue

    data = predictions[
        model_name
    ]

    y_true = data["y_true"]
    y_score = data["y_score"]

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.4f})"
    )


# =====================================
# Random Classifier
# =====================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


# =====================================
# ROC Formatting
# =====================================

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve: GCN vs GAT vs HaGAT"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# =====================================
# Save ROC Curve
# =====================================

roc_file = (
    "results/roc_curve_comparison.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved : {roc_file}"
)

print(
    "\nAdvanced evaluation completed."
)