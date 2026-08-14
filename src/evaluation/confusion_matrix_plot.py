# ============================================================
# Confusion Matrix Plot
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path="experiments/confusion_matrix.png",
):
    """
    Generate and save the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Actual labels.

    y_pred : array-like
        Predicted labels.

    save_path : str
        Location where the confusion matrix image
        will be saved.
    """

    # --------------------------------------------------------
    # Create output directory if it does not exist
    # --------------------------------------------------------

    output_path = Path(save_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create confusion matrix
    # --------------------------------------------------------

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "No Interaction",
            "Interaction",
        ],
        cmap="Blues",
        values_format="d",
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    plt.title(
        "HaGAT Drug-Microbe Interaction Confusion Matrix"
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.tight_layout()

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved: {output_path}"
    )