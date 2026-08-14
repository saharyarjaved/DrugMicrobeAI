import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    y_true,
    y_score,
    save_path="results/roc_curve.png"
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()