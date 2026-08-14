
import os
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)

PREDICTIONS_FILE = "experiments/hagat_clean_predictions.pt"
RESULTS_FILE = "experiments/hagat_clean_holdout_metrics.csv"


def main():
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    data = torch.load(
        PREDICTIONS_FILE,
        map_location="cpu"
    )

    y_true = data["y_true"].cpu().numpy()
    y_pred = data["y_pred"].cpu().numpy()
    y_score = data["y_score"].cpu().numpy()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )
    pr_auc = average_precision_score(
        y_true,
        y_score,
    )
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )
    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    result = {
        "Model": "HaGAT Clean Holdout",
        "Samples": len(y_true),
        "PositiveSamples": int(y_true.sum()),
        "NegativeSamples": int(len(y_true) - y_true.sum()),
        "TrueNegative": int(tn),
        "FalsePositive": int(fp),
        "FalseNegative": int(fn),
        "TruePositive": int(tp),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Balanced_Accuracy": balanced_accuracy,
        "MCC": mcc,
    }

    os.makedirs(
        os.path.dirname(RESULTS_FILE),
        exist_ok=True,
    )

    pd.DataFrame([result]).to_csv(
        RESULTS_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("HaGAT Clean Holdout - Advanced Metrics")
    print("=" * 60)

    print(f"Samples            : {len(y_true)}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"Specificity        : {specificity:.4f}")
    print(f"F1                 : {f1:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"PR-AUC             : {pr_auc:.4f}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
    print(f"MCC                : {mcc:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"TP : {tp}")

    print("\nSaved:")
    print(RESULTS_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
