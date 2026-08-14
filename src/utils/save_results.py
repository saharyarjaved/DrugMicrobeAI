import os
import csv


def save_results(
    experiment,
    model_name,
    hidden,
    output,
    heads,
    learning_rate,
    epochs,
    metrics
):

    # =====================================
    # Folder
    # =====================================

    os.makedirs(
        "experiments",
        exist_ok=True
    )

    results_path = "experiments/results.csv"


    # =====================================
    # Columns
    # =====================================

    fieldnames = [
        "Experiment",
        "Model",
        "Hidden",
        "Output",
        "Heads",
        "LearningRate",
        "Epochs",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]


    # =====================================
    # Check Existing File
    # =====================================

    file_exists = os.path.exists(
        results_path
    )

    file_empty = (
        not file_exists
        or os.path.getsize(results_path) == 0
    )


    # =====================================
    # Save Result
    # =====================================

    with open(
        results_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if file_empty:
            writer.writeheader()

        writer.writerow(
            {
                "Experiment": experiment,
                "Model": model_name,
                "Hidden": hidden,
                "Output": output,
                "Heads": heads,
                "LearningRate": learning_rate,
                "Epochs": epochs,
                "Accuracy": metrics.get("Accuracy"),
                "Precision": metrics.get("Precision"),
                "Recall": metrics.get("Recall"),
                "F1": metrics.get("F1"),
                "ROC_AUC": metrics.get("ROC-AUC")
            }
        )

    print(
        f"Results saved: {results_path}"
    )