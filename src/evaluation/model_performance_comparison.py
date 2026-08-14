import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "experiments/final_model_comparison.csv"
OUTPUT_FILE = (
    "experiments/plots/model_performance_comparison.png"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    # Keep only the final three experiments
    experiments = [
        "GCN_CLEAN",
        "GAT_CLEAN",
        "HAGAT_BASE",
    ]

    df = df[
        df["Experiment"].isin(experiments)
    ].copy()

    # Clean model names for display
    model_names = {
        "GCN_CLEAN": "GCN",
        "GAT_CLEAN": "GAT",
        "HAGAT_BASE": "HaGAT",
    }

    df["DisplayModel"] = df["Experiment"].map(
        model_names
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    os.makedirs(
        "experiments/plots",
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = range(len(df))
    width = 0.15

    for i, metric in enumerate(metrics):

        values = df[metric].tolist()

        positions = [
            value + (i - 2) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=metric
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        df["DisplayModel"].tolist()
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)

    for _, row in df.iterrows():

        print(
            f"{row['DisplayModel']:<8}"
            f" Accuracy={row['Accuracy']:.4f}"
            f" Precision={row['Precision']:.4f}"
            f" Recall={row['Recall']:.4f}"
            f" F1={row['F1']:.4f}"
            f" ROC-AUC={row['ROC_AUC']:.4f}"
        )

    print("=" * 60)
    print(
        f"Chart saved to: {OUTPUT_FILE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()