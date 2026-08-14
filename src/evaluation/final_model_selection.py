import os
import pandas as pd


# =====================================
# Paths
# =====================================

input_file = "experiments/final_analysis.csv"

report_file = "experiments/final_model_report.txt"


# =====================================
# Check File
# =====================================

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File not found: {input_file}"
    )


# =====================================
# Load Results
# =====================================

df = pd.read_csv(input_file)


# =====================================
# Required Columns
# =====================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

required = [
    "Model",
    "Overall_Score",
    "Overall_Rank"
] + metrics


for column in required:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# =====================================
# Best Model
# =====================================

df = df.sort_values(
    "Overall_Rank"
)

best_model = df.iloc[0]["Model"]


# =====================================
# Metric Winners
# =====================================

metric_winners = {}

for metric in metrics:

    best_index = df[metric].idxmax()

    metric_winners[metric] = (
        df.loc[best_index, "Model"]
    )


# =====================================
# Generate Research Report
# =====================================

report = []

report.append(
    "Drug-Microbe Interaction Prediction"
)

report.append(
    "Final Model Selection Report"
)

report.append(
    "=" * 60
)

report.append("")


# =====================================
# Model Results
# =====================================

report.append(
    "MODEL PERFORMANCE"
)

report.append(
    "-" * 60
)

for _, row in df.iterrows():

    report.append(
        f"{row['Model']}: "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Precision={row['Precision']:.4f}, "
        f"Recall={row['Recall']:.4f}, "
        f"F1={row['F1']:.4f}, "
        f"ROC-AUC={row['ROC-AUC']:.4f}"
    )


report.append("")


# =====================================
# Metric Winners
# =====================================

report.append(
    "BEST MODEL BY METRIC"
)

report.append(
    "-" * 60
)

for metric in metrics:

    report.append(
        f"{metric}: "
        f"{metric_winners[metric]}"
    )


report.append("")


# =====================================
# Final Model
# =====================================

report.append(
    "FINAL MODEL"
)

report.append(
    "-" * 60
)

report.append(
    f"Selected Model: {best_model}"
)

report.append(
    f"Overall Score: "
    f"{df.iloc[0]['Overall_Score']:.4f}"
)

report.append("")


# =====================================
# Research Interpretation
# =====================================

report.append(
    "RESEARCH INTERPRETATION"
)

report.append(
    "-" * 60
)

if best_model == "HaGAT":

    report.append(
        "HaGAT achieved the highest overall "
        "performance among the evaluated models."
    )

    report.append(
        "This suggests that explicitly modeling "
        "the heterogeneous drug-microbe graph "
        "structure can improve interaction "
        "prediction performance."
    )

elif best_model == "GAT":

    report.append(
        "GAT achieved the highest overall "
        "performance among the evaluated models."
    )

    report.append(
        "The result indicates that attention-based "
        "message passing provides a strong baseline "
        "for drug-microbe interaction prediction."
    )

else:

    report.append(
        "GCN achieved the highest overall "
        "performance among the evaluated models."
    )

    report.append(
        "The result indicates that graph convolution "
        "provides a strong baseline for the current "
        "dataset and experimental configuration."
    )


# =====================================
# Important Metrics
# =====================================

report.append("")

report.append(
    "The ROC-AUC score is considered important "
    "because it evaluates the model's ability "
    "to distinguish interacting and non-interacting "
    "drug-microbe pairs across classification "
    "thresholds."
)

report.append(
    "F1-score provides a balance between precision "
    "and recall and is particularly useful when "
    "both false positives and false negatives "
    "matter."
)


# =====================================
# Save Report
# =====================================

os.makedirs(
    "experiments",
    exist_ok=True
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# =====================================
# Terminal Output
# =====================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)

print(
    f"Selected Model : {best_model}"
)

print(
    f"Overall Score  : "
    f"{df.iloc[0]['Overall_Score']:.4f}"
)

print("\nBest Model by Metric:")

for metric in metrics:

    print(
        f"{metric:12}: "
        f"{metric_winners[metric]}"
    )

print("=" * 70)

print(
    f"\nReport saved to:"
)

print(
    report_file
)