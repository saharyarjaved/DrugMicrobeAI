import os
import pandas as pd


# =====================================
# Paths
# =====================================

input_file = "experiments/model_comparison.csv"

output_file = "experiments/final_analysis.csv"


# =====================================
# Check File
# =====================================

if not os.path.exists(input_file):

    raise FileNotFoundError(
        f"File not found: {input_file}"
    )


# =====================================
# Load Data
# =====================================

df = pd.read_csv(
    input_file
)


# =====================================
# Metrics
# =====================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]


# =====================================
# Check Columns
# =====================================

for metric in metrics:

    if metric not in df.columns:

        raise ValueError(
            f"Missing metric: {metric}"
        )


# =====================================
# Convert Metrics
# =====================================

for metric in metrics:

    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce"
    )


# =====================================
# Find Best Models
# =====================================

best_models = {}

for metric in metrics:

    index = df[metric].idxmax()

    best_models[metric] = (
        df.loc[index, "Model"]
    )


# =====================================
# Overall Score
# =====================================

df["Overall_Score"] = (
    df[metrics].mean(axis=1)
)


df = df.sort_values(
    "Overall_Score",
    ascending=False
)


# =====================================
# Rank
# =====================================

df["Overall_Rank"] = range(
    1,
    len(df) + 1
)


# =====================================
# Save Analysis
# =====================================

os.makedirs(
    "experiments",
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)


# =====================================
# Print Analysis
# =====================================

print("\n" + "=" * 70)
print("FINAL MODEL ANALYSIS")
print("=" * 70)

print(
    df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Overall_Score",
            "Overall_Rank"
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "-" * 70)
print("BEST MODEL BY METRIC")
print("-" * 70)


for metric in metrics:

    print(
        f"{metric:12} : "
        f"{best_models[metric]}"
    )


# =====================================
# Overall Best Model
# =====================================

best_model = df.iloc[0]["Model"]

best_score = df.iloc[0]["Overall_Score"]


print("\n" + "=" * 70)

print(
    f"Overall Best Model : {best_model}"
)

print(
    f"Overall Score      : {best_score:.4f}"
)

print("=" * 70)


print(
    f"\nSaved : {output_file}"
)