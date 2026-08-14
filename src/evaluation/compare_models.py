import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

results_path = "experiments/results.csv"

comparison_path = (
    "experiments/final_model_comparison.csv"
)

plot_path = (
    "experiments/plots/model_performance_comparison.png"
)

# ============================================================
# Load Results
# ============================================================

if not os.path.exists(results_path):
    raise FileNotFoundError(
        f"Results file not found: {results_path}"
    )

df = pd.read_csv(results_path)

# Normalize column names
df.columns = [
    str(column).strip()
    for column in df.columns
]

# ============================================================
# Required Columns
# ============================================================

required_columns = [
    "Experiment",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}\n"
            f"Available columns: {list(df.columns)}"
        )

# ============================================================
# Select FINAL CLEAN experiments
# ============================================================

target_experiments = [
    "GCN_CLEAN",
    "GAT_CLEAN",
    "HAGAT_BASE",
]

comparison = df[
    df["Experiment"]
    .astype(str)
    .str.strip()
    .str.upper()
    .isin(target_experiments)
].copy()

if comparison.empty:
    raise ValueError(
        "No final clean model results found."
    )

# ============================================================
# Normalize model names
# ============================================================

def normalize_model(row):
    experiment = str(
        row["Experiment"]
    ).strip().upper()

    if experiment == "GCN_CLEAN":
        return "GCN Clean"

    if experiment == "GAT_CLEAN":
        return "GAT Clean"

    if experiment == "HAGAT_BASE":
        return "HaGAT"

    return str(row["Model"])


comparison["Model"] = comparison.apply(
    normalize_model,
    axis=1
)

# ============================================================
# Convert metrics to numeric
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]

for metric in metrics:
    comparison[metric] = pd.to_numeric(
        comparison[metric],
        errors="coerce"
    )

# ============================================================
# Keep one row per model
# ============================================================

comparison = (
    comparison
    .sort_values(
        "ROC_AUC",
        ascending=False
    )
    .drop_duplicates(
        subset=["Model"]
    )
)

# ============================================================
# Model order
# ============================================================

model_order = [
    "GCN Clean",
    "GAT Clean",
    "HaGAT",
]

comparison["Model"] = pd.Categorical(
    comparison["Model"],
    categories=model_order,
    ordered=True
)

comparison = comparison.sort_values(
    "Model"
)

# ============================================================
# Save final comparison CSV
# ============================================================

os.makedirs(
    "experiments",
    exist_ok=True
)

output_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]

comparison[
    output_columns
].to_csv(
    comparison_path,
    index=False
)

# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison[
        output_columns
    ].to_string(
        index=False
    )
)

print("=" * 80)

# ============================================================
# Create performance plot
# ============================================================

os.makedirs(
    "experiments/plots",
    exist_ok=True
)

x = range(
    len(comparison)
)

width = 0.15

plt.figure(
    figsize=(12, 7)
)

for index, metric in enumerate(metrics):

    values = comparison[
        metric
    ].tolist()

    positions = [
        value + (
            index - 2
        ) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

# ============================================================
# Formatting
# ============================================================

plt.xticks(
    list(x),
    comparison["Model"]
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Final Clean Model Performance Comparison"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

# ============================================================
# Save plot
# ============================================================

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Final output
# ============================================================

print(
    f"\nFinal Comparison CSV:"
)

print(
    comparison_path
)

print(
    f"\nPerformance Plot:"
)

print(
    plot_path
)

print(
    "\nFinal model comparison completed successfully."
)