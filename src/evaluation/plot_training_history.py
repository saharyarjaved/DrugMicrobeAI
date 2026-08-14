import os
import pandas as pd
import matplotlib.pyplot as plt


# =====================================
# Paths
# =====================================

history_path = "logs/training_history.csv"
output_path = "results/training_loss.png"


# =====================================
# Check History File
# =====================================

if not os.path.exists(history_path):
    raise FileNotFoundError(
        f"Training history not found: {history_path}"
    )


# =====================================
# Load History
# =====================================

df = pd.read_csv(history_path)


# =====================================
# Validate Columns
# =====================================

required_columns = [
    "Epoch",
    "Training_Loss",
    "Validation_Loss"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )


# =====================================
# Create Results Folder
# =====================================

os.makedirs(
    "results",
    exist_ok=True
)


# =====================================
# Plot
# =====================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Epoch"],
    df["Training_Loss"],
    label="Training Loss"
)

plt.plot(
    df["Epoch"],
    df["Validation_Loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "GCN Training and Validation Loss"
)

plt.legend()
plt.grid(True)


# =====================================
# Save
# =====================================

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================
# Output
# =====================================

print("=" * 50)
print("Loss Curve Generated")
print("=" * 50)
print(f"Saved : {output_path}")
print("=" * 50)