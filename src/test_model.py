# ============================================================
# GCN Model Evaluation
# ============================================================

import torch

from src.config import (
    HIDDEN_DIM,
    OUTPUT_DIM,
    LEARNING_RATE,
    EPOCHS,
)

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gcn import GCNModel
from src.models.edge_decoder import EdgeDecoder

from src.evaluation.evaluate import evaluate
from src.evaluation.confusion_matrix_plot import (
    plot_confusion_matrix,
)
from src.evaluation.roc_curve_plot import (
    plot_roc_curve,
)

from src.utils.save_results import save_results


# ============================================================
# 1. Build Graph
# ============================================================

print("\nBuilding graph...")

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

print(
    f"Graph Nodes : {graph.num_nodes}"
)


# ============================================================
# 2. Load Test Dataset
# ============================================================

print("\nLoading test dataset...")

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/test_split.csv",
    num_drugs=graph.num_drugs,
    num_microbes=graph.num_microbes,
)

# Microbe nodes start after drug nodes
microbe_ids = microbe_ids + graph.num_drugs


# ============================================================
# 3. Load Best Model
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    "saved_models/best_model.pth",
    map_location="cpu",
)


model = GCNModel(
    input_dim=graph.num_nodes,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
)


decoder = EdgeDecoder(
    OUTPUT_DIM
)


model.load_state_dict(
    checkpoint["gcn_state_dict"]
)


decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)


model.eval()
decoder.eval()


print("Model loaded successfully.")


# ============================================================
# 4. Evaluate Model
# ============================================================

print("\nRunning evaluation...")

metrics, y_true, y_pred, y_score = evaluate(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels,
)


# ============================================================
# 5. Print Metrics
# ============================================================

print("\n" + "=" * 50)

print("GCN BASELINE EVALUATION")

print("=" * 50)

for key, value in metrics.items():
    print(
        f"{key:12}: {value:.4f}"
    )

print("=" * 50)


# ============================================================
# 6. Save Experiment Results
# ============================================================

save_results(
    experiment="GCN_BASE",
    model_name="GCN",
    hidden=HIDDEN_DIM,
    output=OUTPUT_DIM,
    heads="-",
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    metrics=metrics,
)


# ============================================================
# 7. Save Confusion Matrix
# ============================================================

print("\nGenerating confusion matrix...")

plot_confusion_matrix(
    y_true,
    y_pred,
    save_path="experiments/confusion_matrix.png",
)


# ============================================================
# 8. Save ROC Curve
# ============================================================

print("\nGenerating ROC curve...")

plot_roc_curve(
    y_true,
    y_score,
    save_path="experiments/roc_curve.png",
)


# ============================================================
# 9. Finished
# ============================================================

print("\n" + "=" * 50)

print("Evaluation completed successfully!")

print("=" * 50)

print(
    "Confusion Matrix : "
    "experiments/confusion_matrix.png"
)

print(
    "ROC Curve        : "
    "experiments/roc_curve.png"
)

print(
    "Results CSV      : "
    "experiments/results.csv"
)

print("=" * 50)