import torch

from src.config import (
    HIDDEN_DIM,
    OUTPUT_DIM,
    HEADS,
    LEARNING_RATE,
    EPOCHS
)

from src.data.build_gcn_graph import build_gcn_graph
from src.data.create_training_data import create_training_dataset

from src.models.gat import GATModel
from src.models.edge_decoder import EdgeDecoder

from src.evaluation.evaluate import evaluate

from src.utils.save_results import save_results


# =====================================
# Build Graph
# =====================================

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

print("=" * 60)
print("GAT Model Evaluation")
print("=" * 60)

print(f"Graph Nodes : {graph.num_nodes}")
print(f"Drug Nodes  : {graph.num_drugs}")
print(f"Microbes    : {graph.num_microbes}")


# =====================================
# Test Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/test_split.csv",
    num_drugs=graph.num_drugs,
    num_microbes=graph.num_microbes
)

# Shift microbe IDs
microbe_ids = (
    microbe_ids + graph.num_drugs
)


# =====================================
# Load GAT Checkpoint
# =====================================

checkpoint = torch.load(
    "saved_models/best_gat_model.pth",
    map_location="cpu"
)

print(
    f"\nCheckpoint Epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint Loss  : "
    f"{checkpoint['loss']:.6f}"
)


# =====================================
# Build GAT Model
# =====================================

model = GATModel(
    input_dim=graph.num_nodes,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    heads=HEADS
)

decoder = EdgeDecoder(
    OUTPUT_DIM
)


# =====================================
# Load Weights
# =====================================

model.load_state_dict(
    checkpoint["gat_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)


# =====================================
# Evaluate
# =====================================

metrics, y_true, y_pred, y_score = evaluate(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels
)


# =====================================
# Print Metrics
# =====================================

print("\n" + "=" * 60)
print("GAT Evaluation Results")
print("=" * 60)

for key, value in metrics.items():

    print(
        f"{key:12}: {value:.4f}"
    )

print("=" * 60)


# =====================================
# Save Results
# =====================================

save_results(
    experiment="GAT_BASE",
    model_name="GAT",
    hidden=HIDDEN_DIM,
    output=OUTPUT_DIM,
    heads=HEADS,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    metrics=metrics
)

print("\nGAT results saved to:")
print("experiments/results.csv")