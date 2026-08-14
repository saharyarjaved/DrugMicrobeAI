import torch

from src.data.build_gcn_graph import build_gcn_graph
from src.models.gcn import GCNModel
from src.models.edge_decoder import EdgeDecoder

# -----------------------
# Build Graph
# -----------------------
graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

# -----------------------
# Load Checkpoint
# -----------------------
checkpoint = torch.load(
    "saved_models/best_model.pth",
    map_location="cpu"
)

# -----------------------
# Build Model
# -----------------------
model = GCNModel(
    input_dim=graph.num_nodes,
    hidden_dim=128,
    output_dim=64
)

decoder = EdgeDecoder(64)

# -----------------------
# Load Weights
# -----------------------
model.load_state_dict(
    checkpoint["gcn_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

print("Graph Nodes :", graph.num_nodes)
print("Epoch :", checkpoint["epoch"])
print("Best Loss :", checkpoint["loss"])