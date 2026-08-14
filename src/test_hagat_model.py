import torch

from src.data.build_hagat_graph import build_hagat_graph
from src.models.hagat import HaGATModel


# =====================================
# Build Heterogeneous Graph
# =====================================

graph = build_hagat_graph(
    "data/processed/train_encoded.csv"
)


# =====================================
# Build Model
# =====================================

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4
)


# =====================================
# Evaluation Mode
# =====================================

model.eval()


# =====================================
# Forward Pass
# =====================================

with torch.no_grad():

    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict
    )


# =====================================
# Extract Embeddings
# =====================================

drug_embeddings = embeddings["drug"]

microbe_embeddings = embeddings["microbe"]


# =====================================
# Print Results
# =====================================

print("\n" + "=" * 60)
print("HaGAT Forward Pass Test")
print("=" * 60)

print(
    f"Drug Input Shape      : "
    f"{graph['drug'].x.shape}"
)

print(
    f"Microbe Input Shape   : "
    f"{graph['microbe'].x.shape}"
)

print(
    f"Drug Embedding Shape  : "
    f"{drug_embeddings.shape}"
)

print(
    f"Microbe Embedding Shape : "
    f"{microbe_embeddings.shape}"
)

print("=" * 60)

print(
    "HaGAT Forward Pass Successful"
)

print("=" * 60)