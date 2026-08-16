import torch

from src.data.build_hagat_graph import build_hagat_graph
from src.models.hagat import HaGATModel


GRAPH_FILE = "data/processed/train_encoded.csv"
MODEL_FILE = "saved_models/best_hagat_model.pth"
OUTPUT_FILE = "saved_models/hagat_embeddings.pth"

HIDDEN_DIM = 256
OUTPUT_DIM = 64
HEADS = 4

device = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("=" * 60)
print("Generating HaGAT inference embeddings")
print("=" * 60)

print("Building graph...")

graph = build_hagat_graph(
    GRAPH_FILE
)

print(
    f"Drug nodes    : {graph['drug'].x.shape[0]}"
)

print(
    f"Microbe nodes : {graph['microbe'].x.shape[0]}"
)

print(
    f"Drug features : {graph['drug'].x.shape[1]}"
)

print(
    f"Microbe features : {graph['microbe'].x.shape[1]}"
)

print("Loading HaGAT model...")

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    heads=HEADS
).to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint["hagat_state_dict"]
)

model.eval()

print("Model loaded.")

print("Generating embeddings...")

with torch.inference_mode():

    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict
    )

    drug_embeddings = (
        embeddings["drug"]
        .detach()
        .cpu()
        .contiguous()
    )

    microbe_embeddings = (
        embeddings["microbe"]
        .detach()
        .cpu()
        .contiguous()
    )

output = {
    "drug_embeddings": drug_embeddings,
    "microbe_embeddings": microbe_embeddings,
    "num_drugs": int(drug_embeddings.shape[0]),
    "num_microbes": int(microbe_embeddings.shape[0]),
    "embedding_dim": int(drug_embeddings.shape[1]),
}

torch.save(
    output,
    OUTPUT_FILE
)

print()
print("=" * 60)
print("Embeddings generated successfully")
print("=" * 60)

print(
    f"Drug embeddings    : {tuple(drug_embeddings.shape)}"
)

print(
    f"Microbe embeddings : {tuple(microbe_embeddings.shape)}"
)

print(
    f"Saved to            : {OUTPUT_FILE}"
)

print("=" * 60)
