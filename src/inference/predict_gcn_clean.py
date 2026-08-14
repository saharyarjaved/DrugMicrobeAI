import torch

from src.data.build_gcn_graph import build_gcn_graph
from src.models.gcn import GCNModel
from src.models.link_predictor import LinkPredictor

GRAPH_FILE = "data/processed/clean_train.csv"
MODEL_FILE = "saved_models/best_gcn_clean_model.pth"

HIDDEN_DIM = 128
OUTPUT_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

graph = build_gcn_graph(GRAPH_FILE)
graph = graph.to(device)

print("Graph Nodes :", graph.x.shape[0])

model = GCNModel(
    input_dim=graph.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM
).to(device)

predictor = LinkPredictor(
    OUTPUT_DIM
).to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint["gcn_state_dict"]
)

predictor.load_state_dict(
    checkpoint["predictor_state_dict"]
)

print("Loaded Epoch :", checkpoint["epoch"])
print("Best Loss :", checkpoint["loss"])

model.eval()
predictor.eval()


@torch.no_grad()
def predict_interaction(drug_id, microbe_id):

    drug_id = int(drug_id)
    microbe_id = int(microbe_id)

    if drug_id < 0 or drug_id >= 1394:
        raise ValueError(f"Invalid drug ID: {drug_id}")

    if microbe_id < 0 or microbe_id >= 180:
        raise ValueError(f"Invalid microbe ID: {microbe_id}")

    embeddings = model(
        graph.x,
        graph.edge_index
    )

    drug_embedding = embeddings[drug_id].unsqueeze(0)

    microbe_index = 1394 + microbe_id
    microbe_embedding = embeddings[microbe_index].unsqueeze(0)

    probability = predictor(
        drug_embedding,
        microbe_embedding
    ).item()

    prediction = (
        "Interaction"
        if probability >= 0.6
        else "No Interaction"
    )

    return {
        "drug_id": drug_id,
        "microbe_id": microbe_id,
        "probability": round(probability, 4),
        "prediction": prediction
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Clean GCN Drug-Microbe Prediction")
    print("=" * 60)

    result = predict_interaction(0, 0)

    print("Drug ID     :", result["drug_id"])
    print("Microbe ID  :", result["microbe_id"])
    print("Probability :", result["probability"])
    print("Prediction  :", result["prediction"])

    print("=" * 60)
