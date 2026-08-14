import torch

from src.data.build_hagat_graph import build_hagat_graph
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder


# =====================================
# Configuration
# =====================================

GRAPH_FILE = (
    "data/processed/train_encoded.csv"
)

MODEL_FILE = (
    "saved_models/best_hagat_model.pth"
)

HIDDEN_DIM = 256
OUTPUT_DIM = 64
HEADS = 4


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================
# Build Graph
# =====================================

graph = build_hagat_graph(
    GRAPH_FILE
)

graph = graph.to(device)


# =====================================
# Build Model
# =====================================

model = HaGATModel(
    drug_input_dim=graph["drug"].x.shape[1],
    microbe_input_dim=graph["microbe"].x.shape[1],
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    heads=HEADS
).to(device)


# =====================================
# Decoder
# =====================================

decoder = EdgeDecoder(
    OUTPUT_DIM
).to(device)


# =====================================
# Load Checkpoint
# =====================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint["hagat_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)


# =====================================
# Evaluation Mode
# =====================================

model.eval()
decoder.eval()


# =====================================
# Prediction Function
# =====================================

@torch.no_grad()
def predict_interaction(
    drug_id,
    microbe_id
):

    drug_id = int(drug_id)
    microbe_id = int(microbe_id)

    # -------------------------------
    # Validate IDs
    # -------------------------------

    if drug_id < 0 or drug_id >= graph["drug"].x.shape[0]:

        raise ValueError(
            f"Invalid drug ID: {drug_id}"
        )

    if microbe_id < 0 or microbe_id >= graph["microbe"].x.shape[0]:

        raise ValueError(
            f"Invalid microbe ID: {microbe_id}"
        )

    # -------------------------------
    # Generate Embeddings
    # -------------------------------

    embeddings = model(
        graph.x_dict,
        graph.edge_index_dict
    )

    drug_embedding = embeddings[
        "drug"
    ][drug_id].unsqueeze(0)

    microbe_embedding = embeddings[
        "microbe"
    ][microbe_id].unsqueeze(0)

    # -------------------------------
    # Decode
    # -------------------------------

    logits = decoder(
        drug_embedding,
        microbe_embedding
    )

    probability = torch.sigmoid(
        logits
    ).item()

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = (
        "Interaction"
        if probability >= 0.5
        else "No Interaction"
    )

    return {
        "drug_id": drug_id,
        "microbe_id": microbe_id,
        "probability": probability,
        "prediction": prediction
    }


# =====================================
# Test Prediction
# =====================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HaGAT Drug-Microbe Prediction")
    print("=" * 60)

    # Example IDs
    drug_id = 0
    microbe_id = 0

    result = predict_interaction(
        drug_id,
        microbe_id
    )

    print(
        f"\nDrug ID      : "
        f"{result['drug_id']}"
    )

    print(
        f"Microbe ID   : "
        f"{result['microbe_id']}"
    )

    print(
        f"Probability  : "
        f"{result['probability']:.4f}"
    )

    print(
        f"Prediction   : "
        f"{result['prediction']}"
    )

    print("=" * 60)
