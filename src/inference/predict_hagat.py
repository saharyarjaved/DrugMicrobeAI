import gc
import torch

from src.models.edge_decoder import EdgeDecoder


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_FILE = (
    "saved_models/hagat_embeddings.pth"
)

MODEL_FILE = (
    "saved_models/best_hagat_model.pth"
)

OUTPUT_DIM = 64


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# Load Precomputed Embeddings
# ============================================================

print("=" * 60, flush=True)
print("Loading precomputed HaGAT embeddings...", flush=True)

embedding_data = torch.load(
    EMBEDDINGS_FILE,
    map_location="cpu"
)

drug_embeddings = (
    embedding_data["drug_embeddings"]
    .contiguous()
)

microbe_embeddings = (
    embedding_data["microbe_embeddings"]
    .contiguous()
)

NUM_DRUGS = int(
    embedding_data["num_drugs"]
)

NUM_MICROBES = int(
    embedding_data["num_microbes"]
)

EMBEDDING_DIM = int(
    embedding_data["embedding_dim"]
)

del embedding_data

gc.collect()

print(
    f"Drug embeddings    : {tuple(drug_embeddings.shape)}",
    flush=True
)

print(
    f"Microbe embeddings : {tuple(microbe_embeddings.shape)}",
    flush=True
)

print(
    f"Embedding dimension : {EMBEDDING_DIM}",
    flush=True
)


# ============================================================
# Load Decoder Only
# ============================================================

print(
    "Loading edge decoder...",
    flush=True
)

decoder = EdgeDecoder(
    OUTPUT_DIM
).to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

del checkpoint

gc.collect()

decoder.eval()

print(
    "Edge decoder loaded successfully.",
    flush=True
)

print("=" * 60, flush=True)


# ============================================================
# Prediction
# ============================================================

@torch.inference_mode()
def predict_interaction(
    drug_id,
    microbe_id
):

    drug_id = int(drug_id)
    microbe_id = int(microbe_id)

    print(
        f"[HaGAT] prediction request: "
        f"drug={drug_id}, microbe={microbe_id}",
        flush=True
    )

    # --------------------------------------------------------
    # Validate IDs
    # --------------------------------------------------------

    if drug_id < 0 or drug_id >= NUM_DRUGS:

        raise ValueError(
            f"Invalid drug ID: {drug_id}. "
            f"Valid range: 0-{NUM_DRUGS - 1}"
        )

    if microbe_id < 0 or microbe_id >= NUM_MICROBES:

        raise ValueError(
            f"Invalid microbe ID: {microbe_id}. "
            f"Valid range: 0-{NUM_MICROBES - 1}"
        )

    # --------------------------------------------------------
    # Get precomputed embeddings
    # --------------------------------------------------------

    drug_embedding = (
        drug_embeddings[drug_id]
        .unsqueeze(0)
    )

    microbe_embedding = (
        microbe_embeddings[microbe_id]
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # Decode interaction
    # --------------------------------------------------------

    logits = decoder(
        drug_embedding,
        microbe_embedding
    )

    probability = float(
        torch.sigmoid(logits).item()
    )

    prediction = (
        "Interaction"
        if probability >= 0.5
        else "No Interaction"
    )

    result = {
        "drug_id": drug_id,
        "microbe_id": microbe_id,
        "probability": round(
            probability,
            6
        ),
        "prediction": prediction
    }

    print(
        f"[HaGAT] Result: {result}",
        flush=True
    )

    return result


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("HaGAT Cached Prediction Test")
    print("=" * 60)

    result = predict_interaction(
        0,
        0
    )

    print()
    print(
        f"Drug ID     : {result['drug_id']}"
    )

    print(
        f"Microbe ID  : {result['microbe_id']}"
    )

    print(
        f"Probability : {result['probability']:.6f}"
    )

    print(
        f"Prediction  : {result['prediction']}"
    )

    print("=" * 60)
