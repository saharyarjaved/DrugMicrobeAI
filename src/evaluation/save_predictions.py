import os
import torch

from src.data.build_gcn_graph import build_gcn_graph
from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset

from src.models.gcn import GCNModel
from src.models.gat import GATModel
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 60)
print("Saving Model Predictions")
print("=" * 60)

print(f"Device : {device}")


# =====================================
# Output Folder
# =====================================

os.makedirs(
    "experiments",
    exist_ok=True
)


# =====================================
# Helper Function
# =====================================

def save_prediction(
    model_name,
    y_true,
    y_pred,
    y_score
):

    return {
        model_name: {
            "y_true": y_true.cpu(),
            "y_pred": y_pred.cpu(),
            "y_score": y_score.cpu()
        }
    }


# =====================================
# Test Dataset
# =====================================

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/test_split.csv",
    num_drugs=1394,
    num_microbes=180
)

drug_ids = drug_ids.long().to(device)

microbe_ids = microbe_ids.long().to(device)

labels = labels.float().to(device)


# =====================================
# Prediction Storage
# =====================================

all_predictions = {}


# ============================================================
# GCN
# ============================================================

print("\nRunning GCN...")

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

graph = graph.to(device)

gcn_model = GCNModel(
    input_dim=graph.x.shape[1],
    hidden_dim=128,
    output_dim=64
).to(device)

gcn_decoder = EdgeDecoder(
    64
).to(device)

checkpoint = torch.load(
    "saved_models/best_model.pth",
    map_location=device
)

gcn_model.load_state_dict(
    checkpoint["gcn_state_dict"]
)

gcn_decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

gcn_model.eval()
gcn_decoder.eval()

# GCN uses combined node IDs
gcn_microbe_ids = (
    microbe_ids + graph.num_drugs
)

with torch.no_grad():

    embeddings = gcn_model(
        graph.x.float(),
        graph.edge_index
    )

    drug_embeddings = embeddings[
        drug_ids
    ]

    microbe_embeddings = embeddings[
        gcn_microbe_ids
    ]

    logits = gcn_decoder(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    scores = torch.sigmoid(
        logits
    )

    preds = (
        scores >= 0.5
    ).int()

all_predictions.update(
    save_prediction(
        "GCN",
        labels,
        preds,
        scores
    )
)

print("GCN predictions saved.")


# ============================================================
# GAT
# ============================================================

print("\nRunning GAT...")

gat_model = GATModel(
    input_dim=graph.x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4
).to(device)

gat_decoder = EdgeDecoder(
    64
).to(device)

checkpoint = torch.load(
    "saved_models/best_gat_model.pth",
    map_location=device
)

gat_model.load_state_dict(
    checkpoint["gat_state_dict"]
)

gat_decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

gat_model.eval()
gat_decoder.eval()

with torch.no_grad():

    embeddings = gat_model(
        graph.x.float(),
        graph.edge_index
    )

    drug_embeddings = embeddings[
        drug_ids
    ]

    microbe_embeddings = embeddings[
        gcn_microbe_ids
    ]

    logits = gat_decoder(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    scores = torch.sigmoid(
        logits
    )

    preds = (
        scores >= 0.5
    ).int()

all_predictions.update(
    save_prediction(
        "GAT",
        labels,
        preds,
        scores
    )
)

print("GAT predictions saved.")


# ============================================================
# HaGAT
# ============================================================

print("\nRunning HaGAT...")

hagat_graph = build_hagat_graph(
    "data/processed/train_encoded.csv"
)

hagat_graph = hagat_graph.to(device)

hagat_model = HaGATModel(
    drug_input_dim=hagat_graph["drug"].x.shape[1],
    microbe_input_dim=hagat_graph["microbe"].x.shape[1],
    hidden_dim=128,
    output_dim=64,
    heads=4
).to(device)

hagat_decoder = EdgeDecoder(
    64
).to(device)

checkpoint = torch.load(
    "saved_models/best_hagat_model.pth",
    map_location=device
)

hagat_model.load_state_dict(
    checkpoint["hagat_state_dict"]
)

hagat_decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)

hagat_model.eval()
hagat_decoder.eval()

with torch.no_grad():

    embeddings = hagat_model(
        hagat_graph.x_dict,
        hagat_graph.edge_index_dict
    )

    drug_embeddings = embeddings[
        "drug"
    ]

    microbe_embeddings = embeddings[
        "microbe"
    ]

    drug_embeddings = drug_embeddings[
        drug_ids
    ]

    microbe_embeddings = microbe_embeddings[
        microbe_ids
    ]

    logits = hagat_decoder(
        drug_embeddings,
        microbe_embeddings
    ).squeeze()

    scores = torch.sigmoid(
        logits
    )

    preds = (
        scores >= 0.5
    ).int()

all_predictions.update(
    save_prediction(
        "HaGAT",
        labels,
        preds,
        scores
    )
)


print("HaGAT predictions saved.")


# =====================================
# Save All Predictions
# =====================================

output_file = (
    "experiments/predictions.pt"
)

torch.save(
    all_predictions,
    output_file
)


# =====================================
# Completed
# =====================================

print("\n" + "=" * 60)
print("Prediction Saving Completed")
print("=" * 60)

print(
    f"Saved : {output_file}"
)

print(
    "Models : GCN, GAT, HaGAT"
)

print("=" * 60)