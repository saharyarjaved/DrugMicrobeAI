import os
import pandas as pd
import torch
from torch.optim import Adam

from src.config import DEVICE, HIDDEN_DIM, OUTPUT_DIM, LEARNING_RATE, EPOCHS
from src.utils.seed import set_seed
from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


HEADS_LIST = [1, 2, 4, 8]

os.makedirs("experiments/ablation", exist_ok=True)
os.makedirs("saved_models/ablation", exist_ok=True)

device = DEVICE

graph = build_hagat_graph(
    "data/processed/train_encoded.csv"
).to(device)

drug_ids, microbe_ids, labels = create_training_dataset(
    "data/processed/train_split.csv",
    num_drugs=1394,
    num_microbes=180
)

drug_ids = drug_ids.long().to(device)
microbe_ids = microbe_ids.long().to(device)
labels = labels.float().to(device)

test_drug_ids, test_microbe_ids, test_labels = create_training_dataset(
    "data/processed/test_split.csv",
    num_drugs=1394,
    num_microbes=180
)

test_drug_ids = test_drug_ids.long().to(device)
test_microbe_ids = test_microbe_ids.long().to(device)
test_labels = test_labels.float().to(device)

results = []

for heads in HEADS_LIST:

    print("\n" + "=" * 60)
    print(f"HaGAT HEAD ABLATION | HEADS = {heads}")
    print("=" * 60)

    # Same seed for fair comparison
    set_seed()

    model = HaGATModel(
        drug_input_dim=graph["drug"].x.shape[1],
        microbe_input_dim=graph["microbe"].x.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        heads=heads
    ).to(device)

    decoder = EdgeDecoder(
        OUTPUT_DIM
    ).to(device)

    optimizer = Adam(
        list(model.parameters()) +
        list(decoder.parameters()),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        decoder.train()

        optimizer.zero_grad()

        embeddings = model(
            graph.x_dict,
            graph.edge_index_dict
        )

        drug_embeddings = embeddings["drug"]
        microbe_embeddings = embeddings["microbe"]

        selected_drug = drug_embeddings[drug_ids]
        selected_microbe = microbe_embeddings[microbe_ids]

        logits = decoder(
            selected_drug,
            selected_microbe
        ).squeeze()

        loss = criterion(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:

            best_loss = current_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "loss": current_loss,
                    "heads": heads,
                    "hagat_state_dict": model.state_dict(),
                    "decoder_state_dict": decoder.state_dict()
                },
                f"saved_models/ablation/best_hagat_heads_{heads}.pth"
            )

    # --------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        f"saved_models/ablation/best_hagat_heads_{heads}.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["hagat_state_dict"]
    )

    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    model.eval()
    decoder.eval()

    with torch.no_grad():

        embeddings = model(
            graph.x_dict,
            graph.edge_index_dict
        )

        drug_embeddings = embeddings["drug"]
        microbe_embeddings = embeddings["microbe"]

        selected_drug = drug_embeddings[test_drug_ids]
        selected_microbe = microbe_embeddings[test_microbe_ids]

        logits = decoder(
            selected_drug,
            selected_microbe
        ).squeeze()

        scores = torch.sigmoid(logits)

        predictions = (
            scores >= 0.5
        ).int()

    y_true = test_labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_score = scores.cpu().numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )

    result = {
        "Model": "HaGAT",
        "Heads": heads,
        "Hidden": HIDDEN_DIM,
        "Output": OUTPUT_DIM,
        "LearningRate": LEARNING_RATE,
        "Epochs": EPOCHS,
        "BestEpoch": best_epoch,
        "BestLoss": best_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }

    results.append(result)

    print(f"Best Epoch : {best_epoch}")
    print(f"Best Loss  : {best_loss:.6f}")
    print(f"Accuracy   : {accuracy:.6f}")
    print(f"Precision  : {precision:.6f}")
    print(f"Recall     : {recall:.6f}")
    print(f"F1         : {f1:.6f}")
    print(f"ROC-AUC    : {roc_auc:.6f}")

df = pd.DataFrame(results)

output = "experiments/ablation/hagat_heads_ablation.csv"

df.to_csv(
    output,
    index=False
)

print("\n" + "=" * 60)
print("HEAD ABLATION COMPLETED")
print("=" * 60)

print(df.round(6).to_string(index=False))
print(f"\nSaved: {output}")
