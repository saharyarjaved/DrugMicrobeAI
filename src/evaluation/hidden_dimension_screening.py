from pathlib import Path
import os
import torch
import pandas as pd

from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_FILE = "data/processed/clean_train.csv"
VAL_FILE = "data/processed/val_split.csv"

EPOCHS = 30
HEADS = 4
OUTPUT_DIM = 64
LEARNING_RATE = 0.001

HIDDEN_VALUES = [64, 128, 256]

OUTPUT_DIR = Path(
    "experiments/hyperparameter"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def evaluate(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels,
):
    model.eval()
    decoder.eval()

    with torch.no_grad():
        embeddings = model(
            graph.x_dict,
            graph.edge_index_dict,
        )

        logits = decoder(
            embeddings["drug"][drug_ids],
            embeddings["microbe"][microbe_ids],
        ).squeeze()

        scores = torch.sigmoid(logits)
        predictions = (scores >= 0.5).int()

    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_score = scores.cpu().numpy()

    return {
        "Accuracy": accuracy_score(
            y_true, y_pred
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            y_score,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            y_score,
        ),
    }


def run_experiment(
    hidden_dim,
    graph,
    drug_ids,
    microbe_ids,
    labels,
):
    model = HaGATModel(
        drug_input_dim=graph["drug"].x.shape[1],
        microbe_input_dim=graph["microbe"].x.shape[1],
        hidden_dim=hidden_dim,
        output_dim=OUTPUT_DIM,
        heads=HEADS,
    ).to(DEVICE)

    decoder = EdgeDecoder(
        OUTPUT_DIM
    ).to(DEVICE)

    optimizer = Adam(
        list(model.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        decoder.train()

        optimizer.zero_grad()

        embeddings = model(
            graph.x_dict,
            graph.edge_index_dict,
        )

        logits = decoder(
            embeddings["drug"][drug_ids],
            embeddings["microbe"][microbe_ids],
        ).squeeze()

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        if loss_value < best_loss:
            best_loss = loss_value

            best_state = {
                "epoch": epoch,
                "loss": loss_value,
                "hagat_state_dict":
                    model.state_dict(),
                "decoder_state_dict":
                    decoder.state_dict(),
            }

    model.load_state_dict(
        best_state["hagat_state_dict"]
    )

    decoder.load_state_dict(
        best_state["decoder_state_dict"]
    )

    metrics = evaluate(
        model,
        decoder,
        graph,
        drug_ids,
        microbe_ids,
        labels,
    )

    return {
        "Hidden": hidden_dim,
        "Heads": HEADS,
        "Output": OUTPUT_DIM,
        "LearningRate": LEARNING_RATE,
        "Epochs": EPOCHS,
        "BestEpoch": best_state["epoch"],
        "BestLoss": best_state["loss"],
        **metrics,
    }


def main():
    print("=" * 70)
    print("HaGAT Hidden-Dimension Screening")
    print("=" * 70)
    print(f"Device : {DEVICE}")

    graph = build_hagat_graph(
        GRAPH_FILE
    ).to(DEVICE)

    drug_ids, microbe_ids, labels = (
        create_training_dataset(
            VAL_FILE,
            num_drugs=1394,
            num_microbes=180,
        )
    )

    drug_ids = drug_ids.long().to(DEVICE)
    microbe_ids = microbe_ids.long().to(DEVICE)
    labels = labels.float().to(DEVICE)

    results = []

    for hidden_dim in HIDDEN_VALUES:
        print(
            f"\nRunning hidden_dim={hidden_dim}"
        )

        result = run_experiment(
            hidden_dim,
            graph,
            drug_ids,
            microbe_ids,
            labels,
        )

        results.append(result)

        torch.save(
            result,
            OUTPUT_DIR
            / f"hidden_{hidden_dim}_screening.pt",
        )

        print(
            f"ROC-AUC : {result['ROC_AUC']:.4f}"
        )
        print(
            f"PR-AUC  : {result['PR_AUC']:.4f}"
        )
        print(
            f"F1      : {result['F1']:.4f}"
        )

    df = pd.DataFrame(results)

    output_csv = (
        OUTPUT_DIR
        / "hidden_dimension_screening.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SCREENING RESULTS")
    print("=" * 70)
    print(
        df[
            [
                "Hidden",
                "BestEpoch",
                "BestLoss",
                "Accuracy",
                "F1",
                "ROC_AUC",
                "PR_AUC",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {output_csv}"
    )


if __name__ == "__main__":
    main()
