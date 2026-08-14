from pathlib import Path
import os
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset
from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GRAPH_FILE = "data/processed/clean_train.csv"
TEST_FILE = "data/processed/clean_test.csv"

HEADS_LIST = [1, 2, 4, 8]

MODEL_TEMPLATE = (
    "saved_models/ablation/best_hagat_heads_{heads}.pth"
)

OUTPUT_FILE = (
    "experiments/ablation/"
    "hagat_heads_clean_holdout.csv"
)


def evaluate_heads():
    graph = build_hagat_graph(
        GRAPH_FILE
    ).to(DEVICE)

    drug_ids, microbe_ids, labels = (
        create_training_dataset(
            TEST_FILE,
            num_drugs=1394,
            num_microbes=180,
        )
    )

    drug_ids = drug_ids.long().to(DEVICE)
    microbe_ids = microbe_ids.long().to(DEVICE)
    labels = labels.float().to(DEVICE)

    results = []

    for heads in HEADS_LIST:

        print("\n" + "=" * 70)
        print(f"Evaluating HaGAT with {heads} attention heads")
        print("=" * 70)

        model = HaGATModel(
            drug_input_dim=graph["drug"].x.shape[1],
            microbe_input_dim=graph["microbe"].x.shape[1],
            hidden_dim=128,
            output_dim=64,
            heads=heads,
        ).to(DEVICE)

        decoder = EdgeDecoder(64).to(DEVICE)

        model_file = MODEL_TEMPLATE.format(
            heads=heads
        )

        checkpoint = torch.load(
            model_file,
            map_location=DEVICE,
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
                graph.edge_index_dict,
            )

            logits = decoder(
                embeddings["drug"][drug_ids],
                embeddings["microbe"][microbe_ids],
            ).squeeze()

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).int()

        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_score = probabilities.cpu().numpy()

        results.append(
            {
                "Model": "HaGAT",
                "Heads": heads,
                "Hidden": 128,
                "Output": 64,
                "Epoch": checkpoint.get("epoch"),
                "Accuracy": accuracy_score(
                    y_true, y_pred
                ),
                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
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
                "Balanced_Accuracy":
                    balanced_accuracy_score(
                        y_true,
                        y_pred,
                    ),
                "MCC": matthews_corrcoef(
                    y_true,
                    y_pred,
                ),
            }
        )

        print(
            f"Accuracy   : {results[-1]['Accuracy']:.4f}"
        )
        print(
            f"Precision  : {results[-1]['Precision']:.4f}"
        )
        print(
            f"Recall     : {results[-1]['Recall']:.4f}"
        )
        print(
            f"F1         : {results[-1]['F1']:.4f}"
        )
        print(
            f"ROC-AUC    : {results[-1]['ROC_AUC']:.4f}"
        )
        print(
            f"PR-AUC     : {results[-1]['PR_AUC']:.4f}"
        )
        print(
            f"Balanced   : {results[-1]['Balanced_Accuracy']:.4f}"
        )
        print(
            f"MCC        : {results[-1]['MCC']:.4f}"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Clean Holdout Ablation Complete")
    print("=" * 70)

    print(
        results_df[
            [
                "Heads",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "PR_AUC",
                "Balanced_Accuracy",
                "MCC",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    evaluate_heads()
