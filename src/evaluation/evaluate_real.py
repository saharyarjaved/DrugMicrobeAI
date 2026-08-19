import torch
import os
import json
import numpy as np

from src.models.hagat import HaGATModel
from src.models.edge_decoder import EdgeDecoder
from src.data.build_hagat_graph import build_hagat_graph
from src.data.create_training_data import create_training_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def evaluate_real_model():

    print("=" * 60)
    print("Running Real Evaluation on Trained HaGAT Checkpoint")
    print("=" * 60)

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}")

    # ============================================================
    # 1. LOAD GRAPH
    # ============================================================

    graph = build_hagat_graph(
        "data/processed/train_encoded.csv"
    )

    graph = graph.to(device)

    print("\n" + "=" * 60)
    print("HaGAT Graph")
    print("=" * 60)

    print(
        f"Drug Nodes    : {graph['drug'].x.shape[0]}"
    )

    print(
        f"Microbe Nodes : {graph['microbe'].x.shape[0]}"
    )

    print(
        f"Drug Features : {graph['drug'].x.shape[1]}"
    )

    print(
        f"Microbe Features : {graph['microbe'].x.shape[1]}"
    )

    # ============================================================
    # 2. LOAD EVALUATION DATA
    # ============================================================

    drug_ids, microbe_ids, labels = create_training_dataset(
        "data/processed/train_split.csv",
        num_drugs=1394,
        num_microbes=180
    )

    drug_ids = drug_ids.long().to(device)
    microbe_ids = microbe_ids.long().to(device)
    labels = labels.float().to(device)

    print("\n[INFO] Evaluation samples:", len(labels))

    # ============================================================
    # 3. SAFETY CHECK
    # ============================================================

    if drug_ids.max().item() >= graph["drug"].x.shape[0]:

        raise ValueError(
            f"Drug ID {drug_ids.max().item()} "
            f"outside graph range."
        )

    if microbe_ids.max().item() >= graph["microbe"].x.shape[0]:

        raise ValueError(
            f"Microbe ID {microbe_ids.max().item()} "
            f"outside graph range."
        )

    # ============================================================
    # 4. INITIALIZE MODEL
    # ============================================================

    model = HaGATModel(
        drug_input_dim=graph["drug"].x.shape[1],
        microbe_input_dim=graph["microbe"].x.shape[1],
        hidden_dim=256,
        output_dim=64,
        heads=4
    ).to(device)

    decoder = EdgeDecoder(
        64
    ).to(device)

    # ============================================================
    # 5. LOAD CHECKPOINT
    # ============================================================

    checkpoint_path = (
        "saved_models/best_hagat_model.pth"
    )

    if not os.path.exists(checkpoint_path):

        print(
            f"[ERROR] Checkpoint not found: "
            f"{checkpoint_path}"
        )

        return

    checkpoint = torch.load(
        checkpoint_path,
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

    print("\n" + "-" * 60)

    print(
        f"[INFO] Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"[INFO] Checkpoint loss: "
        f"{checkpoint['loss']:.6f}"
    )

    # ============================================================
    # 6. REAL INFERENCE
    # ============================================================

    with torch.no_grad():

        embeddings = model(
            graph.x_dict,
            graph.edge_index_dict
        )

        drug_emb = embeddings["drug"][drug_ids]

        microbe_emb = embeddings["microbe"][microbe_ids]

        logits = decoder(
            drug_emb,
            microbe_emb
        ).squeeze()

        probabilities = torch.sigmoid(
            logits
        )

    # ============================================================
    # 7. CONVERT TO NUMPY
    # ============================================================

    y_true = labels.cpu().numpy()

    y_score = probabilities.cpu().numpy()

    # ============================================================
    # 8. FIND BEST F1 THRESHOLD
    # ============================================================

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_true,
            y_score
        )
    )

    f1_scores = (
        2 * precisions * recalls
        / (precisions + recalls + 1e-10)
    )

    # precision_recall_curve returns one extra
    # precision/recall value compared with thresholds.

    if len(thresholds) > 0:

        best_index = np.argmax(
            f1_scores[:-1]
        )

        best_threshold = float(
            thresholds[best_index]
        )

    else:

        best_threshold = 0.5

    # ============================================================
    # 9. PREDICTIONS
    # ============================================================

    predictions = (
        y_score >= best_threshold
    ).astype(int)

    # ============================================================
    # 10. CALCULATE REAL METRICS
    # ============================================================

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )

    auprc = average_precision_score(
        y_true,
        y_score
    )

    # ============================================================
    # 11. SAVE PREDICTIONS
    # ============================================================

    os.makedirs(
        "experiments",
        exist_ok=True
    )

    prediction_file = (
        "experiments/hagat_real_predictions.pt"
    )

    torch.save(
        {
            "y_true": torch.tensor(
                y_true,
                dtype=torch.float32
            ),

            "y_pred": torch.tensor(
                predictions,
                dtype=torch.int32
            ),

            "y_score": torch.tensor(
                y_score,
                dtype=torch.float32
            ),

            "threshold": best_threshold,
        },
        prediction_file
    )

    # ============================================================
    # 12. SAVE METRICS
    # ============================================================

    results_data = {

        "model": "HaGAT",

        "checkpoint":
            checkpoint_path,

        "epoch":
            int(checkpoint["epoch"]),

        "loss":
            float(checkpoint["loss"]),

        "samples":
            int(len(y_true)),

        "threshold":
            round(best_threshold, 6),

        "Accuracy":
            round(accuracy * 100, 2),

        "Precision":
            round(precision * 100, 2),

        "Recall":
            round(recall * 100, 2),

        "F1":
            round(f1 * 100, 2),

        "ROC_AUC":
            round(roc_auc * 100, 2),

        "AUPRC":
            round(auprc * 100, 2),
    }

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    metrics_file = (
        "data/output/metrics.json"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        json.dump(
            results_data,
            f,
            indent=4
        )

    # ============================================================
    # 13. PRINT RESULTS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("REAL COMPUTED METRICS")
    print("=" * 60)

    print(
        f"Checkpoint Epoch : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint Loss  : "
        f"{checkpoint['loss']:.6f}"
    )

    print(
        f"Samples          : "
        f"{len(y_true)}"
    )

    print(
        f"Best Threshold   : "
        f"{best_threshold:.6f}"
    )

    print("-" * 60)

    print(
        f"Accuracy         : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score         : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC          : "
        f"{roc_auc * 100:.2f}%"
    )

    print(
        f"AUPRC            : "
        f"{auprc * 100:.2f}%"
    )

    print("=" * 60)

    print(
        f"[INFO] Metrics saved to: "
        f"{metrics_file}"
    )

    print(
        f"[INFO] Predictions saved to: "
        f"{prediction_file}"
    )

    print(
        "[SUCCESS] Real evaluation finished."
    )


if __name__ == "__main__":

    evaluate_real_model()