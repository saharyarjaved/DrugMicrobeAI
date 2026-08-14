import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


@torch.no_grad()
def evaluate(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels
):

    # Evaluation mode
    model.eval()
    decoder.eval()

    # -------------------------
    # Generate Node Embeddings
    # -------------------------
    embeddings = model(
        graph.x.float(),
        graph.edge_index
    )

    # -------------------------
    # Get Drug & Microbe Embeddings
    # -------------------------
    drug_embedding = embeddings[drug_ids]
    microbe_embedding = embeddings[microbe_ids]

    # -------------------------
    # Predict Interaction Scores
    # -------------------------
    logits = decoder(
        drug_embedding,
        microbe_embedding
    ).squeeze()

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).int()

    # -------------------------
    # Convert to NumPy
    # -------------------------
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_score = probabilities.cpu().numpy()

    # -------------------------
    # Metrics
    # -------------------------
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score)
    }

    # IMPORTANT:
    # Return ONLY 4 values
    return metrics, y_true, y_pred, y_score