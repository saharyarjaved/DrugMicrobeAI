import torch


@torch.no_grad()
def validate(
    model,
    decoder,
    graph,
    drug_ids,
    microbe_ids,
    labels,
):

    model.eval()
    decoder.eval()

    # Node embeddings
    embeddings = model(
        graph.x.float(),
        graph.edge_index
    )

    # Get embeddings of drug and microbe nodes
    drug_embedding = embeddings[drug_ids]
    microbe_embedding = embeddings[microbe_ids]

    # Predict interaction
    logits = decoder(
        drug_embedding,
        microbe_embedding
    ).squeeze()

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= 0.5).float()

    # Accuracy
    accuracy = (
        predictions == labels
    ).float().mean()

    return accuracy.item()