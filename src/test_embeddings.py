import torch

from src.models.embeddings import NodeEmbedding

model = NodeEmbedding(
    num_nodes=1394,
    embedding_dim=64
)

x = torch.tensor([0, 5, 10, 50])

output = model(x)

print(output.shape)