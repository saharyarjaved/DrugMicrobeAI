import torch
import torch.nn as nn


class NodeEmbedding(nn.Module):

    def __init__(self, num_nodes, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

    def forward(self, x):

        return self.embedding(x)