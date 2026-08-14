import torch
import torch.nn as nn


class LinkPredictor(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.linear = nn.Linear(embedding_dim * 2, 1)

    def forward(self, drug_embedding, microbe_embedding):

        x = torch.cat(
            [drug_embedding, microbe_embedding],
            dim=1
        )

        return torch.sigmoid(self.linear(x))