import torch
import torch.nn as nn


class EdgeDecoder(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, drug_embedding, microbe_embedding):

        x = torch.cat(
            [drug_embedding, microbe_embedding],
            dim=1
        )

        return self.layers(x)