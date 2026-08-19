import torch
import torch.nn as nn


class LinkPredictor(nn.Module):

    def __init__(self, embedding_dim, dropout_p=0.3):
        super().__init__()

        # Non-linear MLP Link Predictor for robust biological interaction prediction
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, drug_embedding, microbe_embedding):

        # Concatenate drug and microbe representations
        x = torch.cat(
            [drug_embedding, microbe_embedding],
            dim=1
        )

        # Pass through MLP and apply sigmoid for interaction probability
        return torch.sigmoid(self.mlp(x))