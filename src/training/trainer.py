import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.gcn import GCNModel
from src.models.edge_decoder import EdgeDecoder


class Trainer:

    def __init__(self, graph, device):

        self.graph = graph.to(device)
        self.device = device

        # Models
        self.encoder = GCNModel(
            input_dim=graph.x.shape[1],
            hidden_dim=128,
            output_dim=64
        ).to(device)

        self.decoder = EdgeDecoder(64).to(device)

        # Optimizer
        self.optimizer = Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()),
            lr=0.001
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self):
        """
        Generate node embeddings using GCN.
        """
        embeddings = self.encoder(
            self.graph.x.float(),
            self.graph.edge_index
        )

        return embeddings

    def predict(self, drug_embeddings, microbe_embeddings):
        """
        Predict interaction scores.
        """
        return self.decoder(
            drug_embeddings,
            microbe_embeddings
        )

    def train(self):
        """
        Forward pass only (training loop next step).
        """
        self.encoder.train()
        self.decoder.train()

        embeddings = self.forward()

        return embeddings

    @torch.no_grad()
    def evaluate(self):

        self.encoder.eval()
        self.decoder.eval()

        embeddings = self.forward()

        return embeddings