import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, drug_ids, microbe_ids):
        drug_z = z[drug_ids]
        microbe_z = z[microbe_ids]
        edge_z = torch.cat([drug_z, microbe_z], dim=1)
        return self.mlp(edge_z).squeeze(-1)

class DrugMicrobeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, embedding_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            dropout=dropout
        )
        self.decoder = EdgeDecoder(embedding_dim=embedding_dim, hidden_dim=64)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, drug_ids, microbe_ids):
        return self.decoder(z, drug_ids, microbe_ids)

    def forward(self, x, edge_index, drug_ids, microbe_ids):
        z = self.encode(x, edge_index)
        return self.decode(z, drug_ids, microbe_ids)
