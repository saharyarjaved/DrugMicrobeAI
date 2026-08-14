import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


class GATModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        heads=4
    ):
        super().__init__()

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=0.2
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=0.2
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.gat1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.gat2(
            x,
            edge_index
        )

        return x