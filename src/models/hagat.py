import torch
import torch.nn as nn

from torch_geometric.nn import HeteroConv, GATConv


class HaGATModel(nn.Module):

    def __init__(
        self,
        drug_input_dim,
        microbe_input_dim,
        hidden_dim=128,
        output_dim=64,
        heads=4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads

        # =====================================
        # First Heterogeneous GAT Layer
        # =====================================

        self.conv1 = HeteroConv(
            {
                (
                    "drug",
                    "interacts_with",
                    "microbe"
                ): GATConv(
                    (
                        drug_input_dim,
                        microbe_input_dim
                    ),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    add_self_loops=False
                ),

                (
                    "microbe",
                    "interacts_with",
                    "drug"
                ): GATConv(
                    (
                        microbe_input_dim,
                        drug_input_dim
                    ),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    add_self_loops=False
                )
            },
            aggr="sum"
        )

        # =====================================
        # Second Heterogeneous GAT Layer
        # =====================================

        self.conv2 = HeteroConv(
            {
                (
                    "drug",
                    "interacts_with",
                    "microbe"
                ): GATConv(
                    (
                        hidden_dim,
                        hidden_dim
                    ),
                    output_dim,
                    heads=heads,
                    concat=False,
                    add_self_loops=False
                ),

                (
                    "microbe",
                    "interacts_with",
                    "drug"
                ): GATConv(
                    (
                        hidden_dim,
                        hidden_dim
                    ),
                    output_dim,
                    heads=heads,
                    concat=False,
                    add_self_loops=False
                )
            },
            aggr="sum"
        )

        # =====================================
        # Activation
        # =====================================

        self.activation = nn.ReLU()

        # =====================================
        # Dropout
        # =====================================

        self.dropout = nn.Dropout(
            0.2
        )

    # =====================================
    # Forward
    # =====================================

    def forward(
        self,
        x_dict,
        edge_index_dict
    ):

        # -------------------------------------
        # First GAT Layer
        # -------------------------------------

        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            node_type: self.activation(
                embeddings
            )
            for node_type, embeddings
            in x_dict.items()
        }

        x_dict = {
            node_type: self.dropout(
                embeddings
            )
            for node_type, embeddings
            in x_dict.items()
        }

        # -------------------------------------
        # Second GAT Layer
        # -------------------------------------

        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        return x_dict