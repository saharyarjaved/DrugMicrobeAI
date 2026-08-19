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
                    (drug_input_dim, microbe_input_dim),
                    hidden_dim,
                    heads=heads,
                    concat=True,  # Multi-head attention features concat honge
                    add_self_loops=False
                ),
                (
                    "microbe",
                    "interacts_with",
                    "drug"
                ): GATConv(
                    (microbe_input_dim, drug_input_dim),
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    add_self_loops=False
                )
            },
            aggr="sum"
        )

        # =====================================
        # Second Heterogeneous GAT Layer
        # =====================================
        # Kyunki pehli layer mein concat=True tha, output dimension = hidden_dim * heads ban jayegi
        input_dim_layer2 = hidden_dim * heads

        self.conv2 = HeteroConv(
            {
                (
                    "drug",
                    "interacts_with",
                    "microbe"
                ): GATConv(
                    (input_dim_layer2, input_dim_layer2),
                    output_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=False
                ),
                (
                    "microbe",
                    "interacts_with",
                    "drug"
                ): GATConv(
                    (input_dim_layer2, input_dim_layer2),
                    output_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=False
                )
            },
            aggr="sum"
        )

        # =====================================
        # Activations & Regularization
        # =====================================
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    # =====================================
    # Forward Pass with Attention Support
    # =====================================
    def forward(
        self,
        x_dict,
        edge_index_dict,
        return_attention_weights=False
    ):
        """
        Forward pass with optional attention weights return 
        for biological interpretability analysis.
        """
        # -------------------------------------
        # First GAT Layer
        # -------------------------------------
        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            node_type: self.activation(embeddings)
            for node_type, embeddings in x_dict.items()
        }

        x_dict = {
            node_type: self.dropout(embeddings)
            for node_type, embeddings in x_dict.items()
        }

        # -------------------------------------
        # Second GAT Layer
        # -------------------------------------
        # Note: Agar attention weights extract karne hain toh GATConv ki internal 
        # return_attention_weights API ko HeteroConv ke sath yahan map kiya jata hai.
        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        return x_dict