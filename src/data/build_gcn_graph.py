import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def build_gcn_graph(csv_path):

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(csv_path)

    # Maximum IDs (NOT unique count)
    max_drug_id = int(df["Drug_ID"].max())
    max_microbe_id = int(df["Microbe_ID"].max())

    # Number of nodes
    num_drugs = max_drug_id + 1
    num_microbes = max_microbe_id + 1

    total_nodes = num_drugs + num_microbes

    # -----------------------------
    # Edge Index
    # -----------------------------
    edge_array = np.vstack([
        df["Drug_ID"].to_numpy(),
        df["Microbe_ID"].to_numpy() + num_drugs
    ])

    edge_index = torch.from_numpy(edge_array).long()

    # Undirected Graph
    reverse_edge = edge_index.flip(0)
    edge_index = torch.cat([edge_index, reverse_edge], dim=1)

    # Node Features
    x = torch.eye(total_nodes, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index
    )

    data.num_drugs = num_drugs
    data.num_microbes = num_microbes
    data.total_nodes = total_nodes

    return data