import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_graph(csv_path):

    df = pd.read_csv(csv_path)

    drugs = sorted(df["Drug_ID"].unique())
    microbes = sorted(df["Microbe_ID"].unique())

    data = HeteroData()

    # Number of nodes
    data["drug"].num_nodes = len(drugs)
    data["microbe"].num_nodes = len(microbes)

    # Initial node features
    data["drug"].x = torch.eye(len(drugs))
    data["microbe"].x = torch.eye(len(microbes))

    # Edge index
    edge_index = torch.tensor([
        df["Drug_ID"].values,
        df["Microbe_ID"].values
    ], dtype=torch.long)

    data["drug", "interacts", "microbe"].edge_index = edge_index

    return data