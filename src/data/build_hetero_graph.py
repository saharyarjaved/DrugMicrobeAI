import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv")

# Unique drugs and microbes
drugs = df["Name"].unique()
microbes = df["Microbe"].unique()

# Mapping
drug_to_id = {drug: i for i, drug in enumerate(drugs)}
microbe_to_id = {microbe: i for i, microbe in enumerate(microbes)}

# Create HeteroData
data = HeteroData()

# Number of nodes
data["drug"].num_nodes = len(drugs)
data["microbe"].num_nodes = len(microbes)

# Dummy node features (we'll replace these later)
data["drug"].x = torch.eye(len(drugs))
data["microbe"].x = torch.eye(len(microbes))

# Create edge index
edge_index = []

for _, row in df.iterrows():
    edge_index.append([
        drug_to_id[row["Name"]],
        microbe_to_id[row["Microbe"]]
    ])

edge_index = torch.tensor(edge_index).t().contiguous()

data["drug", "interacts", "microbe"].edge_index = edge_index

print(data)
print("\nDrug Nodes:", data["drug"].num_nodes)
print("Microbe Nodes:", data["microbe"].num_nodes)
print("Interactions:", data["drug", "interacts", "microbe"].edge_index.shape[1])