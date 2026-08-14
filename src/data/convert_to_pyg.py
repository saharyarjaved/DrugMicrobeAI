import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create Graph
G = nx.Graph()

for _, row in df.iterrows():
    drug = row["Name"]
    microbe = row["Microbe"]

    G.add_node(drug, node_type="drug")
    G.add_node(microbe, node_type="microbe")

    G.add_edge(drug, microbe)

# Convert to PyTorch Geometric
data = from_networkx(G)

print("=" * 50)
print("PyTorch Geometric Graph")
print("=" * 50)

print(data)
print("Number of Nodes :", data.num_nodes)
print("Number of Edges :", data.num_edges)