import pandas as pd
import networkx as nx

# Cleaned dataset load karo
df = pd.read_csv("data/processed/cleaned_data.csv")

# Graph banao
G = nx.Graph()

# Drug aur Microbe ko connect karo
for _, row in df.iterrows():
    drug = row["Name"]
    microbe = row["Microbe"]

    G.add_node(drug, node_type="drug")
    G.add_node(microbe, node_type="microbe")

    G.add_edge(drug, microbe)

print("=" * 50)
print("Graph Created Successfully")
print("=" * 50)

print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())