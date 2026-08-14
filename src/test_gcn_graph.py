from src.data.build_gcn_graph import build_gcn_graph

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

print(graph)

print()

print("Nodes :", graph.num_nodes)

print("Edges :", graph.num_edges)

print("Features :", graph.x.shape)