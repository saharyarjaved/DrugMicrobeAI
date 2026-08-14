from src.data.graph_builder import build_graph

graph = build_graph("data/processed/train_encoded.csv")

print(graph)

print()

print(graph["drug"].x.shape)

print(graph["microbe"].x.shape)

print(graph["drug", "interacts", "microbe"].edge_index.shape)