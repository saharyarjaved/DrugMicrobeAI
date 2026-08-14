from src.data.build_hagat_graph import build_hagat_graph


graph = build_hagat_graph(
    "data/processed/train_encoded.csv"
)

print("\nGraph Structure:")
print(graph)