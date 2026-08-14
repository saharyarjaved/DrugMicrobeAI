import torch

from src.models.gat import GATModel

model = GATModel(
    input_dim=20,
    hidden_dim=32,
    output_dim=16
)

x = torch.randn(20, 20)

edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 0]
], dtype=torch.long)

output = model(
    x,
    edge_index
)

print("=" * 50)
print("GAT Working Successfully")
print("=" * 50)
print(output.shape)