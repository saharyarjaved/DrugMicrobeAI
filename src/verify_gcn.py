import torch

from src.models.gcn import GCNModel

model = GCNModel(
    input_dim=64,
    hidden_dim=128,
    output_dim=64
)

print(model)