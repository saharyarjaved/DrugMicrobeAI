import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.models.gcn import GCN

# Dummy graph
num_nodes = 20

x = torch.randn(num_nodes, 16)

edge_index = torch.tensor([
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,0]
], dtype=torch.long)

model = GCN(
    input_dim=16,
    hidden_dim=32,
    output_dim=8
)

output = model(x, edge_index)

print("="*50)
print("GCN Working Successfully")
print("="*50)

print(output.shape)