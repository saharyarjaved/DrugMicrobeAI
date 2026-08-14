import torch
import torch.nn as nn

from src.models.gcn import GCN
from src.training.trainer import Trainer

# Dummy graph
x = torch.randn(20, 16)

edge_index = torch.tensor(
    [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,5,6,7,8,9,0]
    ],
    dtype=torch.long
)

labels = torch.randn(20, 8)

model = GCN(
    input_dim=16,
    hidden_dim=32,
    output_dim=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

trainer = Trainer(
    model,
    optimizer,
    criterion
)

loss = trainer.train_one_epoch(
    x,
    edge_index,
    labels
)

print("=" * 50)
print("Trainer Working Successfully")
print("=" * 50)

print("Loss:", loss)