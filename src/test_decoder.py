import torch

from src.models.edge_decoder import EdgeDecoder

decoder = EdgeDecoder(64)

drug = torch.randn(8, 64)

microbe = torch.randn(8, 64)

output = decoder(drug, microbe)

print(output.shape)
print(output)