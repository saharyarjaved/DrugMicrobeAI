from src.config import DEVICE
from src.data.build_gcn_graph import build_gcn_graph
from src.training.trainer import Trainer

graph = build_gcn_graph(
    "data/processed/train_encoded.csv"
)

trainer = Trainer(graph, DEVICE)

embeddings = trainer.train()

print("Embeddings Shape:", embeddings.shape)