from src.data.dataloader import DrugMicrobeDataset

dataset = DrugMicrobeDataset()

print(dataset.get_train().head())

print()

print(dataset.get_edge_index())