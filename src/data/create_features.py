import torch
import pandas as pd

train = pd.read_csv("data/processed/train_encoded.csv")

num_drugs = train["Drug_ID"].max() + 1
num_microbes = train["Microbe_ID"].max() + 1

drug_features = torch.eye(num_drugs)
microbe_features = torch.eye(num_microbes)

torch.save(drug_features, "data/processed/drug_features.pt")
torch.save(microbe_features, "data/processed/microbe_features.pt")

print("Drug Features:", drug_features.shape)
print("Microbe Features:", microbe_features.shape)