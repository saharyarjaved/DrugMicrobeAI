from src.data.link_dataset import load_link_dataset
from src.data.negative_sampling import generate_negative_edges

drug, microbe, labels = load_link_dataset(
    "data/processed/train_encoded.csv"
)

neg_drug, neg_microbe = generate_negative_edges(
    drug,
    microbe,
    num_drugs=1394,
    num_microbes=180
)

print("Positive Samples :", len(drug))
print("Negative Samples :", len(neg_drug))

print("\nFirst 5 Negative Pairs")

for i in range(5):
    print(
        neg_drug[i].item(),
        neg_microbe[i].item()
    )