import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_hagat_graph(csv_path):

    # =====================================
    # Load Graph Dataset
    # =====================================

    df = pd.read_csv(csv_path)

    # =====================================
    # IMPORTANT:
    # Preserve Original ID Space
    # =====================================

    drugs = sorted(
        df["Drug_ID"].unique()
    )

    microbes = sorted(
        df["Microbe_ID"].unique()
    )

    num_drugs = max(drugs) + 1
    num_microbes = max(microbes) + 1

    # =====================================
    # Identity Mapping
    # =====================================

    drug_mapping = {
        drug_id: drug_id
        for drug_id in drugs
    }

    microbe_mapping = {
        microbe_id: microbe_id
        for microbe_id in microbes
    }

    # =====================================
    # Edge Index
    # =====================================

    drug_indices = torch.tensor(
        df["Drug_ID"].values,
        dtype=torch.long
    )

    microbe_indices = torch.tensor(
        df["Microbe_ID"].values,
        dtype=torch.long
    )

    # =====================================
    # Drug -> Microbe
    # =====================================

    drug_to_microbe_edge_index = torch.stack(
        [
            drug_indices,
            microbe_indices
        ],
        dim=0
    )

    # =====================================
    # Microbe -> Drug
    # =====================================

    microbe_to_drug_edge_index = torch.stack(
        [
            microbe_indices,
            drug_indices
        ],
        dim=0
    )

    # =====================================
    # Heterogeneous Graph
    # =====================================

    data = HeteroData()

    data["drug"].x = torch.eye(
        num_drugs,
        dtype=torch.float
    )

    data["microbe"].x = torch.eye(
        num_microbes,
        dtype=torch.float
    )

    # =====================================
    # Relations
    # =====================================

    data[
        "drug",
        "interacts_with",
        "microbe"
    ].edge_index = (
        drug_to_microbe_edge_index
    )

    data[
        "microbe",
        "interacts_with",
        "drug"
    ].edge_index = (
        microbe_to_drug_edge_index
    )

    # =====================================
    # Metadata
    # =====================================

    data.num_drugs = num_drugs
    data.num_microbes = num_microbes

    data.drug_mapping = drug_mapping
    data.microbe_mapping = microbe_mapping

    # =====================================
    # Information
    # =====================================

    print("\n" + "=" * 60)
    print("HaGAT Heterogeneous Graph")
    print("=" * 60)

    print(
        f"Drug Nodes    : {num_drugs}"
    )

    print(
        f"Microbe Nodes : {num_microbes}"
    )

    print(
        f"Drug ID Range : "
        f"0 - {num_drugs - 1}"
    )

    print(
        f"Microbe ID Range : "
        f"0 - {num_microbes - 1}"
    )

    print(
        f"Drugâ†’Microbe Edges : "
        f"{drug_to_microbe_edge_index.shape[1]}"
    )

    print(
        f"Microbeâ†’Drug Edges : "
        f"{microbe_to_drug_edge_index.shape[1]}"
    )

    print("=" * 60)

    return data
