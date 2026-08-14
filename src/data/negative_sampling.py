import random
import torch


def generate_negative_edges(
    drug_ids,
    microbe_ids,
    num_drugs,
    num_microbes,
    num_samples=None,
    seed=42,
):
    """
    Generate random Drug-Microbe pairs that are not present
    in the supplied positive-edge set.

    A local random generator is used so evaluation and training
    can be made reproducible without modifying global RNG state.
    """

    if num_samples is None:
        num_samples = len(drug_ids)

    positive_edges = set(
        zip(
            drug_ids.tolist(),
            microbe_ids.tolist(),
        )
    )

    rng = random.Random(seed)

    neg_drugs = []
    neg_microbes = []

    while len(neg_drugs) < num_samples:
        d = rng.randint(0, num_drugs - 1)
        m = rng.randint(0, num_microbes - 1)

        if (d, m) not in positive_edges:
            neg_drugs.append(d)
            neg_microbes.append(m)

    return (
        torch.tensor(neg_drugs, dtype=torch.long),
        torch.tensor(neg_microbes, dtype=torch.long),
    )
