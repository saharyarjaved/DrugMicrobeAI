from pathlib import Path
import random
import pandas as pd

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "interactions_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train/validation/test ratios must sum to 1.0")

print("=" * 70)
print("DRUG-MICROBE DATASET SPLIT")
print("=" * 70)

print(f"Reading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "drug_id",
    "drug_name",
    "microbe_id",
    "microbe_name",
    "label",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_columns].copy()

df["drug_id"] = df["drug_id"].astype(int)
df["microbe_id"] = df["microbe_id"].astype(int)
df["label"] = df["label"].astype(int)

if df.duplicated(["drug_id", "microbe_id"]).any():
    raise ValueError("Duplicate drug-microbe pairs detected.")

if not (df["label"] == 1).all():
    raise ValueError("Input file must contain positive interactions only.")

rng = random.Random(SEED)

edges = list(
    zip(
        df["drug_id"].tolist(),
        df["microbe_id"].tolist(),
    )
)

rng.shuffle(edges)

train_edges = set()
remaining_edges = set(edges)

drug_to_edges = {}

for drug_id, microbe_id in edges:
    drug_to_edges.setdefault(drug_id, []).append(
        (drug_id, microbe_id)
    )

for drug_id, candidates in drug_to_edges.items():
    candidates = candidates.copy()
    rng.shuffle(candidates)

    for edge in candidates:
        if edge in remaining_edges:
            train_edges.add(edge)
            remaining_edges.remove(edge)
            break

microbe_to_edges = {}

for drug_id, microbe_id in edges:
    microbe_to_edges.setdefault(microbe_id, []).append(
        (drug_id, microbe_id)
    )

for microbe_id, candidates in microbe_to_edges.items():
    candidates = candidates.copy()
    rng.shuffle(candidates)

    for edge in candidates:
        if edge in remaining_edges:
            train_edges.add(edge)
            remaining_edges.remove(edge)
            break

total_edges = len(edges)

target_train = round(total_edges * TRAIN_RATIO)
target_val = round(total_edges * VAL_RATIO)
target_test = total_edges - target_train - target_val

remaining_list = list(remaining_edges)
rng.shuffle(remaining_list)

needed_for_train = max(
    0,
    target_train - len(train_edges),
)

extra_train = remaining_list[:needed_for_train]

for edge in extra_train:
    train_edges.add(edge)

remaining_list = remaining_list[needed_for_train:]

val_count = min(
    target_val,
    len(remaining_list),
)

val_edges = set(
    remaining_list[:val_count]
)

test_edges = set(
    remaining_list[val_count:]
)

if train_edges & val_edges:
    raise RuntimeError(
        "Train/validation leakage detected."
    )

if train_edges & test_edges:
    raise RuntimeError(
        "Train/test leakage detected."
    )

if val_edges & test_edges:
    raise RuntimeError(
        "Validation/test leakage detected."
    )

edge_to_row = {
    (int(row.drug_id), int(row.microbe_id)): row
    for row in df.itertuples(index=False)
}


def build_dataframe(edge_set):
    rows = [
        edge_to_row[edge]
        for edge in edge_set
    ]

    if not rows:
        return pd.DataFrame(
            columns=required_columns
        )

    return pd.DataFrame(
        [
            {
                "drug_id": row.drug_id,
                "drug_name": row.drug_name,
                "microbe_id": row.microbe_id,
                "microbe_name": row.microbe_name,
                "label": row.label,
            }
            for row in rows
        ]
    ).sort_values(
        ["drug_id", "microbe_id"]
    ).reset_index(drop=True)


train_pos = build_dataframe(train_edges)
val_pos = build_dataframe(val_edges)
test_pos = build_dataframe(test_edges)

all_drugs = sorted(
    df["drug_id"].unique()
)

all_microbes = sorted(
    df["microbe_id"].unique()
)

positive_pairs = set(edges)

all_possible_pairs = [
    (drug_id, microbe_id)
    for drug_id in all_drugs
    for microbe_id in all_microbes
]

negative_candidates = [
    pair
    for pair in all_possible_pairs
    if pair not in positive_pairs
]

rng.shuffle(negative_candidates)


def sample_negatives(count):
    if count > len(negative_candidates):
        raise ValueError(
            "Not enough negative candidate pairs."
        )

    selected = negative_candidates[:count]

    del negative_candidates[:count]

    return selected


drug_name_map = (
    df.drop_duplicates("drug_id")
    .set_index("drug_id")["drug_name"]
    .to_dict()
)

microbe_name_map = (
    df.drop_duplicates("microbe_id")
    .set_index("microbe_id")["microbe_name"]
    .to_dict()
)


def build_negative_dataframe(pairs):
    return pd.DataFrame(
        [
            {
                "drug_id": drug_id,
                "drug_name": drug_name_map[drug_id],
                "microbe_id": microbe_id,
                "microbe_name": microbe_name_map[microbe_id],
                "label": 0,
            }
            for drug_id, microbe_id in pairs
        ]
    )


train_neg = build_negative_dataframe(
    sample_negatives(len(train_pos))
)

val_neg = build_negative_dataframe(
    sample_negatives(len(val_pos))
)

test_neg = build_negative_dataframe(
    sample_negatives(len(test_pos))
)

train = pd.concat(
    [train_pos, train_neg],
    ignore_index=True,
)

val = pd.concat(
    [val_pos, val_neg],
    ignore_index=True,
)

test = pd.concat(
    [test_pos, test_neg],
    ignore_index=True,
)

train = train.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

val = val.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

test = test.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)


def pair_set(dataframe):
    return set(
        zip(
            dataframe["drug_id"],
            dataframe["microbe_id"],
        )
    )


train_pairs = pair_set(train)
val_pairs = pair_set(val)
test_pairs = pair_set(test)

if train_pairs & val_pairs:
    raise RuntimeError(
        "Pair leakage between train and validation."
    )

if train_pairs & test_pairs:
    raise RuntimeError(
        "Pair leakage between train and test."
    )

if val_pairs & test_pairs:
    raise RuntimeError(
        "Pair leakage between validation and test."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

train_file = OUTPUT_DIR / "train.csv"
val_file = OUTPUT_DIR / "val.csv"
test_file = OUTPUT_DIR / "test.csv"

train.to_csv(
    train_file,
    index=False,
    encoding="utf-8-sig",
)

val.to_csv(
    val_file,
    index=False,
    encoding="utf-8-sig",
)

test.to_csv(
    test_file,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"Positive interactions: {len(df):,}")

print()
print("POSITIVE SPLIT")
print(f"Train:      {len(train_pos):,}")
print(f"Validation: {len(val_pos):,}")
print(f"Test:       {len(test_pos):,}")

print()
print("FINAL DATASET")
print(f"Train:      {len(train):,}")
print(f"Validation: {len(val):,}")
print(f"Test:       {len(test):,}")

print()
print("LABEL DISTRIBUTION")

print("\nTrain:")
print(train["label"].value_counts().sort_index())

print("\nValidation:")
print(val["label"].value_counts().sort_index())

print("\nTest:")
print(test["label"].value_counts().sort_index())

print()
print("NODE COVERAGE")

print(
    "Train drugs:",
    train_pos["drug_id"].nunique(),
    "/",
    len(all_drugs),
)

print(
    "Train microbes:",
    train_pos["microbe_id"].nunique(),
    "/",
    len(all_microbes),
)

print()
print("LEAKAGE CHECK: PASSED")

print()
print("FILES:")
print(train_file)
print(val_file)
print(test_file)
