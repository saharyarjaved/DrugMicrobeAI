from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathlib import Path

import pandas as pd

from src.inference.predict_hagat import predict_interaction


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Drug-Microbe AI API",
    description="HaGAT based drug-microbe interaction prediction API",
    version="1.0.0",
)


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_encoded.csv"
)

EXPERIMENTS_DIR = BASE_DIR / "experiments"


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Serve Experiment / Evaluation Files
# ============================================================

if not EXPERIMENTS_DIR.exists():
    print(
        f"WARNING: Experiments directory not found: "
        f"{EXPERIMENTS_DIR}"
    )
else:
    print(
        f"Experiments directory found: "
        f"{EXPERIMENTS_DIR}"
    )

    app.mount(
        "/experiments",
        StaticFiles(
            directory=str(EXPERIMENTS_DIR)
        ),
        name="experiments",
    )


# ============================================================
# Dataset
# ============================================================

try:

    df = pd.read_csv(DATASET_PATH)

    print("==============================================")
    print("Dataset loaded successfully")
    print(f"Dataset rows : {len(df)}")
    print(f"Dataset path : {DATASET_PATH}")
    print("==============================================")

except Exception as error:

    print(
        f"WARNING: Dataset could not be loaded: {error}"
    )

    df = pd.DataFrame()


# ============================================================
# Request Schema
# ============================================================

class PredictionRequest(BaseModel):
    drug_id: int
    microbe_id: int


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def home():

    return {
        "project": "Drug-Microbe AI",
        "model": "HaGAT",
        "status": "running",
        "api_version": "1.0.0",
    }


# ============================================================
# Get Drugs
# ============================================================

@app.get("/drugs")
def get_drugs():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Drug-microbe dataset could not be loaded.",
        )

    required_columns = [
        "Drug_ID",
        "Drug",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise HTTPException(
                status_code=500,
                detail=f"Missing dataset column: {column}",
            )

    drugs = (
        df[["Drug_ID", "Drug"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Drug_ID")
    )

    return [
        {
            "id": int(row["Drug_ID"]),
            "name": str(row["Drug"]),
        }
        for _, row in drugs.iterrows()
    ]


# ============================================================
# Get Microbes
# ============================================================

@app.get("/microbes")
def get_microbes():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Drug-microbe dataset could not be loaded.",
        )

    required_columns = [
        "Microbe_ID",
        "Microbe",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise HTTPException(
                status_code=500,
                detail=f"Missing dataset column: {column}",
            )

    microbes = (
        df[["Microbe_ID", "Microbe"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Microbe_ID")
    )

    return [
        {
            "id": int(row["Microbe_ID"]),
            "name": str(row["Microbe"]),
        }
        for _, row in microbes.iterrows()
    ]


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        result = predict_interaction(
            request.drug_id,
            request.microbe_id,
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# HaGAT Evaluation
# ============================================================

@app.get("/evaluation")
def evaluation():

    return {

        "model": "HaGAT",

        "dataset": "Drug-Microbe Interaction",

        "metrics": {

            "accuracy": 0.6928,

            "precision": 0.7451,

            "recall": 0.5860,

            "f1": 0.6560,

            "roc_auc": 0.7571,
        },
    }


# ============================================================
# GCN vs HaGAT Comparison
# ============================================================

@app.get("/comparison")
def model_comparison():

    return {

        "title": "GCN vs HaGAT",

        "description": (
            "Performance comparison between the "
            "GCN baseline and HaGAT model."
        ),

        "models": {

            "GCN": {

                "accuracy": 0.6928,

                "precision": 0.7451,

                "recall": 0.5860,

                "f1": 0.6560,

                "roc_auc": 0.7571,
            },

            "HaGAT": {

                "accuracy": 0.7058,

                "precision": 0.7353,

                "recall": 0.6429,

                "f1": 0.6860,

                "roc_auc": 0.7797,
            },
        },

        "improvement": {

            "accuracy": round(
                (0.7058 - 0.6928) * 100,
                2,
            ),

            "precision": round(
                (0.7353 - 0.7451) * 100,
                2,
            ),

            "recall": round(
                (0.6429 - 0.5860) * 100,
                2,
            ),

            "f1": round(
                (0.6860 - 0.6560) * 100,
                2,
            ),

            "roc_auc": round(
                (0.7797 - 0.7571) * 100,
                2,
            ),
        },
    }


# ============================================================
# Dataset Statistics
# ============================================================

@app.get("/dataset-stats")
def dataset_stats():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset could not be loaded.",
        )

    unique_drugs = df["Drug_ID"].nunique()

    unique_microbes = df["Microbe_ID"].nunique()

    positive_interactions = None

    negative_interactions = None

    if "Label" in df.columns:

        positive_interactions = int(
            (df["Label"] == 1).sum()
        )

        negative_interactions = int(
            (df["Label"] == 0).sum()
        )

    return {

        "dataset": "train_encoded.csv",

        "total_records": int(
            len(df)
        ),

        "unique_drugs": int(
            unique_drugs
        ),

        "unique_microbes": int(
            unique_microbes
        ),

        "positive_interactions":
            positive_interactions,

        "negative_interactions":
            negative_interactions,
    }


# ============================================================
# Evaluation Files
# ============================================================

@app.get("/evaluation-files")
def evaluation_files():

    confusion_matrix = (
        EXPERIMENTS_DIR
        / "confusion_matrix.png"
    )

    roc_curve = (
        EXPERIMENTS_DIR
        / "roc_curve.png"
    )

    return {

        "confusion_matrix": {
            "available": confusion_matrix.exists(),
            "url": "/experiments/confusion_matrix.png",
        },

        "roc_curve": {
            "available": roc_curve.exists(),
            "url": "/experiments/roc_curve.png",
        },
    }

# ============================================================
# Graph Data - Drugs + Microbes + Interactions
# ============================================================

@app.get("/graph")
def get_graph():

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset could not be loaded.",
        )

    required_columns = [
        "Drug_ID",
        "Drug",
        "Microbe_ID",
        "Microbe",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Missing dataset column: {column}",
            )

    # --------------------------------------------------------
    # Drug nodes
    # --------------------------------------------------------

    drug_nodes = (
        df[["Drug_ID", "Drug"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Drug_ID")
    )

    # --------------------------------------------------------
    # Microbe nodes
    # --------------------------------------------------------

    microbe_nodes = (
        df[["Microbe_ID", "Microbe"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Microbe_ID")
    )

    nodes = []

    # Drug nodes
    for _, row in drug_nodes.iterrows():

        nodes.append({
            "id": f"drug_{int(row['Drug_ID'])}",
            "type": "drug",
            "original_id": int(row["Drug_ID"]),
            "name": str(row["Drug"]),
        })

    # Microbe nodes
    for _, row in microbe_nodes.iterrows():

        nodes.append({
            "id": f"microbe_{int(row['Microbe_ID'])}",
            "type": "microbe",
            "original_id": int(row["Microbe_ID"]),
            "name": str(row["Microbe"]),
        })

    # --------------------------------------------------------
    # Interaction edges
    # --------------------------------------------------------

    edges = []

    interaction_columns = [
        "Drug_ID",
        "Microbe_ID",
    ]

    for column in interaction_columns:

        if column not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Missing interaction column: {column}",
            )

    interaction_df = (
        df[["Drug_ID", "Microbe_ID"]]
        .dropna()
        .drop_duplicates()
    )

    for _, row in interaction_df.iterrows():

        edges.append({
            "id": (
                f"drug_{int(row['Drug_ID'])}"
                f"_microbe_{int(row['Microbe_ID'])}"
            ),
            "source": f"drug_{int(row['Drug_ID'])}",
            "target": f"microbe_{int(row['Microbe_ID'])}",
            "type": "interaction",
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "drug_nodes": len(drug_nodes),
            "microbe_nodes": len(microbe_nodes),
            "interaction_edges": len(edges),
        },
    }

    # ============================================================
# Heterogeneous Graph
# ============================================================

@app.get("/graph")
def get_graph():

    graph_df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "cleaned_data.csv"
    )

    # Remove incomplete rows
    graph_df = graph_df.dropna(
        subset=["Name", "Microbe"]
    )

    # Unique entities
    unique_drugs = (
        graph_df["Name"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    unique_microbes = (
        graph_df["Microbe"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Stable IDs
    drug_to_id = {
        drug: index
        for index, drug in enumerate(unique_drugs)
    }

    microbe_to_id = {
        microbe: index
        for index, microbe in enumerate(unique_microbes)
    }

    nodes = []

    # --------------------------------------------------------
    # Drug nodes
    # --------------------------------------------------------

    for drug, drug_id in drug_to_id.items():

        nodes.append({
            "id": f"drug_{drug_id}",
            "node_id": drug_id,
            "type": "drug",
            "name": drug,
        })

    # --------------------------------------------------------
    # Microbe nodes
    # --------------------------------------------------------

    for microbe, microbe_id in microbe_to_id.items():

        nodes.append({
            "id": f"microbe_{microbe_id}",
            "node_id": microbe_id,
            "type": "microbe",
            "name": microbe,
        })

    # --------------------------------------------------------
    # Interaction edges
    # --------------------------------------------------------

    edges = []

    # Remove duplicate drug-microbe relationships
    interactions = graph_df[
        ["Name", "Microbe"]
    ].drop_duplicates()

    for index, row in interactions.iterrows():

        drug = row["Name"]
        microbe = row["Microbe"]

        edges.append({
            "id": f"interaction_{len(edges)}",
            "source": f"drug_{drug_to_id[drug]}",
            "target": f"microbe_{microbe_to_id[microbe]}",
            "type": "interacts",
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "drug_nodes": len(unique_drugs),
            "microbe_nodes": len(unique_microbes),
            "total_nodes": len(nodes),
            "interaction_edges": len(edges),
        },
    }