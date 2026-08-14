from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathlib import Path

import pandas as pd

from src.inference.predict_hagat import predict_interaction
from src.auth.auth import (
    create_user,
    authenticate_user,
    create_token,
    verify_recovery_code,
    decode_token,
    reset_password,
)


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


def build_graph_explanation(
    drug_id: int,
    microbe_id: int,
):
    # Graph-context explanation.
    # This does not claim to expose internal GAT attention weights.

    if df.empty:
        return {
            "type": "graph_context",
            "drug_neighbors": [],
            "microbe_neighbors": [],
            "common_neighbors": [],
            "drug_neighbor_count": 0,
            "microbe_neighbor_count": 0,
            "common_neighbor_count": 0,
        }

    required = {"Drug_ID", "Microbe_ID", "Drug", "Microbe"}

    if not required.issubset(df.columns):
        return {
            "type": "graph_context",
            "drug_neighbors": [],
            "microbe_neighbors": [],
            "common_neighbors": [],
            "drug_neighbor_count": 0,
            "microbe_neighbor_count": 0,
            "common_neighbor_count": 0,
        }

    drug_rows = df[df["Drug_ID"] == drug_id]

    drug_neighbor_ids = set(
        int(value)
        for value in drug_rows["Microbe_ID"].dropna().tolist()
    )

    microbe_rows = df[df["Microbe_ID"] == microbe_id]

    microbe_neighbor_ids = set(
        int(value)
        for value in microbe_rows["Drug_ID"].dropna().tolist()
    )

    drug_neighbors = (
        df[df["Microbe_ID"].isin(drug_neighbor_ids)]
        [["Microbe_ID", "Microbe"]]
        .drop_duplicates()
        .sort_values("Microbe_ID")
        .head(8)
    )

    microbe_neighbors = (
        df[df["Drug_ID"].isin(microbe_neighbor_ids)]
        [["Drug_ID", "Drug"]]
        .drop_duplicates()
        .sort_values("Drug_ID")
        .head(8)
    )

    common_drug_ids = set()

    for neighbor_microbe_id in drug_neighbor_ids:
        common_drug_ids.update(
            int(value)
            for value in df[
                df["Microbe_ID"] == neighbor_microbe_id
            ]["Drug_ID"].dropna().tolist()
        )

    common_drug_ids.discard(int(drug_id))

    common_neighbors = (
        df[df["Drug_ID"].isin(common_drug_ids)]
        [["Drug_ID", "Drug"]]
        .drop_duplicates()
        .sort_values("Drug_ID")
        .head(8)
    )

    return {
        "type": "graph_context",
        "drug_neighbors": [
            {
                "id": int(row["Microbe_ID"]),
                "name": str(row["Microbe"]),
            }
            for _, row in drug_neighbors.iterrows()
        ],
        "microbe_neighbors": [
            {
                "id": int(row["Drug_ID"]),
                "name": str(row["Drug"]),
            }
            for _, row in microbe_neighbors.iterrows()
        ],
        "common_neighbors": [
            {
                "id": int(row["Drug_ID"]),
                "name": str(row["Drug"]),
            }
            for _, row in common_neighbors.iterrows()
        ],
        "drug_neighbor_count": len(drug_neighbor_ids),
        "microbe_neighbor_count": len(microbe_neighbor_ids),
        "common_neighbor_count": len(common_drug_ids),
    }


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
def predict(
    request: PredictionRequest,
    authorization: str | None = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization token required.",
        )

    token = authorization[7:].strip()
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    user_id = payload.get("user_id")

    if not isinstance(user_id, int):
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload.",
        )

    try:
        result = predict_interaction(
            request.drug_id,
            request.microbe_id,
        )

        probability = float(
            result.get("probability", 0.0)
        )

        result["explanation"] = build_graph_explanation(
            request.drug_id,
            request.microbe_id,
        )

        drug_rows = df.loc[
            df["Drug_ID"] == request.drug_id,
            "Drug",
        ]

        microbe_rows = df.loc[
            df["Microbe_ID"] == request.microbe_id,
            "Microbe",
        ]

        drug_name = (
            str(drug_rows.iloc[0])
            if not drug_rows.empty
            else f"Drug {request.drug_id}"
        )

        microbe_name = (
            str(microbe_rows.iloc[0])
            if not microbe_rows.empty
            else f"Microbe {request.microbe_id}"
        )

        from datetime import datetime, timezone
        from src.auth.database import get_connection

        connection = get_connection()

        try:
            connection.execute(
                """
                INSERT INTO prediction_history (
                    user_id,
                    drug_id,
                    microbe_id,
                    drug_name,
                    microbe_name,
                    prediction,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    request.drug_id,
                    request.microbe_id,
                    drug_name,
                    microbe_name,
                    probability,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            connection.commit()

        finally:
            connection.close()

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
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

            "accuracy": 0.7058,

            "precision": 0.7353,

            "recall": 0.6429,

            "f1": 0.6860,

            "roc_auc": 0.7797,
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

                "accuracy": 0.7058,
            "precision": 0.7353,
            "recall": 0.6429,
            "f1": 0.6860,
            "roc_auc": 0.7797,
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
    graph_df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "cleaned_data.csv"
    )

    graph_df = graph_df.dropna(
        subset=["Name", "Microbe"]
    )

    # --------------------------------------------------------
    # Unique drugs and microbes
    # --------------------------------------------------------

    drugs = (
        graph_df["Name"]
        .drop_duplicates()
        .tolist()
    )

    microbes = (
        graph_df["Microbe"]
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    nodes = []

    drug_to_node = {}

    for index, drug in enumerate(drugs):
        node_id = f"drug_{index}"

        drug_to_node[drug] = node_id

        nodes.append({
            "id": node_id,
            "type": "drug",
            "name": str(drug),
        })

    microbe_to_node = {}

    for index, microbe in enumerate(microbes):
        node_id = f"microbe_{index}"

        microbe_to_node[microbe] = node_id

        nodes.append({
            "id": node_id,
            "type": "microbe",
            "name": str(microbe),
        })

    # --------------------------------------------------------
    # Interaction links
    # --------------------------------------------------------

    interactions = (
        graph_df[["Name", "Microbe"]]
        .drop_duplicates()
    )

    links = []

    for _, row in interactions.iterrows():
        drug = row["Name"]
        microbe = row["Microbe"]

        links.append({
            "source": drug_to_node[drug],
            "target": microbe_to_node[microbe],
        })

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "drug_nodes": len(drugs),
        "microbe_nodes": len(microbes),
        "total_nodes": len(nodes),
        "interactions": len(links),
    }

    return {
        "nodes": nodes,
        "links": links,
        "statistics": stats,
    }

# ============================================================
# Authentication
# ============================================================

class SignupRequest(BaseModel):
    username: str
    password: str
    recovery_code: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RecoveryRequest(BaseModel):
    username: str
    recovery_code: str


@app.post("/auth/signup")
def signup(request: SignupRequest):
    try:
        user = create_user(
            username=request.username,
            password=request.password,
            recovery_code=request.recovery_code,
        )

        token = create_token(
            user_id=user["id"],
            username=user["username"],
        )

        return {
            "success": True,
            "message": "Account created successfully.",
            "user": user,
            "token": token,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/auth/login")
def login(request: LoginRequest):
    user = authenticate_user(
        username=request.username,
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    token = create_token(
        user_id=user["id"],
        username=user["username"],
    )

    return {
        "success": True,
        "message": "Login successful.",
        "user": user,
        "token": token,
    }


@app.post("/auth/recovery/verify")
def verify_recovery(request: RecoveryRequest):
    valid = verify_recovery_code(
        username=request.username,
        recovery_code=request.recovery_code,
    )

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or recovery code.",
        )

    return {
        "success": True,
        "message": "Recovery code verified.",
    }

# ============================================================
# Prediction History
# ============================================================

class HistoryCreateRequest(BaseModel):
    drug_id: int | None = None
    microbe_id: int | None = None
    drug_name: str | None = None
    microbe_name: str | None = None
    prediction: float


@app.post("/history")
def save_history(
    request: HistoryCreateRequest,
    authorization: str | None = Header(default=None),
):
    from datetime import datetime, timezone
    from src.auth.database import get_connection

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization token required.",
        )

    token = authorization[7:].strip()
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    token_user_id = payload.get("user_id")

    if not isinstance(token_user_id, int):
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload.",
        )

    connection = get_connection()

    try:
        user = connection.execute(
            "SELECT id FROM users WHERE id = ?",
            (token_user_id,),
        ).fetchone()

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                user_id,
                drug_id,
                microbe_id,
                drug_name,
                microbe_name,
                prediction,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_user_id,
                request.drug_id,
                request.microbe_id,
                request.drug_name,
                request.microbe_name,
                request.prediction,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        connection.commit()

        return {
            "success": True,
            "history_id": cursor.lastrowid,
            "user_id": token_user_id,
        }

    finally:
        connection.close()



@app.delete("/history")
def clear_history(
    authorization: str | None = Header(default=None),
):
    from src.auth.database import get_connection

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization token required.",
        )

    token = authorization[7:].strip()
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    user_id = payload.get("user_id")

    if not isinstance(user_id, int):
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload.",
        )

    connection = get_connection()

    try:
        connection.execute(
            "DELETE FROM prediction_history WHERE user_id = ?",
            (user_id,),
        )
        connection.commit()

        return {
            "success": True,
            "message": "Prediction history cleared.",
        }

    finally:
        connection.close()


@app.get("/history")
def get_history(
    authorization: str | None = Header(default=None),
):
    from src.auth.database import get_connection

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization token required.",
        )

    token = authorization[7:].strip()
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    user_id = payload.get("user_id")

    if not isinstance(user_id, int):
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload.",
        )

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                drug_id,
                microbe_id,
                drug_name,
                microbe_name,
                prediction,
                created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

        return {
            "success": True,
            "user_id": user_id,
            "history": [dict(row) for row in rows],
        }

    finally:
        connection.close()


# ============================================================
# Password Reset
# ============================================================

class PasswordResetRequest(BaseModel):
    username: str
    recovery_code: str
    new_password: str


@app.post("/auth/password/reset")
def password_reset(request: PasswordResetRequest):

    valid = verify_recovery_code(
        username=request.username,
        recovery_code=request.recovery_code,
    )

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or recovery code.",
        )

    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters.",
        )

    success = reset_password(
        username=request.username,
        new_password=request.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    return {
        "success": True,
        "message": "Password reset successfully.",
    }
