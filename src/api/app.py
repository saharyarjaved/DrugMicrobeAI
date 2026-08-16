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

from src.auth.database import init_db


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Drug-Microbe AI API",
    description="HaGAT based drug-microbe interaction prediction API",
    version="1.0.0",
)

init_db()

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

    print('========== PREDICT REQUEST START ==========', flush=True)
    print(f'drug_id={request.drug_id}, microbe_id={request.microbe_id}', flush=True)

    try:

        print('Calling predict_interaction...', flush=True)

        result = predict_interaction(
            request.drug_id,
            request.microbe_id,
        )

        print('predict_interaction completed successfully', flush=True)
        print(f'Result type: {type(result).__name__}', flush=True)

        # ====================================================
        # Resolve Drug / Microbe Names
        # ====================================================

        drug_rows = df[
            df["Drug_ID"] == request.drug_id
        ][["Drug_ID", "Drug"]].dropna().drop_duplicates()

        microbe_rows = df[
            df["Microbe_ID"] == request.microbe_id
        ][["Microbe_ID", "Microbe"]].dropna().drop_duplicates()

        if drug_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Drug ID not found: {request.drug_id}",
            )

        if microbe_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Microbe ID not found: {request.microbe_id}",
            )

        drug_name = str(
            drug_rows.iloc[0]["Drug"]
        )

        microbe_name = str(
            microbe_rows.iloc[0]["Microbe"]
        )

        probability = float(
            result["probability"]
        )

        # ====================================================
        # Upgraded API Response
        # ====================================================

        response = {
            "drug": {
                "id": int(request.drug_id),
                "name": drug_name,
            },
            "microbe": {
                "id": int(request.microbe_id),
                "name": microbe_name,
            },
            "probability": round(
                probability,
                6
            ),
            "probability_percent": round(
                probability * 100,
                2
            ),
            "prediction": result["prediction"],
        }

        print(
            f"Drug: {drug_name} "
            f"(ID={request.drug_id})",
            flush=True
        )

        print(
            f"Microbe: {microbe_name} "
            f"(ID={request.microbe_id})",
            flush=True
        )

        print(
            f"Probability: {probability:.6f}",
            flush=True
        )

        print(
            f"Prediction: {result['prediction']}",
            flush=True
        )

        return response

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        import traceback
        print('========== PREDICT ERROR ==========', flush=True)
        traceback.print_exc()
        print('===================================', flush=True)

        raise HTTPException(
            status_code=500,
            detail=f'{type(error).__name__}: {error}',
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

    except Exception as error:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Signup database/server error: {type(error).__name__}: {error}",
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

    # --------------------------------------------------------
    # Check Authorization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save Prediction
    # --------------------------------------------------------

    connection = get_connection()

    try:
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
                user_id,
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
            "user_id": user_id,
        }

    except Exception as error:
        connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:
        connection.close()


# ============================================================
# Get Prediction History
# ============================================================

@app.get("/history")
def get_history(
    authorization: str | None = Header(default=None),
):
    from src.auth.database import get_connection

    # --------------------------------------------------------
    # Check Authorization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Get User History
    # --------------------------------------------------------

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                user_id,
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

        history = []

        for row in rows:
            history.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "drug_id": row["drug_id"],
                "microbe_id": row["microbe_id"],
                "drug_name": row["drug_name"],
                "microbe_name": row["microbe_name"],
                "prediction": row["prediction"],
                "created_at": row["created_at"],
            })

        return {
            "success": True,
            "user_id": user_id,
            "total": len(history),
            "history": history,
        }

    finally:
        connection.close()


# ============================================================
# Clear Prediction History
# ============================================================

@app.delete("/history")
def clear_history(
    authorization: str | None = Header(default=None),
):
    from src.auth.database import get_connection

    # --------------------------------------------------------
    # Check Authorization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Delete Only This User's History
    # --------------------------------------------------------

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM prediction_history
            WHERE user_id = ?
            """,
            (user_id,),
        )

        connection.commit()

        return {
            "success": True,
            "message": "Prediction history cleared.",
            "deleted_records": cursor.rowcount,
            "user_id": user_id,
        }

    except Exception as error:
        connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:
        connection.close()
