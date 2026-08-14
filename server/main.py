import os
import sys
from pathlib import Path

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app as base_app

app = FastAPI()

app.mount("/svc/api", base_app)

__all__ = ["app"]
