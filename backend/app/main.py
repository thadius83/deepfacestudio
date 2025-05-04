"""
FastAPI service exposing DeepFace functionality.

Endpoints
---------
POST /reference/{label}   – add one or more reference images under a label
POST /identify            – find every face in a group photo and match to refs
POST /compare             – verify two single‑person photos
POST /analyze             – age / gender / emotion / race for each face
"""

from fastapi import FastAPI
from . import config
from .routes import router

app = FastAPI(title="DeepFace‑API", version="1.0.0")

# Include all routes
app.include_router(router)
